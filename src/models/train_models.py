"""Consolidated trainer: Elo + ML pipeline

Usage:
    python src/models/train_models.py --data data/processed/unified_matches_dataset.csv --out models --calibrate

This script computes sequential Elo ratings, augments matches with Elo and derived features,
and trains an XGBoost classifier. It saves model artifacts under the `models/` directory so
the rest of the project (predictor/evaluator) can load them.
"""

import os
import json
from datetime import datetime, timezone
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import re
import warnings


def try_parse_dates(series):
    """Robust date parsing: try several explicit formats first, then fall back
    to dayfirst/daylast parsing while suppressing pandas format-inference warnings.
    Returns a DatetimeIndex/Series with coerced NaT for unparsable values.
    """
    s = series.astype(str).replace({'nan': '', 'None': ''})
    formats = [
        '%Y-%m-%d', '%Y/%m/%d',
        '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%Y',
        '%m/%d/%Y', '%m-%d-%Y',
        '%d %b %Y', '%d %B %Y'
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(s, format=fmt, errors='coerce')
        except Exception:
            parsed = pd.to_datetime(s, format=fmt, errors='coerce')
        if parsed.notna().sum() > 0:
            return parsed

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parsed = pd.to_datetime(s, dayfirst=True, errors='coerce', infer_datetime_format=False)
        if parsed.notna().sum() > 0:
            return parsed
        parsed = pd.to_datetime(s, dayfirst=False, errors='coerce', infer_datetime_format=False)
        return parsed

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


DEFAULT_DATA_CANDIDATES = [
    os.path.join('data', 'processed', 'unified_matches_dataset.csv'),
    os.path.join('data', 'clean', 'clean_ml_dataset.csv'),
    os.path.join('data', 'predictions', 'predictions_20240101_20261231.csv'),
]


def load_matches(path=None):
    # If an explicit path is provided, prefer that
    if path:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # normalize date-like columns
            for cand in ['date', 'Date', 'match_date', 'MatchDate', 'kickoff', 'Kickoff', 'kickoff_time', 'MatchDay', 'matchday']:
                if cand in df.columns:
                    df['date'] = try_parse_dates(df[cand])
                    break
            # if no date found, try to infer from any column named like date
            if 'date' not in df.columns:
                # try parsing any column that looks like a date
                for col in df.columns:
                    if 'date' in col.lower() or 'day' in col.lower():
                        try:
                            df['date'] = try_parse_dates(df[col])
                            break
                        except Exception:
                            continue
            # if still no date values, create a synthetic increasing date index
            if 'date' not in df.columns or df['date'].isna().all():
                df['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(range(len(df)), unit='D')
            return df
        # allow passing a glob-like path
        import glob
        matches = glob.glob(path)
        if matches:
            return pd.read_csv(matches[0])

    # Try the canonical candidate list
    for p in DEFAULT_DATA_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # same normalization as above
            for cand in ['date', 'Date', 'match_date', 'MatchDate', 'kickoff', 'Kickoff', 'kickoff_time', 'MatchDay', 'matchday']:
                if cand in df.columns:
                    df['date'] = try_parse_dates(df[cand])
                    break
            if 'date' not in df.columns:
                for col in df.columns:
                    if 'date' in col.lower() or 'day' in col.lower():
                        try:
                            df['date'] = try_parse_dates(df[col])
                            break
                        except Exception:
                            continue
            if 'date' not in df.columns or df['date'].isna().all():
                df['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(range(len(df)), unit='D')
            return df

    # Fall back to searching the data/ tree for likely CSVs and concatenate them
    import glob
    candidates = glob.glob(os.path.join('data', '**', '*.csv'), recursive=True)
    if not candidates:
        raise FileNotFoundError('No matches dataset found; provide --data')

    # read and concatenate all CSVs under data/ (this uses all sources), but
    # exclude rows that fall inside the 2025-2026 season window so training
    # data doesn't accidentally include evaluation-season matches.
    parts = []
    for c in candidates:
        try:
            tmp = pd.read_csv(c)
        except Exception:
            continue
        # attempt to normalize date-like columns for each file
        for cand in ['date', 'Date', 'match_date', 'MatchDate', 'kickoff', 'Kickoff', 'kickoff_time', 'MatchDay', 'matchday']:
            if cand in tmp.columns:
                tmp['date'] = try_parse_dates(tmp[cand])
                break
        if 'date' not in tmp.columns:
            for col in tmp.columns:
                if 'date' in col.lower() or 'day' in col.lower():
                    try:
                        tmp['date'] = try_parse_dates(tmp[col])
                        break
                    except Exception:
                        continue
        parts.append(tmp)

    if not parts:
        raise FileNotFoundError('No readable CSVs found under data/')

    df = pd.concat(parts, ignore_index=True, sort=False)

    # drop rows that fall into the 2025-2026 season range (Aug 2025 - Jul 2026)
    try:
        if 'date' in df.columns:
            df['date'] = try_parse_dates(df['date'])
            start = pd.to_datetime('2025-08-01')
            end = pd.to_datetime('2026-07-31')
            df = df[~df['date'].between(start, end)]
    except Exception:
        # if date normalization fails, proceed with the concatenated DF
        pass

    # ensure we have a date column
    if 'date' not in df.columns or df['date'].isna().all():
        df['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(range(len(df)), unit='D')

    return df


def compute_elo(df, k=30, start_rating=1500, home_field_adv=100, base=400, home_col='HomeTeam', away_col='AwayTeam', date_col='date'):
    df = df.copy()
    # ensure date
    if date_col in df.columns:
        df[date_col] = try_parse_dates(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    teams = defaultdict(lambda: start_rating)
    home_ratings = []
    away_ratings = []
    exp_home_probs = []

    for _, row in df.iterrows():
        home = row.get(home_col)
        away = row.get(away_col)
        if pd.isna(home) or pd.isna(away):
            home_ratings.append(start_rating)
            away_ratings.append(start_rating)
            exp_home_probs.append(0.5)
            continue

        r_home = teams[home]
        r_away = teams[away]
        r_home_eff = r_home + home_field_adv
        exp_home = 1.0 / (1.0 + 10 ** ((r_away - r_home_eff) / base))

        home_ratings.append(r_home)
        away_ratings.append(r_away)
        exp_home_probs.append(exp_home)

        # determine result (prefer FTR then goals)
        result = None
        if 'FTR' in row and pd.notna(row['FTR']):
            result = row['FTR']
        elif 'FTHG' in row and 'FTAG' in row and pd.notna(row['FTHG']) and pd.notna(row['FTAG']):
            try:
                hg = int(row['FTHG']); ag = int(row['FTAG'])
                if hg > ag:
                    result = 'H'
                elif hg == ag:
                    result = 'D'
                else:
                    result = 'A'
            except Exception:
                result = None

        if result in ('H', 'D', 'A'):
            s_home = 1.0 if result == 'H' else (0.5 if result == 'D' else 0.0)
            delta_home = k * (s_home - exp_home)
            delta_away = -delta_home
            teams[home] = teams[home] + delta_home
            teams[away] = teams[away] + delta_away

    out = df.copy()
    out['home_elo'] = home_ratings
    out['away_elo'] = away_ratings
    out['elo_diff'] = out['home_elo'] - out['away_elo']
    out['elo_exp_home'] = exp_home_probs
    return out, dict(teams)


def create_features(df):
    df = df.copy()
    # normalize date
    if 'date' in df.columns:
        df['date'] = try_parse_dates(df['date'])
    else:
        df['date'] = pd.NaT

    df['month'] = df['date'].dt.month.fillna(0).astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek.fillna(0).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['season_progress'] = df['month'].apply(lambda x: 1 if x >= 8 else (0.5 if x >= 6 else 0))

    # simple label encoding for teams and league
    encoders = {}
    for col, key in [('HomeTeam', 'home_team'), ('AwayTeam', 'away_team'), ('Div', 'league')]:
        if col in df.columns:
            le = LabelEncoder()
            try:
                df[f'{key}_enc'] = le.fit_transform(df[col].astype(str))
            except Exception:
                df[f'{key}_enc'] = 0
            encoders[key] = le

    # initialize a few advanced stats with safe defaults
    df['home_recent_form'] = 0.0
    df['away_recent_form'] = 0.0
    df['home_goals_avg'] = 0.0
    df['away_goals_avg'] = 0.0
    # weighted recent stats (exponential decay) -- give more importance to latest matches
    df['home_recent_weighted_form'] = 0.0
    df['away_recent_weighted_form'] = 0.0
    df['home_weighted_goals'] = 0.0
    df['away_weighted_goals'] = 0.0

    # compute simple team-level rolling averages using last N matches per team
    # we'll compute per-team last 5 matches averages for goals and simple form
    df = df.sort_values('date').reset_index(drop=True)
    # --- New time-since-last-match features ---
    # For each team, compute days since their previous match and attach to each row
    df['home_days_since_last'] = 0.0
    df['away_days_since_last'] = 0.0
    last_play = {}
    for idx, row in df.iterrows():
        d = row['date']
        home = str(row.get('HomeTeam'))
        away = str(row.get('AwayTeam'))
        # home
        if home in last_play and pd.notna(d) and pd.notna(last_play[home]):
            df.at[idx, 'home_days_since_last'] = (d - last_play[home]).days
        else:
            df.at[idx, 'home_days_since_last'] = 999.0
        # away
        if away in last_play and pd.notna(d) and pd.notna(last_play[away]):
            df.at[idx, 'away_days_since_last'] = (d - last_play[away]).days
        else:
            df.at[idx, 'away_days_since_last'] = 999.0
        # update last played
        last_play[home] = d
        last_play[away] = d
    for team in pd.concat([df['HomeTeam'].astype(str), df['AwayTeam'].astype(str)]).unique():
        team_mask = (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
        team_matches = df[team_mask]
        # iterate indexes in original df order
        for idx in team_matches.index:
            past = team_matches[team_matches.index < idx].tail(8)
            if past.empty:
                continue
            # compute goals for when team was home or away
            hg = 0.0; ag = 0.0; count = 0
            form_vals = []
            # we will compute weighted aggregates with exponential decay (most recent has highest weight)
            weighted_form_vals = []
            weighted_goals_vals = []
            # iterate past in chronological order (older -> newer)
            past_list = list(past.itertuples(index=False, name=None))
            # create weights so that the most recent has highest weight
            n = len(past_list)
            # decay parameter: smaller -> slower decay; tuned modestly
            decay = 0.6
            weights = [np.exp(-decay * (n - 1 - i)) for i in range(n)]  # i= n-1 -> most recent weight=1
            # normalize
            wsum = sum(weights) if sum(weights) != 0 else 1.0
            weights = [w / wsum for w in weights]
            for w, r in zip(weights, past_list):
                # r is a tuple of all columns in order; map by column names via past.columns
                # convert to Series for convenience
                r_ser = pd.Series(r, index=past.columns)
                if r_ser['HomeTeam'] == team:
                    if 'FTHG' in r_ser and not pd.isna(r_ser['FTHG']):
                        hg += float(r_ser['FTHG'])
                        ag += float(r_ser.get('FTAG', 0) or 0)
                else:
                    if 'FTAG' in r_ser and not pd.isna(r_ser['FTAG']):
                        hg += float(r_ser['FTAG'])
                        ag += float(r_ser.get('FTHG', 0) or 0)
                # form
                res = r_ser.get('FTR')
                if pd.notna(res):
                    if res == 'H' and r_ser['HomeTeam'] == team:
                        fv = 3
                    elif res == 'A' and r_ser['AwayTeam'] == team:
                        fv = 3
                    elif res == 'D':
                        fv = 1
                    else:
                        fv = 0
                    form_vals.append(fv)
                    weighted_form_vals.append(w * fv)
                # goals: use team's goals for that match
                gf = None
                if r_ser['HomeTeam'] == team:
                    if 'FTHG' in r_ser and not pd.isna(r_ser['FTHG']):
                        gf = float(r_ser['FTHG'])
                else:
                    if 'FTAG' in r_ser and not pd.isna(r_ser['FTAG']):
                        gf = float(r_ser['FTAG'])
                if gf is not None:
                    weighted_goals_vals.append(w * gf)
                count += 1
            if count > 0:
                avg_goals = hg / count
                avg_form = sum(form_vals) / len(form_vals) if form_vals else 0.0
                weighted_form = sum(weighted_form_vals) if weighted_form_vals else 0.0
                weighted_goals = sum(weighted_goals_vals) if weighted_goals_vals else 0.0
                # assign to the next occurrence rows (we'll set for any row where team plays)
                side = 'home' if df.loc[idx,'HomeTeam']==team else 'away'
                df.loc[idx, f"{side}_goals_avg"] = avg_goals
                df.loc[idx, f"{side}_recent_form"] = avg_form
                df.loc[idx, f"{side}_recent_weighted_form"] = weighted_form
                df.loc[idx, f"{side}_weighted_goals"] = weighted_goals

    # --- Head-to-head aggregates ---
    # Compute simple historical head-to-head stats (last N matches between the two teams)
    df['hh_home_wins'] = 0
    df['hh_away_wins'] = 0
    df['hh_draws'] = 0
    # we'll keep a dict keyed by frozenset({team1,team2}) storing list of past results tuples (home, away, result)
    hh = defaultdict(list)
    for idx, row in df.iterrows():
        home = str(row.get('HomeTeam'))
        away = str(row.get('AwayTeam'))
        if home == 'nan' or away == 'nan':
            continue
        key = (home, away)
        rev_key = (away, home)
        # collect last up to 8 head-to-head matches (both orders)
        past = []
        # look through recorded hh entries (store directional tuples)
        # gather matches where order matches or reversed
        for h,kres in hh.items():
            # h is tuple (hteam, ateam)
            if (h[0] == home and h[1] == away) or (h[0] == away and h[1] == home):
                past.extend(kres)
        # use most recent up to 8
        if past:
            last = past[-8:]
            hh_home = sum(1 for r in last if r[2] == 'H' and r[0] == home)
            hh_away = sum(1 for r in last if r[2] == 'A' and r[1] == away)
            hh_draws = sum(1 for r in last if r[2] == 'D')
            df.at[idx, 'hh_home_wins'] = hh_home
            df.at[idx, 'hh_away_wins'] = hh_away
            df.at[idx, 'hh_draws'] = hh_draws
        else:
            df.at[idx, 'hh_home_wins'] = 0
            df.at[idx, 'hh_away_wins'] = 0
            df.at[idx, 'hh_draws'] = 0

        # now record the current match into hh store for future rows (directional)
        res = None
        if 'FTR' in row and pd.notna(row['FTR']):
            res = row['FTR']
        elif 'FTHG' in row and 'FTAG' in row and pd.notna(row['FTHG']) and pd.notna(row['FTAG']):
            try:
                hg = int(row['FTHG']); ag = int(row['FTAG'])
                if hg > ag:
                    res = 'H'
                elif hg == ag:
                    res = 'D'
                else:
                    res = 'A'
            except Exception:
                res = None
        hh[(home, away)].append((home, away, res))

    # --- Additional features: Elo momentum, hh_has_history, form difference, interactions ---
    # Elo momentum: difference between current elo and elo from 1 and 3 matches ago for each side
    df['home_elo_mom_1'] = 0.0
    df['away_elo_mom_1'] = 0.0
    df['home_elo_mom_3'] = 0.0
    df['away_elo_mom_3'] = 0.0
    # head-to-head history presence flag
    df['hh_has_history'] = 0
    # form difference (home_recent_weighted_form - away_recent_weighted_form)
    df['home_form_diff'] = 0.0

    # Build a lookup of previous elo ratings per team as we iterate
    prev_elos = defaultdict(list)  # team -> list of elos in chronological order
    for idx, row in df.iterrows():
        home = str(row.get('HomeTeam'))
        away = str(row.get('AwayTeam'))
        # assign elo momentum if available
        h_elo = row.get('home_elo', None)
        a_elo = row.get('away_elo', None)
        if h_elo is not None and not pd.isna(h_elo):
            hist = prev_elos.get(home, [])
            if len(hist) >= 1:
                df.at[idx, 'home_elo_mom_1'] = h_elo - hist[-1]
            else:
                df.at[idx, 'home_elo_mom_1'] = 0.0
            if len(hist) >= 3:
                df.at[idx, 'home_elo_mom_3'] = h_elo - hist[-3]
            else:
                df.at[idx, 'home_elo_mom_3'] = 0.0
        if a_elo is not None and not pd.isna(a_elo):
            ahist = prev_elos.get(away, [])
            if len(ahist) >= 1:
                df.at[idx, 'away_elo_mom_1'] = a_elo - ahist[-1]
            else:
                df.at[idx, 'away_elo_mom_1'] = 0.0
            if len(ahist) >= 3:
                df.at[idx, 'away_elo_mom_3'] = a_elo - ahist[-3]
            else:
                df.at[idx, 'away_elo_mom_3'] = 0.0

        # hh_has_history: check if there are any recorded head-to-head entries for these teams
        # We can use the hh dict built above; check either direction
        try:
            if hh.get((home, away)) or hh.get((away, home)):
                df.at[idx, 'hh_has_history'] = 1
            else:
                df.at[idx, 'hh_has_history'] = 0
        except Exception:
            df.at[idx, 'hh_has_history'] = 0

        # form diff
        df.at[idx, 'home_form_diff'] = float(row.get('home_recent_weighted_form', 0.0)) - float(row.get('away_recent_weighted_form', 0.0))

        # update prev_elos store AFTER computing momentum for the current row
        if h_elo is not None and not pd.isna(h_elo):
            prev_elos[home].append(float(h_elo))
        if a_elo is not None and not pd.isna(a_elo):
            prev_elos[away].append(float(a_elo))

    # Interaction features
    df['elo_x_rest'] = 0.0
    df['elo_x_formdiff'] = 0.0
    if 'elo_diff' in df.columns:
        df['elo_x_rest'] = df['elo_diff'] * df.get('home_days_since_last', 0).astype(float)
        df['elo_x_formdiff'] = df['elo_diff'] * df.get('home_form_diff', 0).astype(float)

    # after many item assignments the frame may be fragmented; make a physical copy
    # to avoid PerformanceWarning and SettingWithCopyWarning in later ops
    df = df.copy()

    # fill numeric NA values only (avoid assigning 0 to datetime columns)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)
    # clip extreme days-since-last sentinel to a large but finite value
    if 'home_days_since_last' in df.columns:
        df['home_days_since_last'] = df['home_days_since_last'].clip(0, 365)
    if 'away_days_since_last' in df.columns:
        df['away_days_since_last'] = df['away_days_since_last'].clip(0, 365)
    return df, encoders


def prepare_X_y(df, encoders=None, fit_scaler=True, scaler=None):
    # operate on a copy to avoid SettingWithCopyWarning when assigning new columns
    df = df.copy()
    # ensure mapping
    mapping = {'H': 0, 'D': 1, 'A': 2}
    if 'FTR' not in df.columns:
        # try infer from goals
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            def to_ftr(r):
                try:
                    hg = float(r['FTHG']); ag = float(r['FTAG'])
                    if hg > ag: return 'H'
                    if hg == ag: return 'D'
                    return 'A'
                except Exception:
                    return None
            df['FTR'] = df.apply(to_ftr, axis=1)

    df = df.dropna(subset=['FTR'])
    df['target'] = df['FTR'].map(mapping)

    # drop rows with unmapped/missing target before casting to int
    before = len(df)
    df = df[df['target'].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f'Dropped {dropped} rows with missing target values')

    # Build a comprehensive feature list:
    # - any encoded categorical columns (ending with _enc)
    # - standard temporal columns
    # - elo and derived elo features
    # - rolling stats and any numeric columns except known labels/goals
    candidate_features = []
    # encoded categories
    candidate_features += [c for c in df.columns if str(c).endswith('_enc')]
    # temporal
    for c in ['month', 'day_of_week', 'is_weekend', 'season_progress']:
        if c in df.columns:
            candidate_features.append(c)
    # elo-based
    for c in ['home_elo', 'away_elo', 'elo_diff', 'elo_exp_home']:
        if c in df.columns:
            candidate_features.append(c)
    # rolling stats
    for c in ['home_recent_form', 'away_recent_form', 'home_goals_avg', 'away_goals_avg']:
        if c in df.columns:
            candidate_features.append(c)

    # include any other numeric columns that look useful, excluding raw goal/result columns
    exclude = set(['FTHG', 'FTAG', 'FTR', 'target'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if c not in candidate_features and c not in exclude:
            candidate_features.append(c)

    features = [c for c in candidate_features if c in df.columns]
    X = df[features].copy()

    # sanitize column names and impute
    # convert to strings and replace any non-alphanumeric/_ char with '_'
    raw_cols = [str(c) for c in X.columns]
    clean_cols = []
    seen = {}
    for i, col in enumerate(raw_cols):
        col2 = re.sub(r'[^0-9A-Za-z_]', '_', col)
        if not col2:
            col2 = f'col_{i}'
        # ensure uniqueness
        if col2 in seen:
            seen[col2] += 1
            col2 = f"{col2}_{seen[col2]}"
        else:
            seen[col2] = 0
        clean_cols.append(col2)
    X.columns = clean_cols
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # scale numeric features
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # allow passing a fitted scaler for consistent transform on holdout data
    if num_cols:
        if fit_scaler:
            scaler = scaler or StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])
        else:
            if scaler is None:
                raise ValueError('scaler must be provided when fit_scaler=False')
            X[num_cols] = scaler.transform(X[num_cols])

    y = df['target'].astype(int)
    return X, y, scaler


def train_model(X, y, n_jobs=-1):
    # expanded parameter grid to improve the chance of finding a better model
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.04, 0.1],
        'max_depth': [3, 4, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb = XGBClassifier(objective='multi:softprob', random_state=42, n_jobs=n_jobs, eval_metric='mlogloss')
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(xgb, param_grid, cv=tscv, scoring='accuracy', n_jobs=n_jobs, verbose=1)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y)
    gs.fit(X, y, sample_weight=sample_weight)
    best = gs.best_estimator_
    return best, gs.best_params_, gs.best_score_


def fit_calibration(df):
    mapping = {'H': 0, 'D': 1, 'A': 2}
    if 'FTR' not in df.columns:
        raise ValueError('FTR required for calibration')
    cal_df = df.dropna(subset=['FTR'])
    y = cal_df['FTR'].map(mapping).astype(int)
    X = cal_df[['elo_diff']].values
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    clf.fit(X, y)
    return clf


def main():
    p = argparse.ArgumentParser(description='Train Elo-augmented ML model')
    p.add_argument('--data', help='Path to matches CSV')
    p.add_argument('--out', default='models', help='Output models directory')
    p.add_argument('--k', type=float, default=30.0, help='Elo K-factor')
    p.add_argument('--home-adv', type=float, default=100.0, help='Home advantage in Elo points')
    p.add_argument('--calibrate', action='store_true', help='Fit a logistic calibration on elo_diff')
    args = p.parse_args()

    df = load_matches(args.data)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loaded matches: {len(df)} rows')

    elo_df, teams = compute_elo(df, k=args.k, start_rating=1500, home_field_adv=args.home_adv)
    # Save Elo artifacts with a timestamped filename plus canonical seasoned filenames
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    elo_csv_ts = os.path.join(out_dir, f'elo_matches_{ts}.csv')
    elo_df.to_csv(elo_csv_ts, index=False)

    # infer season year from data if possible, fallback to current year
    try:
        if 'date' in elo_df.columns:
            season_year = int(elo_df['date'].dt.year.max())
        else:
            season_year = datetime.utcnow().year
    except Exception:
        season_year = datetime.utcnow().year

    elo_csv_season = os.path.join(out_dir, f'elo_{season_year}_matches.csv')
    elo_df.to_csv(elo_csv_season, index=False)
    # also keep a stable name
    elo_csv_latest = os.path.join(out_dir, 'elo_matches.csv')
    elo_df.to_csv(elo_csv_latest, index=False)

    ratings_path_ts = os.path.join(out_dir, f'elo_ratings_{ts}.json')
    with open(ratings_path_ts, 'w', encoding='utf-8') as f:
        json.dump(teams, f, indent=2)
    ratings_path_season = os.path.join(out_dir, f'elo_{season_year}_ratings.json')
    with open(ratings_path_season, 'w', encoding='utf-8') as f:
        json.dump(teams, f, indent=2)
    # stable filename for other tools
    ratings_path_latest = os.path.join(out_dir, 'elo_ratings.json')
    with open(ratings_path_latest, 'w', encoding='utf-8') as f:
        json.dump(teams, f, indent=2)

    print('Saved Elo artifacts:', elo_csv_ts, ratings_path_ts, 'and seasoned files', elo_csv_season, ratings_path_season)

    df_feats, encoders = create_features(elo_df)
    X, y, scaler = prepare_X_y(df_feats)

    if X.empty:
        raise RuntimeError('No training data after feature preparation')

    print('Training model on', len(X), 'samples with', X.shape[1], 'features')
    model, best_params, best_score = train_model(X, y, n_jobs=-1)
    print('Best params:', best_params)
    print('CV score:', best_score)

    # evaluate on train set (self-eval)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f'Train accuracy: {acc:.3%}')
    print(classification_report(y, preds, target_names=['Home Win', 'Draw', 'Away Win']))

    # Save artifacts.
    # Produce a canonical "with_elo" bundle so the predictor picks the matching model/scaler/encoders.
    model_path_with = os.path.join(out_dir, 'ensemble_with_elo.joblib')
    enc_path_with = os.path.join(out_dir, 'feature_encoders_with_elo.joblib')
    scaler_path_with = os.path.join(out_dir, 'scaler_with_elo.joblib')

    # Save the canonical artifacts
    joblib.dump(model, model_path_with)
    joblib.dump(encoders, enc_path_with)
    joblib.dump(scaler, scaler_path_with)

    # Also keep legacy filenames for backward compatibility (overwrite)
    model_path_legacy = os.path.join(out_dir, 'ensemble_model.joblib')
    enc_path_legacy = os.path.join(out_dir, 'feature_encoders.joblib')
    scaler_path_legacy = os.path.join(out_dir, 'scaler.joblib')
    joblib.dump(model, model_path_legacy)
    joblib.dump(encoders, enc_path_legacy)
    joblib.dump(scaler, scaler_path_legacy)

    print('Saved model artifacts to', out_dir)

    if args.calibrate:
        try:
            calib = fit_calibration(elo_df)
            calib_path = os.path.join(out_dir, 'elo_calib_model.joblib')
            joblib.dump(calib, calib_path)
            print('Saved calibration model to', calib_path)
        except Exception as e:
            print('Calibration failed:', e)


if __name__ == '__main__':
    main()

