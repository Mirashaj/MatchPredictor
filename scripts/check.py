"""Combined evaluation and analysis helper.

Usage examples:
  python scripts/check.py --evaluate 2024,2025    # evaluate seasons and save CSVs
  python scripts/check.py --analyze predictions/evaluations/eval_2024_2025.csv
  python scripts/check.py --evaluate 2024 --analyze predictions/evaluations/eval_2024_2025.csv
"""
import os
import sys
import glob
import joblib
import pandas as pd
from datetime import datetime
from collections import defaultdict
import importlib.util
import contextlib
import argparse
import subprocess


def load_match_predictor(repo_root, quiet=False):
    mp_path = os.path.join(repo_root, 'src', 'predictions', 'match_predictor.py')
    spec = importlib.util.spec_from_file_location('match_predictor', mp_path)
    mod = importlib.util.module_from_spec(spec)
    # suppress noisy prints from module/class when quiet requested
    if quiet:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                spec.loader.exec_module(mod)
    else:
        spec.loader.exec_module(mod)
    MatchPredictor = getattr(mod, 'MatchPredictor')
    if quiet:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                predictor = MatchPredictor(models_dir=os.path.join(repo_root, 'models'), quiet=True)
    else:
        predictor = MatchPredictor(models_dir=os.path.join(repo_root, 'models'), quiet=False)

    # try loading alternative model files if Dummy is active
    if predictor.best_model_name == 'Dummy':
        extra_candidates = [
            'ensemble_with_elo.joblib', 'ensemble_model.joblib', 'xgboost_model.joblib',
            'elo_calib_model.joblib'
        ]
        for fname in extra_candidates:
            path = os.path.join(repo_root, 'models', fname)
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    name = fname.replace('.joblib', '')
                    predictor.models[name] = model
                    predictor.best_model_name = name
                    break
                except Exception:
                    continue

    return predictor


class SimpleEncoder:
    def __init__(self, classes):
        self.classes_ = sorted(list(classes))
        self._index = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, arr):
        out = []
        for v in arr:
            if v in self._index:
                out.append(self._index[v])
            else:
                out.append(len(self.classes_) // 2)
        return out


def build_team_stats(files):
    stats = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if 'hometeam' not in cols or 'awayteam' not in cols:
            continue
        has_ftr = 'ftr' in cols
        has_scores = 'fthg' in cols and 'ftag' in cols
        for _, r in df.iterrows():
            try:
                home = str(r[cols['hometeam']])
                away = str(r[cols['awayteam']])
            except Exception:
                continue
            if home == '' or away == '':
                continue
            for t in (home, away):
                if t not in stats:
                    stats[t] = {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0.0, 'goals_against': 0.0}
            res = None
            gf = ga = None
            if has_ftr:
                res = str(r.get(cols['ftr'], '')).strip()
            elif has_scores:
                try:
                    gf = float(r.get(cols['fthg'], 0))
                    ga = float(r.get(cols['ftag'], 0))
                    if gf > ga:
                        res = 'H'
                    elif gf == ga:
                        res = 'D'
                    else:
                        res = 'A'
                except Exception:
                    res = ''
            else:
                res = ''
            if gf is None or ga is None:
                try:
                    gf = float(r.get(cols.get('fthg', ''), 0))
                    ga = float(r.get(cols.get('ftag', ''), 0))
                except Exception:
                    gf = ga = 0.0
            stats[home]['matches'] += 1
            stats[away]['matches'] += 1
            stats[home]['goals_for'] += gf if gf is not None else 0.0
            stats[home]['goals_against'] += ga if ga is not None else 0.0
            stats[away]['goals_for'] += ga if ga is not None else 0.0
            stats[away]['goals_against'] += gf if gf is not None else 0.0
            if res == 'H':
                stats[home]['wins'] += 1
                stats[away]['losses'] += 1
            elif res == 'A':
                stats[away]['wins'] += 1
                stats[home]['losses'] += 1
            elif res == 'D':
                stats[home]['draws'] += 1
                stats[away]['draws'] += 1

    for t, v in stats.items():
        m = v['matches'] if v['matches'] else 1
        v['win_rate'] = v['wins'] / m
        v['draw_rate'] = v['draws'] / m
        v['loss_rate'] = v['losses'] / m
        v['goals_for_avg'] = v['goals_for'] / m
        v['goals_against_avg'] = v['goals_against'] / m

    return stats


def fallback_predict(home, away, league, match_date, season_start, team_stats):
    h = team_stats.get(str(home), None)
    a = team_stats.get(str(away), None)
    base = {'Home Win': 0.33, 'Draw': 0.34, 'Away Win': 0.33}
    if not h and not a:
        return {'prediction': 'Draw', 'probabilities': base, 'model_used': 'fallback'}
    def score(team):
        if not team:
            return 0.0
        return team.get('win_rate', 0.0) * 0.6 + (team.get('goals_for_avg', 0.0) - team.get('goals_against_avg', 0.0)) * 0.2 + team.get('draw_rate', 0.0) * 0.2
    hs = score(h)
    as_ = score(a)
    hs += 0.05
    home_prob = max(min(0.5 + (hs - as_) * 0.25, 0.95), 0.01)
    away_prob = max(min(0.5 + (as_ - hs) * 0.25, 0.95), 0.01)
    draw_prob = 1.0 - (home_prob + away_prob)
    if draw_prob < 0:
        s = home_prob + away_prob
        home_prob /= s
        away_prob /= s
        draw_prob = 0.01
    probs = {'Home Win': round(home_prob, 3), 'Draw': round(draw_prob, 3), 'Away Win': round(away_prob, 3)}
    pred = max(probs.items(), key=lambda x: x[1])[0]
    return {'prediction': pred, 'probabilities': probs, 'model_used': 'fallback'}


def find_data_files(repo_root):
    pattern = os.path.join(repo_root, 'data', '**', '*.csv')
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if os.path.normpath(os.path.join(repo_root, 'data', 'predictions')) not in os.path.normpath(f)]
    return files


def infer_season(date):
    if date.month >= 7:
        return date.year
    else:
        return date.year - 1


def map_div_to_league(div):
    m = {'E0': 'EPL', 'E1': 'EPL', 'SP1': 'ES1', 'I1': 'IT1', 'D1': 'DE1', 'F1': 'FR1'}
    return m.get(div, div)


def evaluate(predictor, files, seasons_to_eval=(2024, 2025), repo_root=None):
    season_ranges = {s: (datetime(s, 7, 1), datetime(s + 1, 6, 30)) for s in seasons_to_eval}
    results_by_season = {s: [] for s in seasons_to_eval}

    if not getattr(predictor, 'feature_encoders', None) or not all(k in getattr(predictor, 'feature_encoders', {}) for k in ('home_team', 'away_team', 'league')):
        teams = set()
        leagues = set()
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            if 'hometeam' in cols and 'awayteam' in cols:
                teams.update(df[cols['hometeam']].dropna().astype(str).unique().tolist())
                teams.update(df[cols['awayteam']].dropna().astype(str).unique().tolist())
            if 'div' in cols:
                leagues.update(df[cols['div']].dropna().astype(str).unique().tolist())
        predictor.feature_encoders = {'home_team': SimpleEncoder(teams or ['UnknownHome']), 'away_team': SimpleEncoder(teams or ['UnknownAway']), 'league': SimpleEncoder(leagues or ['UNK'])}

    team_stats = build_team_stats(files)

    all_matches_dfs = []
    for f in files:
        try:
            all_matches_dfs.append(pd.read_csv(f, on_bad_lines='skip'))
        except pd.errors.EmptyDataError:
            continue
    if all_matches_dfs:
        all_matches_df = pd.concat(all_matches_dfs, ignore_index=True)
        all_matches_df['date'] = pd.to_datetime(all_matches_df['Date'], dayfirst=True, errors='coerce')
    else:
        all_matches_df = pd.DataFrame()

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        colmap = {c.lower(): c for c in df.columns}
        if 'date' not in colmap or 'hometeam' not in colmap or 'awayteam' not in colmap:
            continue
        df['date'] = pd.to_datetime(df[colmap['date']], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        if 'ftr' in colmap:
            df['actual_result'] = df[colmap['ftr']].astype(str)
        else:
            if 'fthg' in colmap and 'ftag' in colmap:
                df['fthg'] = pd.to_numeric(df[colmap['fthg']], errors='coerce')
                df['ftag'] = pd.to_numeric(df[colmap['ftag']], errors='coerce')
                def to_ftr(row):
                    if pd.isna(row['fthg']) or pd.isna(row['ftag']):
                        return ''
                    if row['fthg'] > row['ftag']:
                        return 'H'
                    if row['fthg'] == row['ftag']:
                        return 'D'
                    return 'A'
                df['actual_result'] = df.apply(to_ftr, axis=1)
            else:
                df['actual_result'] = ''

        div_col = colmap.get('div')
        for idx, row in df.iterrows():
            match_date = row['date']
            season_start = infer_season(match_date)
            if season_start not in seasons_to_eval:
                continue
            actual = str(row.get('actual_result', '')).strip()
            if actual not in ('H', 'D', 'A'):
                continue
            home = row[colmap['hometeam']]
            away = row[colmap['awayteam']]
            div = row[div_col] if div_col else ''
            league = map_div_to_league(div)
            try:
                pred = predictor.predict_match(home_team=home, away_team=away, league=league, match_date=match_date, season=season_start+1, all_matches_df=all_matches_df)
            except Exception:
                pred = None
            if pred is None:
                pred = fallback_predict(home, away, league, match_date, season_start, team_stats)
            probs = pred.get('probabilities', {})
            results_by_season[season_start].append({
                'date': match_date.strftime('%Y-%m-%d'),
                'league': league,
                'home_team': home,
                'away_team': away,
                'actual': ('Home Win' if actual == 'H' else ('Draw' if actual == 'D' else 'Away Win')),
                'predicted': pred.get('prediction'),
                'model_used': pred.get('model_used'),
                'home_win_prob': probs.get('Home Win', ''),
                'draw_prob': probs.get('Draw', ''),
                'away_win_prob': probs.get('Away Win', '')
            })

    out_dir = os.path.join(repo_root, 'predictions', 'evaluations') if repo_root else 'predictions/evaluations'
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for s, rows in results_by_season.items():
        out_path = os.path.join(out_dir, f'eval_{s}_{s+1}.csv')
        if rows:
            pd.DataFrame(rows).to_csv(out_path, index=False)
        total = len(rows)
        correct = sum(1 for r in rows if r['actual'] == r['predicted'])
        acc = correct / total if total else 0.0
        conf = defaultdict(int)
        for r in rows:
            conf[(r['actual'], r['predicted'])] += 1
        summary[s] = {'total_matches': total, 'correct': correct, 'accuracy': acc, 'confusion': dict(conf), 'out_path': out_path if rows else None}
    return summary


def print_summary(summary):
    for s in sorted(summary.keys()):
        info = summary[s]
        print('\n==== Season {}-{} ===='.format(s, s+1))
        print('Matches evaluated:', info['total_matches'])
        print('Correct predictions:', info['correct'])
        print('Accuracy: {:.2%}'.format(info['accuracy']))
        print('Confusion (actual -> predicted counts):')
        for (act, pred), cnt in sorted(info['confusion'].items(), key=lambda x: (-x[1], x[0])):
            print(f'  {act} -> {pred}: {cnt}')
        if info['out_path']:
            print('Saved CSV:', info['out_path'])


def analyze_evaluation(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    total_matches = len(df)
    correct_predictions = (df['actual'] == df['predicted']).sum()
    overall_accuracy = correct_predictions / total_matches if total_matches > 0 else 0
    print("=== Overall Accuracy ===")
    print(f"Total matches: {total_matches}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print("\n=== Per-League Accuracy ===")
    if 'league' in df.columns:
        leagues = df['league'].unique()
        for league in sorted(leagues):
            league_df = df[df['league'] == league]
            league_total = len(league_df)
            league_correct = (league_df['actual'] == league_df['predicted']).sum()
            league_accuracy = league_correct / league_total if league_total > 0 else 0
            print(f"{league}: {league_correct}/{league_total} ({league_accuracy:.2%})")
    print(f"\nModel used: {df['model_used'].iloc[0] if not df.empty else 'N/A'}")
    print(f"Draw probability unique values: {df['draw_prob'].unique() if 'draw_prob' in df.columns else 'N/A'}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Combined evaluate/analyze helper')
    parser.add_argument('--evaluate', '-e', help='Comma-separated season start years to evaluate (e.g. 2024,2025)')
    parser.add_argument('--analyze', '-a', help='Path to evaluation CSV to analyze')
    parser.add_argument('--train', action='store_true', help='Force running the trainer before evaluation')
    parser.add_argument('--download', action='store_true', help='Force downloading football-data files before evaluation')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (only accuracy)')
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def latest_model_mtime(models_dir):
        # look for common model artifacts
        candidates = ['ensemble_with_elo.joblib', 'ensemble_model.joblib', 'xgboost_model.joblib']
        mt = 0
        for c in candidates:
            p = os.path.join(models_dir, c)
            if os.path.exists(p):
                mt = max(mt, os.path.getmtime(p))
        return mt

    def run_trainer_if_needed(force=False):
        trainer = os.path.join(repo_root, 'src', 'models', 'train_models.py')
        models_dir = os.path.join(repo_root, 'models')
        if not os.path.exists(trainer):
            print('Trainer not found at', trainer)
            return False
        trainer_mtime = os.path.getmtime(trainer)
        model_mtime = latest_model_mtime(models_dir)
        # Only auto-run trainer if explicit force=True or if trainer is newer AND
        # there are candidate data files available. This prevents accidental
        # trainer runs when no dataset is present (which previously caused a
        # FileNotFoundError). If no data files present, return False unless
        # force is True.
        data_files = find_data_files(repo_root)
        if not force and not data_files and trainer_mtime > model_mtime:
            print('Trainer is newer than models but no data files were found; skipping automatic training. Use --train to force.')
            return False

        if force or trainer_mtime > model_mtime:
            if not args.quiet:
                print('Running trainer because --train was passed or trainer is newer than model artifacts')
            cmd = [sys.executable, trainer, '--out', models_dir]
            try:
                if args.quiet:
                    res = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    res = subprocess.run(cmd, check=False)
                if not args.quiet:
                    print('Trainer exit code:', res.returncode)
                return res.returncode == 0
            except Exception as e:
                if not args.quiet:
                    print('Failed to run trainer:', e)
                return False
        else:
            print('Trainer not newer than existing models; skipping training')
            return True

    def run_downloader_if_needed(force=False):
        out_dir = os.path.join(repo_root, 'data', 'football-data')
        # check if folder exists and has csvs
        has_files = False
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                if f.lower().endswith('.csv'):
                    has_files = True
                    break
        if force or not has_files:
            print('Running downloader to ensure football-data CSVs are present')
            downloader = os.path.join(repo_root, 'scripts', 'download_football_data.py')
            if not os.path.exists(downloader):
                print('Downloader script not found at', downloader)
                return False
            cmd = [sys.executable, downloader, '--out', out_dir]
            try:
                res = subprocess.run(cmd, check=False)
                print('Downloader exit code:', res.returncode)
                return res.returncode == 0
            except Exception as e:
                print('Failed to run downloader:', e)
                return False
        else:
            print('Football-data CSVs already present; skipping download')
            return True

    # optionally download data
    if args.download:
        run_downloader_if_needed(force=True)

    # optionally run trainer
    if args.train:
        run_trainer_if_needed(force=True)

    if args.evaluate:
        try:
            seasons = [int(x) for x in args.evaluate.split(',')]
        except Exception:
            seasons = [2024, 2025, 2026]
        # if trainer file is more recent than saved models, run trainer automatically
        run_trainer_if_needed(force=False)

        predictor = load_match_predictor(repo_root, quiet=args.quiet)
        files = find_data_files(repo_root)
        summary = evaluate(predictor, files, seasons_to_eval=tuple(seasons), repo_root=repo_root)
        print_summary(summary) if not args.quiet else print_minimal_summary(summary)

    if args.analyze:
        analyze_evaluation(args.analyze) if not args.quiet else analyze_evaluation_minimal(args.analyze)

    # If no flags were provided, analyze the most recent evaluation CSV if available
    if not args.evaluate and not args.analyze:
        eval_dir = os.path.join(repo_root, 'predictions', 'evaluations')
        if os.path.exists(eval_dir):
            csvs = glob.glob(os.path.join(eval_dir, 'eval_*.csv'))
            if csvs:
                latest = max(csvs, key=os.path.getmtime)
                if not args.quiet:
                    print(f"No flags provided — analyzing latest evaluation: {latest}\n")
                analyze_evaluation(latest) if not args.quiet else analyze_evaluation_minimal(latest)
                return
        print("No evaluation CSV found in predictions/evaluations. Use --evaluate or --analyze <path> to proceed.")


def print_minimal_summary(summary):
    # print one line per season: Accuracy percentage
    for s in sorted(summary.keys()):
        info = summary[s]
        print(f"{s}-{s+1}: {info['accuracy']*100:.2f}%")


def analyze_evaluation_minimal(csv_path):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    total_matches = len(df)
    correct_predictions = (df['actual'] == df['predicted']).sum()
    overall_accuracy = correct_predictions / total_matches if total_matches > 0 else 0
    print(f"Accuracy: {overall_accuracy*100:.2f}%")


if __name__ == '__main__':
    main()
