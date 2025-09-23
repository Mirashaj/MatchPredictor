import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
import contextlib
warnings.filterwarnings('ignore')

class MatchPredictor:
    
    def __init__(self, models_dir="models", quiet=False):
        self.models_dir = models_dir
        self.models = {}
        self.feature_encoders = {}
        self.scaler = None
        self.quiet = bool(quiet)
        # keep track of columns we've already warned about to avoid repeating heavy logs
        self._missing_logged = set()
        # deduplicate prediction errors per model to avoid repeated messages
        self._pred_error_logged = set()

        # simple logger respecting quiet flag
        def _log(*args, **kwargs):
            if not self.quiet:
                print(*args, **kwargs)
        self._log = _log
        self.load_trained_models()
        
    def load_trained_models(self):
        self._log("LOADING TRAINED MODELS")
        self._log("=" * 50)
        # Load possible model files (prefer 'ensemble_with_elo' if present)
        model_files = ['ensemble_with_elo.joblib', 'ensemble_model.joblib', 'xgboost_model.joblib']

        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.joblib', '')
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self._log(f"{model_name} loaded")
                except Exception as e:
                    self._log(f"Unable to load {model_file}: {e} - skipping this model")
            else:
                # don't print repeatedly to avoid noisy output
                pass

        # Load encoders and scaler (prefer 'with_elo' variants when present)
        preferred_enc = os.path.join(self.models_dir, 'feature_encoders_with_elo.joblib')
        alt_enc = os.path.join(self.models_dir, 'feature_encoders.joblib')
        preferred_scaler = os.path.join(self.models_dir, 'scaler_with_elo.joblib')
        alt_scaler = os.path.join(self.models_dir, 'scaler.joblib')

        try:
            if os.path.exists(preferred_enc):
                self.feature_encoders = joblib.load(preferred_enc)
                self._log(f"Feature encoders (with ELO) loaded")
            elif os.path.exists(alt_enc):
                self.feature_encoders = joblib.load(alt_enc)
                self._log(f"Feature encoders loaded")
            else:
                self._log("Feature encoders not found, will use fallback encoders")

            if os.path.exists(preferred_scaler):
                self.scaler = joblib.load(preferred_scaler)
                self._log(f"Scaler (with ELO) loaded")
            elif os.path.exists(alt_scaler):
                self.scaler = joblib.load(alt_scaler)
                self._log(f"Scaler loaded")
            else:
                self._log("Scaler not found, skipping normalization")
        except Exception as e:
            self._log(f"Error loading encoders/scaler: {e}")

        # Load ELO ratings
        elo_ratings_path = f"{self.models_dir}/elo_2025_ratings.json"
        if os.path.exists(elo_ratings_path):
            try:
                with open(elo_ratings_path, 'r') as f:
                    self.elo_ratings = json.load(f)
                self._log(f"ELO ratings loaded")
            except Exception as e:
                self._log(f"Error loading ELO ratings: {e}")
                self.elo_ratings = {}
        else:
            self._log(f"{elo_ratings_path} not found")
            self.elo_ratings = {}

        # If no models were loaded, create a DummyModel as an in-memory fallback
        if not self.models:
            self._log("No ML models loaded: activating DummyModel fallback")

            class DummyModel:
                def predict(self, X):
                    import numpy as _np
                    return _np.array([1 for _ in range(len(X))])

                def predict_proba(self, X):
                    import numpy as _np
                    probs = _np.array([[0.33, 0.34, 0.33] for _ in range(len(X))])
                    return probs

            self.models['Dummy'] = DummyModel()

        # Choose 'ensemble_with_elo' as the primary model if available,
        # otherwise use the first available model key (or 'Dummy').
        self.best_model_name = 'ensemble_with_elo' if 'ensemble_with_elo' in self.models else (list(self.models.keys())[0] if self.models else 'Dummy')
        self._log(f"Primary model: {self.best_model_name}")
    
    def predict_match(self, home_team, away_team, league, all_matches_df, match_date=None, season=2025):
        """
        Predict the outcome of a single match.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League code (EPL, ES1, IT1, DE1, FR1)
            all_matches_df: DataFrame with historical matches
            match_date: Match date (optional, defaults to today)
            season: Season (default 2025)
        """
        if match_date is None:
            match_date = datetime.now()
        elif isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

            self._log(f"\nMATCH PREDICTION")
            self._log(f"Home: {home_team}")
            self._log(f"Away: {away_team}")
            self._log(f"League: {league}")
            self._log(f"Date: {match_date.strftime('%Y-%m-%d')}")

        try:
            # Prepare features (suppress third-party library output when quiet)
            if self.quiet:
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    features = self.prepare_match_features(home_team, away_team, league, match_date, season, all_matches_df)
            else:
                features = self.prepare_match_features(home_team, away_team, league, match_date, season, all_matches_df)
            
            if features is None:
                return None
            
            # Predictions with all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    # Predict class (suppress third-party output when quiet)
                    if self.quiet:
                        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            pred_class = model.predict(features)[0]
                    else:
                        pred_class = model.predict(features)[0]

                    # Probabilities
                    if hasattr(model, 'predict_proba'):
                        if self.quiet:
                            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                pred_proba = model.predict_proba(features)[0]
                        else:
                            pred_proba = model.predict_proba(features)[0]
                        probabilities[model_name] = {
                                'Home Win': pred_proba[0],
                                'Draw': pred_proba[1], 
                                'Away Win': pred_proba[2]
                            }

                    # Convert class to outcome label
                    result_mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
                    predictions[model_name] = result_mapping[pred_class]

                except Exception as e:
                    err_key = (model_name, str(e))
                    if err_key not in self._pred_error_logged:
                        self._log(f"Prediction error for {model_name}: {e}")
                        self._pred_error_logged.add(err_key)
                    # skip this model for future matches to avoid repeated exceptions
                    continue
            
            # Final prediction (best model)
            if self.best_model_name in predictions:
                final_prediction = predictions[self.best_model_name]
                final_probabilities = probabilities.get(self.best_model_name, {})
                
                # Display results
                self._log(f"\nPREDICTION RESULT")
                self._log(f"Prediction: {final_prediction}")
                
                if final_probabilities:
                    self._log(f"Probabilities:")
                    for outcome, prob in final_probabilities.items():
                        self._log(f"   {outcome}: {prob:.1%}")
                
                self._log(f"\nModel consensus:")
                for model_name, prediction in predictions.items():
                    self._log(f"   {model_name}: {prediction}")
                
                return {
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'date': match_date,
                    'prediction': final_prediction,
                    'probabilities': final_probabilities,
                    'all_predictions': predictions,
                    'model_used': self.best_model_name
                }
            
            else:
                self._log(f"Error: primary model not available")
                return None
                
        except Exception as e:
            self._log(f"Prediction error: {e}")
            return None
    
    def prepare_match_features(self, home_team, away_team, league, match_date, season, all_matches_df):
        """Prepare features for a single match using the same features as training."""
        try:
            # Accept many column name variants in the provided historical matches DataFrame.
            # Normalize column names to the canonical names used in training so lookups below
            # are robust (avoids KeyError and boolean-on-Series issues).
            col_map = {}
            if isinstance(all_matches_df, pd.DataFrame):
                for c in all_matches_df.columns:
                    lc = c.lower()
                    if lc in ('home_team', 'home') or ('home' in lc and 'team' in lc):
                        col_map[c] = 'HomeTeam'
                    elif lc in ('away_team', 'away') or ('away' in lc and 'team' in lc):
                        col_map[c] = 'AwayTeam'
                    elif lc in ('div', 'league') or 'div' in lc or 'league' in lc:
                        col_map[c] = 'Div'
                    elif lc in ('date', 'match_date', 'kickoff') or 'date' in lc:
                        col_map[c] = 'date'
                    elif lc in ('ftr', 'result', 'outcome'):
                        col_map[c] = 'FTR'
                    elif lc in ('fthg', 'home_goals'):
                        col_map[c] = 'FTHG'
                    elif lc in ('ftag', 'away_goals'):
                        col_map[c] = 'FTAG'
                    else:
                        # keep other stats columns but map common short names
                        up = c.upper()
                        col_map[c] = up
                # perform renaming safely
                all_matches = all_matches_df.rename(columns=col_map).copy()
            else:
                all_matches = pd.DataFrame()

            # Create a one-row DataFrame for this match
            match_data = {
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Div': league,
                'date': match_date
            }

            df = pd.DataFrame([match_data])

            # Temporal features
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['season_progress'] = df['month'].apply(lambda x: 1 if x >= 8 else (0.5 if x >= 6 else 0))

            # Categorical encoding
            try:
                df['home_team_enc'] = self.feature_encoders['home_team'].transform([home_team])[0]
            except (ValueError, KeyError):
                df['home_team_enc'] = len(self.feature_encoders.get('home_team', {}).classes_ or []) // 2

            try:
                df['away_team_enc'] = self.feature_encoders['away_team'].transform([away_team])[0]
            except (ValueError, KeyError):
                df['away_team_enc'] = len(self.feature_encoders.get('away_team', {}).classes_ or []) // 2

            try:
                df['league_enc'] = self.feature_encoders['league'].transform([league])[0]
            except (ValueError, KeyError):
                df['league_enc'] = 0

            # ELO ratings
            home_elo = self.elo_ratings.get(home_team, 1500) if hasattr(self, 'elo_ratings') else 1500
            away_elo = self.elo_ratings.get(away_team, 1500) if hasattr(self, 'elo_ratings') else 1500
            df['home_elo'] = home_elo
            df['away_elo'] = away_elo
            df['elo_diff'] = home_elo - away_elo
            df['elo_exp_home'] = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + 100)) / 400))

            # Team rating fallbacks
            df['home_rating'] = home_elo / 400
            df['away_rating'] = away_elo / 400
            df['rating_diff'] = df['home_rating'] - df['away_rating']

            # Advanced stats: compute safely with defensive checks to avoid boolean-on-Series issues
            try:
                home_mask = ((all_matches.get('HomeTeam') == home_team) | (all_matches.get('AwayTeam') == home_team)) if not all_matches.empty else pd.Series(dtype=bool)
                away_mask = ((all_matches.get('HomeTeam') == away_team) | (all_matches.get('AwayTeam') == away_team)) if not all_matches.empty else pd.Series(dtype=bool)
            except Exception:
                home_mask = pd.Series(dtype=bool)
                away_mask = pd.Series(dtype=bool)

            home_matches = all_matches[home_mask] if not all_matches.empty else pd.DataFrame()
            away_matches = all_matches[away_mask] if not all_matches.empty else pd.DataFrame()

            def safe_mean(series, transform=None):
                try:
                    if series is None or len(series) == 0:
                        return 0.0
                    if transform is not None:
                        series = series.map(lambda x: transform(x) if pd.notna(x) else None)
                    return float(pd.to_numeric(series, errors='coerce').mean(skipna=True) or 0.0)
                except Exception:
                    return 0.0

            # Recent form: map FTR to numeric values depending on whether the team was home or away
            home_recent = 0.0
            if 'FTR' in home_matches.columns and not home_matches['FTR'].empty:
                # when the team was home, 'H' means 3 points; when away, 'A' means 3 points
                def home_form_val(row):
                    v = row
                    if pd.isna(v):
                        return 0
                    return 3 if v == 'H' else (1 if v == 'D' else 0)
                home_recent = safe_mean(home_matches['FTR'].map(lambda x: x if pd.notna(x) else None))
            df['home_recent_form'] = home_recent

            df['home_goals_avg'] = safe_mean(home_matches.apply(lambda r: (r.get('FTHG') if r.get('HomeTeam') == home_team else r.get('FTAG')), axis=1) if not home_matches.empty else pd.Series(dtype=float))
            df['home_shots_on_target_avg'] = safe_mean(home_matches.get('HST'))
            df['home_shots_avg'] = safe_mean(home_matches.get('HS'))
            df['home_corners_avg'] = safe_mean(home_matches.get('HC'))
            df['home_fouls_avg'] = safe_mean(home_matches.get('HF'))
            df['home_yellow_cards_avg'] = safe_mean(home_matches.get('HY'))
            df['home_red_cards_avg'] = safe_mean(home_matches.get('HR'))

            away_recent = 0.0
            if 'FTR' in away_matches.columns and not away_matches['FTR'].empty:
                away_recent = safe_mean(away_matches['FTR'].map(lambda x: x if pd.notna(x) else None))
            df['away_recent_form'] = away_recent

            df['away_goals_avg'] = safe_mean(away_matches.apply(lambda r: (r.get('FTAG') if r.get('AwayTeam') == away_team else r.get('FTHG')), axis=1) if not away_matches.empty else pd.Series(dtype=float))
            df['away_shots_on_target_avg'] = safe_mean(away_matches.get('AST'))
            df['away_shots_avg'] = safe_mean(away_matches.get('AS'))
            df['away_corners_avg'] = safe_mean(away_matches.get('AC'))
            df['away_fouls_avg'] = safe_mean(away_matches.get('AF'))
            df['away_yellow_cards_avg'] = safe_mean(away_matches.get('AY'))
            df['away_red_cards_avg'] = safe_mean(away_matches.get('AR'))

            df.fillna(0, inplace=True)

            # Ensure all columns expected by the trained model exist. If some are missing
            # (for example when the training pipeline used additional features or odds data),
            # create them with safe default values (0).
            # Normalize column name types to plain Python str to avoid scikit-learn
            # validation errors caused by mixed numpy.str_/str types.
            df.columns = df.columns.astype(str)

            model = self.models.get(self.best_model_name)
            # Ensure model feature names are strings as well
            feature_columns = [str(c) for c in getattr(model, 'feature_names_in_', [])] if model is not None else []
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                # only log new missing column sets to avoid repeating the same heavy message
                missing_key = tuple(sorted(missing))
                if missing_key not in self._missing_logged:
                    self._log(f"Warning: missing columns required by the model: {missing}")
                    self._missing_logged.add(missing_key)
                for c in missing:
                    # create numeric default column (use str name)
                    df[str(c)] = 0

            # Select the expected feature order; if the model lacks `feature_names_in_`, fall back to df.columns
            if feature_columns:
                # ensure the selection uses the stringified feature names
                features_df = df[feature_columns].copy()
            else:
                features_df = df.copy()

            # Normalize all numeric features
            if self.scaler:
                # ensure scaler sees string column names
                features_df.columns = features_df.columns.astype(str)
                # printing full dataframe head is expensive for every match; print only column names
                self._log("Feature columns before scaling:", list(features_df.columns))
                # If scaler was fitted with feature names, check for mismatch and skip scaling if they differ
                try:
                    if hasattr(self.scaler, 'feature_names_in_'):
                        scaler_cols = [str(c) for c in self.scaler.feature_names_in_]
                        if set(scaler_cols) != set(features_df.columns):
                            missing_scaler = sorted(set(scaler_cols) - set(features_df.columns))
                            extra_scaler = sorted(set(features_df.columns) - set(scaler_cols))
                            scaler_mismatch_key = ('scaler_mismatch', tuple(scaler_cols))
                            if scaler_mismatch_key not in self._missing_logged:
                                self._log(f"Warning: scaler was trained with different columns. "
                                          f"Missing in data: {missing_scaler}; Extra in data: {extra_scaler}. "
                                          "Skipping normalization for this and future matches.")
                                self._missing_logged.add(scaler_mismatch_key)
                            # skip scaling to avoid repeated sklearn feature-name errors
                        else:
                            features_df = pd.DataFrame(
                                self.scaler.transform(features_df),
                                columns=features_df.columns,
                                index=features_df.index
                            )
                            self._log("Feature columns after scaling:", list(features_df.columns))
                    else:
                        # scaler has no feature_names_in_; attempt to transform once and catch errors
                        features_df = pd.DataFrame(
                            self.scaler.transform(features_df),
                            columns=features_df.columns,
                            index=features_df.index
                        )
                        self._log("Feature columns after scaling:", list(features_df.columns))
                except Exception as e:
                    err_key = ('scaler_error', str(e))
                    if err_key not in self._missing_logged:
                        self._log(f"Error during normalization (skipped): {e}")
                        self._missing_logged.add(err_key)

            return features_df.values

        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def predict_upcoming_matches(self, data_path="data/processed/unified_matches_dataset.csv"):
        """Predict all upcoming scheduled matches found in the dataset."""
        print(f"\nUPCOMING MATCH PREDICTIONS")
        print("=" * 60)
        
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter future / scheduled matches
            today = datetime.now()
            upcoming_matches = df[
                (df['date'] >= today) | 
                (df['status'] == 'SCHEDULED')
            ].copy()
            
            if upcoming_matches.empty:
                print("No upcoming matches found")
                return pd.DataFrame()
            
            print(f"Upcoming matches found: {len(upcoming_matches)}")
            
            # Make predictions
            predictions_list = []
            
            for idx, match in upcoming_matches.iterrows():
                if idx % 50 == 0:
                    print(f"Progress: {idx}/{len(upcoming_matches)}")
                
                prediction = self.predict_match(
                    home_team=match['home_team'],
                    away_team=match['away_team'],
                    league=match['league'],
                    match_date=match['date'],
                    season=match['season']
                )
                
                if prediction:
                    pred_row = {
                        'match_id': match.get('match_id', f"{idx}"),
                        'date': match['date'],
                        'league': match['league'],
                        'home_team': match['home_team'],
                        'away_team': match['away_team'],
                        'predicted_result': prediction['prediction'],
                        'home_win_prob': prediction['probabilities'].get('Home Win', 0),
                        'draw_prob': prediction['probabilities'].get('Draw', 0),
                        'away_win_prob': prediction['probabilities'].get('Away Win', 0),
                        'model_used': prediction['model_used']
                    }
                    predictions_list.append(pred_row)
            
            if predictions_list:
                predictions_df = pd.DataFrame(predictions_list)
                
                # Save predictions
                output_path = "data/predictions/upcoming_matches_predictions.csv"
                os.makedirs("data/predictions", exist_ok=True)
                predictions_df.to_csv(output_path, index=False)
                
                print(f"\nPredictions completed: {len(predictions_df)} matches")
                print(f"Saved to: {output_path}")
                
                # Show summary
                self.show_predictions_summary(predictions_df)
                
                return predictions_df
            
            else:
                print("No predictions generated")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return pd.DataFrame()
    
    def show_predictions_summary(self, predictions_df):
        """Display a short summary of predictions."""
        print(f"\nPREDICTIONS SUMMARY")
        print("-" * 40)

        # Distribution of predicted results
        result_counts = predictions_df['predicted_result'].value_counts()
        total = len(predictions_df)

        for result, count in result_counts.items():
            pct = count / total * 100
            print(f"{result}: {count} ({pct:.1f}%)")

        # Predictions per league
        print(f"\nPredictions per league:")
        league_summary = predictions_df.groupby('league')['predicted_result'].value_counts().unstack(fill_value=0)
        print(league_summary)

        # High-confidence predictions
        high_confidence = predictions_df[
            (predictions_df['home_win_prob'] > 0.6) |
            (predictions_df['away_win_prob'] > 0.6)
        ]

        print(f"\nHigh-confidence predictions (>60%): {len(high_confidence)}")

def main():
    """Main demo function for predictions"""
    predictor = MatchPredictor()

    print("MATCHPREDICTOR PREDICTION SYSTEM")
    print("=" * 80)

    # Demo single match prediction
    print("DEMO: single match prediction")
    result = predictor.predict_match(
        home_team="Manchester City",
        away_team="Liverpool", 
        league="EPL",
        match_date="2025-08-30"
    )

    # Predict all upcoming matches
    print(f"\nGenerating predictions for all upcoming matches...")
    predictions_df = predictor.predict_upcoming_matches()

    return predictor, predictions_df

if __name__ == "__main__":
    predictor, predictions = main()