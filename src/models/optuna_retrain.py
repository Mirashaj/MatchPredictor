"""Optuna tuning for XGBoost hyperparameters.

Usage:
  python src/models/optuna_retrain.py --trials 30 --out models_optuna --holdout-days 365

This script performs a temporal holdout (last `holdout-days`) for final evaluation, but
uses the remainder for Optuna tuning by splitting that remainder into an internal train/valid split (time-ordered).
It optimizes validation log-loss.
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_models import load_matches, compute_elo, create_features, prepare_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

try:
    import optuna
except Exception:
    optuna = None

from xgboost import XGBClassifier


def to_holdout_split(df, date_col='date', holdout_days=365):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].reset_index(drop=True)
    hold = df[df[date_col] > cutoff].reset_index(drop=True)
    return train, hold


def objective(trial, X_tr, y_tr, X_val, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 30
    }
    clf = XGBClassifier(**param)
    # Train with early stopping (early_stopping_rounds set in constructor); pass eval_set
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    proba = clf.predict_proba(X_val)
    ll = log_loss(y_val, proba)
    return ll


def run_optuna(df, out_dir, trials=30, holdout_days=365):
    os.makedirs(out_dir, exist_ok=True)
    train_df, hold_df = to_holdout_split(df, 'date', holdout_days)
    print('Tuning on', len(train_df), 'rows; holdout set aside:', len(hold_df))

    # compute elo on all data to keep histories aligned
    elo_df, teams = compute_elo(df)
    elo_train = elo_df.loc[train_df.index].reset_index(drop=True)

    feats_train, enc = create_features(elo_train)
    X_all, y_all, scaler = prepare_X_y(feats_train, fit_scaler=True)

    # Use last 10% of X_all as validation (time-ordered)
    split_idx = int(len(X_all) * 0.9)
    X_tr = X_all.iloc[:split_idx]
    y_tr = y_all.iloc[:split_idx]
    X_val = X_all.iloc[split_idx:]
    y_val = y_all.iloc[split_idx:]

    print('Internal train:', X_tr.shape, 'val:', X_val.shape)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_val, y_val), n_trials=trials)

    print('Best trial:', study.best_trial.params)

    # Refit best model on full train (X_all)
    best_params = study.best_trial.params
    best_params.update({'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'mlogloss'})
    best_clf = XGBClassifier(**best_params)
    best_clf.fit(X_all, y_all)

    # Save artifacts
    joblib.dump(best_clf, os.path.join(out_dir, 'optuna_best_xgb.joblib'))
    joblib.dump(study, os.path.join(out_dir, 'optuna_study.joblib'))
    joblib.dump(enc, os.path.join(out_dir, 'optuna_feature_encoders.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'optuna_scaler.joblib'))

    # Evaluate on holdout set if exists
    if len(hold_df) > 0:
        elo_hold = elo_df.loc[hold_df.index].reset_index(drop=True)
        feats_hold, _ = create_features(elo_hold)
        X_hold, y_hold, _ = prepare_X_y(feats_hold, fit_scaler=False, scaler=scaler)
        proba = best_clf.predict_proba(X_hold)
        print('Holdout log-loss:', log_loss(y_hold, proba))
        print('Holdout accuracy:', accuracy_score(y_hold, proba.argmax(axis=1)))

    # Save best params to json
    import json
    with open(os.path.join(out_dir, 'optuna_best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(study.best_trial.params, f, indent=2)

    print('Optuna tuning complete; artifacts saved to', out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='Path to matches CSV')
    p.add_argument('--out', default='models_optuna', help='Output directory')
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--holdout-days', type=int, default=365)
    args = p.parse_args()

    if optuna is None:
        raise SystemExit('Optuna is not installed. Please run: pip install optuna')

    df = load_matches(args.data)
    run_optuna(df, args.out, trials=args.trials, holdout_days=args.holdout_days)


if __name__ == '__main__':
    main()
