"""Quick retrain script: faster single-run retrain to validate feature changes and produce an artifact.

Usage:
    python src/models/quick_retrain.py --data <csv> --out models

This script reuses functions from train_models.py and performs a single XGBoost fit
with fixed hyperparameters for speed.
"""
import os
import sys
import argparse
import joblib
import pandas as pd

# ensure src is on path so imports work when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_models import load_matches, compute_elo, create_features, prepare_X_y
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='Path to matches CSV')
    p.add_argument('--out', default='models', help='Output directory')
    p.add_argument('--k', type=float, default=30.0)
    p.add_argument('--home-adv', type=float, default=100.0)
    args = p.parse_args()

    df = load_matches(args.data)
    os.makedirs(args.out, exist_ok=True)
    print('Loaded', len(df), 'rows')

    elo_df, teams = compute_elo(df, k=args.k, home_field_adv=args.home_adv)
    feats, enc = create_features(elo_df)
    X, y, scaler = prepare_X_y(feats)

    print('Training on', len(X), 'rows with', X.shape[1], 'features')
    # quick XGB with reasonable defaults
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print('Train accuracy:', acc)
    print(classification_report(y, preds, target_names=['Home Win', 'Draw', 'Away Win']))

    # save artifacts
    joblib.dump(model, os.path.join(args.out, 'quick_model.joblib'))
    joblib.dump(enc, os.path.join(args.out, 'quick_feature_encoders.joblib'))
    joblib.dump(scaler, os.path.join(args.out, 'quick_scaler.joblib'))
    print('Saved quick artifacts to', args.out)


if __name__ == '__main__':
    main()
