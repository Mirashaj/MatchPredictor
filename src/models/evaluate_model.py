"""Evaluate saved models on a temporal holdout.

Usage:
  python src/models/evaluate_model.py --data <csv> --models models --holdout-days 365

This script expects artifacts saved by `full_retrain.py` (full_xgb.joblib, full_draw_clf.joblib, full_scaler.joblib, full_feature_encoders.joblib).
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_models import load_matches, compute_elo, create_features, prepare_X_y
from sklearn.metrics import accuracy_score, log_loss, classification_report


def to_holdout_split(df, date_col='date', holdout_days=365):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].reset_index(drop=True)
    hold = df[df[date_col] > cutoff].reset_index(drop=True)
    return train, hold


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='Path to matches CSV')
    p.add_argument('--models', default='models', help='Directory with saved artifacts')
    p.add_argument('--holdout-days', type=int, default=365)
    args = p.parse_args()

    df = load_matches(args.data)
    train_df, hold_df = to_holdout_split(df, 'date', args.holdout_days)
    elo_df, teams = compute_elo(df)

    # map elo rows to holdout rows
    elo_hold = elo_df.loc[hold_df.index].reset_index(drop=True)
    feats_hold, _ = create_features(elo_hold)

    # load artifacts
    xgb = joblib.load(os.path.join(args.models, 'full_xgb.joblib'))
    draw_clf = joblib.load(os.path.join(args.models, 'full_draw_clf.joblib'))
    scaler = joblib.load(os.path.join(args.models, 'full_scaler.joblib'))
    enc = joblib.load(os.path.join(args.models, 'full_feature_encoders.joblib'))

    X_hold, y_hold, _ = prepare_X_y(feats_hold, fit_scaler=False, scaler=scaler)

    preds_xgb_proba = xgb.predict_proba(X_hold)
    preds_draw_prob = draw_clf.predict_proba(X_hold)[:, 1]

    w_xgb = 0.7
    w_draw = 0.3
    final_proba = np.zeros_like(preds_xgb_proba)
    final_proba[:, 1] = w_xgb * preds_xgb_proba[:, 1] + w_draw * preds_draw_prob
    rest = 1.0 - final_proba[:, 1]
    xgb_HA = preds_xgb_proba[:, [0, 2]]
    xgb_HA_sum = xgb_HA.sum(axis=1)
    xgb_HA_sum[xgb_HA_sum == 0] = 1e-9
    final_proba[:, 0] = rest * (xgb_HA[:, 0] / xgb_HA_sum)
    final_proba[:, 2] = rest * (xgb_HA[:, 1] / xgb_HA_sum)

    pred_classes = final_proba.argmax(axis=1)
    acc = accuracy_score(y_hold.values, pred_classes)
    ll = log_loss(y_hold.values, final_proba)

    print('Holdout accuracy:', acc)
    print('Holdout log-loss:', ll)
    print(classification_report(y_hold.values, pred_classes, target_names=['Home Win', 'Draw', 'Away Win']))
    # Save a CSV with predictions + true labels for further analysis
    try:
        out_dir = args.models
        os.makedirs(out_dir, exist_ok=True)
        # attempt to include some original match columns if present
        hold_df = load_matches(args.data)[-len(y_hold):].reset_index(drop=True)
        # Build a DataFrame of predictions
        preds_df = pd.DataFrame({
            'pred_class': pred_classes,
            'pred_label': ["Home Win","Draw","Away Win"][0:1] * len(pred_classes)
        })
    except Exception:
        hold_df = None

    # Create a tidy predictions DataFrame
    cols = ['p_home', 'p_draw', 'p_away']
    proba_df = pd.DataFrame(final_proba, columns=cols)
    results_df = proba_df.copy()
    results_df['pred_class'] = pred_classes
    results_df['pred_label'] = results_df['pred_class'].map({0: 'Home Win', 1: 'Draw', 2: 'Away Win'})
    # add true labels if available
    try:
        results_df['true_class'] = y_hold.values
        results_df['true_label'] = results_df['true_class'].map({0: 'Home Win', 1: 'Draw', 2: 'Away Win'})
    except Exception:
        pass

    # attach some match identifiers if the computed holdout frame had them
    try:
        # Recompute holdoin df via load_matches to preserve original order and columns
        full = load_matches(args.data)
        hold_rows = full.loc[full.index[-len(results_df):], :].reset_index(drop=True)
        # pick common columns
        for c in ['date', 'HomeTeam', 'AwayTeam', 'Div', 'FTHG', 'FTAG', 'FTR']:
            if c in hold_rows.columns:
                results_df[c] = hold_rows[c]
    except Exception:
        pass

    csv_path = os.path.join(args.models, 'holdout_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print('Saved holdout predictions to', csv_path)


if __name__ == '__main__':
    main()
