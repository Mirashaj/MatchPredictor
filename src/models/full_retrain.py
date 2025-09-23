"""Full retrain pipeline

This script performs:
 - temporal holdout (default: last 365 days) for evaluation
 - trains an XGBoost multiclass model with early stopping
 - trains a draw vs non-draw logistic model and a second-stage H/A model
 - trains a simple Elo-based logistic calibration
 - ensembles predictions (weighted) and evaluates on holdout
 - saves artifacts to the models output directory

Usage:
 python src/models/full_retrain.py --out models --holdout-days 365

"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_models import load_matches, compute_elo, create_features, prepare_X_y, fit_calibration
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import train_test_split


def to_holdout_split(df, date_col='date', holdout_days=365):
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError('date column required for temporal split')
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].reset_index(drop=True)
    hold = df[df[date_col] > cutoff].reset_index(drop=True)
    return train, hold


def build_and_eval(df, out_dir, holdout_days=365):
    os.makedirs(out_dir, exist_ok=True)
    # temporal split
    train_df, hold_df = to_holdout_split(df, 'date', holdout_days=holdout_days)
    print('Train rows:', len(train_df), 'Holdout rows:', len(hold_df))

    # compute Elo on combined dataset so elo histories align, but we will only train on train_df
    elo_df, teams = compute_elo(df)
    # merge Elo info onto training and holdout by index: elo_df is same length as df
    elo_train = elo_df.loc[train_df.index].reset_index(drop=True)
    elo_hold = elo_df.loc[hold_df.index].reset_index(drop=True)

    # create features
    feats_train, enc = create_features(elo_train)
    feats_hold, _ = create_features(elo_hold)

    X_train, y_train, scaler = prepare_X_y(feats_train, fit_scaler=True)
    X_hold, y_hold, _ = prepare_X_y(feats_hold, fit_scaler=False, scaler=scaler)

    print('Training X shape:', X_train.shape)

    # Main multiclass XGBoost with early stopping on a small validation split from training
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)

    # Draw vs non-draw classifier
    is_draw_train = (y_train == 1).astype(int)
    draw_clf = LogisticRegression(class_weight='balanced', max_iter=200)
    # train on X_train
    draw_clf.fit(X_train, is_draw_train)

    # Elo logistic calibration on elo_diff (fit on train_df augmented)
    try:
        elo_calib = fit_calibration(elo_train)
    except Exception as e:
        print('Elo calibration failed:', e)
        elo_calib = None

    # Evaluate ensemble on holdout
    preds_xgb_proba = xgb.predict_proba(X_hold)
    preds_draw_prob = draw_clf.predict_proba(X_hold)[:, 1]
    # convert draw prob into three-class by scaling: distribute non-draw prob by xgb's H/A ratio
    # weight parameters (can be tuned)
    w_xgb = 0.7
    w_draw = 0.3
    # xgb returns [P(H), P(D), P(A)]
    # build final probability as weighted average where draw gets an extra weight
    final_proba = np.zeros_like(preds_xgb_proba)
    # draw final prob is weighted combination
    final_proba[:, 1] = w_xgb * preds_xgb_proba[:, 1] + w_draw * preds_draw_prob
    # distribute remaining prob to H and A proportional to xgb's relative probs
    rest = 1.0 - final_proba[:, 1]
    xgb_HA = preds_xgb_proba[:, [0, 2]]
    xgb_HA_sum = xgb_HA.sum(axis=1)
    # avoid division by zero
    xgb_HA_sum[xgb_HA_sum == 0] = 1e-9
    final_proba[:, 0] = rest * (xgb_HA[:, 0] / xgb_HA_sum)
    final_proba[:, 2] = rest * (xgb_HA[:, 1] / xgb_HA_sum)

    # evaluation
    y_hold_true = y_hold.values
    pred_classes = final_proba.argmax(axis=1)
    acc = accuracy_score(y_hold_true, pred_classes)
    ll = log_loss(y_hold_true, final_proba)
    print('Holdout accuracy:', acc)
    print('Holdout log-loss:', ll)
    print(classification_report(y_hold_true, pred_classes, target_names=['Home Win', 'Draw', 'Away Win']))

    # Save artifacts
    joblib.dump(xgb, os.path.join(out_dir, 'full_xgb.joblib'))
    joblib.dump(draw_clf, os.path.join(out_dir, 'full_draw_clf.joblib'))
    if elo_calib is not None:
        joblib.dump(elo_calib, os.path.join(out_dir, 'full_elo_calib.joblib'))
    joblib.dump(enc, os.path.join(out_dir, 'full_feature_encoders.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'full_scaler.joblib'))
    joblib.dump(final_proba, os.path.join(out_dir, 'last_holdout_probs.npy'))
    print('Saved artifacts to', out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='Path to matches CSV')
    p.add_argument('--out', default='models', help='Output directory')
    p.add_argument('--holdout-days', type=int, default=365, help='Days to hold out for evaluation')
    args = p.parse_args()

    df = load_matches(args.data)
    build_and_eval(df, args.out, holdout_days=args.holdout_days)


if __name__ == '__main__':
    main()
