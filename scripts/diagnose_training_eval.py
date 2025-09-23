import pandas as pd
import numpy as np
from pathlib import Path

train_p = Path('data/training_2024_2025.csv')
eval_p = Path('predictions/evaluations/eval_2025_2026.csv')

print('Training file:', train_p.exists(), train_p)
print('Eval file:', eval_p.exists(), eval_p)

if not train_p.exists() or not eval_p.exists():
    raise SystemExit('Missing files; run training/evaluation first')

train = pd.read_csv(train_p)
eval = pd.read_csv(eval_p)

print('\nTrain rows, cols:', train.shape)
print('Eval rows, cols:', eval.shape)

print('\nEval columns:', list(eval.columns)[:50])
print('\nTrain columns sample:', list(train.columns)[:50])

# Candidate features from train_models.prepare_X_y
candidate_features = [
    'league_enc', 'month', 'day_of_week', 'is_weekend', 'season_progress',
    'home_team_enc', 'away_team_enc',
    'home_elo', 'away_elo', 'elo_diff', 'elo_exp_home',
    'home_recent_form', 'away_recent_form', 'home_goals_avg', 'away_goals_avg'
]

print('\nChecking candidate features presence and stats:')
for f in candidate_features:
    in_train = f in train.columns
n_in_train = sum(1 for f in candidate_features if f in train.columns)
print(f'Features present in training: {n_in_train}/{len(candidate_features)}')

# Show numeric feature stats for features present in both
common = [f for f in candidate_features if f in train.columns or f in eval.columns]
print('\nFeature presence (train / eval):')
for f in common:
    print(f, 'train:', f in train.columns, 'eval:', f in eval.columns)

num_common = [f for f in candidate_features if f in train.columns and f in eval.columns]
if num_common:
    print('\nNumeric feature stats (train vs eval):')
    for f in num_common:
        t = pd.to_numeric(train[f], errors='coerce')
        e = pd.to_numeric(eval.get(f, pd.Series(dtype=float)), errors='coerce')
        print(f"{f}: train mean={t.mean():.3f} std={t.std():.3f} missing={t.isna().sum()}  |  eval mean={e.mean():.3f} std={e.std():.3f} missing={e.isna().sum()}")

# Check target / predictions in eval CSV
# possible column names for actual/pred
possible_actual = ['FTR','actual','result','outcome','true']
possible_pred = ['prediction','pred','predicted','predicted_label','model_pred']

actual_col = None
for c in eval.columns:
    if c in possible_actual or c.lower() in [p.lower() for p in possible_actual]:
        actual_col = c
        break
# Fallback: try common names
if actual_col is None:
    for c in eval.columns:
        if 'actual' in c.lower() or 'result' in c.lower() or 'ftr' in c.lower():
            actual_col = c; break

pred_col = None
for c in eval.columns:
    if c in possible_pred or c.lower() in [p.lower() for p in possible_pred]:
        pred_col = c
        break
if pred_col is None:
    for c in eval.columns:
        if 'pred' in c.lower():
            pred_col = c; break

print('\nDetected actual col:', actual_col)
print('Detected pred col:', pred_col)

if actual_col and pred_col:
    # try to map to H/D/A codes
    act = eval[actual_col].astype(str)
    pr = eval[pred_col].astype(str)
    # normalize
    def norm_label(s):
        s = s.strip().upper()
        if s in ['H','HOME','HOME WIN','HOME_WIN']:
            return 'H'
        if s in ['A','AWAY','AWAY WIN','AWAY_WIN']:
            return 'A'
        if s in ['D','DRAW','DRAWN']:
            return 'D'
        return s
    act_n = act.map(norm_label)
    pr_n = pr.map(norm_label)
    both = act_n.notna() & pr_n.notna()
    acc = (act_n[both] == pr_n[both]).mean()
    print(f'Computed eval accuracy from file (on {both.sum()} rows): {acc:.3%}')
    print('\nConfusion table:')
    print(pd.crosstab(act_n[both], pr_n[both]))
else:
    print('Could not detect actual/pred columns; show sample rows:')
    print(eval.head(10).to_string())

# Baseline: most frequent class in training
# determine FTR in training
if 'FTR' in train.columns:
    top = train['FTR'].mode().iloc[0]
    print('\nMost frequent FTR in training:', top)
    # compute baseline accuracy on eval if actual_col detected
    if actual_col:
        baseline_acc = (eval[actual_col].astype(str).str.upper().str.startswith(top[0] if isinstance(top,str) else str(top))).mean()
        print('Baseline (predict most frequent) accuracy on eval (approx):', f'{baseline_acc:.3%}')

print('\nDone diagnostics')
