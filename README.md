# MatchPredictor

MatchPredictor is a local tool to predict football match outcomes using machine learning on locally provided CSV datasets. It includes feature engineering (ELO, rest, head-to-head, momentum), training utilities, evaluation scripts, and a small Flask UI to run predictions.

---

## Quick facts
- Language: Python 3.8+
- Web UI: Flask (src/app.py)
- Models: XGBoost (joblib artifacts in `models/`)
- Data: local CSVs placed under `data/` 
- Virtualenv: `.venv/` (recommended)

---

## Get started (Windows / PowerShell)

1. Clone and enter project folder:

2. Create and activate venv:
```powershell
python -m venv .venv
.venv\Scripts\Activate
```

3. Place your CSV files

Put your historical match CSVs in the `data/` folder.

4. Run the trainer

```powershell
python src/full_retrain.py
```

## Requirements

- Flask
- pandas, numpy
- scikit-learn, xgboost
- joblib

---