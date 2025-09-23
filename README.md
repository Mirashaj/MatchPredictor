# MatchPredictor v2.0 — Project Overview

MatchPredictor is a unified system for predicting football match results for the five major European leagues using machine learning and up-to-date data.

---

## Key Features

### Dataset
- Consolidated historical dataset of matches across the top five European leagues
- Thousands of matches standardized for machine learning
- Focus on recent seasons for better generalization

### Unified Data Sources
- Combines multiple APIs and data sources to maximize coverage and freshness
- Uses API-Sports for stable historical data and Football-Data.org for more recent matches
- Automatic rate-limit handling and retry/fallback logic

### Machine Learning
- Unified preprocessing and feature engineering pipeline
- XGBoost as the primary model for match outcome prediction
- Time-aware validation (temporal holdouts and backtests)

---

## Project Layout

```
MatchPredictor/
├── config/                 # API keys and project configuration
├── data/                   # raw, processed and enriched datasets
├── src/                    # application source code (scrapers, preprocessing, models, predictions)
├── models/                 # trained model artifacts (joblib)
├── logs/                   # training and system logs
├── scripts/                # utility scripts
├── predictions/            # prediction output and evaluations
└── resources/              # static assets (images, etc.)
```

---

## Quick Start (Windows)

1. Clone the repository

```powershell
git clone <repository-url>
cd MatchPredictor
```

2. Create and activate a Python virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure API keys

Create `config/api.env` (or edit `api.env.example`) and add your keys:

```env
FOOTBALL_API_KEY=your_api_sports_key
FOOTBALL_DATA_ORG_KEY=your_football_data_key
```

---

## Common Workflows

Download full dataset:

```powershell
python src/scrapers/unified_scraper.py
```

Preprocess data:

```powershell
python src/preprocessing/unified_preprocessor.py
```

Analyze dataset quality and coverage:

```powershell
python src/analysis/data_analyzer.py
```

Train ML models (XGBoost):

```powershell
python src/models/train_models.py
```

Run the Flask web app (development):

```powershell
python src/app_v2.py
```

---

## Supported Leagues (examples)

```python
LEAGUES = {
    'EPL': 'Premier League',
    'ES1': 'La Liga',
    'IT1': 'Serie A',
    'DE1': 'Bundesliga',
    'FR1': 'Ligue 1'
}
```

---

## Internals and Developer Notes

1. Data collection (`src/scrapers/`) – unified scraping client for multiple APIs with retries and rate-limit handling.
2. Preprocessing (`src/preprocessing/`) – cleaning, normalization and feature engineering for ML.
3. Models (`src/models/`) – training pipelines and utilities (XGBoost, calibration, encoders/scalers).
4. Predictions (`src/predictions/`) – `match_predictor.py` is the runtime predictor used by scripts and the app.
5. Web app (`src/`) – Flask app that serves predictions and basic UI.

Data flow: Raw Data → Preprocessing → Feature Engineering → Model Training → Predictions

---

## Troubleshooting & Tips

- If you hit API rate limits, wait between requests or use multiple API keys.
- If a model training run fails, inspect `logs/train_log.txt` for stack traces and warnings.
- Keep `models/` small and store versioned artifacts (timestamped files are kept in `models/archive/`).

---

## Requirements (high level)

- Flask
- pandas, numpy
- scikit-learn, xgboost
- joblib

Use `requirements.txt` for exact pinned dependencies.

---

## Contributing

Contributions are welcome. To add leagues, update the mapping in `config/project_config.py`. To add models or features, extend files under `src/models/` and `src/preprocessing/`.

---

This README provides a concise English overview for developers and users. For developer-oriented, low-level references, see the `src/` subfolders and inline docstrings.
- **Architettura modulare** per facile manutenzione

Il sistema è **pronto per l'uso** e può essere facilmente esteso con nuove leghe, modelli o funzionalità.
