"""
MatchPredictor Web Interface
Flask application for football match predictions

Data source:
- This deployment uses local CSV files placed in data/ and data/processed/.
- External API ingestion scripts are optional and not required to run the app.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.append('src')
try:
    from predictions.match_predictor import MatchPredictor
    predictor = MatchPredictor()
    print("✅ MatchPredictor initialized")
except Exception as e:
    print(f"❌ Error initializing MatchPredictor: {e}")
    predictor = None

# Use absolute paths for template and static folders so Flask can find them
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, 'html')
static_dir = os.path.join(BASE_DIR, 'css')

# Flask's `static_folder` is used for CSS/JS; keep it pointed at the css folder.
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)  # allow dev React to call API; tighten in production

# Serve project-level resources (images, icons) from the top-level `resources/` folder
RESOURCES_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'resources'))


@app.route('/resources/<path:filename>')
def resources_file(filename):
    """Serve files from the repository-level `resources/` directory.

    Templates reference `/resources/graphic.png`; this route makes that URL work
    without moving the image into the Flask static folder.
    """
    return send_from_directory(RESOURCES_DIR, filename)

@app.route('/')
def home():
    """Home page with prediction interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for match predictions"""
    try:
        data = request.get_json()

        home_team = data.get('home_team')
        away_team = data.get('away_team')
        match_date = data.get('match_date', datetime.now().strftime('%Y-%m-%d'))

        if not predictor:
            return jsonify({'error': 'Prediction system not available'}), 500

        # Make prediction
        prediction = predictor.predict_match(home_team, away_team, match_date)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Expects JSON: {"home_team":"Team A","away_team":"Team B","match_date":"YYYY-MM-DD"}
    Returns JSON: {"p_home":0.7,"p_draw":0.15,"p_away":0.15,"pred":"Home Win"}
    """
    data = request.get_json() or {}
    home = data.get('home_team')
    away = data.get('away_team')
    match_date = data.get('match_date')
    if not home or not away:
        return jsonify({"error": "home_team and away_team required"}), 400

    if not predictor:
        return jsonify({'error': 'Prediction system not available'}), 500

    probs = predictor.predict_prob(home_team=home, away_team=away, match_date=match_date)
    # probs expected as dict {'home':..., 'draw':..., 'away':...}
    p_home, p_draw, p_away = probs['home'], probs['draw'], probs['away']
    pred = "Home Win" if p_home >= max(p_draw, p_away) else ("Draw" if p_draw >= p_away else "Away Win")
    return jsonify({"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "pred": pred})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/match-setup')
def match_setup():
    return render_template('match-setup.html')

@app.route('/prediction-results')
def prediction_results():
    return render_template('prediction-results.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
