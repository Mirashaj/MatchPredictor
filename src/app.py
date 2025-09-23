"""
MatchPredictor Web Interface
Flask application for football match predictions
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
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

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
