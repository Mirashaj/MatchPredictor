from nicegui import ui, app
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configure static files early
app.add_static_files('/resources', 'resources')

# Configurazione delle immagini
def get_hero_image():
    return '/resources/graphic.png'

def get_trophy_image():
    return '/resources/trophy.png'

# Caricamento dati e modello
def get_df():
    data_path = Path('data/cleaned.csv')
    if data_path.exists():
        try:
            return pd.read_csv(data_path)
        except:
            pass
    return pd.DataFrame()

def get_model():
    return None

# Mappa risultati
RESULT_MAP = {1: 'Home Win', 0: 'Draw', -1: 'Away Win'}

# Global state for match setup
match_state = {}

# === Shared Navbar ===
def navbar():
    ui.html('''
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            width: 100vw; 
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .navbar { 
            background: #a31621; 
            color: white; 
            padding: 15px 30px; 
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1000;
        }
        .logo { 
            font-size: 24px; 
            font-weight: bold; 
            width: 200px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .logo img {
            width: 32px;
            height: 32px;
            filter: brightness(0) invert(1);
        }
        .nav-center { 
            display: flex; 
            gap: 30px; 
            flex-grow: 1;
            justify-content: center;
        }
        .nav-right { 
            display: flex; 
            gap: 10px; 
            width: 200px;
            justify-content: flex-end;
        }
        .nav-btn { 
            background: none; 
            color: #fcf7f8; 
            border: none; 
            padding: 10px 0; 
            cursor: pointer;
            text-decoration: none;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .nav-btn:hover { 
            color: #ffffff; 
            transform: translateY(-1px);
        }
        .signin-btn {
            background: #fcf7f8; 
            color: #a31621; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 8px; 
            cursor: pointer;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .signin-btn:hover { 
            background: #f0e8e9; 
            transform: translateY(-1px);
        }
        
        /* Global centering styles */
        .container, .form-container, .results-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 0 auto;
            width: 100%;
            max-width: 1200px;
            padding: 80px 20px 50px;
        }
        
        .title, .subtitle, .form-title, .results-title {
            text-align: center;
            width: 100%;
        }
        
        .cards-container, .match-cards, .results-grid {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 30px;
            margin: 0 auto;
            width: 100%;
        }
        
        .card, .match-card, .result-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .nav-right { 
            display: flex; 
            gap: 10px; 
            width: 200px;
            justify-content: flex-end;
        }
        .nav-btn { 
            background: none; 
            color: #fcf7f8; 
            border: none; 
            padding: 10px 0; 
            cursor: pointer;
            text-decoration: none;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .nav-btn:hover { 
            color: #ffffff; 
            transform: translateY(-1px);
        }
        .signin-btn {
            background: #fcf7f8; 
            color: #a31621; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 8px; 
            cursor: pointer;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .signin-btn:hover { 
            background: #f0e8e9; 
            transform: translateY(-1px);
        }
    </style>
    <div class="navbar">
        <div class="logo">
            <img src="/resources/graphic.png" alt="MatchPredictor Logo">
            MatchPredictor
        </div>
        <div class="nav-center">
            <a href="/" class="nav-btn">Home</a>
            <a href="/simulation" class="nav-btn">Simulation</a>
            <a href="/predictions" class="nav-btn">Predictions</a>
            <a href="/results" class="nav-btn">Results</a>
            <a href="/about" class="nav-btn">About</a>
        </div>
        <div class="nav-right">
            <a href="/signin" class="signin-btn">Sign In</a>
        </div>
    </div>
    ''')

@ui.page('/')
def home_page():
    navbar()
    ui.html('''
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: #a31621 !important; 
            margin: 0; 
            font-family: Arial, sans-serif; 
            width: 100vw;
            overflow-x: hidden;
        }
        .container { 
            background: #a31621; 
            min-height: 100vh; 
            padding: 50px 20px; 
            text-align: center; 
            width: 100%;
            max-width: 100vw;
        }
        .title { 
            color: white; 
            font-size: 3rem; 
            margin-bottom: 20px; 
            font-weight: bold;
        }
        .subtitle { 
            color: white; 
            font-size: 1.2rem; 
            margin-bottom: 40px; 
        }
        .cards-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 50px;
            flex-wrap: wrap;
        }
        .card { 
            background: rgba(255,255,255,0.1); 
            color: white; 
            padding: 30px; 
            border-radius: 15px; 
            cursor: pointer; 
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
            min-width: 250px;
            backdrop-filter: blur(10px);
        }
        .card:hover { 
            background: rgba(255,255,255,0.2); 
            transform: translateY(-5px);
        }
        .card h3 { margin-bottom: 15px; font-size: 1.5rem; }
        .predictions-section {
            margin-top: 60px;
        }
        .predictions-title {
            color: white;
            font-size: 2rem;
            margin-bottom: 30px;
        }
        .match-cards {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        .match-card {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 20px;
            color: white;
            min-width: 280px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .match-card:hover {
            background: rgba(255,255,255,0.2);
            transform: translateY(-3px);
        }
        .match-time { font-size: 0.9rem; opacity: 0.8; }
        .match-teams { font-size: 1.1rem; font-weight: bold; margin: 10px 0; }
        .match-league { font-size: 0.8rem; opacity: 0.7; }
    </style>
    <div class="container">
        <h1 class="title">Welcome to MatchPredictor</h1>
        <p class="subtitle">Predict football match outcomes with advanced analytics</p>
        
        <div class="cards-container">
            <div class="card" onclick="location.href='/match-setup'">
                <h3>🏟️ Match Setup</h3>
                <p>Configure your match prediction with team statistics and historical data</p>
            </div>
            <div class="card" onclick="location.href='/prediction-results'">
                <h3>🏆 Prediction Results</h3>
                <p>View detailed analysis and prediction outcomes for your matches</p>
            </div>
        </div>

        <div class="predictions-section">
            <h2 class="predictions-title">Today's Predictions</h2>
            <div class="match-cards">
                <div class="match-card" onclick="predictMatch('Arsenal', 'Chelsea')">
                    <div class="match-time">15:00</div>
                    <div class="match-teams">Arsenal vs Chelsea</div>
                    <div class="match-league">Premier League</div>
                </div>
                <div class="match-card" onclick="predictMatch('Manchester City', 'Liverpool')">
                    <div class="match-time">17:30</div>
                    <div class="match-teams">Manchester City vs Liverpool</div>
                    <div class="match-league">Premier League</div>
                </div>
                <div class="match-card" onclick="predictMatch('Tottenham', 'Newcastle')">
                    <div class="match-time">20:00</div>
                    <div class="match-teams">Tottenham vs Newcastle</div>
                    <div class="match-league">Premier League</div>
                </div>
            </div>
        </div>
    </div>
    ''')
    
    # Add JavaScript separately using ui.add_body_html()
    ui.add_body_html('''
    <script>
        function predictMatch(home, away) {
            sessionStorage.setItem('matchData', JSON.stringify({
                home: home,
                away: away,
                date: '2025-08-18',
                location: home + ' Stadium'
            }));
            location.href = '/prediction-results';
        }
    </script>
    ''')

@ui.page('/match-setup')
def match_setup_page():
    navbar()
    
    ui.html('''
    <style>
        body { background: #a31621 !important; margin: 0; font-family: Arial, sans-serif; }
        .container { 
            background: #a31621; 
            min-height: 100vh; 
            padding: 50px 20px; 
        }
        .form-container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .form-title {
            color: white;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-label {
            color: white;
            font-size: 1rem;
            margin-bottom: 8px;
            font-weight: 500;
        }
        .form-input, .form-select {
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.9);
            color: #333;
            font-size: 16px;
            outline: none;
        }
        .form-input:focus, .form-select:focus {
            background: white;
            box-shadow: 0 0 0 3px rgba(255,255,255,0.3);
        }
        .predict-btn {
            background: white;
            color: #a31621;
            border: none;
            padding: 18px 40px;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin: 30px auto 0;
            display: block;
            transition: all 0.3s ease;
        }
        .predict-btn:hover {
            background: #f0f0f0;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        @media (max-width: 768px) {
            .form-row { grid-template-columns: 1fr; gap: 20px; }
        }
    </style>
    <div class="container">
        <div class="form-container">
            <h1 class="form-title">Match Setup</h1>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Home Team</label>
                    <select class="form-select" id="homeTeam" required>
                        <option value="">Select Home Team</option>
                        <option value="Arsenal">Arsenal</option>
                        <option value="Chelsea">Chelsea</option>
                        <option value="Liverpool">Liverpool</option>
                        <option value="Manchester City">Manchester City</option>
                        <option value="Manchester United">Manchester United</option>
                        <option value="Tottenham">Tottenham</option>
                        <option value="Newcastle">Newcastle</option>
                        <option value="Brighton">Brighton</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Away Team</label>
                    <select class="form-select" id="awayTeam" required>
                        <option value="">Select Away Team</option>
                        <option value="Arsenal">Arsenal</option>
                        <option value="Chelsea">Chelsea</option>
                        <option value="Liverpool">Liverpool</option>
                        <option value="Manchester City">Manchester City</option>
                        <option value="Manchester United">Manchester United</option>
                        <option value="Tottenham">Tottenham</option>
                        <option value="Newcastle">Newcastle</option>
                        <option value="Brighton">Brighton</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Match Date</label>
                    <input type="date" class="form-input" id="matchDate" required value="2025-08-18">
                </div>
                <div class="form-group">
                    <label class="form-label">Location</label>
                    <input type="text" class="form-input" id="location" placeholder="Stadium name" required>
                </div>
            </div>
            <button type="button" class="predict-btn" onclick="submitPrediction()">Generate Prediction</button>
        </div>
    </div>
    ''')
    
    # Add JavaScript separately using ui.add_body_html()
    ui.add_body_html('''
    <script>
        function submitPrediction() {
            const homeTeam = document.getElementById('homeTeam').value;
            const awayTeam = document.getElementById('awayTeam').value;
            const matchDate = document.getElementById('matchDate').value;
            const location = document.getElementById('location').value;
            
            if (homeTeam && awayTeam && matchDate && location) {
                sessionStorage.setItem('matchData', JSON.stringify({
                    home: homeTeam,
                    away: awayTeam,
                    date: matchDate,
                    location: location
                }));
                location.href = '/prediction-results';
            } else {
                alert('Please fill in all fields');
            }
        }
    </script>
    ''')

@ui.page('/prediction-results')
def prediction_results_page():
    navbar()
    ui.html('''
    <style>
        body { background: #a31621 !important; margin: 0; font-family: Arial, sans-serif; }
        .container { 
            background: #a31621; 
            min-height: 100vh; 
            padding: 50px 20px; 
        }
        .results-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .results-title {
            color: white;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 40px;
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        .result-card {
            background: rgba(255,255,255,0.9);
            color: #a31621;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
        }
        .result-card h3 {
            font-size: 1.8rem;
            margin-bottom: 20px;
        }
        .prediction-result {
            font-size: 1.4rem;
            font-weight: bold;
            margin: 20px 0;
        }
        .probabilities {
            margin-top: 20px;
        }
        .prob-item {
            margin: 5px 0;
            font-size: 1.1rem;
        }
        .stats-card {
            background: rgba(255,255,255,0.1);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
        }
        @media (max-width: 768px) {
            .results-grid { grid-template-columns: 1fr; }
        }
    </style>
    <div class="container">
        <div class="results-container">
            <h1 class="results-title">Prediction Results</h1>
            
            <div class="results-grid">
                <div class="result-card">
                    <h3>🎯 Match Prediction</h3>
                    <div class="prediction-result" id="matchResult">Loading prediction...</div>
                    <div class="probabilities" id="probabilities">
                        <div class="prob-item">Home Win: 45%</div>
                        <div class="prob-item">Draw: 25%</div>
                        <div class="prob-item">Away Win: 30%</div>
                    </div>
                </div>
                
                <div class="result-card">
                    <h3>📊 Key Statistics</h3>
                    <div id="keyStats">
                        <p>Historical head-to-head record</p>
                        <p>Recent form analysis</p>
                        <p>Goal scoring trends</p>
                        <p>Defensive performance</p>
                    </div>
                </div>
            </div>
            
            <div class="stats-card">
                <h3>Team Performance Analysis</h3>
                <p>Detailed statistical breakdown and performance metrics will be displayed here based on historical data and current form.</p>
            </div>
        </div>
    </div>
    ''')
    
    # Add JavaScript separately using ui.add_body_html()
    ui.add_body_html('''
    <script>
        window.onload = function() {
            const matchData = JSON.parse(sessionStorage.getItem('matchData') || '{}');
            if (matchData.home && matchData.away) {
                document.getElementById('matchResult').textContent = 
                    matchData.home + ' vs ' + matchData.away + ': Home Win Predicted';
            }
        };
    </script>
    ''')

@ui.page('/simulation')
def simulation_page():
    navbar()
    ui.html('''
    <div style="background: #a31621; min-height: 100vh; color: #fcf7f8; padding: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <h1 style="text-align: center; margin-bottom: 40px;">Match Simulation</h1>
            <div style="background: rgba(252, 247, 248, 0.1); padding: 30px; border-radius: 15px;">
                <p style="text-align: center; font-size: 18px;">Advanced match simulation features coming soon...</p>
            </div>
        </div>
    </div>
    ''')

@ui.page('/predictions')
def predictions_page():
    navbar()
    ui.html('''
    <div style="background: #a31621; min-height: 100vh; color: #fcf7f8; padding: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <h1 style="text-align: center; margin-bottom: 40px;">Predictions</h1>
            <div style="background: rgba(252, 247, 248, 0.1); padding: 30px; border-radius: 15px;">
                <p style="text-align: center; font-size: 18px;">Match predictions and analytics coming soon...</p>
            </div>
        </div>
    </div>
    ''')

@ui.page('/results')
def results_page():
    navbar()
    ui.html('''
    <div style="background: #a31621; min-height: 100vh; color: #fcf7f8; padding: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <h1 style="text-align: center; margin-bottom: 40px;">Results</h1>
            <div style="background: rgba(252, 247, 248, 0.1); padding: 30px; border-radius: 15px;">
                <p style="text-align: center; font-size: 18px;">Match results and statistics coming soon...</p>
            </div>
        </div>
    </div>
    ''')

@ui.page('/about')
def about_page():
    navbar()
    ui.html('''
    <div style="background: #a31621; min-height: 100vh; color: #fcf7f8; padding: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <h1 style="text-align: center; margin-bottom: 40px;">About MatchPredictor</h1>
            <div style="background: rgba(252, 247, 248, 0.1); padding: 30px; border-radius: 15px;">
                <p style="text-align: center; font-size: 18px; margin-bottom: 20px;">
                    MatchPredictor is an advanced football match prediction platform that uses machine learning 
                    algorithms to analyze team performance, historical data, and statistical patterns.
                </p>
                <p style="text-align: center; font-size: 16px;">
                    Our goal is to provide accurate predictions and insights for football enthusiasts and analysts.
                </p>
            </div>
        </div>
    </div>
    ''')

@ui.page('/signin')
def signin_page():
    navbar()
    ui.html('''
    <div style="background: #a31621; min-height: 100vh; color: #fcf7f8; padding: 40px;">
        <div style="max-width: 400px; margin: 0 auto;">
            <h1 style="text-align: center; margin-bottom: 40px;">Sign In</h1>
            <div style="background: rgba(252, 247, 248, 0.1); padding: 30px; border-radius: 15px;">
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 8px; font-weight: 500;">Email</label>
                    <input type="email" style="width: 100%; padding: 12px; border: none; border-radius: 8px; background: #fcf7f8; color: #a31621;" placeholder="Enter your email">
                </div>
                <div style="margin-bottom: 30px;">
                    <label style="display: block; margin-bottom: 8px; font-weight: 500;">Password</label>
                    <input type="password" style="width: 100%; padding: 12px; border: none; border-radius: 8px; background: #fcf7f8; color: #a31621;" placeholder="Enter your password">
                </div>
                <button style="width: 100%; padding: 12px; background: #fcf7f8; color: #a31621; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                    Sign In
                </button>
                <p style="text-align: center; margin-top: 20px; font-size: 14px;">
                    Don't have an account? <a href="#" style="color: #fcf7f8;">Sign up here</a>
                </p>
            </div>
        </div>
    </div>
    ''')

# Static file serving e avvio
def run_app():
    ui.run(title='MatchPredictor', port=8080, show=False)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()
