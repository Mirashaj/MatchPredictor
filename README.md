# MatchPredictor 📈

A modern web application for predicting football match outcomes using machine learning algorithms.

![MatchPredictor Interface](resources/screenshot.png)

## Features

- **Modern UI**: Clean, responsive design with gradient backgrounds and glassmorphism effects
- **Team Selection**: Choose home and away teams from dropdown menus
- **Match Setup**: Input match date and location
- **AI Predictions**: Get probability predictions for Home Win, Draw, and Away Win
- **Real-time Results**: Instant prediction results with smooth animations

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Data Processing**: pandas, numpy
- **Web Scraping**: requests, beautifulsoup4

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MatchPredictor.git
cd MatchPredictor
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and visit: `http://localhost:5000`

## Project Structure

```
MatchPredictor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── styles.css    # Main stylesheet
│   └── js/
│       └── main.js       # Frontend JavaScript
├── data/                 # Historical match data
├── models/              # Trained ML models
├── src/                 # Source code modules
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

1. **Select Teams**: Choose the home and away teams from the dropdown menus
2. **Set Date**: Select the match date (defaults to tomorrow)
3. **Enter Location**: Specify where the match will be played
4. **Get Prediction**: Click the "Predict" button to get AI-powered predictions

## API Endpoints

### POST /predict
Predicts match outcome based on team and match information.

**Request Body:**
```json
{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2024-03-15",
    "location": "Emirates Stadium"
}
```

**Response:**
```json
{
    "success": true,
    "prediction": {
        "home_win_probability": 45.2,
        "draw_probability": 28.1,
        "away_win_probability": 26.7,
        "predicted_result": "Home Win"
    },
    "match_info": {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": "2024-03-15",
        "location": "Emirates Stadium"
    }
}
```

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## Machine Learning Models

The application uses various ML algorithms to predict match outcomes:
- Historical match data analysis
- Team performance metrics
- Home advantage calculations
- Recent form analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
