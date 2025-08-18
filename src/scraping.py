import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  # legge .env

API_KEY = os.getenv("FOOTBALL_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "FOOTBALL_API_KEY non trovata. Impostala in .env o come variabile d'ambiente."
    )

HEADERS = {"X-RapidAPI-Key": API_KEY}

API_KEY = ""

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {"X-RapidAPI-Key": API_KEY}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_matches(league_id=39, season=2025):
    """Scarica le partite di una lega per una stagione"""
    url = f"{BASE_URL}/fixtures?league={league_id}&season={season}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    fixtures = [f["fixture"] | f["teams"] | f["goals"] for f in data["response"]]  # unisce info
    df = pd.json_normalize(fixtures)
    df.to_csv(f"{DATA_DIR}/matches_{league_id}_{season}.csv", index=False)
    print(f"Salvate {len(df)} partite")
    return df

def get_teams(league_id=39, season=2025):
    """Scarica squadre e giocatori"""
    url = f"{BASE_URL}/teams?league={league_id}&season={season}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    teams = data["response"]
    all_players = []
    for t in teams:
        team_info = t["team"]
        players = t["players"]
        for p in players:
            p["team_id"] = team_info["id"]
            p["team_name"] = team_info["name"]
            all_players.append(p)
    df_players = pd.json_normalize(all_players)
    df_players.to_csv(f"{DATA_DIR}/players_{league_id}_{season}.csv", index=False)
    print(f"Salvati {len(df_players)} giocatori")
    return df_players

if __name__ == "__main__":
    # 39 = Premier League
    get_matches(league_id=39, season=2025)
    get_teams(league_id=39, season=2025)
