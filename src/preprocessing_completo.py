import pandas as pd

# Carica CSV
matches_df = pd.read_csv("data/matches_39_2025.csv")
players_df = pd.read_csv("data/players_39_2025.csv")

# -----------------------------
# 1. Preprocessing di base match
# -----------------------------
# Colonne utili
matches_df = matches_df[[
    "id", "date", "home.id", "home.name", "away.id", "away.name",
    "goals.home", "goals.away"
]]

# Converti la data
matches_df['date'] = pd.to_datetime(matches_df['date'])

# -----------------------------
# 2. Creazione feature squadra
# -----------------------------
# Media gol squadra nelle ultime 5 partite
def calc_recent_goals(df, team_col, goals_col):
    df_sorted = df.sort_values('date')
    rolling = df_sorted.groupby(team_col)[goals_col].rolling(5, min_periods=1).mean().reset_index()
    rolling = rolling.rename(columns={goals_col: f"{team_col}_recent_goals"})
    return rolling[[team_col, 'level_1', f"{team_col}_recent_goals"]]

home_goals = calc_recent_goals(matches_df, 'home.name', 'goals.home')
away_goals = calc_recent_goals(matches_df, 'away.name', 'goals.away')

matches_df = matches_df.merge(home_goals, left_index=True, right_on='level_1')
matches_df = matches_df.merge(away_goals, left_index=True, right_on='level_1')
matches_df.drop(['key_0', 'level_1_x', 'level_1_y'], axis=1, errors='ignore', inplace=True)

# -----------------------------
# 3. Feature giocatori
# -----------------------------
# Numero di top scorer titolari per squadra
# Supponiamo che i top scorer siano i primi 5 giocatori per gol in stagione
top_scorers = players_df.groupby('team_id').apply(
    lambda x: x.nlargest(5, 'goals.total')  # colonna 'goals.total' ipotetica
).reset_index(drop=True)

def count_top_scorers(team_id):
    return top_scorers[top_scorers['team_id'] == team_id].shape[0]

matches_df['home_top_scorers'] = matches_df['home.id'].apply(count_top_scorers)
matches_df['away_top_scorers'] = matches_df['away.id'].apply(count_top_scorers)

# -----------------------------
# 4. Target
# -----------------------------
# 0 = pareggio, 1 = vittoria home, 2 = vittoria away
def result(row):
    if row['goals.home'] > row['goals.away']:
        return 1
    elif row['goals.home'] < row['goals.away']:
        return 2
    else:
        return 0

matches_df['result'] = matches_df.apply(result, axis=1)

# -----------------------------
# 5. Salvataggio CSV pronto per ML
# -----------------------------
matches_df.to_csv("data/matches_features.csv", index=False)
print("Preprocessing completato, CSV pronto per il modello!")
