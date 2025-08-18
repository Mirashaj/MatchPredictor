# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dummy training example
def train():
    # Qui dovresti caricare i tuoi dati reali
    X = pd.DataFrame([[1,2],[3,4]], columns=['a','b'])
    y = [0, 1]
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, '../models/rf_model.pkl', compress=3)

if __name__ == "__main__":
    train()
