import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import os

# --- Nouveau code de chargement sécurisé ---
# 1. Définition du chemin
script_dir = Path(__file__).parent
data_path = script_dir.parent / "data" / "Dataset_IRC.xlsx"

# 2. Vérification et chargement
try:
    data = pd.read_excel(data_path, sheet_name='Données Brutes')
    print("Données chargées avec succès !")
except Exception as e:
    raise FileNotFoundError(f"Erreur de chargement : {e}\nChemin essayé : {data_path}")

# --- Prétraitement ---
X = data.drop(['PatientID', 'Diagnostic (0/1)'], axis=1)
y = data['Diagnostic (0/1)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Entraînement ---
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
model.fit(X_scaled, y)

# --- Sauvegarde sécurisée ---
models_dir = script_dir.parent / "models"
models_dir.mkdir(exist_ok=True)  # Crée le dossier si inexistant

try:
    joblib.dump(model, models_dir / "xgboost_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print(f"Modèles sauvegardés dans {models_dir}")
except Exception as e:
    raise IOError(f"Erreur de sauvegarde : {e}")

print("Pipeline complet exécuté avec succès!")