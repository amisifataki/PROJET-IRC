import joblib
import pandas as pd
from pathlib import Path

def load_ressources():
    """Charge les modèles et le scaler de manière sécurisée"""
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    
    try:
        model = joblib.load(models_dir / "xgboost_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        print("Modèle et scaler chargés avec succès")
        return model, scaler
    except Exception as e:
        raise FileNotFoundError(f"""
        Erreur de chargement des ressources: {e}
        Chemin essayé: {models_dir}
        Vérifiez que:
        1. Les fichiers .pkl existent bien
        2. La structure des dossiers est correcte
        """)

# Chargement une seule fois au démarrage
model, scaler = load_ressources()

def predict_irc(input_data):
    """Fonction de prédiction sécurisée"""
    try:
        input_df = pd.DataFrame([input_data])
        scaled_data = scaler.transform(input_df)
        
        prediction = model.predict(scaled_data)
        proba = model.predict_proba(scaled_data)[0][1]
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(proba)
        }
    except Exception as e:
        raise ValueError(f"Erreur de prédiction: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    test_patient = {
        'GFR (mL/min)': 45.7,
        'Créatinine (mg/dL)': 4.96,
        'ACR (mg/g)': 123.8,
        'Hypertension (0/1)': 0,
        'Diabète (0/1)': 1,
        'BMI (kg/m²)': 31.1,
        'Âge': 71,
        'Sexe (0=H, 1=F)': 0,
        'NSAIDs (score)': 4.56,
        'Œdème (0/1)': 0
    }
    
    try:
        result = predict_irc(test_patient)
        print(f"Résultat: {result}")
    except Exception as e:
        print(f"ERREUR: {e}")