import streamlit as st
try:
# En tout premier ligne :
import sys
print(f"Python version: {sys.version}", file=sys.stderr)  # Pour vérification

try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
import pandas as pd
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Détection IRC - Interface Complète",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Charge le modèle et le scaler avec gestion robuste des chemins"""
    try:
        script_dir = Path(__file__).parent
        models_dir = script_dir.parent / "models"
        model = joblib.load(models_dir / "xgboost_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"""
        **Erreur de chargement des ressources**:
        {str(e)}
        
        Vérifiez que:
        1. Les fichiers .pkl existent dans `projet_irc/models/`
        2. La structure des dossiers est correcte
        """)
        st.stop()

# Chargement des ressources
model, scaler = load_model()

# =============================================
# SECTION INTERFACE UTILISATEUR
# =============================================

st.title("🩺 Modèle de Détection d'IRC")
st.markdown("""
    *Remplissez tous les champs ci-dessous pour obtenir une évaluation du risque*  
    *Les champs marqués d'un astérisque (*) sont obligatoires*
""")

with st.form("patient_form"):
    # ============= COLONNE 1 =============
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations démographiques")
        age = st.slider("Âge*", 18, 120, 50)
        sexe = st.radio("Sexe*", ["Homme", "Femme"], horizontal=True)
        bmi = st.number_input("IMC (kg/m²)*", 10.0, 50.0, 25.0, step=0.1)
        
        st.subheader("Paramètres cliniques")
        hypertension = st.radio("Hypertension artérielle*", ["Non", "Oui"], horizontal=True)
        diabete = st.radio("Diabète*", ["Non", "Oui"], horizontal=True)
        oedeme = st.radio("Œdème visible*", ["Non", "Oui"], horizontal=True)

    # ============= COLONNE 2 =============
    with col2:
        st.subheader("Marqueurs biologiques")
        gfr = st.number_input("DFG (mL/min/1.73m²)*", 0.0, 200.0, 90.0, step=0.1)
        creatinine = st.number_input("Créatinine (mg/dL)*", 0.1, 20.0, 0.9, step=0.1)
        acr = st.number_input("ACR (mg/g)*", 0.0, 5000.0, 30.0, step=1.0)
        
        st.subheader("Traitements")
        nsaids = st.slider(
    "Score d'utilisation d'AINS* (0-9.9)",
    min_value=0.0,
    max_value=9.9,
    value=0.0,
    step=0.1,
    format="%.1f",
    help="""Échelle complète du score :
    0 = Aucun - 3 = Usage modéré - 6 = Usage important - 9.9 = Usage très intensif"""
)

       
    # Bouton de soumission
    submitted = st.form_submit_button("Prédire le risque", type="primary")

# =============================================
# SECTION PRÉDICTION
# =============================================
if submitted:
    try:
        # Conversion des données
        input_data = {
            'GFR (mL/min)': gfr,
            'Créatinine (mg/dL)': creatinine,
            'ACR (mg/g)': acr,
            'Hypertension (0/1)': 1 if hypertension == "Oui" else 0,
            'Diabète (0/1)': 1 if diabete == "Oui" else 0,
            'BMI (kg/m²)': bmi,
            'Âge': age,
            'Sexe (0=H, 1=F)': 1 if sexe == "Femme" else 0,
            'NSAIDs (score)': nsaids,
            'Œdème (0/1)': 1 if oedeme == "Oui" else 0
        }

        # Vérification des données
        required_fields = ['GFR (mL/min)', 'Créatinine (mg/dL)', 'ACR (mg/g)', 'BMI (kg/m²)']
        for field in required_fields:
            if input_data[field] <= 0:
                st.warning(f"La valeur de {field} doit être positive!")
                st.stop()

        # Prétraitement et prédiction
        input_df = pd.DataFrame([input_data])
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        proba = model.predict_proba(scaled_data)[0][1]

        # Affichage des résultats
        st.divider()
        st.subheader("📊 Résultats de l'évaluation")

        if prediction == 1:
            st.error(f"## Risque élevé d'IRC ({proba:.1%} de probabilité)")
            st.progress(proba, text="Niveau de risque")
            
            st.markdown("""
                **Recommandations:**
                - Consultation néphrologique urgente
                - Bilan rénal complet
                - Surveillance tensionnelle rapprochée
            """)
        else:
            st.success(f"## Risque faible d'IRC ({(1-proba):.1%} de probabilité)")
            st.progress(1-proba, text="Niveau de risque")
            
            st.markdown("""
                **Recommandations:**
                - Surveillance annuelle de la fonction rénale
                - Maintenir une hydratation adéquate
                - Éviter l'automédication par AINS
            """)

        # Détails techniques (optionnel)
        with st.expander("🔍 Afficher les données techniques"):
            st.json(input_data)
            st.write("Valeurs standardisées:", scaled_data.tolist()[0])

    except Exception as e:
        st.error(f"Une erreur est survenue: {str(e)}")
        st.warning("Veuillez vérifier toutes les entrées et réessayer")

# Pied de page
st.divider()
st.caption("""
    *Système expert développé par [Votre Nom] - Utilisation réservée aux professionnels de santé*  
    *Les résultats doivent être interprétés dans un contexte clinique global*
""")
