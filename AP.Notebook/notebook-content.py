# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

# Cellule 1: Installation des packages
%pip install streamlit==1.28.0
%pip install catboost==1.2.0
%pip install shap==0.42.0
%pip install plotly==5.15.0

# Pour Fabric, utilisez %pip au lieu de requirements.txt
print("✅ Packages installés!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ajouter le chemin pour importer utils
sys.path.append('/lakehouse/default/Files')

# Importations après installation
import joblib
import shap
import matplotlib.pyplot as plt

# Configuration Streamlit
st.set_page_config(
    page_title="Cervical Cancer Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cellule 3: Chargement du modèle depuis le Lakehouse
def load_model():
    """Charge le modèle depuis le Lakehouse Fabric"""
    try:
        # Chemin dans le Lakehouse
        model_path = "/lakehouse/default/Files/models/cancerPredictionModel_catBoost.cbm"
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Modèle chargé depuis le Lakehouse")    
        else:
                raise FileNotFoundError("Modèle non trouvé")
        
        # Noms des features (adapter selon votre modèle)
        feature_names = [
            'Age', 'NumOfPregnancies', 'FirstSexualIntercourse', 'NumberOfSexualPartners',
            'Schiller', 'Hinselmann', 'Citology', 'DxCancer', 'HormonalContraceptives',
            'HormonalContraceptivesYears', 'IUDYears','STDsGenitalHerpes', 'SmokesPacksYear'
        ]
        
        
        return model, feature_names
        
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        return None, []

# Charger le modèle
model, feature_names = load_model()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# SOLUTION SIMPLE - Obtenir l'URL d'accès
import socket
from IPython.display import display, HTML

# Obtenir les informations de connexion
hostname = socket.gethostname()
try:
    ip = socket.gethostbyname(hostname)
except:
    ip = "localhost"

# Générer le lien
streamlit_url = f"https://{ip}:8501"

print("🎯 STREAMLIT EST PRÊT !")
print("📋 Copiez et collez cette URL dans votre navigateur :")
print(f"🔗 {streamlit_url}")

# Lien cliquable
display(HTML(f'''
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h3>🚀 Accéder à l'Application Streamlit</h3>
    <p><a href="{streamlit_url}" target="_blank" style="color: #1f77b4; font-size: 16px; text-decoration: none;">
        Cliquez ici pour ouvrir l'application
    </a></p>
    <p><small>Si le lien ne fonctionne pas, copiez-collez : <code>{streamlit_url}</code></small></p>
</div>
'''))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def predict_risk(model, patient_data, feature_names):
    """Fait une prédiction de risque"""
    try:
        # Convertir en DataFrame
        df = pd.DataFrame([patient_data])
        
        # S'assurer que toutes les features sont présentes
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[feature_names]
        
        # Prédiction
        probability = model.predict_proba(df)[0, 1]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Erreur prédiction: {e}")
        return 0, 0.0

def explain_prediction(model, patient_data, feature_names):
    """Génère une explication SHAP"""
    try:
        df = pd.DataFrame([patient_data])
        df = df[feature_names]
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        
        # Graphique force plot
        st.subheader("📊 Impact des caractéristiques sur la prédiction")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Erreur explication: {e}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# Configuration de la page
st.set_page_config(
    page_title="Cervical Cancer Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .feature-importance {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">🏥 Cervical Cancer Risk Prediction Tool</div>', 
            unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", 
                       ["Risk Prediction", "Model Explanation", "About"])

# Charger le modèle (avec cache)
@st.cache_resource
def load_app_model():
    return load_model()

model, feature_names = load_app_model()

if page == "Risk Prediction":
    # Section de prédiction de risque
    st.header("Patient Risk Assessment")
    
    # Formulaire en deux colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Medical History")
        
        # Données médicales
        age = st.number_input("Age", min_value=15, max_value=60, value=35)
        num_pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=10, value=1)
        first_sexual_intercourse = st.number_input("Age at First Sexual Intercourse", 
                                                 min_value=10, max_value=30, value=18)
        num_sexual_partners = st.number_input("Number of Sexual Partners", 
                                            min_value=0, max_value=20, value=1)
        
        # Tests médicaux
        schiller = st.selectbox("Schiller Test Result", 
                               ["Normal", "Abnormal", "Not Performed"])
        hinselmann = st.selectbox("Hinselmann Test Result", 
                                 ["Normal", "Abnormal", "Not Performed"])
        citology = st.selectbox("Citology Result", 
                               ["Normal", "Abnormal", "Not Performed"])
        DxCancer = st.selectbox("Cancer Result", 
                               ["Normal", "Abnormal", "Not Performed"])                       

    
    with col2:
        st.subheader("Behavioral Factors")
        
        # Contraception
        hormonal_contraceptives = st.selectbox("Hormonal Contraceptives Use", 
                                              ["No", "Yes", "Former User"])
        hormonal_contraceptives_years = st.number_input("Years of Hormonal Contraceptives Use", 
                                                       min_value=0, max_value=30, value=0)
        iud_years = st.number_input("Years of IUD Use", min_value=0, max_value=30, value=0)
        
        # STD History
        stds_genital_herpes = st.selectbox("Genital Herpes History", ["No", "Yes"])
        
        # Lifestyle
        smokes_packs_year = st.number_input("Smoking (Pack-Years)", min_value=0.0, max_value=50.0, value=0.0)
    
    # Bouton de prédiction
    if st.button("Assess Cancer Risk", type="primary"):
        # Préparer les données
        patient_data = {
            'Age': age,
            'NumOfPregnancies': num_pregnancies,
            'FirstSexualIntercourse': first_sexual_intercourse,
            'NumberOfSexualPartners': num_sexual_partners,
            'Schiller': 1 if schiller == "Abnormal" else 0,
            'Hinselmann': 1 if hinselmann == "Abnormal" else 0,
            'Citology': 1 if citology == "Abnormal" else 0,
            'DxCancer': 1 if DxCancer == "Abnormal" else 0,
            'HormonalContraceptives': 1 if hormonal_contraceptives in ["Yes", "Former User"] else 0,
            'HormonalContraceptivesYears': hormonal_contraceptives_years,
            'IUDYears': iud_years,
            'STDsGenitalHerpes': 1 if stds_genital_herpes == "Yes" else 0,
            'SmokesPacksYear': smokes_packs_year
        }
        
        # Faire la prédiction
        prediction, probability = predict_risk(model, patient_data, feature_names)
        
        # Afficher les résultats
        st.markdown("---")
        st.subheader("🔍 Risk Assessment Results")
        
        # Affichage visuel du risque
        risk_percentage = probability * 100
        
        if prediction == 1:
            st.markdown(f'<div class="risk-high">🚨 HIGH RISK: {risk_percentage:.1f}% probability</div>', 
                       unsafe_allow_html=True)
            st.progress(probability)
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK: {risk_percentage:.1f}% probability</div>', 
                       unsafe_allow_html=True)
            st.progress(probability)
        
        # Détails de la prédiction
        with st.expander("View Prediction Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Probability", f"{risk_percentage:.1f}%")
                st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
            with col2:
                st.metric("Confidence", f"{max(probability, 1-probability)*100:.1f}%")
                st.metric("Threshold", "> 50%")
        
        # Explication SHAP
        st.subheader("📊 Prediction Explanation")
        explain_prediction(model, patient_data, feature_names)

elif page == "Model Explanation":
    # Page d'explication du modèle
    st.header("Model Interpretation")
    
    st.info("""
    This section explains how the model makes predictions using SHAP (SHapley Additive exPlanations). 
    SHAP values show the contribution of each feature to the final prediction.
    """)
    
    # Importance globale des features
    st.subheader("Global Feature Importance")
    
    # Calculer l'importance SHAP globale
    @st.cache_data
    def get_global_shap():
        # Générer des données d'exemple pour le calcul SHAP
        background_data = shap.sample(pd.DataFrame([{f: 0 for f in feature_names}]), 100)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background_data)
        return explainer, shap_values, background_data
    
    explainer, shap_values, background_data = get_global_shap()
    
    # Graphique d'importance globale
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, background_data, feature_names=feature_names, show=False)
    st.pyplot(fig)
    
    # Explication détaillée
    st.subheader("How to Interpret the Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Feature Importance:**
        - Features are ordered by importance
        - Higher on the list = more impact on prediction
        - Color shows feature value (red=high, blue=low)
        """)
    
    with col2:
        st.markdown("""
        **SHAP Values:**
        - Right side = increases cancer risk
        - Left side = decreases cancer risk
        - Longer bars = stronger effect
        """)
    
    # Dependence plots pour les top features
    st.subheader("Feature Effects Analysis")
    
    top_features = ['Schiller', 'Age', 'HormonalContraceptivesYears', 'Hinselmann']
    selected_feature = st.selectbox("Select feature to analyze:", top_features)
    
    if selected_feature:
        fig_dependence, ax_dependence = plt.subplots(figsize=(10, 6))
        feature_index = feature_names.index(selected_feature)
        shap.dependence_plot(feature_index, shap_values, background_data, 
                           feature_names=feature_names, show=False)
        st.pyplot(fig_dependence)

elif page == "About":
    # Page d'information
    st.header("About This Tool")
    
    st.markdown("""
    ## Cervical Cancer Risk Prediction Tool
    
    This application uses machine learning to assess cervical cancer risk based on 
    patient medical history and behavioral factors.
    
    ### How It Works
    - **Model**: CatBoost Classifier trained on medical dataset
    - **Features**: 25 clinical and behavioral factors
    - **Interpretation**: SHAP explanations for transparency
    
    ### Intended Use
    - For healthcare professionals only
    - Supplementary tool for clinical decision making
    - Not a replacement for medical diagnosis
    
    ### Model Performance
    - Recall: 88% (detects 88% of high-risk cases)
    - Precision: 78% (78% of high-risk predictions are correct)
    - F1-Score: 82% (balanced performance metric)
    """)
    
    # Information technique
    with st.expander("Technical Details"):
        st.write(f"**Features used:** {len(feature_names)}")
        st.write("**Top 5 most important features:**")
        st.write("1. Schiller Test Result")
        st.write("2. Age")
        st.write("3. Hormonal Contraceptives Years")
        st.write("4. Hinselmann Test Result")
        st.write("5. First Sexual Intercourse Age")

# Footer
st.markdown("---")
st.markdown("*For medical professionals only. Always confirm with clinical assessment.*")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cellule séparée: Sauvegarde du modèle dans le Lakehouse
def save_model_to_fabric(model, model_name="catboost_model"):
    """Sauvegarde le modèle dans le Lakehouse Fabric"""
    
    # Créer le dossier models s'il n'existe pas
    models_dir = "/lakehouse/default/Files/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Chemin de sauvegarde
    model_path = f"{models_dir}/{model_name}.joblib"
    
    # Sauvegarder
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")
    return model_path

# Exemple d'utilisation (à exécuter une fois)
# save_model_to_fabric(votre_modele_entraine)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************



# Vérifier l'installation
!streamlit version
if __name__ == "__main__":
    # Cette commande lance Streamlit dans Fabric
    !streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

!python -m pip install --upgrade streamlit

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
