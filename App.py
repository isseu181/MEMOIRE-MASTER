# ================================
# app.py - Déploiement Streamlit
# ================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================================
# 1️⃣ Chargement du modèle et du scaler
# ================================
model_loaded = joblib.load("random_forest_model.pkl")
scaler_loaded = joblib.load("scaler.pkl")
features_loaded = joblib.load("features.pkl")  # Colonnes utilisées lors de l'entraînement
optimal_threshold = 0.56  # Remplacer par le seuil trouvé sur validation

# ================================
# 2️⃣ Titre de l'application
# ================================
st.title("Prédiction de l'évolution des patients drépanocytaires")
st.write("Entrez les informations du patient pour prédire l'évolution (Favorable ou Complications).")

# ================================
# 3️⃣ Création du formulaire d'entrée
# ================================
def user_input_features():
    data = {}
    for col in features_loaded:
        # Pour les colonnes binaires
        if "_OUI" in col or col in ['Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
                                     'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
                                     'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
                                     'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque']:
            data[col] = st.selectbox(col, [0,1])
        # Pour les colonnes numériques
        else:
            data[col] = st.number_input(col, value=0)
    return pd.DataFrame([data])

new_data = user_input_features()

# ================================
# 4️⃣ Assurer correspondance des colonnes
# ================================
# Ajouter les colonnes manquantes
for col in features_loaded:
    if col not in new_data.columns:
        new_data[col] = 0

# Réordonner les colonnes
new_data = new_data[features_loaded]

# Convertir toutes les colonnes en float
for col in new_data.columns:
    new_data[col] = pd.to_numeric(new_data[col])

# ================================
# 5️⃣ Standardisation et prédiction
# ================================
new_data_scaled = scaler_loaded.transform(new_data)
pred_proba = model_loaded.predict_proba(new_data_scaled)[:,1]
pred_class = (pred_proba >= optimal_threshold).astype(int)

# ================================
# 6️⃣ Affichage des résultats
# ================================
st.subheader("Résultat de la prédiction")
st.write(f"Classe prédite : {pred_class[0]}  (0=Favorable, 1=Complications)")
st.write(f"Probabilité de complications : {pred_proba[0]:.2f}")
