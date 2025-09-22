import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Chargement du modèle et scaler
# -------------------------------
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
optimal_threshold = 0.56

st.title("Prédiction de l'évolution de la drépanocytose")

# -------------------------------
# Formulaire de saisie
# -------------------------------
st.header("Entrez les informations du patient")
input_data = {}

for feature in features:
    input_data[feature] = st.text_input(feature, "0")  # par défaut "0"

# -------------------------------
# Bouton prédiction
# -------------------------------
if st.button("Prédire"):

    # Convertir en DataFrame
    new_data = pd.DataFrame([input_data])
    
    # Conversion en float
    for col in new_data.columns:
        new_data[col] = pd.to_numeric(new_data[col])
    
    # Ajouter les colonnes manquantes
    for col in features:
        if col not in new_data.columns:
            new_data[col] = 0

    # Réordonner
    new_data = new_data[features]
    
    # Standardisation
    new_data_scaled = scaler.transform(new_data)
    
    # Prédiction
    pred_proba = model.predict_proba(new_data_scaled)[:,1]
    pred_class = (pred_proba >= optimal_threshold).astype(int)
    
    st.write(f"**Classe prédite** : {pred_class[0]}")
    st.write(f"**Probabilité** : {pred_proba[0]:.2f}")
