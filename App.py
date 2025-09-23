# ================================
# app.py - Déploiement Streamlit (version améliorée)
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
st.title("🔬 Prédiction de l'évolution des patients drépanocytaires")
st.write("Remplissez les informations ci-dessous pour estimer le risque de complications.")

# ================================
# 3️⃣ Formulaire utilisateur
# ================================
def user_input_features():
    st.subheader("📋 Informations patient")

    data = {}

    # --- Variables numériques importantes ---
    data['Âge de début des signes (en mois)'] = st.slider("Âge de début des signes (mois)", 0, 200, 60)
    data['Âge du debut d etude en mois (en janvier 2023)'] = st.slider("Âge de l’étude (mois)", 0, 300, 120)
    data['Âge de découverte de la drépanocytose (en mois)'] = st.slider("Âge découverte drépanocytose (mois)", 0, 200, 50)
    data['GR (/mm3)'] = st.number_input("Globules rouges (GR /mm3)", min_value=1000000, max_value=7000000, value=4500000)
    data['GB (/mm3)'] = st.number_input("Globules blancs (GB /mm3)", min_value=2000, max_value=50000, value=10000)
    data['PLT (/mm3)'] = st.number_input("Plaquettes (PLT /mm3)", min_value=10000, max_value=1000000, value=300000)
    data['HB (g/dl)'] = st.slider("Hémoglobine (g/dl)", 0.0, 20.0, 10.0, 0.1)
    data["Taux d'Hb (g/dL)"] = st.slider("Taux Hb (g/dl)", 0.0, 20.0, 10.0, 0.1)
    data['CRP Si positive (Valeur)'] = st.slider("CRP (mg/L)", 0, 300, 10)
    data['% d\'Hb S'] = st.slider("% Hb S", 0, 100, 70)
    data['% d\'Hb F'] = st.slider("% Hb F", 0, 100, 5)

    # --- Variables catégorielles binaires ---
    binary_vars = {
        'Pâleur': "Présence de pâleur",
        'Souffle systolique fonctionnel': "Souffle systolique fonctionnel",
        'Vaccin contre méningocoque': "Vaccin contre le méningocoque",
        'Splénomégalie': "Splénomégalie",
        'Prophylaxie à la pénicilline': "Prophylaxie à la pénicilline",
        'Parents Salariés': "Parents salariés",
        'Prise en charge Hospitalisation': "Hospitalisation prise en charge",
        'Radiographie du thorax Oui ou Non': "Radiographie thorax",
        'Douleur provoquée (Os.Abdomen)': "Douleur provoquée",
        'Vaccin contre pneumocoque': "Vaccin contre le pneumocoque"
    }
    for col, label in binary_vars.items():
        data[col] = st.selectbox(label, ["NON", "OUI"])
        data[col] = 1 if data[col] == "OUI" else 0

    # --- Niveau d'urgence ---
    niveau_urgence = st.selectbox("Niveau d'urgence", ["Urgence1","Urgence2","Urgence3","Urgence4","Urgence5","Urgence6"])
    data['NiveauUrgence'] = int(niveau_urgence.replace("Urgence", ""))

    # --- Niveau d’instruction ---
    niveau_sco = st.selectbox("Niveau de scolarité", ["NON","Maternelle ","Elémentaire ","Secondaire","Enseignement Supérieur "])
    mapping_sco = {"NON":0, "Maternelle ":1, "Elémentaire ":2, "Secondaire":3, "Enseignement Supérieur ":4}
    data["Niveau d'instruction scolarité"] = mapping_sco[niveau_sco]

    return pd.DataFrame([data])

# Récupération des données utilisateur
new_data = user_input_features()

# ================================
# 4️⃣ Alignement des colonnes
# ================================
for col in features_loaded:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[features_loaded]  # Réordonne les colonnes
new_data = new_data.astype(float)     # Conversion en float

# ================================
# 5️⃣ Standardisation + prédiction
# ================================
new_data_scaled = scaler_loaded.transform(new_data)
pred_proba = model_loaded.predict_proba(new_data_scaled)[:,1]
pred_class = (pred_proba >= optimal_threshold).astype(int)

# ================================
# 6️⃣ Résultats
# ================================
st.subheader("🩺 Résultat de la prédiction")
if pred_class[0] == 0:
    st.success(f"✅ Évolution prédite : Favorable (probabilité de complications = {pred_proba[0]:.2f})")
else:
    st.error(f"⚠️ Évolution prédite : Complications (probabilité = {pred_proba[0]:.2f})")
