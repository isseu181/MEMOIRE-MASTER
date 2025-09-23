# utils/classification.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

def show_classification():
    st.subheader("🔬 Prédiction de l'évolution des patients drépanocytaires")

    # Charger modèle + scaler
    try:
        model_loaded = joblib.load("fichiers modèles/random_forest_model.pkl")
        scaler_loaded = joblib.load("fichiers modèles/scaler.pkl")
        features_loaded = joblib.load("fichiers modèles/features.pkl")
        optimal_threshold = 0.56
        st.success("✅ Modèle et scaler chargés")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou du scaler : {e}")
        return

    st.write("Remplissez les informations ci-dessous pour estimer le risque de complications.")

    def user_input_features():
        st.subheader("📋 Informations patient")
        data = {}
        # Variables numériques
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

        # Variables binaires
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

        # Urgence
        niveau_urgence = st.selectbox("Niveau d'urgence", ["Urgence1","Urgence2","Urgence3","Urgence4","Urgence5","Urgence6"])
        data['NiveauUrgence'] = int(niveau_urgence.replace("Urgence", ""))

        # Scolarité
        niveau_sco = st.selectbox("Niveau de scolarité", ["NON","Maternelle ","Elémentaire ","Secondaire","Enseignement Supérieur "])
        mapping_sco = {"NON":0, "Maternelle ":1, "Elémentaire ":2, "Secondaire":3, "Enseignement Supérieur ":4}
        data["Niveau d'instruction scolarité"] = mapping_sco[niveau_sco]

        return pd.DataFrame([data])

    new_data = user_input_features()

    # Alignement des colonnes
    for col in features_loaded:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[features_loaded]
    new_data = new_data.astype(float)

    # Prédiction
    new_data_scaled = scaler_loaded.transform(new_data)
    pred_proba = model_loaded.predict_proba(new_data_scaled)[:,1]
    pred_class = (pred_proba >= optimal_threshold).astype(int)

    st.subheader("🩺 Résultat de la prédiction")
    if pred_class[0] == 0:
        st.success(f"✅ Évolution prédite : Favorable (probabilité de complications = {pred_proba[0]:.2f})")
    else:
        st.error(f"⚠️ Évolution prédite : Complications (probabilité = {pred_proba[0]:.2f})")

