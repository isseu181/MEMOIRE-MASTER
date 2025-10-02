
# ====================================
# deployment.py - Déploiement Random Forest
# ====================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def show_deployment():
    st.set_page_config(page_title="Déploiement Random Forest", layout="wide")
    st.markdown("<h1 style='text-align:center;color:darkgreen;'>Déploiement - Random Forest</h1>", unsafe_allow_html=True)
    
    # -------------------------------
    # Charger le modèle et le scaler
    # -------------------------------
    model = joblib.load("random_forest_model.pkl")  # Modèle Random Forest sauvegardé
    scaler = joblib.load("scaler.pkl")             # Scaler utilisé pour les variables quantitatives

    # Variables quantitatives
    quantitative_vars = [
        'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
        'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
    ]

    # Variables binaires
    binary_vars = [
        'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
        'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
        'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
        'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque'
    ]

    st.markdown("### Formulaire de saisie pour prédire l’évolution d’un patient")

    # Extraire les catégories exactes du modèle entraîné
    model_features = model.feature_names_in_
    diagnostic_cols = [c for c in model_features if "Diagnostic Catégorisé_" in c]
    mois_cols = [c for c in model_features if "Mois_" in c]

    # Extraire les catégories originales pour le formulaire
    diagnostic_categories = [c.replace("Diagnostic Catégorisé_", "") for c in diagnostic_cols]
    mois_categories = [c.replace("Mois_", "") for c in mois_cols]

    # ============================
    # Formulaire Streamlit
    # ============================
    with st.form("patient_form"):
        st.subheader("Variables quantitatives")
        quantitative_inputs = {var: st.number_input(var, value=0.0) for var in quantitative_vars}

        st.subheader("Variables binaires (OUI=1, NON=0)")
        binary_inputs = {var: st.selectbox(var, options=[0,1]) for var in binary_vars}

        st.subheader("Variables ordinales")
        niveau_urgence = st.slider("Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 1, 6, 1)
        niveau_instruction = st.selectbox(
            "Niveau d'instruction scolarité (0=NON,1=Maternelle,2=Elémentaire,3=Secondaire,4=Supérieur)",
            options=[0,1,2,3,4]
        )

        st.subheader("Variables catégorielles")
        diagnostic = st.selectbox("Diagnostic Catégorisé", options=diagnostic_categories)
        mois = st.selectbox("Mois", options=mois_categories)

        submitted = st.form_submit_button("Prédire")

    # ============================
    # Traitement et prédiction
    # ============================
    if submitted:
        # Préparer les données sous forme de DataFrame
        input_dict = {**quantitative_inputs, **binary_inputs,
                      'NiveauUrgence': niveau_urgence,
                      "Niveau d'instruction scolarité": niveau_instruction,
                      "Diagnostic Catégorisé": diagnostic,
                      "Mois": mois}

        input_df = pd.DataFrame([input_dict])

        # Encodage dummies identique à l'entraînement
        input_df = pd.get_dummies(input_df, columns=["Diagnostic Catégorisé","Mois"], drop_first=True)

        # Ajouter les colonnes manquantes avec 0
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Réordonner les colonnes pour correspondre au modèle
        input_df = input_df[model_features]

        # Standardisation des variables quantitatives
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])

        # Prédiction
        pred_proba = model.predict_proba(input_df)[:,1][0]
        pred_class = model.predict(input_df)[0]

        # ============================
        # Sauvegarde automatique
        # ============================
        output_df = input_df.copy()
        output_df["Prédiction"] = pred_class
        output_df["Probabilité_Complication"] = pred_proba

        if not os.path.exists("predictions"):
            os.makedirs("predictions")

        csv_file = "predictions/predictions_random_forest.csv"

        if os.path.exists(csv_file):
            output_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            output_df.to_csv(csv_file, index=False)

        # ============================
        # Affichage résultat
        # ============================
        st.subheader("📌 Résultat de la prédiction")
        col_res1, col_res2 = st.columns([2,1])

        if pred_class == 0:
            col_res1.success(f"✅ Évolution prévue : **Favorable** (Probabilité de complication : {pred_proba:.2f})")
        else:
            col_res1.error(f"⚠️ Évolution prévue : **Complications attendues** (Probabilité : {pred_proba:.2f})")

        col_res2.metric("Probabilité de complication", f"{pred_proba*100:.1f} %")

        


