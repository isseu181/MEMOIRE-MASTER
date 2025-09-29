# deployment.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def show_deployment():
    # =====================
    # Chargement modèle/scaler
    # =====================
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")

    df = pd.read_excel("fichier_nettoye.xlsx")

    diagnostic_categories = ["ANÉMIE","AUTRES","AVC","CVO","INFECTIONS","STA"]
    mois_categories = sorted(df["Mois"].dropna().unique())

    quantitative_vars = [
        'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
        'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
    ]

    binary_vars = [
        'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
        'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
        'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
        'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque'
    ]

    st.markdown("### Formulaire de saisie des données du patient")
    with st.form("patient_form"):
        quantitative_inputs = {var: st.number_input(var, value=0.0) for var in quantitative_vars}
        binary_inputs = {var: st.selectbox(var, options=[0,1]) for var in binary_vars}
        niveau_urgence = st.slider("Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 1, 6, 1)
        niveau_instruction = st.selectbox(
            "Niveau d'instruction scolarité",
            options=[0,1,2,3,4]
        )
        diagnostic = st.selectbox("Diagnostic Catégorisé", options=diagnostic_categories)
        mois = st.selectbox("Mois", options=mois_categories)
        submitted = st.form_submit_button("Prédire")

    if submitted:
        input_dict = {**quantitative_inputs, **binary_inputs,
                      'NiveauUrgence': niveau_urgence,
                      "Niveau d'instruction scolarité": niveau_instruction,
                      "Diagnostic Catégorisé": diagnostic,
                      "Mois": mois}
        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df, columns=["Diagnostic Catégorisé","Mois"], drop_first=True)

        # Ajouter les colonnes manquantes
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])

        pred_proba = model.predict_proba(input_df)[:,1][0]
        pred_class = model.predict(input_df)[0]

        st.subheader("Résultat de la prédiction")
        if pred_class == 0:
            st.success(f"Évolution prévue : **Favorable** ✅ (Probabilité de complication : {pred_proba:.2f})")
        else:
            st.error(f"Évolution prévue : **Complications** ⚠️ (Probabilité : {pred_proba:.2f})")
