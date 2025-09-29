# deployment.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def show_deployment():
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

    # Variables ordinales
    # NiveauUrgence: 1=Urgence1, 2=Urgence2, 3=Urgence3, 4=Urgence4, 5=Urgence5, 6=Urgence6
    # Niveau d'instruction scolarité: 0=NON, 1=Maternelle, 2=Elémentaire, 3=Secondaire, 4=Enseignement Supérieur
    ordinal_vars = ["NiveauUrgence", "Niveau d'instruction scolarité"]

    # Variables catégorielles encodées en dummies
    diagnostic_categories = ['CVO', 'Anémie', 'Anémie','AVC', 'Autres']  
    mois_categories = ['Janvier','Février','Mars','Avril','Mai','Juin','Juillet',
                       'Aout','Septembre','Octobre','Novembre','Decembre']

    st.markdown("### Formulaire de saisie des données du patient")

    with st.form("patient_form"):
        st.subheader("Variables quantitatives")
        quantitative_inputs = {}
        for var in quantitative_vars:
            quantitative_inputs[var] = st.number_input(var, value=0.0)

        st.subheader("Variables binaires (OUI=1, NON=0)")
        binary_inputs = {}
        for var in binary_vars:
            binary_inputs[var] = st.selectbox(var, options=[0,1])

        st.subheader("Variables ordinales")
        niveau_urgence = st.slider("Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 1, 6, 1)
        niveau_instruction = st.selectbox("Niveau d'instruction scolarité (0=NON,1=Maternelle,2=Elémentaire,3=Secondaire,4=Supérieur)", options=[0,1,2,3,4])

        st.subheader("Variables catégorielles")
        diagnostic = st.selectbox("Diagnostic Catégorisé", options=diagnostic_categories)
        mois = st.selectbox("Mois", options=mois_categories)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        # Préparer les données sous forme de DataFrame
        input_dict = {**quantitative_inputs, **binary_inputs,
                      'NiveauUrgence': niveau_urgence,
                      "Niveau d'instruction scolarité": niveau_instruction,
                      "Diagnostic Catégorisé": diagnostic,
                      "Mois": mois}

        input_df = pd.DataFrame([input_dict])

        # Encodage dummies pour Diagnostic et Mois
        input_df = pd.get_dummies(input_df, columns=["Diagnostic Catégorisé","Mois"], drop_first=True)

        # Vérifier que toutes les colonnes attendues par le modèle sont présentes
        # (par exemple, lors de l'entraînement, X.columns a été utilisé)
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0  # Ajouter les colonnes manquantes avec 0

        # Réordonner les colonnes
        input_df = input_df[model_features]

        # Standardisation des variables quantitatives
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])

        # Prédiction
        pred_proba = model.predict_proba(input_df)[:,1][0]
        pred_class = model.predict(input_df)[0]

        # Résultat
        st.subheader("Résultat de la prédiction")
        if pred_class == 0:
            st.success(f"Évolution prévue : **Favorable** ✅ (Probabilité de complication : {pred_proba:.2f})")
        else:
            st.error(f"Évolution prévue : **Complications** ⚠️ (Probabilité : {pred_proba:.2f})")

