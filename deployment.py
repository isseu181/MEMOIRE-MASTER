# ================================
# deployment.py - Déploiement Random Forest (compact)
# ================================
import streamlit as st
import pandas as pd
import joblib

def show_deployment():
    st.set_page_config(page_title="Déploiement Random Forest", layout="wide")
    st.markdown("<h1 style='text-align:center;color:darkgreen;'> Déploiement - Modèle Random Forest</h1>", unsafe_allow_html=True)

    # Charger le modèle et le scaler
    try:
        model = joblib.load("random_forest_model.pkl")  
        scaler = joblib.load("scaler.pkl")              
    except:
        st.error(" Impossible de charger le modèle ou le scaler.")
        return

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

    model_features = model.feature_names_in_
    diagnostic_categories = [c.replace("Diagnostic Catégorisé_", "") for c in model_features if "Diagnostic Catégorisé_" in c]
    mois_categories = [c.replace("Mois_", "") for c in model_features if "Mois_" in c]

    st.markdown("### Remplissez le formulaire pour prédire l’évolution clinique d’un patient")

    with st.form("patient_form"):
        inputs = {}

        # Quantitatives
        for var in quantitative_vars:
            inputs[var] = st.number_input(var, value=0.0, format="%.2f")

        # Qualitatives binaires + ordinales + catégorielles
        # Binaires
        for var in binary_vars:
            inputs[var] = st.selectbox(f"{var} (OUI=1, NON=0)", options=[0,1])

        # Ordinales
        inputs['NiveauUrgence'] = st.slider("Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 1, 6, 1)
        inputs["Niveau d'instruction scolarité"] = st.selectbox(
            "Niveau d'instruction scolarité",
            options=[0,1,2,3,4],
            format_func=lambda x: ["Non","Maternelle","Elémentaire","Secondaire","Supérieur"][x]
        )

        # Catégorielles
        inputs["Diagnostic Catégorisé"] = st.selectbox("Diagnostic Catégorisé", options=diagnostic_categories)
        inputs["Mois"] = st.selectbox("Mois", options=mois_categories)

        submitted = st.form_submit_button("🔮 Prédire")

    if submitted:
        input_df = pd.DataFrame([inputs])
        input_df = pd.get_dummies(input_df, columns=["Diagnostic Catégorisé","Mois"])

        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_features]
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])

        pred_proba = model.predict_proba(input_df)[:,1][0]
        pred_class = model.predict(input_df)[0]

        st.subheader(" Résultat de la prédiction")
        if pred_class == 0:
            st.success(f"✅ Évolution prévue : **Favorable** (Probabilité de complication : {pred_proba:.2f})")
        else:
            st.error(f"⚠️ Évolution prévue : **Complications attendues** (Probabilité : {pred_proba:.2f})")


