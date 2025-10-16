# ================================
# deployment.py - Déploiement Random Forest 
# ================================
import streamlit as st
import pandas as pd
import joblib

def show_deployment():
    st.set_page_config(page_title="Déploiement Random Forest", layout="wide")

    # --- Style CSS ---
    st.markdown("""
        <style>
        body {
            background-color: #f5f9f6;
        }
        .stApp {
            background-color: #f5f9f6;
        }
        h1 {
            text-align: center;
            color: #0b6e4f;
            font-weight: bold;
        }
        .prediction-card {
            padding: 20px;
            border-radius: 15px;
            background-color: #e8f5e9;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .prediction-card-bad {
            padding: 20px;
            border-radius: 15px;
            background-color: #ffebee;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .reco {
            background-color: #ffffff;
            border-left: 5px solid #0b6e4f;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .reco-bad {
            background-color: #ffffff;
            border-left: 5px solid #b71c1c;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>🩺 Déploiement du Modèle Random Forest</h1>", unsafe_allow_html=True)

    # Charger modèle et scaler
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except:
        st.error("❌ Impossible de charger le modèle ou le scaler.")
        return

    # --- Variables ---
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

    st.markdown("### 🧾 Remplissez le formulaire du patient pour estimer l’évolution clinique")

    # --- FORMULAIRE ---
    with st.form("patient_form"):
        inputs = {}
        col1, col2 = st.columns(2)

        with col1:
            for var in quantitative_vars[:len(quantitative_vars)//2]:
                inputs[var] = st.number_input(var, value=0.0, format="%.2f")
            for var in binary_vars[:len(binary_vars)//2]:
                inputs[var] = st.selectbox(f"{var} (OUI=1, NON=0)", options=[0,1])

        with col2:
            for var in quantitative_vars[len(quantitative_vars)//2:]:
                inputs[var] = st.number_input(var, value=0.0, format="%.2f")
            for var in binary_vars[len(binary_vars)//2:]:
                inputs[var] = st.selectbox(f"{var} (OUI=1, NON=0)", options=[0,1])

            inputs['NiveauUrgence'] = st.slider("Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 1, 6, 1)
            inputs["Niveau d'instruction scolarité"] = st.selectbox(
                "Niveau d'instruction scolarité",
                options=[0,1,2,3,4],
                format_func=lambda x: ["Non","Maternelle","Élémentaire","Secondaire","Supérieur"][x]
            )
            inputs["Diagnostic Catégorisé"] = st.selectbox("Diagnostic Catégorisé", options=diagnostic_categories)
            inputs["Mois"] = st.selectbox("Mois", options=mois_categories)

        submitted = st.form_submit_button("🔮 Prédire")

    # --- PREDICTION ---
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

        # --- Résultats et recommandations ---
        if pred_class == 0:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>✅ Évolution prévue : <b>Favorable</b></h3>
                <p>Probabilité de complication : <b>{pred_proba:.2f}</b></p>
            </div>
            <div class="reco">
                <h4>Recommandations cliniques :</h4>
                <ul>
                    <li>Poursuivre le suivi médical régulier selon le protocole en vigueur.</li>
                    <li>Vérifier la bonne observance de la prophylaxie et des traitements préventifs.</li>
                    <li>Confirmer que les vaccinations spécifiques sont à jour.</li>
                    <li>Encourager une hydratation et une hygiène de vie adaptées.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="prediction-card-bad">
                <h3>⚠️ Évolution prévue : <b>Complications possibles</b></h3>
                <p>Probabilité estimée : <b>{pred_proba:.2f}</b></p>
            </div>
            <div class="reco-bad">
                <h4>Recommandations cliniques :</h4>
                <ul>
                    <li>Envisager une évaluation clinique approfondie et un bilan complémentaire.</li>
                    <li>Mettre en place une surveillance rapprochée des paramètres cliniques et biologiques.</li>
                    <li>Réévaluer la prophylaxie, le traitement de fond et l’observance thérapeutique.</li>
                    <li>Adapter la prise en charge selon l’évolution clinique et le contexte du patient.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
