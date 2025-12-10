# ================================
# deployment.py - Déploiement Random Forest 
# ================================
import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import io

def show_deployment():
    st.set_page_config(page_title="Déploiement Random Forest", layout="wide")

    # ---  Style CSS  ---
    st.markdown(""" 
        /* ... ton CSS existant ... */ 
    """, unsafe_allow_html=True)

    st.markdown("<h1>🩺 Déploiement du Modèle Random Forest</h1>", unsafe_allow_html=True)

    # Charger modèle et scaler
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except:
        st.error("Impossible de charger le modèle ou le scaler.")
        return

    # --- Variables (quantitative_vars, binary_vars, model_features, etc.) ---
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

    st.markdown("###  Remplissez le formulaire du patient pour estimer son évolution clinique")

    # --- FORMULAIRE (ton code existant) ---
    with st.form("patient_form"):
        inputs = {}
        col1, col2 = st.columns(2)
        # --- Colonne 1 ---
        with col1:
            for var in quantitative_vars[:len(quantitative_vars)//2]:
                inputs[var] = st.number_input(var, value=0.0, format="%.2f")
            for var in binary_vars[:len(binary_vars)//2]:
                inputs[var] = st.selectbox(f"{var} (OUI=1, NON=0)", options=[0,1])
        # --- Colonne 2 ---
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

    # --- Fonction pour générer PDF ---
    def generate_pdf(inputs, pred_class, pred_proba):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Rapport de Prédiction Random Forest", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Prédiction: {'Favorable' if pred_class==0 else 'Complications possibles'}", ln=True)
        pdf.cell(0, 10, f"Probabilité: {pred_proba:.2f}", ln=True)
        pdf.ln(10)

        pdf.cell(0, 10, "Données du patient:", ln=True)
        pdf.ln(5)
        for key, value in inputs.items():
            pdf.cell(0, 8, f"{key} : {value}", ln=True)

        pdf.ln(10)
        pdf.cell(0, 10, "Recommandations :", ln=True)
        pdf.ln(5)
        if pred_class == 0:
            reco = [
                "Maintenir le suivi médical régulier selon le protocole établi",
                "Poursuivre la prophylaxie médicamenteuse et la couverture vaccinale",
                "Surveiller périodiquement les constantes biologiques (Hb, GB, PLT, CRP)",
                "Documenter toute modification clinique dans le dossier patient"
            ]
        else:
            reco = [
                "Renforcer le suivi médical rapproché et la fréquence des bilans",
                "Réévaluer la prophylaxie, le traitement de fond et l’observance thérapeutique",
                "Surveiller de près les signes cliniques d’alerte : fièvre, pâleur, douleurs osseuses ou abdominales",
                "Envisager une adaptation thérapeutique (transfusions, traitement symptomatique, hospitalisation préventive)",
                "Consigner et communiquer toute évolution clinique significative"
            ]
        for r in reco:
            pdf.multi_cell(0, 8, f"- {r}")

        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer

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

        # --- Affichage résultat existant ---
        if pred_class == 0:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>✅ Évolution prévue : <b>Favorable</b></h3>
                <p>Probabilité de complication : <b>{pred_proba:.2f}</b></p>
            </div>
            <div class="reco">
                <h4> Recommandations :</h4>
                <ul>
                    <li>Maintenir le suivi médical régulier selon le protocole établi</li>
                    <li>Poursuivre la prophylaxie médicamenteuse et la couverture vaccinale</li>
                    <li>Surveiller périodiquement les constantes biologiques (Hb, GB, PLT, CRP)</li>
                    <li>Documenter toute modification clinique dans le dossier patient</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-bad">
                <h3> Évolution prévue : <b>Complications possibles</b></h3>
                <p>Probabilité : <b>{pred_proba:.2f}</b></p>
            </div>
            <div class="reco-bad">
                <h4> Recommandations :</h4>
                <ul>
                    <li>Renforcer le suivi médical rapproché et la fréquence des bilans</li>
                    <li>Réévaluer la prophylaxie, le traitement de fond et l’observance thérapeutique</li>
                    <li>Surveiller de près les signes cliniques d’alerte : fièvre, pâleur, douleurs osseuses ou abdominales</li>
                    <li>Envisager une adaptation thérapeutique (transfusions, traitement symptomatique, hospitalisation préventive)</li>
                    <li>Consigner et communiquer toute évolution clinique significative</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)  

        # --- Bouton téléchargement PDF ---
        pdf_buffer = generate_pdf(inputs, pred_class, pred_proba)
        st.download_button(
            label="📄 Télécharger le rapport PDF",
            data=pdf_buffer,
            file_name="rapport_prediction.pdf",
            mime="application/pdf"
        )
