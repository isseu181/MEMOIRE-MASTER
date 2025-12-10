# ================================
# deployment.py - Déploiement Random Forest 
# ================================
import streamlit as st
import pandas as pd
import joblib

def show_deployment():
    st.set_page_config(page_title="Déploiement Random Forest", layout="wide")

    # ---  Style CSS  ---
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
        st.error("Impossible de charger le modèle ou le scaler.")
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

    st.markdown("###  Remplissez le formulaire du patient pour estimer son évolution clinique")

    # --- Dictionnaire des variables ---
    with st.expander(" Voir les définitions des variables"):
        st.markdown("""
        **Variables biologiques :**
        - **GB (/mm³)** : Valeur du nombre de globules blancs mesuré **en situation d’urgence**.
        - **PLT (/mm³)** :  Valeur du nombre de plaquettes mesuré **en urgence**.
        - **Nbre de GB (/mm³)** : Valeur du nombre de globules blancs lors du **suivi régulier**.
        - **Nbre de PLT (/mm³)** : Valeur du nombre de plaquettes lors du **suivi régulier**.
        - **HB (g/dl)** : Taux d’hémoglobine mesuré.
        - **CRP Si positive (Valeur)** : Valeur de la protéine C-réactive lorsqu’elle est positive.
        - **% d’Hb S / % d’Hb F** : Répartition des fractions d’hémoglobine.
        - **GR (/mm3)** : Nombre de globules rouges.
        - **VGM (fl/u3)** : Volume globulaire moyen.
        - **TCMH (g/dl)** : Teneur corpusculaire moyenne en hémoglobine.
        

        **Variables cliniques :**
        - **Pâleur**, **Splénomégalie**, **Souffle systolique fonctionnel** : Observations cliniques binaires (1 = Oui, 0 = Non).
        - **Niveau d’urgence** : Cotation de 1 à 6 indiquant la gravité clinique.
        - **Niveau d’instruction scolarité** : Niveau de scolarisation du patient.

        **Autres :**
        - **Diagnostic catégorisé** : Type principal de diagnostic.
        - **Mois** : Mois de la consultation ou du suivi.
        """)

    # --- FORMULAIRE ---
    with st.form("patient_form"):
        inputs = {}
        col1, col2 = st.columns(2)

        # --- Colonne 1 ---
        with col1:
            for var in quantitative_vars[:len(quantitative_vars)//2]:
                help_text = None
                if var == "GB (/mm3)":
                    help_text = "Valeur du nombre de globules blancs mesuré en urgence."
                elif var == "PLT (/mm3)":
                    help_text = "Valeur du nombre de plaquettes mesuré en urgence."
                elif var == "Nbre de GB (/mm3)":
                    help_text = "Valeur du nombre de globules blancs en suivi régulier."
                elif var == "Nbre de PLT (/mm3)":
                    help_text = "Valeur du nombre de plaquettes en suivi régulier."
                inputs[var] = st.number_input(var, value=0.0, format="%.2f", help=help_text)

            for var in binary_vars[:len(binary_vars)//2]:
                inputs[var] = st.selectbox(
                    f"{var} (OUI=1, NON=0)", 
                    options=[0,1],
                    help=f"Indique la présence ou non de {var.lower()}."
                )

        # --- Colonne 2 ---
        with col2:
            for var in quantitative_vars[len(quantitative_vars)//2:]:
                help_text = None
                if var == "GB (/mm3)":
                    help_text = "Taux de globules blancs mesuré en urgence."
                elif var == "PLT (/mm3)":
                    help_text = "Taux de plaquettes mesuré en urgence."
                elif var == "Nbre de GB (/mm3)":
                    help_text = "Valeur du nombre de globules blancs en suivi régulier."
                elif var == "Nbre de PLT (/mm3)":
                    help_text = "Valeur du nombre de plaquettes en suivi régulier."
                inputs[var] = st.number_input(var, value=0.0, format="%.2f", help=help_text)

            for var in binary_vars[len(binary_vars)//2:]:
                inputs[var] = st.selectbox(
                    f"{var} (OUI=1, NON=0)", 
                    options=[0,1],
                    help=f"Indique la présence ou non de {var.lower()}."
                )

            inputs['NiveauUrgence'] = st.slider(
                "Niveau d'urgence (1=Urgence1 ... 6=Urgence6)", 
                1, 6, 1,
                help="Échelle d’évaluation de la gravité clinique (1 = plus urgente, 6 = moins urgente)."
            )

            inputs["Niveau d'instruction scolarité"] = st.selectbox(
                "Niveau d'instruction scolarité",
                options=[0,1,2,3,4],
                format_func=lambda x: ["Non","Maternelle","Élémentaire","Secondaire","Supérieur"][x],
                help="Niveau de scolarisation du patient."
            )

            inputs["Diagnostic Catégorisé"] = st.selectbox(
                "Diagnostic Catégorisé", 
                options=diagnostic_categories,
                help="Type de diagnostic principal observé."
            )
            inputs["Mois"] = st.selectbox(
                "Mois", 
                options=mois_categories,
                help="Mois de référence de la consultation."
            )

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
