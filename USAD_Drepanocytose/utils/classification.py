import streamlit as st
import pandas as pd
import joblib

# ================================
# Fonction principale
# ================================
def show_classification():
    st.subheader("🧠 Prédiction de l'évolution (Favorable vs Complications)")

    # Charger modèle, scaler et features
    try:
        model_loaded = joblib.load("fichiers modèles/random_forest_model.pkl")
        scaler_loaded = joblib.load("fichiers modèles/scaler.pkl")
        features_loaded = joblib.load("fichiers modèles/features.pkl")
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle ou les fichiers nécessaires : {e}")
        return

    # ================================
    # Menus déroulants pour Oui/Non
    # ================================
    def oui_non(label):
        return st.selectbox(label, ["OUI", "NON"])

    paleur = oui_non("Pâleur")
    souffle = oui_non("Souffle systolique fonctionnel")
    vaccin_meningo = oui_non("Vaccin contre méningocoque")
    spleno = oui_non("Splénomégalie")
    penicilline = oui_non("Prophylaxie à la pénicilline")
    parents = oui_non("Parents Salariés")
    hospit = oui_non("Prise en charge Hospitalisation")
    radio = oui_non("Radiographie du thorax Oui ou Non")
    douleur = oui_non("Douleur provoquée (Os.Abdomen)")
    vaccin_pneumo = oui_non("Vaccin contre pneumocoque")
    hdj = oui_non("HDJ")

    # ================================
    # Niveau d’urgence et scolarité
    # ================================
    urgence = st.selectbox("🚨 Niveau d’urgence", ["Urgence1", "Urgence2", "Urgence3", "Urgence4", "Urgence5", "Urgence6"])
    scolarite = st.selectbox(
        "🎓 Niveau d'instruction scolarité",
        ["NON", "Maternelle", "Elémentaire", "Secondaire", "Enseignement Supérieur"]
    )

    # ================================
    # Mois consultation
    # ================================
    mois_dict = {
        "Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4,
        "Mai": 5, "Juin": 6, "Juillet": 7, "Août": 8,
        "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12
    }
    mois_choisi = st.selectbox("📅 Mois consultation", list(mois_dict.keys()))

    # ================================
    # Champs numériques principaux
    # ================================
    age_signes = st.number_input("Âge de début des signes (en mois)", min_value=0, value=12)
    age_etude = st.number_input("Âge du début d'étude (en mois)", min_value=0, value=24)
    age_decouverte = st.number_input("Âge de découverte de la drépanocytose (en mois)", min_value=0, value=6)

    gr = st.number_input("GR (/mm3)", min_value=0.0, value=4.5)
    gb = st.number_input("GB (/mm3)", min_value=0.0, value=7.2)
    plt = st.number_input("PLT (/mm3)", min_value=0.0, value=300.0)
    hb = st.number_input("HB (g/dl)", min_value=0.0, value=10.0)
    vgm = st.number_input("VGM (fl/u3)", min_value=0.0, value=85.0)
    tcmh = st.number_input("TCMH (g/dl)", min_value=0.0, value=30.0)
    taux_hb = st.number_input("Taux d'Hb (g/dL)", min_value=0.0, value=10.0)

    crp = st.number_input("CRP Si positive (Valeur)", min_value=0.0, value=5.0)
    hb_s = st.number_input("% d'Hb S", min_value=0.0, value=90.0)
    hb_f = st.number_input("% d'Hb F", min_value=0.0, value=10.0)

    hospit_avant2017 = st.number_input("Nbre d'hospitalisations avant 2017", min_value=0, value=1)
    hospit_apres2017 = st.number_input("Nbre d'hospitalisations entre 2017 et 2023", min_value=0, value=0)
    transfusion_avant2017 = st.number_input("Nbre de transfusion avant 2017", min_value=0, value=0)
    transfusion_apres2017 = st.number_input("Nbre de transfusion entre 2017 et 2023", min_value=0, value=0)

    # ================================
    # Construction du DataFrame
    # ================================
    if st.button("⚡ Lancer la prédiction"):
        new_data = pd.DataFrame([{
            "Âge de début des signes (en mois)": age_signes,
            "Âge du debut d etude en mois (en janvier 2023)": age_etude,
            "Âge de découverte de la drépanocytose (en mois)": age_decouverte,
            "GR (/mm3)": gr,
            "GB (/mm3)": gb,
            "PLT (/mm3)": plt,
            "HB (g/dl)": hb,
            "VGM (fl/u3)": vgm,
            "TCMH (g/dl)": tcmh,
            "Taux d'Hb (g/dL)": taux_hb,
            "CRP Si positive (Valeur)": crp,
            "% d'Hb S": hb_s,
            "% d'Hb F": hb_f,
            "Nbre d'hospitalisations avant 2017": hospit_avant2017,
            "Nbre d'hospitalisations entre 2017 et 2023": hospit_apres2017,
            "Nbre de transfusion avant 2017": transfusion_avant2017,
            "Nbre de transfusion Entre 2017 et 2023": transfusion_apres2017,
            "Pâleur": 1 if paleur=="OUI" else 0,
            "Souffle systolique fonctionnel": 1 if souffle=="OUI" else 0,
            "Vaccin contre méningocoque": 1 if vaccin_meningo=="OUI" else 0,
            "Splénomégalie": 1 if spleno=="OUI" else 0,
            "Prophylaxie à la pénicilline": 1 if penicilline=="OUI" else 0,
            "Parents Salariés": 1 if parents=="OUI" else 0,
            "Prise en charge Hospitalisation": 1 if hospit=="OUI" else 0,
            "Radiographie du thorax Oui ou Non": 1 if radio=="OUI" else 0,
            "Douleur provoquée (Os.Abdomen)": 1 if douleur=="OUI" else 0,
            "Vaccin contre pneumocoque": 1 if vaccin_pneumo=="OUI" else 0,
            "HDJ": 1 if hdj=="OUI" else 0,
            "NiveauUrgence": int(urgence.replace("Urgence","")),
            "Niveau d'instruction scolarité": {
                "NON":0, "Maternelle":1, "Elémentaire":2,
                "Secondaire":3, "Enseignement Supérieur":4
            }[scolarite],
            "Mois": mois_dict[mois_choisi]
        }])

        # Réalignement avec les features d’entraînement
        new_data = new_data.reindex(columns=features_loaded, fill_value=0)

        # Standardisation des variables quantitatives
        quantitative_vars = [
            'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
            'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
            'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
            'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
            "Nbre d'hospitalisations entre 2017 et 2023",
            'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
            'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
        ]
        try:
            new_data[quantitative_vars] = scaler_loaded.transform(new_data[quantitative_vars])
        except Exception as e:
            st.warning(f"⚠️ Attention : certaines variables quantitatives n'ont pas été trouvées ({e})")

        # Prédiction
        pred_proba = model_loaded.predict_proba(new_data)[:,1][0]
        seuil = 0.56
        pred_class = "Complications" if pred_proba >= seuil else "Favorable"

        st.success(f"✅ Résultat : **{pred_class}**")
        st.metric("📊 Probabilité de complications", f"{pred_proba:.2f}")
