# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

st.set_page_config(page_title="Classification Patients", layout="wide")

def show_classification():
    st.title("Classification Patients - Evolution clinique")

    # ================================
    # 1. Chargement des fichiers
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")  # ton fichier nettoyé
    best_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    # Sélection des colonnes du modèle
    df_selected = df.copy()
    # Encodage identique à l'entraînement
    binary_mappings = {
        'Pâleur': {'OUI':1, 'NON':0},
        'Souffle systolique fonctionnel': {'OUI':1, 'NON':0},
        'Vaccin contre méningocoque': {'OUI':1, 'NON':0},
        'Splénomégalie': {'OUI':1, 'NON':0},
        'Prophylaxie à la pénicilline': {'OUI':1, 'NON':0},
        'Parents Salariés': {'OUI':1, 'NON':0},
        'Prise en charge Hospitalisation': {'OUI':1, 'NON':0},
        'Radiographie du thorax Oui ou Non': {'OUI':1, 'NON':0},
        'Douleur provoquée (Os.Abdomen)': {'OUI':1, 'NON':0},
        'Vaccin contre pneumocoque': {'OUI':1, 'NON':0},
    }
    df_selected.replace(binary_mappings, inplace=True)

    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)

    df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé', 'Mois'], drop_first=True)

    # Assurer que toutes les colonnes du modèle sont présentes
    for col in features:
        if col not in df_selected.columns:
            df_selected[col] = 0
    X = df_selected[features]

    # Standardiser les quantitatives
    quantitative_vars = [
        'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
        'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
        'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
    ]
    X[quantitative_vars] = scaler.transform(X[quantitative_vars])

    # Variable cible
    y = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})

    # ================================
    # 2. Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Métriques du modèle Random Forest")
        y_proba = best_model.predict_proba(X)[:,1]
        threshold = 0.56  # seuil optimal
        y_pred = (y_proba >= threshold).astype(int)

        # Matrice de confusion
        cm = confusion_matrix(y, y_pred)
        st.write("Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        st.pyplot(fig)

        # Rapport de classification
        st.write("Rapport de classification")
        st.text(classification_report(y, y_pred))

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel("Taux de faux positifs (1-Spécificité)")
        ax2.set_ylabel("Taux de vrais positifs (Sensibilité)")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Importance des variables")
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            feat_importance = pd.DataFrame({"Variable": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
            st.dataframe(feat_importance)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x="Importance", y="Variable", data=feat_importance, palette="viridis", ax=ax)
            ax.set_title("Variables importantes")
            st.pyplot(fig)
        else:
            st.info("Le modèle sélectionné ne fournit pas d'importance des variables.")

    # --- Onglet 3 : Simulateur ---
    with tabs[2]:
        st.subheader("Simulateur de prédiction")
        st.markdown("Entrez les valeurs des variables pour prédire l'évolution clinique")

        user_input = {}
        for feat in features:
            if feat in binary_mappings.keys():
                user_input[feat] = st.selectbox(feat, options=[0,1], format_func=lambda x: "Oui" if x==1 else "Non")
            else:
                min_val = float(X[feat].min())
                max_val = float(X[feat].max())
                mean_val = float(X[feat].mean())
                user_input[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Prédire l'évolution"):
            X_new = pd.DataFrame([user_input])
            # Ajouter colonnes manquantes
            for col in features:
                if col not in X_new.columns:
                    X_new[col] = 0
            X_new = X_new[features]
            # Standardiser quantitatives
            X_new[quantitative_vars] = scaler.transform(X_new[quantitative_vars])
            y_new_proba = best_model.predict_proba(X_new)[:,1]
            y_new_pred = (y_new_proba >= threshold).astype(int)
            st.write("**Probabilité d'évolution vers complications :**", round(float(y_new_proba),3))
            st.write("**Prédiction finale :**", "Complications" if y_new_pred[0]==1 else "Favorable")
