# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def show_classification():
    st.title("Classification de l'évolution des patients")

    # ================================
    # 1. Chargement des fichiers pré-entraînés
    # ================================
    chemin_sauvegarde = "./"  # dossier contenant les fichiers
    best_model = joblib.load(f"{chemin_sauvegarde}/random_forest_model.pkl")
    scaler = joblib.load(f"{chemin_sauvegarde}/scaler.pkl")
    features = joblib.load(f"{chemin_sauvegarde}/features.pkl")

    # ================================
    # 2. Chargement des données
    # ================================
    st.info("Chargement automatique de la base de données : fichier_nettoye.xlsx")
    df = pd.read_excel("fichier_nettoye.xlsx")
    df_selected = df[features].copy()

    # ================================
    # 3. Standardisation
    # ================================
    df_selected_scaled = scaler.transform(df_selected)

    # ================================
    # 4. Variable cible
    # ================================
    y_true = df['Evolution'].map({'Favorable':0,'Complications':1})

    # ================================
    # 5. Prédictions
    # ================================
    y_proba = best_model.predict_proba(df_selected_scaled)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)  # seuil par défaut 0.5

    # ================================
    # 6. Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Performance du modèle Random Forest")

        # Matrice de confusion
        st.write("### Matrice de confusion")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        st.pyplot(plt)
        plt.clf()

        # Courbe ROC
        st.write("### Courbe ROC")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title('Courbe ROC')
        plt.legend(loc='lower right')
        st.pyplot(plt)
        plt.clf()

        # Métriques
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics_df = pd.DataFrame([{
            "Accuracy": report['accuracy'],
            "Precision": report['macro avg']['precision'],
            "Recall": report['macro avg']['recall'],
            "F1-Score": report['macro avg']['f1-score'],
            "AUC-ROC": roc_auc
        }])
        st.write("### Métriques du modèle")
        st.dataframe(metrics_df)

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Variables importantes (Random Forest)")
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({"Variable": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
        st.dataframe(imp_df)
        plt.figure(figsize=(8,6))
        sns.barplot(x="Importance", y="Variable", data=imp_df)
        plt.title("Importance des variables")
        st.pyplot(plt)
        plt.clf()

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie")
        st.markdown("""
        - Prétraitement et sélection des variables
        - Encodage des variables binaires (0=Non, 1=Oui) et ordinales
        - Standardisation des variables quantitatives
        - Utilisation d'un modèle Random Forest pré-entraîné
        - Évaluation via Accuracy, Precision, Recall, F1-Score, AUC-ROC
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction")
        st.info("Saisissez les caractéristiques d'un nouveau patient")

        # Créer un dictionnaire pour stocker les valeurs
        input_data = {}
        for col in features:
            if df[col].dtype == 'object' or 'Oui' in df[col].unique() or 'Non' in df[col].unique():
                val = st.selectbox(f"{col} (0=Non, 1=Oui)", options=[0,1])
                input_data[col] = val
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                val = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=min_val)
                input_data[col] = val

        if st.button("Prédire l'évolution"):
            new_df = pd.DataFrame([input_data])
            new_scaled = scaler.transform(new_df)
            pred_proba = best_model.predict_proba(new_scaled)[:,1][0]
            pred_class = int(pred_proba >= 0.5)
            st.write(f"**Probabilité de complications : {pred_proba:.3f}**")
            st.write(f"**Prédiction : {'Complications' if pred_class==1 else 'Favorable'}**")
