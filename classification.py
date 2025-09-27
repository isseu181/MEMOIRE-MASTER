# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def show_classification():
    st.title("Classification - Prédiction de l'évolution des patients")

    # ================================
    # 1. Chargement des fichiers sauvegardés
    # ================================
    model_file = "random_forest_model.pkl"
    scaler_file = "scaler.pkl"
    features_file = "features.pkl"

    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        features = joblib.load(features_file)
    except Exception as e:
        st.error(f"Erreur lors du chargement des fichiers : {e}")
        return

    # ================================
    # 2. Chargement des données
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")
    df_selected = df.copy()

    # ================================
    # 3. Onglets horizontaux
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles et métriques")

        # Pour cet exemple, on fixe les résultats connus
        results_df = pd.DataFrame({
            "Modèle":["LightGBM","Random Forest","Decision Tree","SVM"],
            "Accuracy":[0.963,0.984,0.915,0.746],
            "Precision":[0.966,0.984,0.915,0.746],
            "Recall":[0.963,0.984,0.915,0.746],
            "F1-Score":[0.963,0.984,0.915,0.746],
            "AUC-ROC":[0.998,0.997,0.915,0.809],
            "Seuil optimal":[0.999,0.560,1.0,0.456]
        })
        st.dataframe(results_df)

        # Choix du meilleur modèle (max sur AUC-ROC et autres métriques)
        metrics = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
        results_df["Score Moyenne"] = results_df[metrics].mean(axis=1)
        best_row = results_df.loc[results_df["Score Moyenne"].idxmax()]
        st.success(f"✅ Modèle retenu : {best_row['Modèle']}")

        st.write("### Matrice de confusion et courbe ROC du modèle retenu")

        # Utilisation du modèle pour générer matrice ROC (sur tout le dataset pour démo)
        X_demo = df_selected[features].copy()
        X_scaled = scaler.transform(X_demo)
        y_demo = df_selected['Evolution'].map({'Favorable':0,'Complications':1})

        y_proba = model.predict_proba(X_scaled)[:,1]
        y_pred = (y_proba >= best_row["Seuil optimal"]).astype(int)

        # Matrice de confusion
        cm = confusion_matrix(y_demo, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")
        st.pyplot(fig)

        # Courbe ROC
        fpr, tpr, thresholds = roc_curve(y_demo, y_proba)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel("Taux de faux positifs (1-Spécificité)")
        ax2.set_ylabel("Taux de vrais positifs (Sensibilité)")
        ax2.set_title("Courbe ROC")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Importance des variables (Random Forest)")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Variable":features,
            "Importance":importances
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)

        # Graphique importance
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.barplot(x="Importance", y="Variable", data=importance_df, ax=ax3)
        ax3.set_title("Variables importantes")
        st.pyplot(fig3)

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie utilisée")
        st.markdown("""
        1. **Prétraitement** : nettoyage, sélection des variables, encodage binaire (0=Non, 1=Oui) et catégoriel.  
        2. **Standardisation** : mise à l'échelle des variables quantitatives avec StandardScaler.  
        3. **Gestion du déséquilibre** : SMOTETomek pour équilibrer la variable cible.  
        4. **Modélisation** : entraînement de Decision Tree, Random Forest, SVM et LightGBM.  
        5. **Évaluation** : Accuracy, Precision, Recall, F1-Score et AUC-ROC.  
        6. **Sélection du meilleur modèle** : moyenne des métriques.  
        7. **Visualisation** : matrice de confusion et courbe ROC-AUC.  
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction")
        st.write("Remplir les informations du patient :")

        input_data = {}
        for feat in features:
            if df_selected[feat].dtype in [np.float64, np.int64]:
                input_data[feat] = st.number_input(f"{feat}", value=float(df_selected[feat].median()))
            else:
                # choix Oui/Non pour les variables binaires
                if set(df_selected[feat].dropna().unique()) <= {0,1}:
                    input_data[feat] = st.selectbox(f"{feat} (0=Non, 1=Oui)", options=[0,1])
                else:
                    # autres variables catégorielles
                    input_data[feat] = st.selectbox(f"{feat}", options=df_selected[feat].dropna().unique())

        if st.button("Prédire l'évolution"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred_proba = model.predict_proba(input_scaled)[:,1][0]
            pred_class = int(pred_proba >= best_row["Seuil optimal"])
            st.write(f"**Probabilité de complication : {pred_proba:.3f}**")
            st.write(f"**Prédiction : {'Complications' if pred_class==1 else 'Favorable'}**")
