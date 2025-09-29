# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb

st.set_page_config(page_title="Classification Patients", layout="wide")

def show_classification():
    st.title("Classification Patients - Evolution clinique")

    # ================================
    # 1️⃣ Chargement des fichiers
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")
    best_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    df_selected = df.copy()

    # Colonnes quantitatives utilisées pour la standardisation
    quantitative_vars = [
        'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
        'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
        'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
    ]

    # Reindexer selon les features sauvegardés (remplit les manquantes avec 0)
    X = df_selected.reindex(columns=features, fill_value=0)

    # Standardiser les quantitatives
    X[quantitative_vars] = scaler.transform(X[quantitative_vars])

    y = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})

    # ================================
    # 2️⃣ Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles et métriques")

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": best_model,
            "SVM": SVC(probability=True, random_state=42),
            "LightGBM": lgb.LGBMClassifier()
        }

        results = []
        for name, mdl in models.items():
            if name != "Random Forest":
                mdl.fit(X, y)

            y_proba = mdl.predict_proba(X)[:,1]
            fpr, tpr, thresholds = roc_curve(y, y_proba)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            y_pred = (y_proba >= optimal_threshold).astype(int)

            results.append({
                "Modèle": name,
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred),
                "Recall": recall_score(y, y_pred),
                "F1-Score": f1_score(y, y_pred),
                "AUC-ROC": roc_auc_score(y, y_proba),
                "Seuil optimal": optimal_threshold
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        metrics = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
        results_df["Score Moyenne"] = results_df[metrics].mean(axis=1)
        best_row = results_df.loc[results_df["Score Moyenne"].idxmax()]
        st.success(f"✅ Modèle retenu : {best_row['Modèle']}")

        # Matrice de confusion
        best_model_final = models[best_row["Modèle"]]
        y_proba_best = best_model_final.predict_proba(X)[:,1]
        y_pred_best = (y_proba_best >= best_row["Seuil optimal"]).astype(int)

        cm = confusion_matrix(y, y_pred_best)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(f"Matrice de confusion - {best_row['Modèle']}")
        st.pyplot(fig)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y, y_proba_best)
        roc_auc = roc_auc_score(y, y_proba_best)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel("Taux de faux positifs (1-Spécificité)")
        ax2.set_ylabel("Taux de vrais positifs (Sensibilité)")
        ax2.set_title(f"Courbe ROC - {best_row['Modèle']}")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Importance des variables")
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            feat_importance = pd.DataFrame({"Variable": features, "Importance": importances})
            feat_importance = feat_importance.sort_values(by="Importance", ascending=False)
            st.dataframe(feat_importance)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x="Importance", y="Variable", data=feat_importance, ax=ax, palette="viridis")
            ax.set_title("Variables importantes")
            st.pyplot(fig)
        else:
            st.info("Le modèle sélectionné ne fournit pas d'importance des variables.")

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie utilisée")
        st.markdown("""
        1. Prétraitement des données et sélection des variables pertinentes
        2. Encodage des variables qualitatives et standardisation des quantitatives
        3. Entraînement des modèles : Random Forest, Decision Tree, SVM, LightGBM
        4. Évaluation et comparaison via Accuracy, Precision, Recall, F1-Score, AUC-ROC et seuil optimal
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction")
        st.markdown("Entrez les valeurs des variables pour prédire l'évolution clinique")

        # Identifier les colonnes One-Hot pour Mois et Diagnostic
        mois_cols = [col for col in features if col.startswith("Mois_")]
        diag_cols = [col for col in features if col.startswith("Diagnostic Catégorisé_")]

        mois_values = [col.replace("Mois_", "") for col in mois_cols]
        diag_values = [col.replace("Diagnostic Catégorisé_", "") for col in diag_cols]

        user_input = {}

        for feat in features:
            if feat in quantitative_vars:
                min_val = float(df_selected[feat].min())
                max_val = float(df_selected[feat].max())
                mean_val = float(df_selected[feat].mean())
                user_input[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=mean_val)
            elif feat in mois_cols or feat in diag_cols:
                user_input[feat] = 0
            else:
                user_input[feat] = st.selectbox(feat, options=[0,1], format_func=lambda x: "Oui" if x==1 else "Non")

        if mois_cols:
            selected_mois = st.selectbox("Mois", options=mois_values)
            for col in mois_cols:
                user_input[col] = 1 if col == f"Mois_{selected_mois}" else 0

        if diag_cols:
            selected_diag = st.selectbox("Diagnostic Catégorisé", options=diag_values)
            for col in diag_cols:
                user_input[col] = 1 if col == f"Diagnostic Catégorisé_{selected_diag}" else 0

        if st.button("Prédire l'évolution"):
            X_new = pd.DataFrame([user_input])
            for col in features:
                if col not in X_new.columns:
                    X_new[col] = 0
            X_new = X_new[features]
            X_new[quantitative_vars] = scaler.transform(X_new[quantitative_vars])
            y_new_proba = best_model.predict_proba(X_new)[:,1]
            y_new_pred = (y_new_proba >= 0.56).astype(int)
            st.write("**Probabilité d'évolution vers complications :**", round(float(y_new_proba),3))
            st.write("**Prédiction finale :**", "Complications" if y_new_pred[0]==1 else "Favorable")
