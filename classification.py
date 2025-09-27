# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
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
    # 1. Chargement des fichiers
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")
    best_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    df_selected = df.copy()

    # 🔹 Correction KeyError : créer les colonnes manquantes si nécessaire
    for col in features:
        if col not in df_selected.columns:
            df_selected[col] = 0

    X = df_selected[features]
    y = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})

    X_scaled = scaler.transform(X)

    # ================================
    # 2. Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles et métriques")

        # Définition des modèles
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": best_model,
            "SVM": SVC(probability=True, random_state=42),
            "LightGBM": lgb.LGBMClassifier()
        }

        results = []
        for name, mdl in models.items():
            if name != "Random Forest":
                mdl.fit(X_scaled, y)

            y_proba = mdl.predict_proba(X_scaled)[:,1]
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

        # Détermination du meilleur modèle selon moyenne métriques
        metrics = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
        results_df["Score Moyenne"] = results_df[metrics].mean(axis=1)
        best_row = results_df.loc[results_df["Score Moyenne"].idxmax()]
        st.success(f"✅ Modèle retenu : {best_row['Modèle']}")

        # Matrice de confusion
        best_model_final = models[best_row["Modèle"]]
        y_proba_best = best_model_final.predict_proba(X_scaled)[:,1]
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
            # Graphique
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
        1. **Prétraitement des données** : nettoyage et sélection des variables pertinentes.
        2. **Encodage des variables qualitatives** : 0 = Non, 1 = Oui, et encodage ordinal si nécessaire.
        3. **Standardisation** : mise à l’échelle des variables quantitatives via StandardScaler.
        4. **Sélection des features** : sauvegardées dans `features.pkl`.
        5. **Entraînement des modèles** : Random Forest, Decision Tree, SVM, LightGBM.
        6. **Evaluation** : Accuracy, Precision, Recall, F1-Score, AUC-ROC, seuil optimal.
        7. **Comparaison des modèles** : choix du meilleur modèle par moyenne des métriques.
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction")
        st.markdown("Entrez les valeurs des variables pour prédire l'évolution clinique")

        user_input = {}
        for feat in features:
            # Si variable binaire
            if df_selected[feat].nunique() == 2:
                user_input[feat] = st.selectbox(feat, options=[0,1], format_func=lambda x: "Oui" if x==1 else "Non")
            else:
                min_val = float(df_selected[feat].min())
                max_val = float(df_selected[feat].max())
                mean_val = float(df_selected[feat].mean())
                user_input[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Prédire l'évolution"):
            X_new = pd.DataFrame([user_input])
            X_new_scaled = scaler.transform(X_new)
            y_new_proba = best_model.predict_proba(X_new_scaled)[:,1]
            y_new_pred = (y_new_proba >= best_row["Seuil optimal"]).astype(int)
            st.write("**Probabilité d'évolution vers complications :**", round(float(y_new_proba),3))
            st.write("**Prédiction finale :**", "Complications" if y_new_pred[0]==1 else "Favorable")
