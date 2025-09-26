# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# ================================
# 1. Chargement modèle, scaler et features
# ================================
chemin_sauvegarde = r"C:\Users\DELL\Desktop\stage sante"
best_model = joblib.load(f"{chemin_sauvegarde}\\random_forest_model.pkl")
scaler = joblib.load(f"{chemin_sauvegarde}\\scaler.pkl")
features = joblib.load(f"{chemin_sauvegarde}\\features.pkl")

# ================================
# 2. Chargement données
# ================================
df = pd.read_excel("fichier_nettoye.xlsx")

# Variables sélectionnées (quantitatives et binaires)
variables_selection = features + ["Evolution"]
df_selected = df[variables_selection].copy()

# Définir variable cible
df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
y = df_selected['Evolution_Cible']

# Standardisation
quantitative_vars = [col for col in X.columns if np.issubdtype(X[col].dtype, np.number)]
X[quantitative_vars] = scaler.transform(X[quantitative_vars])

# ================================
# 3. Onglets Streamlit
# ================================
st.title("Classification de l'évolution des patients")

tabs = st.tabs(["Performance Modèles", "Variables Importantes", "Méthodologie", "Simulateur"])

# --- Onglet 1 : Performance Modèles ---
with tabs[0]:
    st.subheader("Comparaison des modèles et résultats du meilleur modèle")
    
    # Résultats simulés
    summary_df = pd.DataFrame({
        "Modèle":["LightGBM","Random Forest","Decision Tree","SVM"],
        "Accuracy":[0.963,0.984,0.915,0.746],
        "Precision":[0.966,0.984,0.915,0.746],
        "Recall":[0.963,0.984,0.915,0.746],
        "F1-Score":[0.963,0.984,0.915,0.746],
        "AUC-ROC":[0.998,0.997,0.915,0.809],
        "Seuil optimal":[0.999,0.560,1.000,0.456]
    })

    st.write("### Résumé des modèles")
    st.dataframe(summary_df)

    # Choix du meilleur modèle selon moyenne métriques
    metrics = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
    summary_df["Moyenne_metrics"] = summary_df[metrics].mean(axis=1)
    best_model_name = summary_df.sort_values(by="Moyenne_metrics", ascending=False).iloc[0]["Modèle"]

    st.success(f"Meilleur modèle sélectionné : {best_model_name}")

    # Affichage matrice et courbe ROC pour Random Forest
    y_pred = best_model.predict(X)
    y_pred_proba = best_model.predict_proba(X)[:,1]

    # Matrice de confusion
    cm = confusion_matrix(y, y_pred)
    st.write("### Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    st.pyplot(fig)

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    st.write("### Courbe ROC - Random Forest")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel("Taux de faux positifs (1-Spécificité)")
    ax.set_ylabel("Taux de vrais positifs (Sensibilité)")
    ax.set_title("ROC Curve - Random Forest")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Calcul métriques principales
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    spec = cm[0,0]/(cm[0,0]+cm[0,1])
    st.write("### Métriques du modèle retenu (Random Forest)")
    st.write(f"**Accuracy :** {acc:.3f}")
    st.write(f"**Précision :** {prec:.3f}")
    st.write(f"**Recall (Sensibilité) :** {rec:.3f}")
    st.write(f"**F1-Score :** {f1:.3f}")
    st.write(f"**Spécificité :** {spec:.3f}")
    st.write(f"**AUC-ROC :** {roc_auc:.3f}")

# --- Onglet 2 : Variables importantes ---
with tabs[1]:
    st.subheader("Variables importantes - Random Forest")
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({"Variable": X.columns, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="Importance", y="Variable", data=importance_df, ax=ax)
    ax.set_title("Importance des variables")
    st.pyplot(fig)

# --- Onglet 3 : Méthodologie ---
with tabs[2]:
    st.subheader("Méthodologie utilisée")
    st.markdown("""
    1. **Prétraitement des données** : nettoyage, sélection et encodage des variables.
    2. **Encodage** : transformation des variables binaires et catégorielles.
    3. **Standardisation** : mise à l’échelle des variables quantitatives.
    4. **SMOTETomek** : pour équilibrer les classes.
    5. **Division train/val/test** : entraînement et validation du modèle.
    6. **Entraînement des modèles** : Decision Tree, Random Forest, SVM, LightGBM.
    7. **Évaluation** : Accuracy, Precision, Recall, F1-Score, AUC-ROC, matrice de confusion et courbe ROC.
    8. **Sélection du meilleur modèle** selon la moyenne des métriques principales.
    """)

# --- Onglet 4 : Simulateur ---
with tabs[3]:
    st.subheader("Simulateur - Prédiction pour un nouveau patient")
    st.write(" Remplissez les informations du patient :")
    st.write(" Pour les variables binaires : 0 = Non, 1 = Oui")

    user_input = {}
    # Quantitatives
    for var in quantitative_vars:
        min_val = float(X[var].min())
        max_val = float(X[var].max())
        mean_val = float(X[var].mean())
        user_input[var] = st.number_input(f"{var}", value=mean_val, min_value=min_val, max_value=max_val)
    # Binaires
    binary_vars = [col for col in X.columns if col not in quantitative_vars]
    for var in binary_vars:
        user_input[var] = st.selectbox(f"{var}", options=[0,1], index=0)

    if st.button("Prédire l'évolution"):
        input_df = pd.DataFrame([user_input])
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])
        pred = best_model.predict(input_df)[0]
        proba = best_model.predict_proba(input_df)[:,1][0]

        st.write("### Résultat de la prédiction :")
        if pred==0:
            st.success("Évolution prévue : Favorable")
        else:
            st.error("Évolution prévue : Complications")
        st.info(f"Probabilité d'évolution vers complication : {proba:.2f}")
