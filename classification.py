# classification.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Classification Patients", layout="wide")

# ================================
# 1. Chargement des données
# ================================
st.title("Classification des Patients - Évolution")

df = pd.read_excel("fichier_nettoye.xlsx")

# ================================
# 2. Variables sélectionnées
# ================================
variables_selection = [
    'Âge de début des signes (en mois)', 'NiveauUrgence', 'GR (/mm3)', 'GB (/mm3)',
    "Nbre d'hospitalisations avant 2017", 'CRP Si positive (Valeur)', 'Pâleur',
    'Âge du debut d etude en mois (en janvier 2023)', 'Souffle systolique fonctionnel',
    'VGM (fl/u3)', 'HB (g/dl)', 'Vaccin contre méningocoque', 'Nbre de GB (/mm3)',
    "% d'Hb S", 'Âge de découverte de la drépanocytose (en mois)', 'Splénomégalie',
    'Prophylaxie à la pénicilline', "Taux d'Hb (g/dL)", 'Parents Salariés',
    'PLT (/mm3)', 'Diagnostic Catégorisé', 'Prise en charge Hospitalisation',
    'Nbre de PLT (/mm3)', 'TCMH (g/dl)', 'Nbre de transfusion avant 2017',
    'Radiographie du thorax Oui ou Non', "Niveau d'instruction scolarité",
    "Nbre d'hospitalisations entre 2017 et 2023", "% d'Hb F",
    'Douleur provoquée (Os.Abdomen)', 'Mois', 'Vaccin contre pneumocoque',
    'HDJ', 'Nbre de transfusion Entre 2017 et 2023', 'Evolution'
]
df_selected = df[variables_selection].copy()

# ================================
# 3. Encodage
# ================================
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

# ================================
# 4. Standardisation
# ================================
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]
scaler = StandardScaler()
df_selected[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

# ================================
# 5. Variable cible
# ================================
df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
y = df_selected['Evolution_Cible']

# ================================
# 6. SMOTETomek
# ================================
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X, y)

# ================================
# 7. Division train/val/test
# ================================
X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ================================
# 8. Modèles
# ================================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)[:,1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_test_proba = model.predict_proba(X_test)[:,1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_proba)
    results[name] = {"model": model, "y_test_pred": y_test_pred, "y_test_proba": y_test_proba, "cm": cm, "auc": auc_score, "threshold": optimal_threshold}

# ================================
# Onglets Streamlit
# ================================
tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

# --- Onglet 1 : Performance ---
with tabs[0]:
    st.subheader("Comparaison des modèles")
    summary_metrics = []
    for name, res in results.items():
        report = classification_report(y_test, res["y_test_pred"], output_dict=True)
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        auc = res['auc']
        summary_metrics.append({
            "Modèle": name,
            "Accuracy": round(accuracy,3),
            "Precision": round(precision,3),
            "Recall": round(recall,3),
            "F1-Score": round(f1,3),
            "AUC-ROC": round(auc,3),
            "Seuil optimal": round(res["threshold"],3)
        })
    summary_df = pd.DataFrame(summary_metrics).sort_values(by="AUC-ROC", ascending=False)
    st.dataframe(summary_df)

    # Modèle retenu
    best_model_name = summary_df.iloc[0]["Modèle"]
    st.write(f"**Modèle retenu : {best_model_name}**")
    best_res = results[best_model_name]

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, best_res["y_test_proba"])
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {best_res["auc"]:.3f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("1 - Spécificité")
    plt.ylabel("Sensibilité")
    plt.title("Courbe ROC-AUC")
    plt.legend()
    st.pyplot(plt)

    # Matrice de confusion
    plt.figure(figsize=(5,4))
    sns.heatmap(best_res["cm"], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {best_model_name}")
    st.pyplot(plt)

# --- Onglet 2 : Variables importantes ---
with tabs[1]:
    st.subheader("Importance des variables (Random Forest et LightGBM)")
    for name in ["Random Forest", "LightGBM"]:
        if name in models:
            model = models[name]
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                st.write(f"### {name}")
                st.dataframe(importances)

# --- Onglet 3 : Méthodologie ---
with tabs[2]:
    st.subheader("Méthodologie utilisée")
    st.markdown("""
    1. Prétraitement et sélection des variables.
    2. Encodage des variables qualitatives et ordinales.
    3. Standardisation des variables quantitatives.
    4. Gestion du déséquilibre avec SMOTETomek.
    5. Division en ensembles train/validation/test.
    6. Modélisation avec Decision Tree, Random Forest, SVM et LightGBM.
    7. Évaluation des modèles sur l'ensemble test (Accuracy, AUC-ROC, Sensibilité, Spécificité).
    8. Sélection du meilleur modèle basé sur l'AUC-ROC.
    """)

# --- Onglet 4 : Simulateur ---
with tabs[3]:
    st.subheader("Simulateur de prédiction")
    st.write("Renseignez les caractéristiques du patient :")
    input_data = {}
    for col in X_train.columns:
        if col in quantitative_vars:
            val = st.number_input(f"{col}", value=0.0)
        else:
            val = st.selectbox(f"{col}", [0,1])
        input_data[col] = val
    if st.button("Prédire l'évolution"):
        input_df = pd.DataFrame([input_data])
        pred_prob = best_res["model"].predict_proba(input_df)[:,1][0]
        pred_class = int(pred_prob >= best_res["threshold"])
        st.write(f"Probabilité de complication : {pred_prob:.3f}")
        st.write(f"Classe prédite : {'Complications' if pred_class==1 else 'Favorable'}")
