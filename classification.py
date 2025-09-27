# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1. Chargement des fichiers
# ================================
df = pd.read_excel("fichier_nettoye.xlsx")

# Charger modèle, scaler et features
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ================================
# 2. Variables sélectionnées
# ================================
variables_selection = features + ['Evolution']  # utiliser les features sauvegardés
df_selected = df[variables_selection].copy()

# ================================
# 3. Encodage binaire
# ================================
binary_cols = [
    'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque','Splénomégalie',
    'Prophylaxie à la pénicilline','Parents Salariés','Prise en charge Hospitalisation',
    'Radiographie du thorax Oui ou Non','Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque'
]
for col in binary_cols:
    df_selected[col] = df_selected[col].map({'OUI':1, 'NON':0})

# ================================
# 4. Encodage ordinal
# ================================
ordinal_cols = {
    'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
    "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
}
for col, mapping in ordinal_cols.items():
    if col in df_selected.columns:
        df_selected[col] = df_selected[col].map(mapping)

# ================================
# 5. Standardisation des quantitatives
# ================================
quantitative_vars = [
    'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
    'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
    "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
]
df_selected[quantitative_vars] = scaler.transform(df_selected[quantitative_vars])

# ================================
# 6. Définir X et y
# ================================
X = df_selected[features]
y = df_selected['Evolution'].map({'Favorable':0,'Complications':1})

# ================================
# 7. Création de l'app Streamlit
# ================================
def show_classification():
    st.title("Classification de l'évolution des patients")

    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Performance des modèles")

        # On affiche les résultats fictifs (d'après ton tableau fourni)
        results_df = pd.DataFrame({
            "Modèle": ["LightGBM","Random Forest","Decision Tree","SVM"],
            "Accuracy":[0.963,0.984,0.915,0.746],
            "Precision":[0.966,0.984,0.915,0.746],
            "Recall":[0.963,0.984,0.915,0.746],
            "F1-Score":[0.963,0.984,0.915,0.746],
            "AUC-ROC":[0.998,0.997,0.915,0.809],
            "Seuil optimal":[0.999,0.560,1.0,0.456]
        })

        st.dataframe(results_df.style.format({
            "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
            "F1-Score":"{:.3f}","AUC-ROC":"{:.3f}","Seuil optimal":"{:.3f}"
        }))

        # Détermination du meilleur modèle par combinaison de métriques (ici, on choisit Random Forest)
        st.markdown("### Modèle sélectionné : Random Forest")

        # Matrice de confusion
        y_pred_proba = rf_model.predict_proba(X)[:,1]
        optimal_threshold = 0.56  # d'après ton tableau
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        cm = confusion_matrix(y, y_pred)
        st.write("Matrice de confusion : 0=Favorable, 1=Complications")
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        st.pyplot(plt)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        st.write(f"AUC-ROC : {roc_auc:.3f}")
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('1 - Spécificité (Faux positifs)')
        plt.ylabel('Sensibilité (Vrais positifs)')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # Affichage des métriques
        st.write("### Précision et autres métriques")
        st.write(f"Accuracy : 0.984, Precision : 0.984, Recall (Sensibilité) : 0.984")
        st.write(f"Seuil optimal : {optimal_threshold}")

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Variables importantes du modèle Random Forest")
        importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
        st.dataframe(importances)

        plt.figure(figsize=(8,5))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title("Importance des variables")
        st.pyplot(plt)

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie utilisée")
        st.markdown("""
        1. **Prétraitement des données** : nettoyage et sélection des variables.  
        2. **Encodage** : transformation des variables qualitatives en binaire (0=NON, 1=OUI) et ordinal.  
        3. **Standardisation** : mise à l'échelle des variables quantitatives avec un scaler sauvegardé.  
        4. **Classification** : Random Forest pour prédire l'évolution des patients.  
        5. **Évaluation** : Accuracy, Precision, Recall, F1-Score, AUC-ROC, matrice de confusion et courbe ROC.
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction pour un nouveau patient")
        new_data = {}
        for col in features:
            if col in binary_cols:
                new_data[col] = st.selectbox(f"{col} (0=NON,1=OUI)", [0,1])
            elif col in quantitative_vars:
                new_data[col] = st.number_input(f"{col}", value=float(df_selected[col].mean()))
            elif col in ordinal_cols:
                new_data[col] = st.selectbox(f"{col}", list(ordinal_cols[col].values()))
            else:
                new_data[col] = st.number_input(f"{col}", value=float(df_selected[col].mean()))

        if st.button("Prédire l'évolution"):
            new_df = pd.DataFrame([new_data])
            new_df_scaled = scaler.transform(new_df[features])
            proba = rf_model.predict_proba(new_df_scaled)[:,1][0]
            pred = int(proba >= 0.56)
            st.write(f"Prédiction : {'Complications' if pred==1 else 'Favorable'}")
            st.write(f"Probabilité de complications : {proba:.3f}")

