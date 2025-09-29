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

    # ================================
    # 2️⃣ Sélection des colonnes
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
    # 3️⃣ Encodage
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
    # 4️⃣ Variable cible
    # ================================
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})

    # Reindexer pour correspondre aux features sauvegardées
    X = df_selected.reindex(columns=features, fill_value=0)
    # Convertir en float uniquement pour les colonnes numériques
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    y = df_selected['Evolution_Cible']

    # ================================
    # 5️⃣ Standardisation
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
    X[quantitative_vars] = scaler.transform(X[quantitative_vars])

    # ================================
    # 6️⃣ Onglets Streamlit
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

        # Matrice de confusion pour le meilleur modèle
        best_row = results_df.loc[results_df["AUC-ROC"].idxmax()]
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
        1. Prétraitement des données : nettoyage et sélection des variables pertinentes.
        2. Encodage des variables qualitatives et binaires.
        3. Standardisation via StandardScaler.
        4. SMOTETomek pour équilibrage des classes.
        5. Division train/validation/test.
        6. Entraînement des modèles : Random Forest, Decision Tree, SVM, LightGBM.
        7. Evaluation : Accuracy, Precision, Recall, F1-Score, AUC-ROC.
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction")
        st.info("Entrez les caractéristiques du patient pour obtenir une prédiction d'évolution.")

        # Exemple : saisie utilisateur
        user_input = {}
        for var in features:
            user_input[var] = st.number_input(var, value=0.0)

        new_data = pd.DataFrame([user_input])
        # Standardiser les variables quantitatives
        new_data[quantitative_vars] = scaler.transform(new_data[quantitative_vars])
        # Ajouter les colonnes manquantes
        for col in features:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[features]

        pred_proba = best_model.predict_proba(new_data)[:,1]
        pred_class = (pred_proba >= 0.56).astype(int)

        st.write("Probabilité de complication :", pred_proba[0])
        st.write("Classe prédite :", "Complications" if pred_class[0]==1 else "Favorable")
