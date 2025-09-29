# ================================
# classification.py pour Streamlit
# ================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

def show_classification():
    st.title("Classification Supervisée - Analyse des Modèles")

    # ================================
    # 1️⃣ Chargement des données
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # ================================
    # 2️⃣ Variables sélectionnées
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
    # 4️⃣ Standardisation
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
    # 5️⃣ Variable cible
    # ================================
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_selected.drop(['Evolution', 'Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']

    # ================================
    # 6️⃣ SMOTETomek
    # ================================
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    # ================================
    # 7️⃣ Division train/val/test
    # ================================
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    st.subheader("Taille des ensembles")
    st.write(f"Train : {X_train.shape}, Validation : {X_val.shape}, Test : {X_test.shape}")

    # ================================
    # 8️⃣ Modèles
    # ================================
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
    }

    # ================================
    # 9️⃣ Entraînement & évaluation
    # ================================
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_val_proba = model.predict_proba(X_val)[:,1]
        fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]

        y_test_proba = model.predict_proba(X_test)[:,1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        cm = confusion_matrix(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_test_proba)

        results[name] = {
            "Confusion Matrix": cm,
            "Classification Report": report,
            "AUC-ROC": auc,
            "Optimal Threshold": optimal_threshold,
            "Model Object": model
        }

    # ================================
    # 10️⃣ Comparaison des modèles
    # ================================
    summary_metrics = []
    for name, res in results.items():
        report = res['Classification Report']
        summary_metrics.append({
            "Modèle": name,
            "Accuracy": round(report['accuracy'],3),
            "Precision": round(report['macro avg']['precision'],3),
            "Recall": round(report['macro avg']['recall'],3),
            "F1-Score": round(report['macro avg']['f1-score'],3),
            "AUC-ROC": round(res['AUC-ROC'],3)
        })
    summary_df = pd.DataFrame(summary_metrics)

    # ================================
    # 11️⃣ Graphe comparatif Plotly
    # ================================
    fig = px.bar(summary_df, x="Modèle", y=["AUC-ROC","Precision"], barmode='group',
                 title="Comparaison des modèles selon AUC et Precision")
    st.plotly_chart(fig)

    # ================================
    # 12️⃣ Meilleur modèle selon toutes les métriques
    # ================================
    # Moyenne des métriques pour choisir le meilleur modèle
    summary_df['Mean Metric'] = summary_df[['Accuracy','Precision','Recall','F1-Score','AUC-ROC']].mean(axis=1)
    best_model_name = summary_df.loc[summary_df['Mean Metric'].idxmax(), 'Modèle']
    st.subheader(f"Meilleur modèle : {best_model_name}")
    st.write("Matrice de confusion du meilleur modèle :")
    st.write(results[best_model_name]["Confusion Matrix"])
    st.write("AUC-ROC :", results[best_model_name]["AUC-ROC"])

    # ================================
    # 13️⃣ Méthodologie
    # ================================
    st.subheader("Méthodologie et étapes")
    st.markdown("""
    - Chargement et exploration des données brutes.
    - Sélection des variables pertinentes pour l'analyse.
    - Encodage des variables binaires et ordinales.
    - Création de variables factices pour les catégories.
    - Standardisation des variables quantitatives.
    - Définition de la variable cible et encodage.
    - Gestion du déséquilibre avec SMOTETomek.
    - Division en ensembles train/validation/test.
    - Définition et entraînement de plusieurs modèles supervisés.
    - Évaluation des modèles sur plusieurs métriques (Accuracy, Precision, Recall, F1, AUC).
    - Comparaison visuelle des modèles.
    - Sélection du meilleur modèle basé sur l’ensemble des métriques.
    """)

    # ================================
    # 14️⃣ Variables importantes pour le meilleur modèle (si applicable)
    # ================================
    st.subheader("Variables importantes du meilleur modèle")
    best_model = results[best_model_name]["Model Object"]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False).head(15)
        st.bar_chart(importances)
    else:
        st.write("Pas de variable importance disponible pour ce modèle.")
