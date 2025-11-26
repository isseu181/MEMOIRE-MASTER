# ================================
# classification.py  
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb


# ===================================================
# FONCTION PRINCIPALE STREAMLIT
# ===================================================
def show_classification():

    st.markdown("<h1 style='text-align:center;color:darkblue;'>Classification Supervisée - Modèles et Résultats</h1>",
                unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # 1️ Chargement des données
    # ----------------------------------------------------------------------
    df = pd.read_excel("fichier_nettoye.xlsx")

    # ----------------------------------------------------------------------
    # 2️ Sélection des variables
    # ----------------------------------------------------------------------
    vars_select = [
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

    df_sel = df[vars_select].copy()

    # ----------------------------------------------------------------------
    # 3️ Encodage
    # ----------------------------------------------------------------------
    binary_map = {
        'Pâleur': {'OUI':1,'NON':0},
        'Souffle systolique fonctionnel': {'OUI':1,'NON':0},
        'Vaccin contre méningocoque': {'OUI':1,'NON':0},
        'Splénomégalie': {'OUI':1,'NON':0},
        'Prophylaxie à la pénicilline': {'OUI':1,'NON':0},
        'Parents Salariés': {'OUI':1,'NON':0},
        'Prise en charge Hospitalisation': {'OUI':1,'NON':0},
        'Radiographie du thorax Oui ou Non': {'OUI':1,'NON':0},
        'Douleur provoquée (Os.Abdomen)': {'OUI':1,'NON':0},
        'Vaccin contre pneumocoque': {'OUI':1,'NON':0},
    }
    df_sel.replace(binary_map, inplace=True)

    ordinal_map = {
        "NiveauUrgence": {'Urgence1':1,'Urgence2':2,'Urgence3':3,'Urgence4':4,'Urgence5':5,'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1,'Elémentaire ':2,'Secondaire':3,'Enseignement Supérieur ':4,'NON':0}
    }
    df_sel.replace(ordinal_map, inplace=True)

    df_sel = pd.get_dummies(df_sel, columns=["Diagnostic Catégorisé", "Mois"])

    # ----------------------------------------------------------------------
    # 4️ Variable cible
    # ----------------------------------------------------------------------
    df_sel['Cible'] = df_sel['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_sel.drop(['Evolution', 'Cible'], axis=1)
    y = df_sel['Cible']

    # ----------------------------------------------------------------------
    # 5️ Division Train/Validation/Test 
    # ----------------------------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # ----------------------------------------------------------------------
    # 6️ Standardisation 
    # ----------------------------------------------------------------------
    quantitative = [
        'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)', 'HB (g/dl)',
        'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)', 'TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017", "Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
    ]

    scaler = StandardScaler()
    X_train_val[quantitative] = scaler.fit_transform(X_train_val[quantitative])
    X_test[quantitative] = scaler.transform(X_test[quantitative])

    # ----------------------------------------------------------------------
    # 7️ Seconde division Train/Val
    # ----------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )

    # ----------------------------------------------------------------------
    # 8️ SMOTETomek uniquement sur TRAIN
    # ----------------------------------------------------------------------
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

    # ----------------------------------------------------------------------
    # 9️ Fonction d’évaluation 
    # ----------------------------------------------------------------------
    def evaluate(model, name):

        # Calibration si modèle sans probas
        if not hasattr(model, "predict_proba"):
            model = CalibratedClassifierCV(model, method="sigmoid", cv=5)

        model.fit(X_train_res, y_train_res)

        y_proba = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_proba)
        fpr, tpr, thr = roc_curve(y_val, y_proba)
        best_idx = np.argmax(tpr - fpr)
        threshold = thr[best_idx]

        y_pred = (y_proba >= threshold).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True)
        cm = confusion_matrix(y_val, y_pred)

        return {
            "AUC": auc,
            "Report": report,
            "Threshold": threshold,
            "CM": cm,
            "Model": model
        }

    # ----------------------------------------------------------------------
    # 10 Modèles 
    # ----------------------------------------------------------------------
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15,
                                                min_samples_split=10, random_state=42),
        "SVM": SVC(C=1.0, kernel='rbf', gamma='scale', probability=False, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                       num_leaves=31, random_state=42, verbose=-1),
    }

    results = {name: evaluate(model, name) for name, model in models.items()}

    # ----------------------------------------------------------------------
    # 11 Tableau récapitulatif 
    # ----------------------------------------------------------------------
    summary = []
    for name, res in results.items():
        r = res["Report"]
        summary.append({
            "Modèle": name,
            "Accuracy": round(r["accuracy"], 3),
            "Precision": round(r["weighted avg"]["precision"], 3),
            "Recall": round(r["weighted avg"]["recall"], 3),
            "F1-Score": round(r["weighted avg"]["f1-score"], 3),
            "AUC-ROC": round(res["AUC"], 3),
            "Seuil optimal": round(res["Threshold"], 3)
        })

    summary_df = pd.DataFrame(summary).sort_values(by="AUC-ROC", ascending=False)

    # ----------------------------------------------------------------------
    # 12 Interface Streamlit 
    # ----------------------------------------------------------------------
    st.subheader("Résumé des performances des modèles")
    st.dataframe(summary_df)

    best_model_name = summary_df.iloc[0]["Modèle"]
    best_result = results[best_model_name]

    st.markdown(f"###  Meilleur modèle : **{best_model_name}**")

    st.write("Matrice de confusion :")
    st.write(best_result["CM"])

    # ROC curve
    y_proba = best_result["Model"].predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    roc_fig.add_trace(go.Scatter(x=[0 (Evolution favorable), 1 (complication)], y=[0, 1],
                                 mode='lines', name='Random', line=dict(dash='dash')))
    st.plotly_chart(roc_fig)





