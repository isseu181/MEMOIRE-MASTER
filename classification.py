# ================================
# classification.py complet
# ================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def show_classification():
    st.title("Classification supervisée")

    # ================================
    # 1️⃣ Chargement des données
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")

    # ================================
    # 2️⃣ Sélection des variables
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
    # Binaire
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

    # Ordinal
    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)

    # One-hot encoding
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

    # ================================
    # 8️⃣ Définition des modèles
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
            "Model": model,
            "CM": cm,
            "Report": report,
            "AUC": auc,
            "Optimal_Threshold": optimal_threshold
        }

    # ================================
    # 1️⃣ Onglet comparaison
    # ================================
    tab1, tab2, tab3 = st.tabs(["Comparer les modèles", "Méthodologie", "Meilleur modèle"])
    
    # --------------------
    # Comparer les modèles
    # --------------------
    with tab1:
        summary = []
        for name,res in results.items():
            r = res["Report"]
            summary.append({
                "Modèle":name,
                "Accuracy":round(r['accuracy'],3),
                "Precision":round(r['macro avg']['precision'],3),
                "Recall":round(r['macro avg']['recall'],3),
                "F1":round(r['macro avg']['f1-score'],3),
                "AUC":round(res['AUC'],3)
            })
        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df)

        # Plotly comparatif
        fig = px.bar(summary_df, x="Modèle", y=["AUC","Precision"], barmode='group', title="Comparaison des modèles (AUC + Precision)")
        st.plotly_chart(fig)

    # --------------------
    # Méthodologie
    # --------------------
    with tab2:
        st.header("Méthodologie et étapes")
        st.write("""
        1️⃣ Chargement et sélection des variables pertinentes.  
        2️⃣ Encodage des variables catégorielles et binaires.  
        3️⃣ Standardisation des variables quantitatives.  
        4️⃣ Application de SMOTETomek pour équilibrer les classes.  
        5️⃣ Division en jeux Train, Validation et Test.  
        6️⃣ Entraînement de plusieurs modèles supervisés : Decision Tree, Random Forest, SVM, LightGBM.  
        7️⃣ Évaluation sur le jeu de Test avec toutes les métriques : Accuracy, Precision, Recall, F1, AUC.  
        8️⃣ Sélection du meilleur modèle selon toutes les métriques.  
        Note : des valeurs manquantes ou des '-' dans la base ont été gérées lors de l'encodage et du remplissage.
        """)

    # --------------------
    # Meilleur modèle
    # --------------------
    with tab3:
        summary_df['Mean'] = summary_df[['Accuracy','Precision','Recall','F1','AUC']].mean(axis=1)
        best_name = summary_df.loc[summary_df['Mean'].idxmax(),'Modèle']
        st.write(f"Meilleur modèle selon toutes les métriques : **{best_name}**")
        st.write("Matrice de confusion :")
        st.write(results[best_name]["CM"])
       

        # Courbe ROC
        best_model = results[best_name]["Model"]
        y_test_proba = best_model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title=f'Courbe ROC - {best_name}', xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig_roc)

