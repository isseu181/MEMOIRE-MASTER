# ================================
# classification.py pour Streamlit avec onglets
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
    # 2️⃣ Variables sélectionnées et encodage
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

    # Encodage binaire
    binary_mappings = {col: {'OUI':1,'NON':0} for col in [
        'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
        'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
        'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
        'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque']}
    df_selected.replace(binary_mappings, inplace=True)

    # Encodage ordinal
    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1,'Urgence2':2,'Urgence3':3,'Urgence4':4,'Urgence5':5,'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1,'Elémentaire ':2,'Secondaire':3,'Enseignement Supérieur ':4,'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)

    # Variables catégorielles en dummies
    df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé','Mois'], drop_first=True)

    # Standardisation
    quantitative_vars = [
        'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
        'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
    ]
    scaler = StandardScaler()
    df_selected[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

    # Variable cible
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0,'Complications':1})
    X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']

    # SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X,y)

    # Division train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X_res,y_res,test_size=0.4,stratify=y_res,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,stratify=y_temp,random_state=42)

    # ================================
    # Modèles
    # ================================
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='binary',learning_rate=0.05,num_leaves=31,n_estimators=500,random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:,1]
        fpr,tpr,thresholds = roc_curve(y_val,y_val_proba)
        optimal_threshold = thresholds[np.argmax(tpr-fpr)]
        y_test_proba = model.predict_proba(X_test)[:,1]
        y_test_pred = (y_test_proba>=optimal_threshold).astype(int)
        cm = confusion_matrix(y_test,y_test_pred)
        report = classification_report(y_test,y_test_pred,output_dict=True)
        auc = roc_auc_score(y_test,y_test_proba)
        results[name] = {"CM":cm,"Report":report,"AUC":auc,"Threshold":optimal_threshold,"Model":model}

    # ================================
    # Onglets Streamlit
    # ================================
    tab1, tab2, tab3 = st.tabs(["Comparaison des modèles","Méthodologie","Variables importantes"])

    # ================================
    # Onglet 1 - Comparaison des modèles
    # ================================
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

        # Plotly comparatif AUC + Precision
        fig = px.bar(summary_df, x="Modèle", y=["AUC","Precision"], barmode='group', title="Comparaison des modèles")
        st.plotly_chart(fig)

        # Meilleur modèle selon moyenne métriques
        summary_df['Mean'] = summary_df[['Accuracy','Precision','Recall','F1','AUC']].mean(axis=1)
        best_name = summary_df.loc[summary_df['Mean'].idxmax(),'Modèle']
        st.write(f"Meilleur modèle selon toutes les métriques : **{best_name}**")
        st.write("Matrice de confusion :")
        st.write(results[best_name]["CM"])
        st.write("AUC-ROC :", results[best_name]["AUC"])

    # ================================
    # Onglet 2 - Méthodologie
    # ================================
    with tab2:
        st.markdown("""
        ### Étapes méthodologiques
        1. Chargement et exploration des données brutes.
        2. Sélection des variables pertinentes.
        3. Encodage des variables binaires et ordinales.
        4. Création de variables factices pour les catégories.
        5. Standardisation des variables quantitatives.
        6. Définition de la variable cible et encodage.
        7. Gestion du déséquilibre avec SMOTETomek.
        8. Division en ensembles train/validation/test.
        9. Définition et entraînement de plusieurs modèles supervisés.
        10. Évaluation des modèles sur plusieurs métriques (Accuracy, Precision, Recall, F1, AUC).
        11. Comparaison visuelle des modèles.
        12. Sélection du meilleur modèle basé sur l’ensemble des métriques.
        """)

    # ================================
    # Onglet 3 - Variables importantes
    # ================================
    with tab3:
        st.subheader("Variables importantes du meilleur modèle")
        best_model = results[best_name]["Model"]
        if hasattr(best_model,"feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            st.bar_chart(importances)
        else:
            st.write("Pas de variable importance disponible pour ce modèle.")
