# ================================
# classification.py
# ================================
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def show_classification():
    st.title("Classification des patients - Évolution clinique")

    # ================================
    # Chargement des données
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")

    # ================================
    # Variables sélectionnées
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
    # Encodage des variables binaires 0 = Non, 1 = Oui
    # ================================
    binary_vars = [
        'Pâleur', 'Souffle systolique fonctionnel', 'Vaccin contre méningocoque', 'Splénomégalie',
        'Prophylaxie à la pénicilline', 'Parents Salariés', 'Prise en charge Hospitalisation',
        'Radiographie du thorax Oui ou Non', 'Douleur provoquée (Os.Abdomen)', 'Vaccin contre pneumocoque'
    ]
    for col in binary_vars:
        df_selected[col] = df_selected[col].map({'OUI':1, 'NON':0})

    # ================================
    # Encodage ordinal
    # ================================
    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)

    # ================================
    # One-Hot Encoding pour Diagnostic et Mois
    # ================================
    df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé','Mois'], drop_first=True)

    # ================================
    # Standardisation des variables quantitatives
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
    # Variable cible
    # ================================
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_selected.drop(['Evolution', 'Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']

    # ================================
    # SMOTETomek
    # ================================
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    # ================================
    # Division Train / Test / Validation
    # ================================
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    st.success("✅ Préprocessing terminé avec succès !")
    st.write("Nombre de patients dans chaque classe (0 = Favorable, 1 = Complications) :")
    st.dataframe(pd.Series(y.value_counts()))

    # ================================
    # Modèle Random Forest par défaut
    # ================================
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_test_proba = rf_model.predict_proba(X_test)[:,1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_test_pred)
    st.write("### Matrice de confusion (0=Non,1=Oui)")
    st.dataframe(cm)

    # Classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    st.write("### Précision / Recall / F1-Score")
    st.dataframe(pd.DataFrame(report).T)

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    st.write(f"### AUC-ROC : {auc_score:.3f}")
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('1 - Spécificité (Faux positifs)')
    plt.ylabel('Sensibilité (Vrais positifs)')
    plt.title('Courbe ROC - Random Forest')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # ================================
    # Sauvegarde du modèle et scaler
    # ================================
    chemin_sauvegarde = "modeles"
    joblib.dump(rf_model, f"{chemin_sauvegarde}/random_forest_model.pkl")
    joblib.dump(scaler, f"{chemin_sauvegarde}/scaler.pkl")
    features = X_train.columns.tolist()
    joblib.dump(features, f"{chemin_sauvegarde}/features.pkl")
    st.success("Modèle, scaler et features sauvegardés !")
