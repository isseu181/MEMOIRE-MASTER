# classification.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def show_classification():
    st.title("Classification des patients - Modèles prédictifs")

    # ================================
    # 1. Chargement automatique des données
    # ================================
    st.info("Chargement automatique de la base : fichier_nettoye.xlsx")
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
    # 3. Encodage des variables
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
    # 5. Variable cible et SMOTETomek
    # ================================
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']

    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # ================================
    # 6. Définition des modèles
    # ================================
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
    }

    results = {}

    # ================================
    # 7. Onglets horizontaux
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles")
        for name, model in models.items():
            model.fit(X_train, y_train)

            # ROC & seuil optimal
            y_val_proba = model.predict_proba(X_val)[:,1]
            fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]

            y_test_proba = model.predict_proba(X_test)[:,1]
            y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

            cm = confusion_matrix(y_test, y_test_pred)
            auc_score = roc_auc_score(y_test, y_test_proba)
            report = classification_report(y_test, y_test_pred, output_dict=True)

            st.write(f"### {name}")
            st.write(f"- Accuracy : {report['accuracy']:.3f}")
            st.write(f"- AUC-ROC : {auc_score:.3f}")
            st.write(f"- Sensibilité : {report['1']['recall']:.3f}")
            st.write(f"- Spécificité : {report['0']['recall']:.3f}")

            # Matrice de confusion
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            ax.set_title(f"Matrice de confusion - {name}")
            st.pyplot(fig)

            # Courbe ROC
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (AUC = {auc_score:.3f})')
            ax2.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0,1])
            ax2.set_ylim([0,1.05])
            ax2.set_xlabel('1 - Spécificité')
            ax2.set_ylabel('Sensibilité')
            ax2.set_title(f'Courbe ROC - {name}')
            ax2.legend(loc="lower right")
            st.pyplot(fig2)

            results[name] = {"model": model, "optimal_threshold": optimal_threshold}

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Variables importantes pour Random Forest")
        rf_model = results["Random Forest"]["model"]
        if hasattr(rf_model, "feature_importances_"):
            importances = pd.DataFrame({
                "Variable": X_train.columns,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.dataframe(importances)

            fig, ax = plt.subplots(figsize=(8,6))
            sns.barplot(x="Importance", y="Variable", data=importances)
            ax.set_title("Importance des variables")
            st.pyplot(fig)

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie utilisée")
        st.markdown("""
        1. Prétraitement des données (Nettoyage et sélection des variables).  
        2. Encodage des variables qualitatives et catégorielles.  
        3. Standardisation des variables quantitatives.  
        4. Rééquilibrage des classes avec SMOTETomek.  
        5. Division Train/Validation/Test.  
        6. Entraînement de plusieurs modèles (Decision Tree, Random Forest, SVM, LightGBM).  
        7. Évaluation avec Accuracy, Sensibilité, Spécificité, AUC-ROC.  
        8. Sélection du meilleur modèle et interprétation.  
        9. Simulation de prédiction pour nouveaux patients.  
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction pour un patient")
        st.write("Sélectionner les valeurs des variables :")
        input_data = {}
        for col in X_train.columns:
            if col in quantitative_vars:
                val = st.number_input(f"{col}", float(df[col].mean()))
            else:
                val = st.selectbox(f"{col}", [0,1])
            input_data[col] = val
        input_df = pd.DataFrame([input_data])
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])
        rf_model = results["Random Forest"]["model"]
        pred_prob = rf_model.predict_proba(input_df)[:,1][0]
        pred_class = "Complications" if pred_prob>=results["Random Forest"]["optimal_threshold"] else "Favorable"
        st.write(f"Prédiction : {pred_class} (Probabilité : {pred_prob:.3f})")

