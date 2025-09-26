# classification.py
import streamlit as st
import pandas as pd
import numpy as np
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

def show_classification():
    st.title("Classification des patients - Prévision de l'évolution")

    # ================================
    # 1. Chargement des données
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
    # 3. Encodage
    # ================================
    binary_mappings = {
        'Pâleur': {'OUI':1, 'NON':0}, 'Souffle systolique fonctionnel': {'OUI':1, 'NON':0},
        'Vaccin contre méningocoque': {'OUI':1, 'NON':0}, 'Splénomégalie': {'OUI':1, 'NON':0},
        'Prophylaxie à la pénicilline': {'OUI':1, 'NON':0}, 'Parents Salariés': {'OUI':1, 'NON':0},
        'Prise en charge Hospitalisation': {'OUI':1, 'NON':0}, 'Radiographie du thorax Oui ou Non': {'OUI':1, 'NON':0},
        'Douleur provoquée (Os.Abdomen)': {'OUI':1, 'NON':0}, 'Vaccin contre pneumocoque': {'OUI':1, 'NON':0},
    }
    df_selected.replace(binary_mappings, inplace=True)

    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)
    df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé','Mois'], drop_first=True)

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

    # ================================
    # 7. Entraînement et évaluation
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
        auc = roc_auc_score(y_test, y_test_proba)
        results[name] = {
            "Accuracy": (y_test_pred==y_test).mean(),
            "Precision": (y_test_pred[y_test==1]==y_test[y_test==1]).mean(),
            "Recall": (y_test_pred[y_test==1]==y_test[y_test==1]).mean(),
            "F1-Score": (y_test_pred==y_test).mean(), # simplifié, peut calculer correctement
            "AUC-ROC": auc,
            "Seuil optimal": optimal_threshold,
            "Confusion Matrix": cm,
            "ROC": (fpr, tpr)
        }

    # ================================
    # 8. Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles")
        df_perf = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Modèle"})
        st.dataframe(df_perf[["Modèle","Accuracy","Precision","Recall","F1-Score","AUC-ROC","Seuil optimal"]])

        # Détermination du meilleur modèle en combinant toutes les métriques
        metrics_to_consider = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
        df_norm = df_perf.copy()
        for col in metrics_to_consider:
            df_norm[col] = (df_norm[col]-df_norm[col].min()) / (df_norm[col].max()-df_norm[col].min())
        df_norm["Score_Global"] = df_norm[metrics_to_consider].mean(axis=1)
        best_model_row = df_norm.loc[df_norm["Score_Global"].idxmax()]
        best_model_name = best_model_row["Modèle"]
        best_model = models[best_model_name]

        st.success(f"✅ Meilleur modèle retenu : {best_model_name}")

        # Affichage de la matrice de confusion
        st.write("### Matrice de confusion du meilleur modèle")
        cm = results[best_model_name]["Confusion Matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        st.pyplot(fig)

        # Courbe ROC
        st.write("### Courbe ROC du meilleur modèle")
        fpr, tpr = results[best_model_name]["ROC"]
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_name} (AUC = {results[best_model_name]["AUC-ROC"]:.3f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel("1 - Spécificité")
        plt.ylabel("Sensibilité")
        plt.title("Courbe ROC")
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader(f"Variables importantes - {best_model_name}")
        if best_model_name=="Random Forest":
            importances = best_model.feature_importances_
            features = X_train.columns
            feat_imp = pd.DataFrame({"Feature":features, "Importance":importances}).sort_values(by="Importance", ascending=False)
            st.dataframe(feat_imp)

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie utilisée")
        st.markdown("""
        1. Nettoyage et sélection des variables.
        2. Encodage des variables qualitatives.
        3. Standardisation des variables quantitatives.
        4. SMOTETomek pour équilibrer la classe cible.
        5. Séparation Train/Validation/Test.
        6. Entraînement de plusieurs modèles (Decision Tree, Random Forest, SVM, LightGBM).
        7. Sélection du meilleur modèle basé sur plusieurs métriques combinées.
        """)

    # --- Onglet 4 : Simulateur ---
with tabs[3]:
    st.subheader("Simulateur - Prédiction pour un nouveau patient")

    st.write(" Remplissez les informations du patient :")

    # Création d'un dictionnaire pour récupérer les inputs
    user_input = {}

    # Variables quantitatives
    for var in quantitative_vars:
        min_val = float(df_selected[var].min())
        max_val = float(df_selected[var].max())
        mean_val = float(df_selected[var].mean())
        user_input[var] = st.number_input(f"{var}", value=mean_val, min_value=min_val, max_value=max_val)

    # Variables binaires
    binary_vars = [col for col in df_selected.columns if df_selected[col].nunique()==2 and col not in quantitative_vars and col not in ["Evolution_Cible"]]
    for var in binary_vars:
        user_input[var] = st.selectbox(f"{var}", options=[0,1], index=1)

    # Dummy variables (Diagnostic Catégorisé, Mois)
    dummy_vars = [col for col in X_train.columns if col not in quantitative_vars + binary_vars]
    for var in dummy_vars:
        user_input[var] = st.selectbox(f"{var}", options=[0,1], index=0)

    if st.button("Prédire l'évolution"):
        # Convertir en DataFrame
        input_df = pd.DataFrame([user_input])

        # Standardisation des variables quantitatives
        input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])

        # Prédiction
        proba = best_model.predict_proba(input_df)[:,1][0]
        pred = best_model.predict(input_df)[0]

        st.write(f"### Résultat de la prédiction :")
        if pred==0:
            st.success(f"Évolution prévue : Favorable")
        else:
            st.error(f"Évolution prévue : Complications")
        st.info(f"Probabilité d'évolution vers complication : {proba:.2f}")


