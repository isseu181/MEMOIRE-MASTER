# classification.py
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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def show_classification():
    st.title("Classification de l'évolution des patients")

    # ================================
    # 1. Chargement automatique des données
    # ================================
    st.info("Chargement automatique de la base de données : fichier_nettoye.xlsx")
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
    # 3. Encodage binaire et ordinal
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
    # 4. Standardisation des variables quantitatives
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
    X = df_selected.drop(['Evolution', 'Evolution_Cible'], axis=1)
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
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:,1]
        fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        y_test_proba = model.predict_proba(X_test)[:,1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_proba)
        report = classification_report(y_test, y_test_pred, output_dict=True)
        results[name] = {
            "Confusion Matrix": cm,
            "Classification Report": report,
            "AUC-ROC": auc,
            "Optimal Threshold": optimal_threshold,
            "Accuracy": report['accuracy'],
            "Precision": report['macro avg']['precision'],
            "Recall": report['macro avg']['recall'],
            "F1-Score": report['macro avg']['f1-score']
        }

    # ================================
    # 7. Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles")
        # Affichage tableau comparatif
        summary_df = pd.DataFrame([{**{"Modèle": k}, **{m: v[m] for m in ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]}} for k,v in results.items()])
        st.dataframe(summary_df)

        # Sélection du meilleur modèle (basé sur la somme des métriques)
        summary_df['Score_total'] = summary_df[['Accuracy','Precision','Recall','F1-Score','AUC-ROC']].sum(axis=1)
        best_model_name = summary_df.loc[summary_df['Score_total'].idxmax(), 'Modèle']
        best_model = models[best_model_name]
        st.success(f"Le meilleur modèle sélectionné est : {best_model_name}")

        # Affichage matrice de confusion et courbe ROC du meilleur modèle
        y_test_proba = best_model.predict_proba(X_test)[:,1]
        optimal_threshold = results[best_model_name]['Optimal Threshold']
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)

        st.write("### Matrice de confusion")
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        st.pyplot(plt)

        st.write("### Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title('Courbe ROC')
        plt.legend(loc='lower right')
        st.pyplot(plt)

        # Affichage des métriques du meilleur modèle
        st.write("### Métriques du meilleur modèle")
        metrics_df = pd.DataFrame([{
            "Accuracy": results[best_model_name]['Accuracy'],
            "Precision": results[best_model_name]['Precision'],
            "Recall": results[best_model_name]['Recall'],
            "F1-Score": results[best_model_name]['F1-Score'],
            "AUC-ROC": results[best_model_name]['AUC-ROC'],
            "Seuil optimal": results[best_model_name]['Optimal Threshold']
        }])
        st.dataframe(metrics_df)

    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Variables importantes (Random Forest)")
        importances = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
        if importances is not None:
            imp_df = pd.DataFrame({"Variable": X_train.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
            st.dataframe(imp_df)
            plt.figure(figsize=(8,6))
            sns.barplot(x="Importance", y="Variable", data=imp_df)
            plt.title("Importance des variables")
            st.pyplot(plt)
        else:
            st.info("Le modèle choisi n'a pas de méthode feature_importances_.")

    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie")
        st.markdown("""
        - Prétraitement et sélection des variables
        - Encodage des variables binaires et ordinales (0=Non, 1=Oui)
        - Standardisation des variables quantitatives
        - Équilibrage de classes avec SMOTETomek
        - Division train/validation/test
        - Entraînement de Decision Tree, Random Forest, SVM, LightGBM
        - Évaluation selon Accuracy, Precision, Recall, F1-Score, AUC-ROC
        - Sélection du meilleur modèle basé sur la somme de ces métriques
        """)

    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur")
        st.info("A compléter : interface pour tester de nouvelles données et prédire l'évolution")
