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

def show_classification():
    st.title("Analyse et Classification des données")

    # ================================
    # Chargement des données et prétraitement
    # ================================
    df = pd.read_excel("fichier_nettoye.xlsx")

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

    # Encodage
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

    # Standardisation
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

    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']

    # SMOTE
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # ================================
    # Onglets
    # ================================
    onglets = st.tabs(["Méthodologie & Prétraitement", "Comparaison des modèles", "Meilleur modèle"])

    # ----------------
    # Onglet 1 : Méthodologie & Prétraitement
    # ----------------
    with onglets[0]:
        st.subheader("Aperçu des données et prétraitement")
        st.dataframe(df_selected.head())
        st.text(f"Taille du dataset après SMOTE : {X_res.shape}")
        st.text(f"Distribution de la cible :\n{pd.Series(y_res).value_counts()}")

    # ----------------
    # Onglet 2 : Comparaison des modèles
    # ----------------
    with onglets[1]:
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
        }

        results = {}
        summary_metrics = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:,1]
            fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            y_test_proba = model.predict_proba(X_test)[:,1]
            y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

            auc = roc_auc_score(y_test, y_test_proba)
            results[name] = {
                "model": model,
                "Optimal Threshold": optimal_threshold,
                "y_test": y_test,
                "y_test_pred": y_test_pred,
                "y_test_proba": y_test_proba,
            }

            report = classification_report(y_test, y_test_pred, output_dict=True)
            summary_metrics.append({
                "Modèle": name,
                "Accuracy": round(report['accuracy'],3),
                "Precision": round(report['macro avg']['precision'],3),
                "Recall": round(report['macro avg']['recall'],3),
                "F1-Score": round(report['macro avg']['f1-score'],3),
                "AUC-ROC": round(auc,3),
            })

        summary_df = pd.DataFrame(summary_metrics).sort_values(by="AUC-ROC", ascending=False)
        st.subheader("Comparaison des modèles selon leurs métriques")
        st.dataframe(summary_df)

        st.subheader("Graphique AUC-ROC")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x="Modèle", y="AUC-ROC", data=summary_df, ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)

    # ----------------
    # Onglet 3 : Meilleur modèle
    # ----------------
    with onglets[2]:
        best_model_name = summary_df.iloc[0]["Modèle"]
        best = results[best_model_name]
        st.subheader(f"Meilleur modèle : {best_model_name}")

        # Matrice de confusion
        cm = confusion_matrix(best["y_test"], best["y_test_pred"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")
        st.pyplot(fig)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(best["y_test"], best["y_test_proba"])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(best['y_test'], best['y_test_proba']):.3f}")
        ax.plot([0,1],[0,1],'--', color='gray')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Courbe ROC")
        ax.legend()
        st.pyplot(fig)
