# classification.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def show_classification():
    st.title("Classification des évolutions de patients")
    
    # ================================
    # 1. Chargement automatique
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
    # 3. Encodage
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
    # 5. Variable cible
    # ================================
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']
    
    # ================================
    # 6. SMOTETomek
    # ================================
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    
    # ================================
    # 7. Train/Test/Validation
    # ================================
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # ================================
    # 8. Modèles
    # ================================
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
    }
    
    # ================================
    # Onglets Streamlit
    # ================================
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])
    
    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Comparaison des modèles")
        results = {}
        auc_scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:,1]
            fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]
            y_test_proba = model.predict_proba(X_test)[:,1]
            y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
            cm = confusion_matrix(y_test, y_test_pred)
            auc_score = roc_auc_score(y_test, y_test_proba)
            
            report = classification_report(y_test, y_test_pred, output_dict=True)
            results[name] = {"model": model, "CM": cm, "report": report, "AUC": auc_score, "Threshold": optimal_threshold}
            auc_scores[name] = auc_score
        
        # Meilleur modèle
        best_model_name = max(auc_scores, key=auc_scores.get)
        best = results[best_model_name]
        st.write(f"**Meilleur modèle : {best_model_name}** (AUC-ROC = {best['AUC']:.3f})")
        
        # Affichage métriques
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Précision", "Sensibilité (Recall)", "F1-score", "AUC-ROC", "Seuil optimal"],
            "Value": [
                best['report']['accuracy'],
                best['report']['macro avg']['precision'],
                best['report']['macro avg']['recall'],
                best['report']['macro avg']['f1-score'],
                best['AUC'],
                best['Threshold']
            ]
        })
        st.table(metrics_df)
        
        # Matrice de confusion
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(best['CM'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        st.pyplot(fig)
        
        # Courbe ROC
        st.write("### Courbe ROC")
        y_test_proba = best['model'].predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"{best_model_name} (AUC = {best['AUC']:.3f})")
        ax.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel("1 - Spécificité")
        ax.set_ylabel("Sensibilité")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    
    # --- Onglet 2 : Variables importantes ---
    with tabs[1]:
        st.subheader("Variables importantes")
        if best_model_name in ["Random Forest","LightGBM"]:
            importances = best['model'].feature_importances_
            imp_df = pd.DataFrame({"Variable": X_train.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
            st.dataframe(imp_df)
        else:
            st.info("Importance des variables non disponible pour ce modèle.")
    
    # --- Onglet 3 : Méthodologie ---
    with tabs[2]:
        st.subheader("Méthodologie utilisée")
        st.markdown("""
        1. Prétraitement des données : sélection des variables pertinentes et encodage.  
        2. Standardisation des variables quantitatives.  
        3. Rééquilibrage avec SMOTETomek.  
        4. Division Train/Validation/Test.  
        5. Entraînement des modèles : Decision Tree, Random Forest, SVM, LightGBM.  
        6. Évaluation avec matrice de confusion, classification report et AUC-ROC.  
        7. Sélection du meilleur modèle selon l'AUC-ROC.  
        """)
    
    # --- Onglet 4 : Simulateur ---
    with tabs[3]:
        st.subheader("Simulateur de prédiction individuelle")
        st.info("Entrer les valeurs pour chaque variable sélectionnée pour obtenir une prédiction.")
        input_data = {}
        for col in X_train.columns:
            if X_train[col].nunique() <= 5:
                val = st.selectbox(f"{col}", X_train[col].unique())
            else:
                val = st.number_input(f"{col}", float(X_train[col].min()), float(X_train[col].max()))
            input_data[col] = val
        
        if st.button("Prédire l'évolution"):
            input_df = pd.DataFrame([input_data])
            input_df[quantitative_vars] = scaler.transform(input_df[quantitative_vars])
            pred_proba = best['model'].predict_proba(input_df)[:,1][0]
            pred_class = int(pred_proba >= best['Threshold'])
            st.write(f"Probabilité de complications : {pred_proba:.3f}")
            st.write(f"Classe prédite : {'Complications' if pred_class==1 else 'Favorable'}")
