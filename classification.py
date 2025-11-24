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

# Plotly pour les visualisations interactives dans Streamlit
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Modèles et utilitaires
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import warnings

# Supprimer les avertissements de LightGBM pour une interface plus propre
warnings.filterwarnings('ignore')

# Définition des paramètres de modèles (Fidèle à la demande)
MODEL_PARAMS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=10, random_state=42),
    # SVC doit être calibré s'il est utilisé pour predict_proba avec probability=False
    "SVM (SVC)": SVC(C=1.0, kernel='rbf', gamma='scale', probability=False, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
}

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Entraîne un modèle, trouve le seuil optimal sur l'ensemble de validation 
    et évalue la performance finale sur l'ensemble de test.
    """
    
    # Gérer les modèles sans predict_proba nativement
    if not hasattr(model, "predict_proba"):
        model_calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5, ensemble=False)
        model_calibrated.fit(X_train, y_train)
        model = model_calibrated
    else:
        model.fit(X_train, y_train)
    
    # 1. Détermination du seuil optimal sur l'ensemble de VALIDATION
    y_val_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    # 2. Évaluation finale sur l'ensemble de TEST
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)
    
    return {
        "CM": cm, 
        "Report": report, 
        "AUC": auc, 
        "Threshold": optimal_threshold, 
        "Model": model,
        "Y_Test_Proba": y_test_proba # Ajout pour la courbe ROC finale
    }

def show_classification():
    st.set_page_config(page_title="Classification Supervisée", layout="wide")
    st.markdown("<h1 style='text-align:center;color:darkblue;'>Classification Supervisée - Analyse des Modèles</h1>", unsafe_allow_html=True)

    # -------------------------------
    # 1️⃣ Chargement et Préparation
    # -------------------------------
    try:
        df = pd.read_excel("fichier_nettoye.xlsx")
    except FileNotFoundError:
        st.error("Erreur: Le fichier 'fichier_nettoye.xlsx' est introuvable. Veuillez vérifier le chemin.")
        return

    # Définition des variables (Doit être complet pour que X.columns soit correct)
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

    # Encodage (Binaire, Ordinal, One-Hot)
    binary_mappings = {col: {'OUI':1,'NON':0} for col in [
        'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque',
        'Splénomégalie','Prophylaxie à la pénicilline','Parents Salariés',
        'Prise en charge Hospitalisation','Radiographie du thorax Oui ou Non',
        'Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque']}
    df_selected.replace(binary_mappings, inplace=True)
    ordinal_mappings = {
        'NiveauUrgence': {'Urgence1':1,'Urgence2':2,'Urgence3':3,'Urgence4':4,'Urgence5':5,'Urgence6':6},
        "Niveau d'instruction scolarité": {'Maternelle ':1,'Elémentaire ':2,'Secondaire':3,'Enseignement Supérieur ':4,'NON':0}
    }
    df_selected.replace(ordinal_mappings, inplace=True)
    df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé','Mois'])

    # Séparation X et y
    df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0,'Complications':1})
    X = df_selected.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_selected['Evolution_Cible']
    
    # Variables quantitatives pour la standardisation
    quantitative_vars = [
        'Âge de début des signes (en mois)','GR (/mm3)','GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)','VGM (fl/u3)','HB (g/dl)',
        'Nbre de GB (/mm3)','PLT (/mm3)','Nbre de PLT (/mm3)','TCMH (g/dl)',
        "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017','Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)',"Taux d'Hb (g/dL)","% d'Hb S","% d'Hb F"
    ]

    # Division Train/Validation/Test (80% / 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

    # Standardisation (Fit sur Train UNIQUEMENT)
    scaler = StandardScaler()
    X_train[quantitative_vars] = scaler.fit_transform(X_train[quantitative_vars])
    X_val[quantitative_vars] = scaler.transform(X_val[quantitative_vars])
    X_test[quantitative_vars] = scaler.transform(X_test[quantitative_vars])

    # SMOTETomek (Appliqué UNIQUEMENT sur X_train/y_train)
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

    X_train = X_train_res
    y_train = y_train_res

    # -------------------------------
    # 3️⃣ Entraînement et Évaluation
    # -------------------------------
    
    results = {}
    st.info("Entraînement et évaluation des modèles en cours (cela peut prendre quelques secondes)...")
    
    # Utilisation des paramètres de modèles définis dans MODEL_PARAMS
    for name, model_instance in MODEL_PARAMS.items():
        results[name] = evaluate_model(model_instance, X_train, y_train, X_val, y_val, X_test, y_test)

    # -------------------------------
    # 4️⃣ Résumé des métriques
    # -------------------------------
    summary = []
    for name,res in results.items():
        r = res["Report"]
        summary.append({
            "Modèle":name,
            "Accuracy":round(r['accuracy'],3),
            "Precision (Macro)":round(r['macro avg']['precision'],3),
            "Recall (Macro)":round(r['macro avg']['recall'],3),
            "F1 (Macro)":round(r['macro avg']['f1-score'],3),
            "AUC":round(res['AUC'],3)
        })
    summary_df = pd.DataFrame(summary)
    summary_df['Mean Score'] = summary_df[['F1 (Macro)','AUC']].mean(axis=1) 
    
    # Sélection du meilleur modèle basé sur le "Mean Score" (F1 + AUC)
    best_name = summary_df.loc[summary_df['Mean Score'].idxmax(),'Modèle']
    best_result = results[best_name]
    best_model = best_result["Model"]
    
    summary_df.drop(columns=['Mean Score'], inplace=True)
    summary_df = summary_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)

    # -------------------------------
    # 5️⃣ Onglets Streamlit
    # -------------------------------
    tab1, tab2, tab3 = st.tabs(["Comparaison des modèles","Méthodologie","Analyse du meilleur modèle"])

    # --- Onglet 1 - Comparaison des modèles ---
    with tab1:
        st.subheader("Comparaison des modèles (Test Set)")

        # 1️⃣ DataFrame du résumé des métriques
        st.dataframe(summary_df.style.background_gradient(cmap='Blues'), use_container_width=True)

        colA, colB = st.columns(2)
        
        with colA:
            # 2️⃣ Barplot comparatif AUC + Precision
            fig_bar = px.bar(summary_df, x="Modèle", y=["AUC","Precision (Macro)"],
                            barmode='group', title="Comparaison des modèles : AUC vs Precision (Macro)",
                            color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with colB:
            # 3️⃣ Heatmap des métriques
            metrics_heatmap = summary_df.set_index('Modèle')
            heatmap_fig = ff.create_annotated_heatmap(
                z=np.round(metrics_heatmap.values,3),
                x=metrics_heatmap.columns.tolist(),
                y=metrics_heatmap.index.tolist(),
                colorscale='Viridis', showscale=True, reversescale=True,
                zmin=0, zmax=1
            )
            heatmap_fig.update_layout(title='Heatmap des métriques (Macro) des modèles')
            st.plotly_chart(heatmap_fig, use_container_width=True)

    # --- Onglet 2 - Méthodologie ---
    with tab2:
        st.markdown("### Étapes méthodologique")
        st.markdown("""
        Le pipeline de modélisation suit une approche rigoureuse pour garantir la validité des résultats :

        * **Préparation des données :** Encodage des variables (binaire, ordinal, One-Hot).
        * **Division Stratifiée :** Séparation en **Train (60%) / Validation (20%) / Test (20%)** pour garantir l'équité des sous-échantillons.
        * **Standardisation :** Le `StandardScaler` est ajusté (**fit**) uniquement sur l'ensemble d'entraînement pour éviter la fuite de données, puis appliqué aux ensembles de validation et de test.
        * **Gestion du Déséquilibre ($\text{SMOTETomek}$) :** Appliqué **uniquement** sur l'ensemble d'entraînement pour équilibrer la distribution de la classe minoritaire "Complications".
        * **Entraînement des Modèles :** Les modèles sont entraînés sur l'ensemble d'entraînement rééchantillonné.
        * **Optimisation du Seuil :** Le seuil de classification optimal est déterminé en maximisant l'indice de Youden sur l'ensemble de **validation**.
        * **Évaluation Finale :** Les métriques finales ($\text{AUC}$, $\text{F1-score}$, $\text{Rappel}$, etc.) sont calculées sur l'ensemble de **test** (données jamais vues) en utilisant le seuil optimal.
        """)
        
    # --- Onglet 3 - Analyse du meilleur modèle ---
    with tab3:
        st.markdown(f"### Analyse détaillée du meilleur modèle : <span style='color:darkgreen;'>{best_name}</span>", unsafe_allow_html=True)
        
        # Affichage des métriques du meilleur modèle
        best_row = summary_df[summary_df['Modèle']==best_name].iloc[0]
        st.markdown(f"""
        - **Accuracy** : **{best_row['Accuracy']}**
        - **Precision (Macro)** : **{best_row['Precision (Macro)']}**
        - **Recall (Macro)** : **{best_row['Recall (Macro)']}**
        - **F1-score (Macro)** : **{best_row['F1 (Macro)']}**
        - **AUC** : **{best_row['AUC']}**
        - **Seuil Optimal** (trouvé sur Val) : **{best_result['Threshold']:.3f}**
        """)
        
        colC, colD = st.columns(2)

        with colC:
            # Matrice de confusion du meilleur modèle
            st.markdown("#### Matrice de Confusion (Test Set)")
            cm_labels = ['Favorable (0)', 'Complications (1)']
            fig_cm = ff.create_annotated_heatmap(
                z=best_result["CM"], x=cm_labels, y=cm_labels, colorscale='Blues', showscale=False,
                zmin=0, zmax=np.max(best_result["CM"])
            )
            fig_cm.update_layout(xaxis_title="Prédit", yaxis_title="Réel")
            st.plotly_chart(fig_cm, use_container_width=True)

        with colD:
            # Courbe ROC (Utilisant X_test)
            st.markdown("#### Courbe ROC (Test Set)")
            fpr, tpr, _ = roc_curve(y_test, best_result["Y_Test_Proba"])
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve', line=dict(color='darkblue', width=3)))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Hasard (AUC=0.5)', line=dict(color='red', width=2, dash='dash')))
            roc_fig.update_layout(title=f'AUC = {best_result["AUC"]:.3f}', xaxis_title='FPR', yaxis_title='TPR')
            st.plotly_chart(roc_fig, use_container_width=True)


        # Variables importantes
        st.markdown("#### Importance des Caractéristiques")
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
            fig_imp = go.Figure(go.Bar(
                x=importances.values,
                y=importances.index,
                orientation='h',
                marker=dict(color=importances.values, colorscale='Viridis'),
            ))
            fig_imp.update_layout(xaxis_title='Importance', yaxis_title='Variables')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("L'importance des variables n'est pas directement disponible pour ce type de modèle (e.g., SVM).")


if __name__ == '__main__':
    show_classification()
