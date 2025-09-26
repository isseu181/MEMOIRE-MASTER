import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

# ==============================================================================
# CONFIGURATION ET NETTOYAGE
# ==============================================================================
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Analyse Drépanocytose - Segmentation & Prédiction",
    layout="wide"
)

# --- Fichiers d'entrée (Doivent être présents dans le dépôt) ---
CLUSTER_DATA_PATH = "segmentation.xlsx"
PREDICT_DATA_PATH = "fichier_nettoye.xlsx"

# --- Chemins des modèles sauvegardés (Doivent être créés AVANT le déploiement) ---
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

# --- Constantes pour la prédiction (doivent correspondre à vos résultats) ---
# **REMPLACER CES VALEURS par les résultats de votre script d'entraînement (step 11/13)**
BEST_MODEL_NAME = "Random Forest"
BEST_THRESHOLD = 0.56 # Remplacer par votre seuil optimal
SUMMARY_DF_RAW = {
    "Modèle": ["Random Forest", "LightGBM", "Decision Tree", "SVM"],
    "AUC-ROC": [0.965, 0.958, 0.880, 0.945], # Exemple : REMPLACER PAR VOS VRAIES VALEURS
    "Seuil optimal": [0.450, 0.520, 0.500, 0.510] # Exemple : REMPLACER PAR VOS VRAIES VALEURS
}
summary_df = pd.DataFrame(SUMMARY_DF_RAW).sort_values(by="AUC-ROC", ascending=False).reset_index(drop=True)

# --- Variables quantitatives pour Standardisation (doivent être les mêmes dans les deux parties) ---
QUANTITATIVE_VARS_PREDICT = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]

# ==============================================================================
# FONCTIONS CACHÉES (CLUSTERING)
# ==============================================================================

@st.cache_data(show_spinner="Nettoyage des données et calcul du clustering...")
def preparer_et_calculer_coude(chemin_fichier):
    # --- 1. Chargement et Sélection ---
    try:
        df = pd.read_excel(chemin_fichier)
    except FileNotFoundError:
        return None, None, None, "Fichier de clustering non trouvé."

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Liste de variables (du script de clustering)
    variables_selected = [
        "Âge du debut d etude en mois (en janvier 2023)", "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)", "Âge de début des signes (en mois)",
        "Âge de découverte de la drépanocytose (en mois)", "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023", "Nbre de transfusion avant 2017",
        "Nbre de transfusion Entre 2017 et 2023", "HDJ",
        "Sexe", "Origine Géographique", "Parents Salariés", "PEV Complet", 
        "Vaccin contre pneumocoque", "Vaccin contre méningocoque", "Vaccin contre Les salmonelles",
        "L'hydroxyurée", "Echange transfusionnelle", "Prise en charge", "Prophylaxie à la pénicilline",
        "CVO", "Anémie", "AVC", "STA", "Priapisme", "Infections", "Ictère", "Type de drépanocytose",
    ]
    
    df_selected = df[variables_selected].fillna(0).copy() 
    
    # --- 2. Encodage (Identique à votre script de clustering) ---
    binary_mappings = {
        "Sexe": {"Masculin": 1, "Féminin": 0, "M": 1, "F": 0}, "Parents Salariés": {"OUI": 1, "NON": 0},
        "PEV Complet": {"OUI": 1, "NON": 0}, "Vaccin contre pneumocoque": {"OUI": 1, "NON": 0},
        "Vaccin contre méningocoque": {"OUI": 1, "NON": 0}, "Vaccin contre Les salmonelles": {"OUI": 1, "NON": 0},
        "L'hydroxyurée": {"OUI": 1, "NON": 0}, "Echange transfusionnelle": {"OUI": 1, "NON": 0},
        "Prophylaxie à la pénicilline": {"OUI": 1, "NON": 0}, "CVO": {"OUI": 1, "NON": 0},
        "Anémie": {"OUI": 1, "NON": 0}, "AVC": {"OUI": 1, "NON": 0}, "STA": {"OUI": 1, "NON": 0},
        "Priapisme": {"OUI": 1, "NON": 0}, "Infections": {"OUI": 1, "NON": 0}, "Ictère": {"OUI": 1, "NON": 0},
    }
    df_selected.replace(binary_mappings, inplace=True)
    df_selected = pd.get_dummies(df_selected, columns=["Origine Géographique"], drop_first=False)
    df_selected = pd.get_dummies(df_selected, columns=["Prise en charge"], drop_first=True)
    df_selected = pd.get_dummies(df_selected, columns=["Type de drépanocytose"], drop_first=False)

    # --- 3. Standardisation ---
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)", "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)", "Âge de début des signes (en mois)",
        "Âge de découverte de la drépanocytose (en mois)", "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023", "Nbre de transfusion avant 2017",
        "Nbre de transfusion Entre 2017 et 2023", "HDJ"
    ]
    scaler = StandardScaler()
    df_selected[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars].fillna(0))
    df_final = df_selected.dropna(axis=1) # Supprime les colonnes totalement NaN après OHE/clean
    
    if df_final.empty or df_final.shape[1] == 0:
        return None, None, None, "Le DataFrame final est vide après le prétraitement."

    # --- 4. Graphe du coude ---
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_test.fit(df_final)
        inertia.append(kmeans_test.inertia_)

    fig_coude, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertia, marker='o')
    ax.set_xlabel('Nombre de clusters (k)')
    ax.set_ylabel('Inertia (SSE)')
    ax.set_title('Graphe du coude pour KMeans')
    ax.set_xticks(K_range)
    ax.grid(True)
    
    return df_final, fig_coude, df, None

# ==============================================================================
# FONCTIONS CACHÉES (PRÉDICTION)
# ==============================================================================

@st.cache_resource
def load_predictive_resources():
    """Charge le meilleur modèle, le scaler et les features pour la prédiction."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, features
    except FileNotFoundError as e:
        return None, None, None
    except Exception as e:
        return None, None, None

model_loaded, scaler_loaded, features_loaded = load_predictive_resources()

def prepare_and_predict(input_data):
    """Prépare les données et retourne la probabilité/classe."""
    new_data = pd.DataFrame([input_data])
    
    # 1. Ajouter les colonnes manquantes (OHE) et s'assurer du bon ordre
    for col in features_loaded:
        if col not in new_data.columns:
            new_data[col] = 0
            
    new_data = new_data[features_loaded]
    
    # 2. Standardisation
    new_data_scaled = new_data.copy()
    cols_to_scale = [col for col in QUANTITATIVE_VARS_PREDICT if col in new_data_scaled.columns]
    
    if not cols_to_scale or scaler_loaded is None:
        return None, None

    new_data_scaled[cols_to_scale] = scaler_loaded.transform(new_data[cols_to_scale])

    # 3. Prédiction
    pred_proba = model_loaded.predict_proba(new_data_scaled)[:, 1][0]
    pred_class = (pred_proba >= BEST_THRESHOLD).astype(int)
    
    return pred_proba, pred_class

# ==============================================================================
# INTERFACE UTILISATEUR STREAMLIT
# ==============================================================================

st.title("Projet Data Science : Drépanocytose")
st.markdown("Analyse de Segmentation et Modèle Prédictif du Risque de Complications")

tab_cluster, tab_predict_eval, tab_predict_interface = st.tabs([
    "📊 Segmentation K-Means", 
    "📈 Évaluation du Modèle", 
    "🧪 Interface de Prédiction"
])

# --------------------------------------------------------------------------
# ONGLET 1 : CLUSTERING
# --------------------------------------------------------------------------
with tab_cluster:
    st.header("Analyse de Segmentation (Clustering K-Means)")
    df_final, fig_coude, df_original, error_msg = preparer_et_calculer_coude(CLUSTER_DATA_PATH)
    
    if error_msg:
        st.error(error_msg)
    elif df_final is None:
        st.warning("Vérifiez la structure de votre fichier 'segmentation.xlsx'.")
    else:
        st.subheader("Étape 1 : Détermination du Nombre Optimal de Clusters (K)")
        col_coude, col_k_choice = st.columns([2, 1])

        with col_coude:
            st.pyplot(fig_coude)
            st.caption("Le 'coude' suggère le nombre optimal de clusters.")

        with col_k_choice:
            K_optimal = st.slider("Choisissez le nombre de clusters (K) :", min_value=2, max_value=10, value=3, step=1)
            st.info(f"Modèle entraîné avec **K = {K_optimal}** clusters.")

        # Exécution de K-Means final
        kmeans = KMeans(n_clusters=K_optimal, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_final)

        df_clusters = df_original.copy()
        df_clusters['Cluster'] = clusters

        st.subheader(f"Étape 2 : Profils des {K_optimal} Clusters")
        
        # Calcul des métriques clés par cluster pour l'interprétation
        df_summary = df_clusters.groupby('Cluster').agg({
            'Âge du debut d etude en mois (en janvier 2023)': 'mean',
            "Taux d'Hb (g/dL)": 'mean',
            "Nbre d'hospitalisations entre 2017 et 2023": 'mean',
            "L'hydroxyurée": 'mean', # Proportion de Oui
            "Sexe": 'mean' # Proportion de Masculin
        })
        df_summary['Taille'] = df_clusters['Cluster'].value_counts().sort_index()

        st.dataframe(df_summary.rename(columns={"L'hydroxyurée": "Prop. Hydroxyurée (1=Oui)", "Sexe": "Prop. Masculin (1=M)"}), use_container_width=True)

        st.subheader("Étape 3 : Visualisation des Clusters (PCA)")
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_final)
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters

        fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
        scatter = ax_pca.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', s=50, alpha=0.7)
        ax_pca.set_xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax_pca.set_ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax_pca.set_title(f'Visualisation des {K_optimal} Clusters (K-Means + PCA)')
        ax_pca.legend(*scatter.legend_elements(), title="Clusters")
        st.pyplot(fig_pca)

# --------------------------------------------------------------------------
# ONGLET 2 : ÉVALUATION DU MODÈLE
# --------------------------------------------------------------------------
with tab_predict_eval:
    st.header("Évaluation des Modèles de Prédiction (Évolution = Complications)")
    
    if model_loaded is None:
        st.error("Les fichiers du modèle (Random Forest) n'ont pas été trouvés. Veuillez exécuter le script d'entraînement une fois pour générer les fichiers `.pkl`.")
    else:
        st.subheader("Tableau de Synthèse des Performances")
        st.dataframe(
            summary_df.set_index('Modèle').style.background_gradient(subset=["AUC-ROC", "Accuracy"], cmap='Blues'),
            use_container_width=True
        )
        
        st.subheader(f"Comparaison de l'AUC-ROC (Meilleur Modèle : {BEST_MODEL_NAME})")
        fig_auc, ax_auc = plt.subplots(figsize=(10,6))
        sns.barplot(x="Modèle", y="AUC-ROC", data=summary_df, ax=ax_auc)
        ax_auc.set_title("Comparaison des modèles selon l'AUC-ROC")
        ax_auc.set_ylim(0,1)
        st.pyplot(fig_auc)

# --------------------------------------------------------------------------
# ONGLET 3 : PRÉDICTION INTERACTIVE
# --------------------------------------------------------------------------
with tab_predict_interface:
    st.header("Simulateur de Risque de Complications")
    
    if model_loaded is None:
        st.warning("Impossible d'effectuer des prédictions sans les fichiers du modèle.")
    else:
        st.markdown(f"**Modèle utilisé : {BEST_MODEL_NAME}** (Seuil optimal : **{BEST_THRESHOLD:.3f}**)")
        st.info("Entrez les données du patient pour obtenir une estimation du risque d'évolution vers des complications.")

        # --- Saisie des données ---
        input_data = {}
        
        st.subheader("Paramètres Clés du Patient")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data['Âge du debut d etude en mois (en janvier 2023)'] = st.number_input("Âge (mois)", min_value=1, value=24)
            input_data["Taux d'Hb (g/dL)"] = st.number_input("Taux d'Hb (g/dL)", min_value=1.0, max_value=20.0, value=9.5, format="%.2f")
            input_data["% d'Hb S"] = st.number_input("% d'Hb S", min_value=0.0, max_value=100.0, value=85.0, format="%.1f")
            
        with col2:
            input_data["Nbre d'hospitalisations entre 2017 et 2023"] = st.number_input("Nbre d'Hospitalisations (2017-2023)", min_value=0, value=0)
            input_data['CRP Si positive (Valeur)'] = st.number_input("CRP (mg/L)", min_value=0.0, max_value=500.0, value=5.0, format="%.1f")
            input_data['Pâleur'] = st.selectbox("Pâleur (Oui=1/Non=0)", options=[1, 0], format_func=lambda x: 'Oui' if x == 1 else 'Non')
            
        with col3:
            # Saisie de la catégorie de diagnostic (pour l'OHE)
            diag_cat = st.selectbox("Diagnostic Catégorisé", options=['Autres', 'CVO', 'Infections', 'AVC', 'STA', 'Anémie'])
            # Création des colonnes OHE nécessaires
            for cat in ['CVO', 'Infections', 'AVC', 'STA', 'Anémie']:
                input_data[f'Diagnostic Catégorisé_{cat}'] = 1 if diag_cat == cat else 0

            # Saisie d'un mois (pour l'OHE)
            mois_input = st.selectbox("Mois de l'Urgence (1=Janvier...)", options=list(range(1, 13)))
            for m in range(2, 13):
                input_data[f'Mois_{m}'] = 1 if mois_input == m else 0
            
            input_data['Prise en charge Hospitalisation'] = st.selectbox("Hospitalisation requise", options=[1, 0], format_func=lambda x: 'Oui' if x == 1 else 'Non')

        st.markdown("---")
        
        if st.button("Évaluer le Risque", type="primary"):
            
            # --- Ajout des variables non demandées (avec valeur par défaut 0) ---
            final_input_data = {}
            for feature in features_loaded:
                # Priorité aux valeurs saisies par l'utilisateur
                if feature in input_data and not isinstance(input_data[feature], str):
                    final_input_data[feature] = input_data[feature]
                elif feature.startswith('Diagnostic Catégorisé_') and feature in input_data:
                     final_input_data[feature] = input_data[feature]
                elif feature.startswith('Mois_') and feature in input_data:
                     final_input_data[feature] = input_data[feature]
                else:
                    # Défaut pour les variables binaires/quantitatives non saisies
                    final_input_data[feature] = 0

            prob, classe = prepare_and_predict(final_input_data)
            
            if prob is not None:
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric(label="Probabilité de Complications", value=f"{prob*100:.1f}%")
                    
                with col_res2:
                    if classe == 1:
                        st.error("Résultat : **RISQUE ÉLEVÉ DE COMPLICATIONS**", icon="⚠️")
                    else:
                        st.success("Résultat : **ÉVOLUTION FAVORABLE PRÉDITE**", icon="✅")
                        
                with col_res3:
                    st.metric(label="Seuil de Décision", value=f"{BEST_THRESHOLD:.3f}")

