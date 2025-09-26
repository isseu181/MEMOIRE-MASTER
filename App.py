# app.py
import streamlit as st
import pandas as pd

# Import des modules depuis le dossier utils
from utils import eda, clustering, classification

# Configuration de la page
st.set_page_config(
    page_title="Analyse USAD Drépanocytose",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Barre latérale de navigation
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", [
    "Chapitre 1 : Cadre théorique",
    "Chapitre 2 : Analyse exploratoire",
    "Chapitre 3 : Classification non supervisée",
    "Chapitre 4 : Classification supervisée",
    "Déploiement du modèle"
])

# ============================
# Chapitre 1 : Cadre théorique
# ============================
if page == "Chapitre 1 : Cadre théorique":
    st.title("Cadre théorique et conceptuel")
    st.markdown("""
    - Présentation de l’USAD  
    - Généralités sur la drépanocytose  
    - Principes de l’intelligence artificielle appliquée à la santé
    """)

# ============================
# Chapitre 2 : Analyse exploratoire
# ============================
elif page == "Chapitre 2 : Analyse exploratoire":
    st.title("Analyse exploratoire des données")
    eda.show_eda()

# ============================
# Chapitre 3 : Classification non supervisée
# ============================
elif page == "Chapitre 3 : Classification non supervisée":
    st.title("Classification non supervisée")
    clustering.show_clustering()

# ============================
# Chapitre 4 : Classification supervisée
# ============================
elif page == "Chapitre 4 : Classification supervisée":
    st.title("Classification supervisée")
    classification.show_classification()

# ============================
# Déploiement du modèle
# ============================
elif page == "Déploiement du modèle":
    st.title("Déploiement du modèle")
    st.markdown("""
    La base de données de test est déjà intégrée.  
    Vous pouvez directement utiliser le modèle pour prédire sur de nouvelles données internes.
    """)
    try:
        df_test = pd.read_excel("data/Base_de_donnees_USAD_URGENCES1.xlsx", sheet_name=None)
        for sheet_name, df_sheet in df_test.items():
            st.subheader(f"Aperçu de la feuille : {sheet_name}")
            st.dataframe(df_sheet.head())
    except FileNotFoundError:
        st.error("❌ Le fichier de test est introuvable dans 'data/Base_de_donnees_USAD_URGENCES1.xlsx'.")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la base intégrée : {e}")
