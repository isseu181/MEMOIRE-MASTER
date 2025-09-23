# app.py
import streamlit as st
from utils import eda, clustering, classification

st.set_page_config(page_title="Analyse USAD Drépanocytose", layout="wide")

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
    st.markdown("La base de données est chargée automatiquement depuis le dossier `data/`.")
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
    import pandas as pd
    try:
        df_test = pd.read_excel("data/Base_de_donnees_USAD_URGENCES1.xlsx", sheet_name=None)
        st.write("Aperçu des données intégrées :", {k: v.head() for k, v in df_test.items()})
    except Exception as e:
        st.error(f"Impossible de charger la base intégrée : {e}")

