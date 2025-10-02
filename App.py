# app.py
import streamlit as st
import eda, clustering, classification, deployment, tableau_de_bord

st.set_page_config(page_title="Analyse USAD Drépanocytose", layout="wide")

# ============================
# Barre latérale de navigation
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", [
    "A Propos",
    "Analyse exploratoire",
    "Classification non supervisée",
    "Classification supervisée",
    "Déploiement du modèle",
    "Tableau de bord"
])

# ============================
# Chapitre 1 : Cadre théorique
# ============================
if page == "A Propos":
    st.title("Analyse et prédiction de l’évolution des urgences drépanocytaires chez les enfants")
    st.markdown("""
    - Présentation de l’USAD  
    - Généralités sur la drépanocytose  
    - Principes de l’intelligence artificielle appliquée à la santé
    """)

# ============================
# Chapitre 2 : Analyse exploratoire
# ============================
elif page == "Chapitre 2 : Analyse exploratoire":
    eda.show_eda()

# ============================
# Chapitre 3 : Classification non supervisée
# ============================
elif page == "Chapitre 3 : Classification non supervisée":
    clustering.show_clustering()

# ============================
# Chapitre 4 : Classification supervisée
# ============================
elif page == "Chapitre 4 : Classification supervisée":
    classification.show_classification()

# ============================
# Déploiement du modèle
# ============================
elif page == "Déploiement du modèle":
    deployment.show_deployment()

# ============================
# Tableau de bord
# ============================
elif page == "Tableau de bord":
    tableau_de_bord.show_dashboard()
