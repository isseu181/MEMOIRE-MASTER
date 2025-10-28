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
# A Propos
# ============================
if page == "A Propos":
    st.title("Analyse et prédiction de l’évolution des urgences drépanocytaires chez les enfants")
    
    # Ajout de l image
    st.image("drepano.png", caption="Urgences drépanocytaires chez les enfants", use_column_width=True)
    # 

    st.markdown("""
    Ce projet a pour objectif d’analyser les urgences drépanocytaires , 
    d’identifier leurs caractéristiques cliniques, biologiques et temporelles, et de prédire leur évolution 
    à l’aide de méthodes d’intelligence artificielle.
    
    **Points clés du projet :**
    - Analyse descriptive (socio-démographique, clinique, temporelle et biologique)
    - Classification non supervisée pour détecter des profils de patients
    - Classification supervisée pour prédire l’évolution 
    - Déploiement d’un outil interactif permettant aux médecins de visualiser et d’exploiter les résultats
    """)

# ============================
# Analyse exploratoire
# ============================
elif page == "Analyse exploratoire":
    eda.show_eda()

# ============================
# Classification non supervisée
# ============================
elif page == "Classification non supervisée":
    clustering.show_clustering()

# ============================
# Classification supervisée
# ============================
elif page == "Classification supervisée":
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
