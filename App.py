# app.py
import streamlit as st
from PIL import Image
import eda, clustering, classification, deployment, tableau_de_bord

st.set_page_config(page_title="Analyse USAD Drépanocytose", layout="wide")

# ============================
# Barre latérale de navigation
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", [
    "A Propos",
    "Analyse exploratoire",
    "Prédiction",
    "Tableau de bord"
])

# ============================
# A Propos
# ============================
if page == "A Propos":
    # --- Titre réduit et centré ---
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Analyse et prédiction de l’évolution des urgences drépanocytaires chez les enfants</h2>",
        unsafe_allow_html=True
    )
    
    st.markdown("""
    Ce projet a pour objectif d’analyser les urgences drépanocytaires, 
    d’identifier leurs caractéristiques cliniques, biologiques et temporelles, 
    et de prédire leur évolution à l’aide de méthodes d’intelligence artificielle.
    
    **Points clés du projet :**
    - Analyse descriptive (socio-démographique, clinique, temporelle et biologique)
    - Classification supervisée pour prédire l’évolution 
    - Déploiement d’un outil interactif permettant aux médecins de visualiser et d’exploiter les résultats
    """)

    # --- Image centrée et large ---
    image = Image.open("drepano.png")
    col1, col2, col3 = st.columns([1, 3, 1])  # colonne centrale large
    with col2:
        st.image(
            image,
            caption="Urgences drépanocytaires : analyse et prédiction",
            width=800  # largeur augmentée
        )

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
