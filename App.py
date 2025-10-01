# App.py
import streamlit as st
from eda import show_eda
from clustering import show_clustering
from classification import show_classification

st.set_page_config(page_title="Mémoire - Dashboard", layout="wide")

st.title("📊 Tableau de bord global - Mémoire")

# ============================
# Menu principal
# ============================
menu = ["Exploration des données (EDA)", "Clustering", "Classification supervisée"]
choix = st.sidebar.radio("Sélectionnez une section :", menu)

if choix == "Exploration des données (EDA)":
    show_eda()

elif choix == "Clustering":
    show_clustering()

elif choix == "Classification supervisée":
    show_classification()
