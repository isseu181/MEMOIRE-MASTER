# App.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Importer ton module eda.py
import eda

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
    eda.show_eda()

# ============================
# Chapitre 3 : Classification non supervisée
# ============================
elif page == "Chapitre 3 : Classification non supervisée":
    st.title("Classification non supervisée (Clustering)")

    # Charger les données
    try:
        df = pd.read_excel("segmentation.xlsx").applymap(lambda x: x.strip() if isinstance(x,str) else x)
    except FileNotFoundError:
        st.error("Fichier 'segmentation.xlsx' introuvable.")
        st.stop()

    variables = ["Âge du debut d etude en mois (en janvier 2023)",
                 "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                 "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    df_selected = df[variables].copy()
    scaler = StandardScaler()
    df_selected[variables] = scaler.fit_transform(df_selected)

    # Méthode du coude
    inertia = [KMeans(n_clusters=k, random_state=42).fit(df_selected).inertia_ for k in range(1,11)]
    fig_coude = px.line(x=list(range(1,11)), y=inertia, markers=True, labels={"x":"k","y":"Inertia"}, title="Méthode du coude")
    st.plotly_chart(fig_coude, use_container_width=True)

    k_optimal = st.slider("Choisir le nombre de clusters", 2, 10, 3)
    df_selected["Cluster"] = KMeans(n_clusters=k_optimal, random_state=42).fit_predict(df_selected)

    st.subheader("Résumé clusters")
    st.dataframe(df_selected.groupby("Cluster")[variables].agg(["mean","median","max"]).round(2))

    # PCA interactive
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_selected[variables])
    df_selected["PCA1"] = components[:,0]
    df_selected["PCA2"] = components[:,1]
    fig_pca = px.scatter(df_selected, x="PCA1", y="PCA2", color="Cluster", hover_data=variables,
                         title="Visualisation PCA des clusters", color_continuous_scale=px.colors.qualitative.Bold)
    st.plotly_chart(fig_pca, use_container_width=True)

# ============================
# Chapitre 4 : Classification supervisée
# ============================
elif page == "Chapitre 4 : Classification supervisée":
    st.title("Classification supervisée (Analyse binaire)")

    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
    except FileNotFoundError:
        st.error("Fichier 'fichier_nettoye.xlsx' introuvable.")
        st.stop()

    cible = "Evolution"
    if cible in df_nettoye.columns:
        variables = ["Type de drépanocytose","Sexe","Âge du debut d etude en mois (en janvier 2023)",
                     "Origine Géographique","Prise en charge","Diagnostic Catégorisé"]
        for var in variables:
            if var in df_nettoye.columns:
                st.subheader(f"{var} vs {cible}")

                if df_nettoye[var].dtype=="object":
                    cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                    st.dataframe(cross_tab.round(2))
                    fig = px.bar(cross_tab, barmode="group", text_auto=".2f",
                                 title=f"{var} vs {cible}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
                    st.table(stats_group)

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
        df_test = pd.read_excel("Base_de_donnees_USAD_URGENCES1.xlsx", sheet_name=None)
        st.write("Aperçu des données intégrées :", {k: v.head() for k, v in df_test.items()})
    except Exception as e:
        st.error(f"Impossible de charger la base intégrée : {e}")
