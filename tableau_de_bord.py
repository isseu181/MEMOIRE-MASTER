# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Tableau de bord - Mémoire", layout="wide")

# ============================
# Chargement des données
# ============================
try:
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
except:
    st.error("⚠️ fichier_nettoye.xlsx introuvable")
    st.stop()

try:
    df_cluster = pd.read_excel("segmentation.xlsx")
except:
    st.warning("⚠️ segmentation.xlsx introuvable")
    df_cluster = None

# ============================
# Indicateurs colorés en haut
# ============================
patients_total = len(df_cluster) if df_cluster is not None else len(df_eda)
urgences_total = df_eda.shape[0]
evol_favorable = df_eda['Evolution'].value_counts(normalize=True).get('Favorable', 0) * 100
complications = df_eda['Evolution'].value_counts(normalize=True).get('Complications', 0) * 100

cols = st.columns(4)
indicators = [
    ("Patients Total ", patients_total, "#1f77b4"),
    ("Urgences Total", urgences_total, "#ff7f0e"),
    ("Évolution Favorable", f"{evol_favorable:.1f}%", "#2ca02c"),
    ("Complications", f"{complications:.1f}%", "#d62728"),
]

for col, (title, value, color) in zip(cols, indicators):
    col.markdown(f"""
        <div style="
            background-color:{color};
            color:white;
            text-align:center;
            padding:15px;
            border-radius:10px;
            font-size:18px;
        ">
            <strong>{value}</strong><br>{title}
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================
# Graphiques alignés en grille
# ============================
grid_cols = 2  # 2 graphiques par ligne

# Sexe
if 'Sexe' in df_eda.columns:
    fig_sexe = px.pie(df_eda['Sexe'].value_counts(), names=df_eda['Sexe'].value_counts().index,
                       values=df_eda['Sexe'].value_counts().values,
                       title="Répartition par sexe")
    cols = st.columns(grid_cols)
    cols[0].plotly_chart(fig_sexe, use_container_width=True)

# Type de drépanocytose
if 'Type de drépanocytose' in df_eda.columns:
    fig_type = px.pie(df_eda['Type de drépanocytose'].value_counts(),
                      names=df_eda['Type de drépanocytose'].value_counts().index,
                      values=df_eda['Type de drépanocytose'].value_counts().values,
                      title="Type de drépanocytose")
    cols[1].plotly_chart(fig_type, use_container_width=True)

# Diagnostiques vs Evolution (seulement graphiques)
qual_vars = ["Sexe","Origine Géographique","Diagnostic Catégorisé"]
cible = "Evolution"
for var in qual_vars:
    if var in df_eda.columns and cible in df_eda.columns:
        fig = px.bar(pd.crosstab(df_eda[var], df_eda[cible]), barmode="group",
                     title=f"{var} vs {cible}")
        st.plotly_chart(fig, use_container_width=True)

# ============================
# Représentation temporelle
# ============================
if 'Mois' in df_eda.columns:
    mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                  "Juillet","Aout","Septembre","Octobre","Novembre","Décembre"]
    df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
    mois_counts = df_eda['Mois'].value_counts().sort_index()
    fig_mois = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                       title="Nombre de consultations par mois",
                       labels={"x":"Mois", "y":"Nombre de consultations"})
    st.plotly_chart(fig_mois, use_container_width=True)

    # Diagnostics par mois
    if 'Diagnostic Catégorisé' in df_eda.columns:
        diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
        fig_diag = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                           title="Évolution des diagnostics par mois")
        st.plotly_chart(fig_diag, use_container_width=True)

# ============================
# Moyennes des biomarqueurs style PowerBI
# ============================
st.subheader("Biomarqueurs (moyennes)")

bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
            "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]

bio_data = {}
for col in bio_cols:
    if col in df_eda.columns:
        df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
        bio_data[col] = df_eda[col].mean()

if bio_data:
    bio_items = list(bio_data.items())
    n_cols = 3  # colonnes par ligne
    for i in range(0, len(bio_items), n_cols):
        cols = st.columns(n_cols)
        for j, col_st in enumerate(cols):
            if i+j < len(bio_items):
                name, value = bio_items[i+j]
                col_st.markdown(f"""
                    <div style="
                        background-color:#1f77b4;
                        color:white;
                        text-align:center;
                        padding:15px;
                        border-radius:10px;
                        font-size:18px;
                    ">
                        <strong>{value:.2f}</strong><br>{name}
                    </div>
                    """, unsafe_allow_html=True)

# ============================
# Clustering
# ============================
if df_cluster is not None:
    st.subheader("Clustering KMeans")
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"
    ]
    df_cluster_scaled = df_cluster[quantitative_vars].copy()
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster_scaled)

    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

    # PCA 2D
    st.subheader("Visualisation PCA 2D")
    from sklearn.decomposition import PCA
    components = PCA(n_components=2).fit_transform(df_cluster_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df_cluster['Cluster']

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

    # Profil détaillé
    st.subheader("Profil détaillé des clusters")
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Nombre de patients"))
    cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars].mean())
    st.dataframe(cluster_means.round(2))

