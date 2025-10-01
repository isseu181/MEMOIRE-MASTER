# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Tableau de bord - Mémoire", layout="wide")

# ============================
# Chargement des données
# ============================
try:
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
    st.success("✅ Données principales chargées")
except:
    st.warning("⚠️ fichier_nettoye.xlsx introuvable")
    st.stop()

try:
    df_cluster = pd.read_excel("segmentation.xlsx")
    st.success("✅ Données clustering chargées")
except:
    st.warning("⚠️ segmentation.xlsx introuvable")
    df_cluster = None

# ============================
# Indicateurs clés
# ============================
st.markdown("## Tableau de bord")
col1, col2, col3, col4 = st.columns(4)

# Nombre total de patients
patients_total = len(df_cluster) if df_cluster is not None else len(df_eda)
col1.metric("Patients Total / Suivis 2023", patients_total)

# Nombre total d'urgences
urgences_total = len(df_eda)
col2.metric("Urgences Total", urgences_total)

# Évolution favorable
if "Evolution" in df_eda.columns:
    evo_counts = df_eda['Evolution'].value_counts(normalize=True) * 100
    evo_favorable = round(evo_counts.get('Favorable', 0), 1)
    col3.metric("Évolution Favorable", f"{evo_favorable}%")
    evo_complications = round(evo_counts.get('Complications', 0), 1)
    col4.metric("Complications", f"{evo_complications}%")

st.markdown("---")

# ============================
# Graphiques Démographiques
# ============================
st.subheader("Données Démographiques")
fig_width = 700
fig_height = 400

# Répartition des âges à l’inclusion
age_col = "Âge du debut d etude en mois (en janvier 2023)"
if age_col in df_eda.columns:
    df_eda[age_col] = pd.to_numeric(df_eda[age_col], errors='coerce')
    fig_age = px.histogram(df_eda, x=age_col, nbins=15,
                           title="Répartition des âges à l’inclusion")
    fig_age.update_layout(width=fig_width, height=fig_height)
    st.plotly_chart(fig_age)

# Type de drépanocytose
if "Type de drépanocytose" in df_eda.columns:
    type_counts = df_eda['Type de drépanocytose'].value_counts()
    fig_type = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                      title="Répartition des types de drépanocytose")
    fig_type.update_layout(width=fig_width, height=fig_height)
    st.plotly_chart(fig_type)

# Sexe
if 'Sexe' in df_eda.columns:
    sexe_counts = df_eda['Sexe'].value_counts()
    fig_sexe = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                      title="Répartition par sexe")
    fig_sexe.update_layout(width=fig_width, height=fig_height)
    st.plotly_chart(fig_sexe)

# Origine Géographique
if 'Origine Géographique' in df_eda.columns:
    origine_counts = df_eda['Origine Géographique'].value_counts()
    fig_origine = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                         title="Répartition par origine géographique")
    fig_origine.update_layout(width=fig_width, height=fig_height)
    st.plotly_chart(fig_origine)

st.markdown("---")

# ============================
# Graphiques Cliniques
# ============================
st.subheader("Données Cliniques & Biomarqueurs")
cible = "Evolution"

# Graphiques qualitatives vs Evolution
qualitative_vars = ["Sexe", "Origine Géographique", "Diagnostic Catégorisé"]
for var in qualitative_vars:
    if var in df_eda.columns and cible in df_eda.columns:
        cross_tab = pd.crosstab(df_eda[var], df_eda[cible], normalize="index")*100
        fig = px.bar(cross_tab, barmode="group", text_auto=".1f",
                     title=f"{var} vs {cible}")
        fig.update_layout(width=fig_width, height=fig_height)
        st.plotly_chart(fig)

# Biomarqueurs (tableau avec moyennes)
bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
            "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
bio_data = {}
for col in bio_cols:
    if col in df_eda.columns:
        df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
        bio_data[col] = df_eda[col].mean()
if bio_data:
    bio_df = pd.DataFrame(bio_data.items(), columns=["Biomarqueur", "Moyenne"])
    st.table(bio_df.round(2))

st.markdown("---")

# ============================
# Analyse Temporelle
# ============================
st.subheader("Analyse Temporelle")

# Nombre de consultations par mois
if 'Mois' in df_eda.columns:
    mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août",
                  "Septembre","Octobre","Novembre","Décembre"]
    df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
    mois_counts = df_eda['Mois'].value_counts().sort_index()
    fig = px.bar(x=mois_counts.index, y=mois_counts.values, text=mois_counts.values,
                 title="Nombre de consultations par mois")
    fig.update_layout(width=fig_width, height=fig_height)
    st.plotly_chart(fig)

# Diagnostics par mois
if 'Diagnostic Catégorisé' in df_eda.columns and 'Mois' in df_eda.columns:
    diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
    fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                  title="Évolution des diagnostics par mois")
    fig.update_layout(width=fig_width, height=fig_height)
    st.plotly_chart(fig)

st.markdown("---")

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
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster[quantitative_vars])

    # Graphe du coude
    inertia = [KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_ for k in range(1,11)]
    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertia, marker='o')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Inertia (SSE)')
    st.pyplot(fig)

    # KMeans + PCA 2D
    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

    st.subheader("Visualisation PCA 2D")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_cluster_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df_cluster['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

    # Profil détaillé
    st.subheader("Profil détaillé des clusters")
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Nombre de patients"))
    cluster_means = df_cluster.groupby('Cluster')[quantitative_vars].mean()
    st.dataframe(cluster_means.round(2))
