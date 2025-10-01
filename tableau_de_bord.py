# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Tableau de bord USAD", layout="wide")

# ============================
# Chargement des données
# ============================
try:
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
except:
    st.warning("⚠️ fichier_nettoye.xlsx introuvable")
    st.stop()

try:
    df_cluster = pd.read_excel("segmentation.xlsx")
except:
    st.warning("⚠️ segmentation.xlsx introuvable")
    df_cluster = None

# ============================
# Indicateurs clés
# ============================
st.markdown("## Tableau de bord USAD")
col1, col2, col3, col4 = st.columns(4)

with col1:
    patients_total = df_cluster.shape[0] if df_cluster is not None else df_eda.shape[0]
    st.metric("Patients Total / Suivis 2023", patients_total)

with col2:
    urgences_total = df_eda.shape[0]
    st.metric("Urgences Total", urgences_total)

with col3:
    evolution_favorable = round(df_eda['Evolution'].value_counts(normalize=True).get('Favorable',0)*100,1)
    st.metric("Évolution Favorable", f"{evolution_favorable}%")

with col4:
    complications = round(df_eda['Evolution'].value_counts(normalize=True).get('Complications',0)*100,1)
    st.metric("Complications", f"{complications}%")

st.markdown("---")

# ============================
# Graphiques Démographiques
# ============================
st.header("📊 Données Démographiques")
graph_height = 400  # Hauteur uniforme pour tous les graphes

# Sexe
if 'Sexe' in df_eda.columns:
    sexe_counts = df_eda['Sexe'].value_counts()
    fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                 title="Répartition par sexe", height=graph_height)
    st.plotly_chart(fig, use_container_width=True)

# Origine Géographique
if 'Origine Géographique' in df_eda.columns:
    origine_counts = df_eda['Origine Géographique'].value_counts()
    fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                 title="Répartition par origine géographique", height=graph_height)
    st.plotly_chart(fig, use_container_width=True)

# Âge
age_col = "Âge du debut d etude en mois (en janvier 2023)"
if age_col in df_eda.columns:
    df_eda[age_col] = pd.to_numeric(df_eda[age_col], errors='coerce')
    fig = px.histogram(df_eda, x=age_col, nbins=15, title="Répartition des âges à l’inclusion",
                       height=graph_height)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================
# Graphiques Cliniques & Biomarqueurs
# ============================
st.header("💉 Données Cliniques & Biomarqueurs")

# Sexe vs Evolution
if 'Sexe' in df_eda.columns and 'Evolution' in df_eda.columns:
    cross_tab = pd.crosstab(df_eda['Sexe'], df_eda['Evolution'], normalize='index')*100
    fig = px.bar(cross_tab, barmode="group", text_auto=".1f", title="Sexe vs Evolution",
                 height=graph_height)
    st.plotly_chart(fig, use_container_width=True)

# Biomarqueurs : tableau descriptif
bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
bio_data = {}
for col in bio_cols:
    if col in df_eda.columns:
        df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
        bio_data[col] = {
            "Moyenne": round(df_eda[col].mean(),2),
            "Médiane": round(df_eda[col].median(),2),
            "Min": round(df_eda[col].min(),2),
            "Max": round(df_eda[col].max(),2)
        }
if bio_data:
    st.subheader("Biomarqueurs - statistiques descriptives")
    st.table(pd.DataFrame(bio_data).T)

st.markdown("---")

# ============================
# Graphiques Temporels
# ============================
st.header("📅 Analyse Temporelle")

# Nombre de consultations par mois
if 'Mois' in df_eda.columns:
    mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
    mois_counts = df_eda['Mois'].value_counts().sort_index()
    fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True, title="Nombre de consultations par mois",
                  height=graph_height)
    st.plotly_chart(fig, use_container_width=True)

    # Diagnostics par mois
    if 'Diagnostic Catégorisé' in df_eda.columns:
        diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
        fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True, title="Diagnostics par mois",
                      height=graph_height)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================
# Clustering
# ============================
if df_cluster is not None:
    st.header("🧩 Clustering KMeans")
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
    ]
    df_cluster_scaled = df_cluster[quantitative_vars].copy()
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster_scaled)

    # Clustering et PCA 2D
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

    st.subheader("Visualisation PCA 2D")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_cluster_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df_cluster['Cluster']
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    ax.set_title("Clusters PCA 2D", fontsize=14)
    st.pyplot(fig)

    # Profil détaillé
    st.subheader("Profil des clusters")
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Nombre de patients"))
    cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars].mean())
    st.dataframe(cluster_means.round(2))
