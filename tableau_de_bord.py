import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Tableau de bord", layout="wide")

# ---------------------------
# Chargement des données
# ---------------------------
try:
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
except:
    st.warning("fichier_nettoye.xlsx introuvable")
    st.stop()

try:
    df_cluster = pd.read_excel("segmentation.xlsx")
except:
    st.warning("segmentation.xlsx introuvable")

# ---------------------------
# KPI en haut
# ---------------------------
total_patients = df_eda.shape[0]
patients_2023 = df_eda['Mois'].notna().sum() if 'Mois' in df_eda.columns else "N/A"
urgences_total = sum(df_eda[[f'Urgence{i}' for i in range(1,7)]].notna().sum()) if all(f'Urgence{i}' in df_eda.columns for i in range(1,7)) else "N/A"
evol_favorable = round((df_eda['Evolution']=="Favorable").mean()*100,1) if 'Evolution' in df_eda.columns else "N/A"
evol_complications = round((df_eda['Evolution']=="Complications").mean()*100,1) if 'Evolution' in df_eda.columns else "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Patients Total", total_patients)
col2.metric("Patients suivis 2023", patients_2023)
col3.metric("Urgences Total", urgences_total)
col4.metric("Évolution Favorable", f"{evol_favorable}%")
col5.metric("Complications", f"{evol_complications}%")

st.markdown("---")

# ---------------------------
# Grille 3x3 pour graphiques
# ---------------------------
# Définir les colonnes
rows = []

# Ligne 1 : Démographique
cols = st.columns(3)
rows.append(cols)
if 'Sexe' in df_eda.columns:
    sexe_counts = df_eda['Sexe'].value_counts()
    fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                 title="Répartition par Sexe")
    cols[0].plotly_chart(fig, use_container_width=True)

if 'Origine Géographique' in df_eda.columns:
    origine_counts = df_eda['Origine Géographique'].value_counts()
    fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                 title="Répartition par Origine Géographique")
    cols[1].plotly_chart(fig, use_container_width=True)

# Ligne 2 : Biomarqueurs
cols = st.columns(3)
rows.append(cols)
bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
for i, col_name in enumerate(bio_cols[:3]):
    if col_name in df_eda.columns:
        fig = px.histogram(df_eda, x=col_name, nbins=15, title=col_name)
        cols[i].plotly_chart(fig, use_container_width=True)

cols = st.columns(3)
rows.append(cols)
for i, col_name in enumerate(bio_cols[3:6]):
    if col_name in df_eda.columns:
        fig = px.histogram(df_eda, x=col_name, nbins=15, title=col_name)
        cols[i].plotly_chart(fig, use_container_width=True)

# Ligne 3 : Temporel
cols = st.columns(3)
rows.append(cols)
if 'Mois' in df_eda.columns:
    mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
    mois_counts = df_eda['Mois'].value_counts().sort_index()
    fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True, title="Nombre de consultations par mois")
    cols[0].plotly_chart(fig, use_container_width=True)

    if 'Diagnostic Catégorisé' in df_eda.columns:
        diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
        fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True, title="Diagnostics par mois")
        cols[1].plotly_chart(fig, use_container_width=True)

# Ligne 4 : Clustering PCA
quantitative_vars = [
    "Âge du debut d etude en mois (en janvier 2023)",
    "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
    "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"
]
df_cluster_scaled = StandardScaler().fit_transform(df_cluster[quantitative_vars])
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)
pca = PCA(n_components=2)
components = pca.fit_transform(df_cluster_scaled)
df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
df_pca['Cluster'] = df_cluster['Cluster']

cols = st.columns(1)
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
ax.set_title("Clustering PCA 2D")
cols[0].pyplot(fig)
