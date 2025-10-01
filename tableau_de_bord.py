# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tableau de bord - Mémoire", layout="wide")

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
    st.stop()

# ============================
# KPI en haut
# ============================
patients_total = df_cluster.shape[0]  # Segmentation.xlsx
urgences_cols = [f'Urgence{i}' for i in range(1,7) if f'Urgence{i}' in df_eda.columns]
urgences_total = df_eda[urgences_cols].notna().sum().sum() if urgences_cols else "N/A"
evol_favorable = round((df_eda['Evolution']=="Favorable").sum()/df_eda.shape[0]*100,1) if 'Evolution' in df_eda.columns else "N/A"
evol_complications = round((df_eda['Evolution']=="Complications").sum()/df_eda.shape[0]*100,1) if 'Evolution' in df_eda.columns else "N/A"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Patients Total / suivis 2023", patients_total)
kpi2.metric("Urgences Total", urgences_total)
kpi3.metric("Évolution Favorable", f"{evol_favorable}%")
kpi4.metric("Complications", f"{evol_complications}%")

st.markdown("---")

# ============================
# Démographique
# ============================
st.header("Données Démographiques")
col1, col2, col3 = st.columns(3)

with col1:
    if 'Sexe' in df_eda.columns:
        sexe_counts = df_eda['Sexe'].value_counts()
        fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values, title="Sexe")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'Origine Géographique' in df_eda.columns:
        origine_counts = df_eda['Origine Géographique'].value_counts()
        fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values, title="Origine Géographique")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with col3:
    age_col = "Âge du debut d etude en mois (en janvier 2023)"
    if age_col in df_eda.columns:
        df_eda[age_col] = pd.to_numeric(df_eda[age_col], errors='coerce')
        fig = px.histogram(df_eda, x=age_col, nbins=15, title="Répartition des âges")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================
# Clinique & Biomarqueurs
# ============================
st.header("Données Cliniques & Biomarqueurs")

# Qualitatives vs Evolution
cible = "Evolution"
qualitative_vars = ["Sexe","Origine Géographique","Diagnostic Catégorisé"]
for var in qualitative_vars:
    if var in df_eda.columns and cible in df_eda.columns:
        st.subheader(f"{var} vs {cible}")
        cross_tab = pd.crosstab(df_eda[var], df_eda[cible], normalize="index")*100
        st.dataframe(cross_tab.round(2))
        fig = px.bar(cross_tab, barmode="group", text_auto=".2f", title=f"{var} vs {cible}")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# Biomarqueurs
st.subheader("Biomarqueurs")
bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
            "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
bio_data = {}
for col in bio_cols:
    if col in df_eda.columns:
        df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
        bio_data[col] = {
            "Moyenne": df_eda[col].mean(),
            "Médiane": df_eda[col].median(),
            "Min": df_eda[col].min(),
            "Max": df_eda[col].max()
        }
if bio_data:
    bio_df = pd.DataFrame(bio_data).T.round(2)
    st.table(bio_df)

st.markdown("---")

# ============================
# Temporel
# ============================
st.header("Analyse Temporelle")

# Consultations par urgence
nombre_consultations = {}
for u in urgences_cols:
    nombre_consultations[u] = df_eda[u].notna().sum()
if nombre_consultations:
    temp_df = pd.DataFrame.from_dict(nombre_consultations, orient='index', columns=['Nombre de consultations'])
    fig = px.bar(temp_df, y='Nombre de consultations', x=temp_df.index, text='Nombre de consultations', title="Consultations par urgence")
    fig.update_layout(height=400)
    st.plotly_chart(fig)

# Consultations par mois
if 'Mois' in df_eda.columns:
    mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
    mois_counts = df_eda['Mois'].value_counts().sort_index()
    fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True, title="Consultations par mois")
    fig.update_layout(height=400)
    st.plotly_chart(fig)

    # Diagnostics par mois
    if 'Diagnostic Catégorisé' in df_eda.columns:
        diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
        st.subheader("Diagnostics par mois")
        st.dataframe(diag_month)
        fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True, title="Évolution des diagnostics")
        fig.update_layout(height=400)
        st.plotly_chart(fig)

st.markdown("---")

# ============================
# Clustering
# ============================
st.header("Clustering KMeans")
quantitative_vars = [
    "Âge du debut d etude en mois (en janvier 2023)",
    "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
    "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"
]
df_cluster_scaled = StandardScaler().fit_transform(df_cluster[quantitative_vars])

# Graphe du coude
inertia = []
for k in range(1,11):
    inertia.append(KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_)
fig, ax = plt.subplots()
ax.plot(range(1,11), inertia, marker='o')
ax.set_xlabel('Nombre de clusters')
ax.set_ylabel('Inertia (SSE)')
st.pyplot(fig)

# PCA 2D
n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)
pca = PCA(n_components=2)
components = pca.fit_transform(df_cluster_scaled)
df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
df_pca['Cluster'] = df_cluster['Cluster']

fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
st.pyplot(fig)

# Profil détaillé
cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
st.dataframe(cluster_counts.rename("Nombre de patients"))
cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars].mean())
st.dataframe(cluster_means.round(2))
