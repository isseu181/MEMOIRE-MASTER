# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def show_dashboard():
    st.set_page_config(page_title="Tableau de bord - Mémoire", layout="wide")
    st.title("📊 Tableau de bord USAD Drépanocytose")

    # ---------------------------
    # Chargement des données
    # ---------------------------
    try:
        df_eda = pd.read_excel("fichier_nettoye.xlsx")
    except:
        st.warning("⚠️ fichier_nettoye.xlsx introuvable")
        st.stop()

    try:
        df_cluster = pd.read_excel("segmentation.xlsx")
    except:
        st.warning("⚠️ segmentation.xlsx introuvable")

    st.markdown("---")

    # ---------------------------
    # Section 1 : Données démographiques
    # ---------------------------
    st.header("1️⃣ Données démographiques")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'Sexe' in df_eda.columns:
            sexe_counts = df_eda['Sexe'].value_counts()
            fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values, title="Répartition par sexe")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Origine Géographique' in df_eda.columns:
            origine_counts = df_eda['Origine Géographique'].value_counts()
            fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values, title="Origine géographique")
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if "Niveau d'instruction scolarité" in df_eda.columns:
            scolar_counts = df_eda["Niveau d'instruction scolarité"].value_counts()
            fig = px.pie(scolar_counts, names=scolar_counts.index, values=scolar_counts.values, title="Scolarité")
            st.plotly_chart(fig, use_container_width=True)

    # Âge
    age_col = "Âge du debut d etude en mois (en janvier 2023)"
    if age_col in df_eda.columns:
        df_eda[age_col] = pd.to_numeric(df_eda[age_col], errors='coerce')
        fig = px.histogram(df_eda, x=age_col, nbins=15, title="Répartition des âges à l’inclusion")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------
    # Section 2 : Clinique et biomarqueurs
    # ---------------------------
    st.header("2️⃣ Données cliniques & biomarqueurs")
    cible = "Evolution"

    # Qualitatives vs Evolution
    qualitative_vars = ["Sexe","Origine Géographique","Diagnostic Catégorisé"]
    for var in qualitative_vars:
        if var in df_eda.columns and cible in df_eda.columns:
            cross_tab = pd.crosstab(df_eda[var], df_eda[cible], normalize="index")*100
            fig = px.bar(cross_tab, barmode="group", text_auto=".2f", title=f"{var} vs {cible}")
            st.plotly_chart(fig, use_container_width=True)

    # Quantitatives vs Evolution
    quantitative_vars = ["Âge du debut d etude en mois (en janvier 2023)", "GR (/mm3)", "GB (/mm3)", "HB (g/dl)"]
    for var in quantitative_vars:
        if var in df_eda.columns and cible in df_eda.columns:
            stats_group = df_eda.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
            st.table(stats_group)

    # Biomarqueurs
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
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
        st.subheader("📈 Statistiques descriptives des biomarqueurs")
        st.table(bio_df)

    st.markdown("---")

    # ---------------------------
    # Section 3 : Analyse temporelle
    # ---------------------------
    st.header("3️⃣ Analyse temporelle")
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True, title="Nombre de consultations par mois")
        st.plotly_chart(fig)

        # Diagnostics par mois
        if 'Diagnostic Catégorisé' in df_eda.columns:
            diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
            fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True, title="Évolution des diagnostics par mois")
            st.plotly_chart(fig)

    st.markdown("---")

    # ---------------------------
    # Section 4 : Clustering KMeans
    # ---------------------------
    st.header("4️⃣ Clustering KMeans")
    quantitative_vars_cluster = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
    ]
    df_cluster_scaled = df_cluster[quantitative_vars_cluster].copy()
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster_scaled)

    # Graphe du coude
    inertia = [KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_ for k in range(1,11)]
    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertia, marker='o')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Inertia (SSE)')
    st.pyplot(fig)

    # Clustering et PCA
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

    # Profil détaillé des clusters
    cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars_cluster].mean()).round(2)
    st.subheader("Moyennes des variables par cluster")
    st.dataframe(cluster_means)
