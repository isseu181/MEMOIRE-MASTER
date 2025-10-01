# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import plotly

def show_dashboard():
    st.set_page_config(page_title="Tableau de bord USAD Drépanocytose", layout="wide")
    st.title("📊 Tableau de bord USAD Drépanocytose")

    # ============================
    # Chargement des données
    # ============================
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
    df_cluster = pd.read_excel("segmentation.xlsx")

    # ============================
    # Indicateurs clés colorés
    # ============================
    total_patients = df_cluster.shape[0]
    urgences_total = df_eda.shape[0]
    evolution_favorable = round(df_eda['Evolution'].value_counts(normalize=True).get('Favorable',0)*100,1)
    complications = round(df_eda['Evolution'].value_counts(normalize=True).get('Complications',0)*100,1)

    # Couleurs custom
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]

    cols = st.columns(4)
    cols[0].markdown(f"<div style='background-color:{colors[0]};padding:10px;border-radius:5px;text-align:center;color:white;'><h4>Patients Total / Suivis 2023</h4><h2>{total_patients}</h2></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div style='background-color:{colors[1]};padding:10px;border-radius:5px;text-align:center;color:white;'><h4>Urgences Total</h4><h2>{urgences_total}</h2></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div style='background-color:{colors[2]};padding:10px;border-radius:5px;text-align:center;color:white;'><h4>Évolution Favorable (%)</h4><h2>{evolution_favorable}</h2></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div style='background-color:{colors[3]};padding:10px;border-radius:5px;text-align:center;color:white;'><h4>Complications (%)</h4><h2>{complications}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # Graphiques démographiques et cliniques
    # ============================
    plots = []

    # Sexe
    if 'Sexe' in df_eda.columns:
        sexe_counts = df_eda['Sexe'].value_counts()
        fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                     title="Répartition par sexe")
        plots.append(fig)

    # Type de drépanocytose
    if 'Type de drépanocytose' in df_eda.columns:
        type_counts = df_eda['Type de drépanocytose'].value_counts()
        fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                     title="Répartition par type de drépanocytose")
        plots.append(fig)

    # Temporel - consultations par mois
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août",
                      "Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                      title="Nombre de consultations par mois")
        plots.append(fig)

    # Temporel - diagnostics par mois
    if 'Diagnostic Catégorisé' in df_eda.columns and 'Mois' in df_eda.columns:
        diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
        fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                      title="Évolution des diagnostics par mois")
        plots.append(fig)

    # Affichage 2 graphiques par ligne
    for i in range(0, len(plots), 2):
        cols_graph = st.columns(2)
        for j, col in enumerate(cols_graph):
            if i+j < len(plots):
                plot = plots[i+j]
                if isinstance(plot, plotly.graph_objs._figure.Figure):
                    col.plotly_chart(plot, use_container_width=True)
                elif isinstance(plot, Figure):
                    col.pyplot(plot)

    st.markdown("---")

    # ============================
    # Moyennes biomarqueurs
    # ============================
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    bio_data = {}
    for col in bio_cols:
        if col in df_eda.columns:
            df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
            bio_data[col] = round(df_eda[col].mean(),2)

    if bio_data:
        st.subheader("Moyennes des biomarqueurs")
        bio_df = pd.DataFrame.from_dict(bio_data, orient='index', columns=['Moyenne'])
        st.table(bio_df)

    st.markdown("---")

    # ============================
    # Clustering
    # ============================
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"
    ]
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster[quantitative_vars])
    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

    # Graphe du coude
    inertia = []
    for k in range(1,11):
        inertia.append(KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertia, marker='o')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Inertia (SSE)')
    ax.set_title("Graphe du coude pour KMeans")
    st.pyplot(fig)

    # PCA 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_cluster_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df_cluster['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    ax.set_title("Visualisation PCA 2D")
    st.pyplot(fig)

    # Profil détaillé
    st.subheader("Profil détaillé des clusters")
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Nombre de patients"))
    cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars].mean())
    st.dataframe(cluster_means.round(2))
