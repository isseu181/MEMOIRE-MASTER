# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Tableau de bord - Mémoire", layout="wide")

def show_dashboard():
    # ============================
    # Chargement des données
    # ============================
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
    df_cluster = pd.read_excel("segmentation.xlsx")

    # ============================
    # Titre principal
    # ============================
    st.title("📊 Tableau de Bord - Analyse USAD Drépanocytose")

    # ============================
    # Indicateurs clés en haut
    # ============================
    total_patients = df_cluster.shape[0]
    urgences_total = df_eda.shape[0]
    evolution_favorable = round((df_eda['Evolution'] == 'Favorable').mean() * 100, 1)
    complications = round((df_eda['Evolution'] == 'Complications').mean() * 100, 1)

    kpi_cols = st.columns(4)
    kpi_vals = [total_patients, urgences_total, f"{evolution_favorable}%", f"{complications}%"]
    kpi_labels = ["Patients Total / Suivis 2023", "Consultations d'Urgence", "Évolution Favorable", "Complications"]
    kpi_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(4):
        kpi_cols[i].markdown(
            f"<div style='background-color:{kpi_colors[i]};padding:15px;border-radius:8px;text-align:center;color:white;'>"
            f"<h5>{kpi_labels[i]}</h5><h3>{kpi_vals[i]}</h3></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ============================
    # Graphiques univariés
    # ============================
    plots = []

    # Sexe
    if 'Sexe' in df_eda.columns:
        sexe_counts = df_eda['Sexe'].value_counts()
        fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values, title="Répartition par Sexe")
        plots.append(fig)

    # Origine Géographique
    if 'Origine Géographique' in df_eda.columns:
        origine_counts = df_eda['Origine Géographique'].value_counts()
        fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                     title="Répartition par Origine Géographique")
        plots.append(fig)

    # Diagnostic Catégorisé
    if 'Diagnostic Catégorisé' in df_eda.columns:
        diag_counts = df_eda['Diagnostic Catégorisé'].value_counts()
        fig = px.pie(diag_counts, names=diag_counts.index, values=diag_counts.values,
                     title="Répartition par Type de Drépanocytose")
        plots.append(fig)

    # Nombre de consultations par urgence
    urgences = [f'Urgence{i}' for i in range(1, 7)]
    nombre_consultations = {u: df_eda[u].notna().sum() for u in urgences if u in df_eda.columns}
    if nombre_consultations:
        temp_df = pd.DataFrame.from_dict(nombre_consultations, orient='index', columns=['Nombre de consultations'])
        fig = px.bar(temp_df, y='Nombre de consultations', x=temp_df.index,
                     text='Nombre de consultations', title="Nombre de Consultations par Urgence")
        plots.append(fig)

    # Nombre de consultations par mois
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                      "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                      title="Nombre de Consultations par Mois")
        plots.append(fig)

        # Diagnostics par mois
        if 'Diagnostic Catégorisé' in df_eda.columns:
            diag_month = df_eda.groupby(['Mois', 'Diagnostic Catégorisé']).size().unstack(fill_value=0)
            fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                          title="Diagnostics par Mois")
            plots.append(fig)

    # Affichage 2 par ligne
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(plots):
                col.plotly_chart(plots[i + j], use_container_width=True)

    st.markdown("---")

    # ============================
    # Moyennes des biomarqueurs
    # ============================
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    bio_data = {col: round(pd.to_numeric(df_eda[col], errors='coerce').mean(), 2)
                for col in bio_cols if col in df_eda.columns}

    if bio_data:
        st.subheader("🧪 Moyennes des Biomarqueurs")
        colors_bio = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        cols = st.columns(len(bio_data))
        for i, (col_name, value) in enumerate(bio_data.items()):
            cols[i].markdown(
                f"<div style='background-color:{colors_bio[i]};padding:15px;border-radius:8px;text-align:center;color:white;'>"
                f"<h5>{col_name}</h5><h3>{value}</h3></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ============================
    # Analyses bivariées
    # ============================
    st.subheader("🔄 Analyses Bivariées")
    bivariate_cols = st.columns(2)

    # Sexe vs Evolution
    if 'Sexe' in df_eda.columns and 'Evolution' in df_eda.columns:
        cross_tab = pd.crosstab(df_eda['Sexe'], df_eda['Evolution'], normalize='index') * 100
        fig = px.bar(cross_tab, barmode='group', text_auto=".1f", title="Sexe vs Evolution (%)")
        bivariate_cols[0].plotly_chart(fig, use_container_width=True)

    # Diagnostic Catégorisé vs Evolution
    if 'Diagnostic Catégorisé' in df_eda.columns and 'Evolution' in df_eda.columns:
        cross_tab = pd.crosstab(df_eda['Diagnostic Catégorisé'], df_eda['Evolution'], normalize='index') * 100
        fig = px.bar(cross_tab, barmode='group', text_auto=".1f", title="Diagnostic vs Evolution (%)")
        bivariate_cols[1].plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ============================
    # Clustering KMeans
    # ============================
    st.subheader("🤖 Clustering KMeans")
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
    ]
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster[quantitative_vars])

    # Graphe du coude
    inertia = [KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_ for k in range(1, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertia, marker='o')
    ax.set_xlabel('Nombre de Clusters')
    ax.set_ylabel('Inertie (SSE)')
    st.pyplot(fig)

    # Clustering final
    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

    st.subheader("Visualisation PCA 2D")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_cluster_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df_cluster['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

    st.subheader("Profil détaillé des Clusters")
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Nombre de Patients"))
    cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars].mean())
    st.dataframe(cluster_means.round(2))
