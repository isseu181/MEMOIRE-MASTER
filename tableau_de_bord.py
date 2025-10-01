# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def show_dashboard():
    st.set_page_config(page_title="Tableau de bord - Mémoire", layout="wide")

    # ============================
    # Chargement des données
    # ============================
    try:
        df_eda = pd.read_excel("fichier_nettoye.xlsx")
    except:
        st.warning("⚠️ fichier_nettoye.xlsx introuvable")
        df_eda = pd.DataFrame()

    try:
        df_cluster = pd.read_excel("segmentation.xlsx")
    except:
        st.warning("⚠️ segmentation.xlsx introuvable")
        df_cluster = pd.DataFrame()

    # ============================
    # Indicateurs colorés en haut
    # ============================
    patients_total = len(df_cluster) if not df_cluster.empty else len(df_eda)
    urgences_total = len(df_eda)
    if "Evolution" in df_eda.columns:
        evo_counts = df_eda['Evolution'].value_counts(normalize=True) * 100
        evo_favorable = round(evo_counts.get('Favorable', 0), 1)
        evo_complications = round(evo_counts.get('Complications', 0), 1)
    else:
        evo_favorable = 0
        evo_complications = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients Total / Suivis 2023", patients_total)
    col2.metric("Urgences Total", urgences_total)
    col3.metric("Évolution Favorable", f"{evo_favorable}%")
    col4.metric("Complications", f"{evo_complications}%")

    st.markdown("---")

    # ============================
    # Grille de graphiques Démographiques
    # ============================
    st.subheader("Données Démographiques")
    fig_width, fig_height = 550, 400

    demo_graphs = []
    if 'Sexe' in df_eda.columns:
        sexe_counts = df_eda['Sexe'].value_counts()
        fig_sexe = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values, title="Sexe")
        fig_sexe.update_layout(width=fig_width, height=fig_height)
        demo_graphs.append(fig_sexe)

    if 'Origine Géographique' in df_eda.columns:
        origine_counts = df_eda['Origine Géographique'].value_counts()
        fig_origine = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values, title="Origine géographique")
        fig_origine.update_layout(width=fig_width, height=fig_height)
        demo_graphs.append(fig_origine)

    if "Type de drépanocytose" in df_eda.columns:
        type_counts = df_eda['Type de drépanocytose'].value_counts()
        fig_type = px.pie(type_counts, names=type_counts.index, values=type_counts.values, title="Type de drépanocytose")
        fig_type.update_layout(width=fig_width, height=fig_height)
        demo_graphs.append(fig_type)

    # Affichage des graphiques démographiques en grille
    for i in range(0, len(demo_graphs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(demo_graphs):
                col.plotly_chart(demo_graphs[i+j])

    st.markdown("---")

    # ============================
    # Biomarqueurs (moyennes)
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
        bio_df = pd.DataFrame(bio_data.items(), columns=["Biomarqueur", "Moyenne"])
        st.table(bio_df.round(2))

    st.markdown("---")

    # ============================
    # Analyse Temporelle
    # ============================
    st.subheader("Analyse Temporelle")
    time_graphs = []

    # Consultations par mois
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout",
                      "Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig_consult = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                              title="Nombre de consultations par mois", text=mois_counts.values)
        fig_consult.update_layout(width=fig_width, height=fig_height)
        time_graphs.append(fig_consult)

    # Diagnostics par mois
    if 'Diagnostic Catégorisé' in df_eda.columns and 'Mois' in df_eda.columns:
        diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
        fig_diag = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                           title="Diagnostics par mois")
        fig_diag.update_layout(width=fig_width, height=fig_height)
        time_graphs.append(fig_diag)

    # Affichage des graphiques temporels en grille
    for i in range(0, len(time_graphs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(time_graphs):
                col.plotly_chart(time_graphs[i+j])

    st.markdown("---")

    # ============================
    # Clustering
    # ============================
    if df_cluster is not None and not df_cluster.empty:
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
