# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def show_dashboard():
    st.set_page_config(page_title="Tableau de bord USAD Drépanocytose", layout="wide")
    
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
    # KPI en haut
    # ============================
    st.title("Tableau de bord USAD Drépanocytose")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Patients Total
    patients_total = df_cluster.shape[0] if df_cluster is not None else 0
    col1.metric("Patients Total / Suivis 2023", f"{patients_total}")

    # Urgences Total
    urgences_cols = [c for c in df_eda.columns if "Urgence" in c]
    urgences_total = df_eda[urgences_cols].notna().sum().sum() if urgences_cols else 0
    col2.metric("Urgences Total", f"{urgences_total}")

    # Evolution Favorable
    if 'Evolution' in df_eda.columns:
        evolution_counts = df_eda['Evolution'].value_counts(normalize=True) * 100
        evol_fav = round(evolution_counts.get('Favorable', 0),1)
        evol_comp = round(evolution_counts.get('Complications', 0),1)
    else:
        evol_fav = 0
        evol_comp = 0
    col3.metric("Évolution Favorable", f"{evol_fav}%")
    col4.metric("Complications", f"{evol_comp}%")

    st.markdown("---")
    
    # ============================
    # Données démographiques
    # ============================
    st.header("Données Démographiques")
    demo_cols = ['Sexe', 'Origine Géographique', "Niveau d'instruction scolarité"]
    demo_graphs = []
    for col in demo_cols:
        if col in df_eda.columns:
            counts = df_eda[col].value_counts()
            fig = px.pie(counts, names=counts.index, values=counts.values,
                         title=col, hole=0.3)
            fig.update_layout(height=400, width=400)
            demo_graphs.append(fig)

    col1, col2, col3 = st.columns(3)
    for i, fig in enumerate(demo_graphs):
        if i==0: col1.plotly_chart(fig, use_container_width=True)
        elif i==1: col2.plotly_chart(fig, use_container_width=True)
        else: col3.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ============================
    # Données Cliniques & Biomarqueurs
    # ============================
    st.header("Données Cliniques & Biomarqueurs")

    # Sexe vs Evolution (graph seulement)
    if 'Sexe' in df_eda.columns and 'Evolution' in df_eda.columns:
        cross_tab = pd.crosstab(df_eda['Sexe'], df_eda['Evolution'])
        fig = px.bar(cross_tab, barmode="group", text_auto=True,
                     title="Sexe vs Evolution")
        fig.update_layout(height=400, width=700)
        st.plotly_chart(fig, use_container_width=True)

    # Biomarqueurs : table uniquement
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    bio_data = {}
    for col in bio_cols:
        if col in df_eda.columns:
            bio_data[col] = {
                "Moyenne": df_eda[col].mean(),
                "Médiane": df_eda[col].median(),
                "Min": df_eda[col].min(),
                "Max": df_eda[col].max()
            }
    if bio_data:
        bio_df = pd.DataFrame(bio_data).T.round(2)
        st.subheader("Biomarqueurs - Statistiques descriptives")
        st.table(bio_df)

    st.markdown("---")

    # ============================
    # Analyse Temporelle
    # ============================
    st.header("Analyse Temporelle")
    # Nombre de consultations par mois
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                      title="Nombre de consultations par mois")
        fig.update_layout(height=400, width=700)
        st.plotly_chart(fig, use_container_width=True)

        # Diagnostics par mois
        if 'Diagnostic Catégorisé' in df_eda.columns:
            diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
            fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                          title="Évolution des diagnostics par mois")
            fig.update_layout(height=400, width=700)
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Clustering
    # ============================
    if df_cluster is not None:
        st.header("Clustering KMeans")
        quantitative_vars = [
            "Âge du debut d etude en mois (en janvier 2023)",
            "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
            "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"
        ]
        df_cluster_scaled = df_cluster[quantitative_vars].copy()
        df_cluster_scaled = StandardScaler().fit_transform(df_cluster_scaled)

        # Graphe du coude
        inertia = [KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_ for k in range(1,11)]
        fig, ax = plt.subplots()
        ax.plot(range(1,11), inertia, marker='o')
        ax.set_xlabel('Nombre de clusters')
        ax.set_ylabel('Inertia (SSE)')
        st.pyplot(fig)

        # Clustering et PCA 2D
        n_clusters = 3
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
        st.subheader("Profil des clusters")
        cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename("Nombre de patients"))
        cluster_means = df_cluster.groupby('Cluster')[quantitative_vars].mean()
        st.dataframe(cluster_means.round(2))
