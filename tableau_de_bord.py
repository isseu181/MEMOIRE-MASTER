# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def show_dashboard():
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
    # Cartes en haut
    # ============================
    col1, col2, col3, col4 = st.columns(4)
    patients_total = df_cluster.shape[0] if df_cluster is not None else df_eda.shape[0]
    urgences_total = df_eda.shape[0]
    evolution_favorable = round(df_eda['Evolution'].value_counts(normalize=True).get('Favorable',0)*100,1)
    complications = round(df_eda['Evolution'].value_counts(normalize=True).get('Complications',0)*100,1)

    col1.markdown(f"<div style='background-color:#1f77b4;padding:15px;border-radius:10px;color:white;text-align:center;'>"
                  f"<h5>Patients Total / suivis 2023</h5><h3>{patients_total}</h3></div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='background-color:#ff7f0e;padding:15px;border-radius:10px;color:white;text-align:center;'>"
                  f"<h5>Urgences Total</h5><h3>{urgences_total}</h3></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='background-color:#2ca02c;padding:15px;border-radius:10px;color:white;text-align:center;'>"
                  f"<h5>Évolution Favorable</h5><h3>{evolution_favorable}%</h3></div>", unsafe_allow_html=True)
    col4.markdown(f"<div style='background-color:#d62728;padding:15px;border-radius:10px;color:white;text-align:center;'>"
                  f"<h5>Complications</h5><h3>{complications}%</h3></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # Graphiques Démographiques
    # ============================
    st.subheader("Données Démographiques")
    fig_width = 700
    fig_height = 400

    if 'Sexe' in df_eda.columns:
        sexe_counts = df_eda['Sexe'].value_counts()
        fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                     title="Répartition par sexe")
        fig.update_layout(width=fig_width, height=fig_height)
        st.plotly_chart(fig)

    if 'Origine Géographique' in df_eda.columns:
        origine_counts = df_eda['Origine Géographique'].value_counts()
        fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                     title="Répartition par origine géographique")
        fig.update_layout(width=fig_width, height=fig_height)
        st.plotly_chart(fig)

    # Âge
    age_col = "Âge du debut d etude en mois (en janvier 2023)"
    if age_col in df_eda.columns:
        df_eda[age_col] = pd.to_numeric(df_eda[age_col], errors='coerce')
        fig = px.histogram(df_eda, x=age_col, nbins=15, title="Répartition des âges à l’inclusion")
        fig.update_layout(width=fig_width, height=fig_height)
        st.plotly_chart(fig)

    st.markdown("---")

    # ============================
    # Données Cliniques & Biomarqueurs
    # ============================
    st.subheader("Données Cliniques & Biomarqueurs")

    cible = "Evolution"
    qualitative_vars = ["Sexe","Origine Géographique","Diagnostic Catégorisé"]
    for var in qualitative_vars:
        if var in df_eda.columns and cible in df_eda.columns:
            fig = px.bar(pd.crosstab(df_eda[var], df_eda[cible]),
                         barmode="group", text_auto=True,
                         title=f"{var} vs {cible}")
            fig.update_layout(width=fig_width, height=fig_height)
            st.plotly_chart(fig)

    # Biomarqueurs : seulement les moyennes
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    bio_data = {col: df_eda[col].mean() for col in bio_cols if col in df_eda.columns}
    if bio_data:
        bio_df = pd.DataFrame.from_dict(bio_data, orient='index', columns=['Moyenne']).round(2)
        st.subheader("Moyennes des Biomarqueurs")
        st.table(bio_df)

    st.markdown("---")

    # ============================
    # Analyse Temporelle
    # ============================
    st.subheader("Analyse Temporelle")
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig = px.bar(x=mois_counts.index, y=mois_counts.values, text=mois_counts.values,
                     title="Nombre de consultations par mois")
        fig.update_layout(width=fig_width, height=fig_height)
        st.plotly_chart(fig)

        # Diagnostics par mois
        if 'Diagnostic Catégorisé' in df_eda.columns:
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
        quantitative_vars = ["Âge du debut d etude en mois (en janvier 2023)",
                             "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                             "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"]
        df_cluster_scaled = df_cluster[quantitative_vars].copy()
        df_cluster_scaled = StandardScaler().fit_transform(df_cluster_scaled)

        inertia = [KMeans(n_clusters=k, random_state=42).fit(df_cluster_scaled).inertia_ for k in range(1,11)]
        fig, ax = plt.subplots()
        ax.plot(range(1,11), inertia, marker='o')
        ax.set_xlabel('Nombre de clusters')
        ax.set_ylabel('Inertia (SSE)')
        st.pyplot(fig)

        n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        # PCA 2D
        st.subheader("Visualisation PCA 2D")
        pca = PCA(n_components=2)
        components = pca.fit_transform(df_cluster_scaled)
        df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
        df_pca['Cluster'] = df_cluster['Cluster']
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
        st.pyplot(fig)

        # Profil détaillé des clusters
        st.subheader("Profil des clusters")
        cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename("Nombre de patients"))
        cluster_means = pd.DataFrame(df_cluster.groupby('Cluster')[quantitative_vars].mean()).round(2)
        st.dataframe(cluster_means)
