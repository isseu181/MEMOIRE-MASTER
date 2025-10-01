# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def show_dashboard():
    st.set_page_config(page_title="Tableau de bord Drépanocytose", layout="wide")
    st.title("📊 Tableau de bord Drépanocytose - USAD")

    # ============================
    # Chargement des données
    # ============================
    try:
        df_eda = pd.read_excel("fichier_nettoye.xlsx")
    except:
        st.error("⚠️ fichier_nettoye.xlsx introuvable")
        st.stop()

    try:
        df_cluster = pd.read_excel("segmentation.xlsx")
    except:
        st.warning("⚠️ segmentation.xlsx introuvable")
        df_cluster = None

    # ============================
    # Indicateurs colorés en haut
    # ============================
    patients_total = len(df_cluster) if df_cluster is not None else len(df_eda)
    urgences_total = df_eda.shape[0]
    evol_favorable = df_eda['Evolution'].value_counts(normalize=True).get('Favorable', 0) * 100
    complications = df_eda['Evolution'].value_counts(normalize=True).get('Complications', 0) * 100

    cols = st.columns(4)
    indicators = [
        ("Patients Total ", patients_total, "#1f77b4"),
        ("Urgences Total", urgences_total, "#ff7f0e"),
        ("Évolution Favorable", f"{evol_favorable:.1f}%", "#2ca02c"),
        ("Complications", f"{complications:.1f}%", "#d62728"),
    ]

    for col, (title, value, color) in zip(cols, indicators):
        col.markdown(f"""
            <div style="
                background-color:{color};
                color:white;
                text-align:center;
                padding:15px;
                border-radius:10px;
                font-size:18px;
            ">
                <strong>{value}</strong><br>{title}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # Graphiques cliniques et biomarqueurs
    # ============================
    st.header("Données Cliniques & Biomarqueurs")

    # Graphique Sexe vs Evolution
    if 'Sexe' in df_eda.columns and 'Evolution' in df_eda.columns:
        sexe_counts = pd.crosstab(df_eda['Sexe'], df_eda['Evolution'], normalize='index')*100
        fig = px.bar(sexe_counts, barmode='group', text_auto='.1f', title="Sexe vs Evolution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Type de drépanocytose
    if 'Type de drépanocytose' in df_eda.columns:
        type_counts = df_eda['Type de drépanocytose'].value_counts()
        fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                     title="Répartition par type de drépanocytose")
        st.plotly_chart(fig, use_container_width=True)

    # Moyennes des biomarqueurs
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
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
        bio_df = pd.DataFrame(bio_data).T
        st.subheader("Moyennes des biomarqueurs")
        st.table(bio_df)

    # ============================
    # Graphiques temporels
    # ============================
    st.header("Analyse Temporelle")
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Aout","Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)

        # Nombre de consultations par mois
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                      title="Nombre de consultations par mois")
        fig.update_layout(height=400, xaxis_title="Mois", yaxis_title="Nombre de consultations")
        st.plotly_chart(fig, use_container_width=True)

        # Diagnostics par mois
        if 'Diagnostic Catégorisé' in df_eda.columns:
            diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
            fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                          title="Diagnostics par mois")
            fig.update_layout(height=400, xaxis_title="Mois", yaxis_title="Nombre de cas")
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Clustering KMeans
    # ============================
    if df_cluster is not None:
        st.header("Clustering KMeans")
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

        # Clustering PCA 2D
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)
        pca = PCA(n_components=2)
        df_pca = pd.DataFrame(pca.fit_transform(df_cluster_scaled), columns=['PC1','PC2'])
        df_pca['Cluster'] = df_cluster['Cluster']

        fig, ax = plt.subplots()
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
        st.pyplot(fig)

        st.subheader("Profil détaillé des clusters")
        cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename("Nombre de patients"))
        cluster_means = df_cluster.groupby('Cluster')[quantitative_vars].mean().round(2)
        st.dataframe(cluster_means)
