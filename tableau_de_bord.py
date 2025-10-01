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
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
    try:
        df_cluster = pd.read_excel("segmentation.xlsx")
    except:
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
        ("Patients Total / suivis 2023", patients_total, "#1f77b4"),
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
                padding:20px;
                border-radius:10px;
                font-size:20px;
            ">
                <strong>{value}</strong><br>{title}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # Graphiques en grille (2 par ligne)
    # ============================
    plots = []

    # Sexe vs Evolution
    if 'Sexe' in df_eda.columns and 'Evolution' in df_eda.columns:
        sexe_counts = pd.crosstab(df_eda['Sexe'], df_eda['Evolution'], normalize='index')*100
        fig1 = px.bar(sexe_counts, barmode='group', text_auto='.1f', title="Sexe vs Evolution")
        fig1.update_layout(height=400)
        plots.append(fig1)

    # Type de drépanocytose
    if 'Type de drépanocytose' in df_eda.columns:
        type_counts = df_eda['Type de drépanocytose'].value_counts()
        fig2 = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                      title="Répartition par type de drépanocytose")
        fig2.update_layout(height=400)
        plots.append(fig2)

    # Nombre de consultations par mois
    if 'Mois' in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
        df_eda['Mois'] = pd.Categorical(df_eda['Mois'], categories=mois_ordre, ordered=True)
        mois_counts = df_eda['Mois'].value_counts().sort_index()
        fig3 = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                       title="Nombre de consultations par mois")
        fig3.update_layout(height=400, xaxis_title="Mois", yaxis_title="Nombre de consultations")
        plots.append(fig3)

        # Diagnostics par mois
        if 'Diagnostic Catégorisé' in df_eda.columns:
            diag_month = df_eda.groupby(['Mois','Diagnostic Catégorisé']).size().unstack(fill_value=0)
            fig4 = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                           title="Diagnostics par mois")
            fig4.update_layout(height=400, xaxis_title="Mois", yaxis_title="Nombre de cas")
            plots.append(fig4)

    # Clustering KMeans
    if df_cluster is not None:
        quantitative_vars = [
            "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
            "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
        ]
        df_cluster_scaled = StandardScaler().fit_transform(df_cluster[quantitative_vars])
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)
        pca = PCA(n_components=2)
        df_pca = pd.DataFrame(pca.fit_transform(df_cluster_scaled), columns=['PC1','PC2'])
        df_pca['Cluster'] = df_cluster['Cluster']
        fig5, ax = plt.subplots()
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
        plots.append(fig5)

    # ============================
    # Affichage des graphiques en grille 2 par ligne
    # ============================
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(plots):
                plot = plots[i+j]
                if isinstance(plot, px.Figure):
                    col.plotly_chart(plot, use_container_width=True)
                else:
                    col.pyplot(plot)

    st.markdown("---")

    # ============================
    # Biomarqueurs - cartes avec moyennes
    # ============================
    st.subheader("Moyennes des biomarqueurs")
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]

    bio_data = {}
    for col in bio_cols:
        if col in df_eda.columns:
            df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
            bio_data[col] = round(df_eda[col].mean(), 2)

    bio_cols_split = list(bio_data.items())
    n_cols = 3
    for i in range(0, len(bio_cols_split), n_cols):
        cols = st.columns(n_cols)
        for j, col_name in enumerate(bio_cols_split[i:i+n_cols]):
            title, value = col_name
            cols[j].markdown(f"""
                <div style="
                    background-color:#1f77b4;
                    color:white;
                    text-align:center;
                    padding:15px;
                    border-radius:10px;
                    font-size:18px;
                ">
                    <strong>{value}</strong><br>{title}
                </div>
            """, unsafe_allow_html=True)
