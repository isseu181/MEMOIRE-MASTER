# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================
# Fonction principale
# ============================
def show_dashboard():
    st.title("📊 Tableau de Bord - Analyse USAD Drépanocytose")

    # ============================
    # Chargement des données
    # ============================
    try:
        df_eda = pd.read_excel("fichier_nettoye.xlsx")
    except:
        st.error("⚠️ fichier_nettoye.xlsx introuvable")
        return

    try:
        df_cluster = pd.read_excel("segmentation.xlsx")
    except:
        df_cluster = None
        st.warning("⚠️ segmentation.xlsx introuvable")

    # ============================
    # Indicateurs clés en haut
    # ============================
    col1, col2, col3, col4 = st.columns(4)

    patients_total = len(df_eda)
    urgences_total = df_eda[[c for c in df_eda.columns if "Urgence" in c]].count().sum()
    evolution_fav = df_eda["Evolution"].value_counts(normalize=True).get("Favorable", 0) * 100
    complications = 100 - evolution_fav

    with col1:
        st.metric("Patients Total", patients_total)
    with col2:
        st.metric("Urgences Total", urgences_total)
    with col3:
        st.metric("Évolution Favorable", f"{evolution_fav:.1f}%")
    with col4:
        st.metric("Complications", f"{complications:.1f}%")

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

    # Origine géographique
    if 'Origine Géographique' in df_eda.columns:
        origine_counts = df_eda['Origine Géographique'].value_counts()
        fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                     title="Répartition par origine géographique")
        plots.append(fig)

    # Diagnostic
    if 'Diagnostic Catégorisé' in df_eda.columns:
        diag_counts = df_eda['Diagnostic Catégorisé'].value_counts()
        fig = px.pie(diag_counts, names=diag_counts.index, values=diag_counts.values,
                     title="Répartition des diagnostics")
        plots.append(fig)

    # Sexe vs Evolution
    if "Sexe" in df_eda.columns and "Evolution" in df_eda.columns:
        cross_tab = pd.crosstab(df_eda["Sexe"], df_eda["Evolution"], normalize="index") * 100
        fig = px.bar(cross_tab, barmode="group", title="Sexe vs Evolution")
        plots.append(fig)

    # Origine vs Evolution
    if "Origine Géographique" in df_eda.columns and "Evolution" in df_eda.columns:
        cross_tab = pd.crosstab(df_eda["Origine Géographique"], df_eda["Evolution"], normalize="index") * 100
        fig = px.bar(cross_tab, barmode="group", title="Origine géographique vs Evolution")
        plots.append(fig)

    # ============================
    # Affichage des graphiques en grille
    # ============================
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(plots):
                col.plotly_chart(plots[i + j], use_container_width=True)

    st.markdown("---")

    # ============================
    # Moyennes des biomarqueurs (cartes)
    # ============================
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]

    bio_data = {}
    for col in bio_cols:
        if col in df_eda.columns:
            df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
            bio_data[col] = round(df_eda[col].mean(), 2)

    if bio_data:
        st.subheader("🔬 Moyennes des indicateurs biologiques")
        bio_cols_disp = st.columns(len(bio_data))
        for i, (name, val) in enumerate(bio_data.items()):
            bio_cols_disp[i].metric(name, val)

    st.markdown("---")

    # ============================
    # Analyse temporelle
    # ============================
    st.subheader("📈 Analyse temporelle")

    # Nombre de consultations par urgence
    urgences = [c for c in df_eda.columns if "Urgence" in c]
    consultations = {u: df_eda[u].notna().sum() for u in urgences}
    if consultations:
        temp_df = pd.DataFrame.from_dict(consultations, orient="index", columns=["Consultations"])
        fig = px.line(temp_df, y="Consultations", x=temp_df.index,
                      markers=True, title="Nombre de consultations par urgence")
        st.plotly_chart(fig, use_container_width=True)

    # Consultations par mois
    if "Mois" in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
        df_eda["Mois"] = pd.Categorical(df_eda["Mois"], categories=mois_ordre, ordered=True)
        mois_counts = df_eda["Mois"].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                      title="Consultations par mois")
        st.plotly_chart(fig, use_container_width=True)

        if "Diagnostic Catégorisé" in df_eda.columns:
            diag_month = df_eda.groupby(["Mois", "Diagnostic Catégorisé"]).size().unstack(fill_value=0)
            fig = px.line(diag_month, x=diag_month.index, y=diag_month.columns, markers=True,
                          title="Diagnostics par mois")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ============================
    # Clustering
    # ============================
    st.subheader("🔎 Clustering")

    if df_cluster is not None:
        quantitative_vars = [
            "Âge du debut d etude en mois (en janvier 2023)",
            "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
            "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)"
        ]

        # Répartition des diagnostics
        if "Diagnostic Catégorisé" in df_cluster.columns:
            diag_counts = df_cluster["Diagnostic Catégorisé"].value_counts()
            fig = px.pie(diag_counts, names=diag_counts.index, values=diag_counts.values,
                         title="Répartition des types de drépanocytose (segmentation)")
            st.plotly_chart(fig, use_container_width=True)

        # Moyennes standardisées
        scaler = StandardScaler()
        df_scaled = df_cluster[quantitative_vars].dropna()
        df_scaled_z = pd.DataFrame(scaler.fit_transform(df_scaled),
                                   columns=quantitative_vars)
        df_scaled_z["Cluster"] = df_cluster.loc[df_scaled.index, "Cluster"]

        cluster_means = df_scaled_z.groupby("Cluster").mean()
        st.subheader("📊 Moyennes standardisées (Z-scores) des variables par cluster")
        st.dataframe(cluster_means.round(2))

        # Heatmap
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(cluster_means, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Profil des clusters (Z-scores)")
        st.pyplot(fig)

        # Interprétation
        st.subheader("📝 Interprétation des clusters")
        for cluster, row in cluster_means.iterrows():
            interpretation = []
            if row["Taux d'Hb (g/dL)"] > 0:
                interpretation.append("Hb plus élevée")
            else:
                interpretation.append("Hb plus faible")
            if row["% d'Hb F"] > 0:
                interpretation.append("HbF élevée (protectrice)")
            if row["% d'Hb S"] > 0:
                interpretation.append("HbS élevée (forme sévère)")
            if row["Nbre de GB (/mm3)"] > 0:
                interpretation.append("GB élevés (inflammation)")
            if row["Nbre de PLT (/mm3)"] > 0:
                interpretation.append("PLT élevés (risque thrombotique)")
            st.markdown(f"**Cluster {cluster} :** " + ", ".join(interpretation))
