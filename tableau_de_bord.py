# tableau_de_bord.py
import streamlit as st
import pandas as pd
import plotly.express as px

# ============================
# CONFIGURATION
# ============================
st.set_page_config(page_title="Tableau de bord - Drépanocytose", layout="wide")

# ============================
# CHARGEMENT DES DONNÉES
# ============================
try:
    df_eda = pd.read_excel("fichier_nettoye.xlsx")
except:
    st.error("❌ fichier_nettoye.xlsx introuvable")
    st.stop()

try:
    df_cluster = pd.read_excel("segmentation.xlsx")
except:
    df_cluster = None

# ============================
# FONCTION TABLEAU DE BORD
# ============================
def show_dashboard():
    st.title("📊 Tableau de bord - Suivi drépanocytose")

    # ============================
    # INDICATEURS EN HAUT
    # ============================
    total_patients = len(df_cluster) if df_cluster is not None else len(df_eda)
    total_urgences = len(df_eda)
    evolution_fav = (df_eda["Evolution"].eq("Favorable").mean()*100
                     if "Evolution" in df_eda.columns else None)
    complications = 100 - evolution_fav if evolution_fav is not None else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients Total", total_patients, "↗ Suivis 2023")
    col2.metric("Urgences Total", total_urgences, "↗ Consultations")
    col3.metric("Évolution Favorable", f"{evolution_fav:.1f}%" if evolution_fav else "N/A")
    col4.metric("Complications", f"{complications:.1f}%" if complications else "N/A")

    # ============================
    # GRAPHIQUES (2 PAR LIGNE)
    # ============================
    plots = []

    # 1. Répartition par sexe
    if "Sexe" in df_cluster.columns:
        sexe_counts = df_cluster["Sexe"].value_counts()
        fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                     title="Répartition par sexe")
        plots.append(fig)

    # 2. Origine géographique
    if "Origine Géographique" in df_cluster.columns:
        origine_counts = df_cluster["Origine Géographique"].value_counts()
        fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                     title="Répartition par origine géographique")
        plots.append(fig)

    # 3. Diagnostics
    if "Diagnostic Catégorisé" in df_eda.columns:
        diag_counts = df_eda["Diagnostic Catégorisé"].value_counts()
        fig = px.pie(diag_counts, names=diag_counts.index, values=diag_counts.values,
                     title="Répartition des diagnostics")
        plots.append(fig)

    # 4. Type de drépanocytose (depuis segmentation)
    if df_cluster is not None and "Type de drépanocytose" in df_cluster.columns:
        type_counts = df_cluster["Type de drépanocytose"].value_counts()
        fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                     title="Répartition des types de drépanocytose")
        plots.append(fig)

    # 5. Courbe consultations par mois
    if "Mois" in df_eda.columns:
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet",
                      "Aout","Septembre","Octobre","Novembre","Décembre"]
        df_eda["Mois"] = pd.Categorical(df_eda["Mois"], categories=mois_ordre, ordered=True)
        mois_counts = df_eda.groupby("Mois").size().reindex(mois_ordre)
        fig = px.line(x=mois_counts.index, y=mois_counts.values, markers=True,
                      labels={"x":"Mois","y":"Nombre de consultations"},
                      title="Nombre de consultations par mois")
        plots.append(fig)

    # 6. Courbe empilée diagnostics par mois
    if "Mois" in df_eda.columns and "Diagnostic Catégorisé" in df_eda.columns:
        mois_diag = df_eda.groupby(["Mois","Diagnostic Catégorisé"]).size().reset_index(name="Count")
        mois_diag["Mois"] = pd.Categorical(mois_diag["Mois"], categories=mois_ordre, ordered=True)
        fig = px.area(mois_diag, x="Mois", y="Count", color="Diagnostic Catégorisé",
                      title="Évolution des diagnostics par mois")
        plots.append(fig)

    # 7. Analyse bivariée : Sexe vs Évolution
    if "Sexe" in df_cluster.columns and "Evolution" in df_eda.columns:
        cross_tab = pd.crosstab(df_cluster["Sexe"], df_eda["Evolution"], normalize="index")*100
        fig = px.bar(cross_tab, barmode="group", title="Sexe vs Évolution (%)")
        plots.append(fig)

    # 8. Analyse bivariée : Diagnostic vs Évolution
    if "Diagnostic Catégorisé" in df_eda.columns and "Evolution" in df_eda.columns:
        cross_tab = pd.crosstab(df_eda["Diagnostic Catégorisé"], df_eda["Evolution"], normalize="index")*100
        fig = px.bar(cross_tab, barmode="group", title="Diagnostic vs Évolution (%)")
        plots.append(fig)

    # --- affichage 2 par ligne ---
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(plots):
                col.plotly_chart(plots[i+j], use_container_width=True)

    # ============================
    # BIOMARQUEURS EN CARTES SIMPLES
    # ============================
    st.subheader("🧪 Moyennes des biomarqueurs")
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]

    bio_means = {}
    for col in bio_cols:
        if col in df_eda.columns:
            df_cluster[col] = pd.to_numeric(df_cluster[col], errors="coerce")
            bio_means[col] = df_cluster[col].mean()

    if bio_means:
        cols = st.columns(len(bio_means))
        for (name, val), col in zip(bio_means.items(), cols):
            with col:
                st.metric(name, f"{val:.2f}")

# ============================
# EXECUTION DU DASHBOARD
# ============================
if __name__ == "__main__":
    show_dashboard()


