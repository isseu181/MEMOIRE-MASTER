# eda.py
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ============================
# Fonctions utilitaires
# ============================
def oui_non_vers_binaire(valeur):
    if isinstance(valeur, str) and valeur.strip().lower() in ["oui", "o"]:
        return 1
    elif isinstance(valeur, str) and valeur.strip().lower() in ["non", "n"]:
        return 0
    return valeur

def convertir_df_oui_non(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns or []
    for col in df.columns:
        if col not in exclude_columns and df[col].isin(
            ["Oui", "Non", "OUI", "NON", "oui", "non", "O", "N"]
        ).any():
            df[col] = df[col].apply(oui_non_vers_binaire)
    return df

# ============================
# Page Streamlit
# ============================
def show_eda():
    st.title("Analyse exploratoire des données")

    # ============================
    # Charger les fichiers
    # ============================
    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
    except:
        df_nettoye = pd.DataFrame()

    try:
        df_seg = pd.read_excel("segmentation.xlsx")
    except:
        df_seg = pd.DataFrame()

    # ============================
    # Onglets horizontaux
    # ============================
    onglets = st.tabs(["Démographique", "Clinique", "Temporel", "Biomarqueurs"])

    # ============================
    # Onglet Démographique
    # ============================
    with onglets[0]:
        st.header("1️⃣ Données démographiques")

        # Sexe
        if "Sexe" in df_seg.columns:
            sexe_counts = df_seg["Sexe"].value_counts()
            fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                         title="Répartition par sexe", color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textinfo="percent+label", pull=0.05)
            st.plotly_chart(fig, use_container_width=True)

        # Origine géographique
        if "Origine Géographique" in df_seg.columns:
            origine_counts = df_seg["Origine Géographique"].value_counts()
            fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                         title="Répartition par origine géographique", color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_traces(textinfo="percent+label", pull=0.05)
            st.plotly_chart(fig, use_container_width=True)

        # Scolarité
        if "Niveau d'instruction scolarité" in df_seg.columns:
            scolar_counts = df_seg["Niveau d'instruction scolarité"].value_counts()
            fig = px.pie(scolar_counts, names=scolar_counts.index, values=scolar_counts.values,
                         title="Répartition de la scolarisation", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textinfo="percent+label", pull=0.05)
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Clinique
    # ============================
    with onglets[1]:
        st.header("2️⃣ Données cliniques")

        # Type de drépanocytose
        if "Type de drépanocytose" in df_seg.columns:
            type_counts = df_seg["Type de drépanocytose"].value_counts()
            fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                         title="Répartition des types de drépanocytose")
            st.plotly_chart(fig, use_container_width=True)

        # Analyse bivariée qualitatives vs Evolution
        cible = "Evolution"
        qualitative_vars = ["Sexe", "Origine Géographique", "Diagnostic Catégorisé"]
        for var in qualitative_vars:
            if var in df_nettoye.columns and cible in df_nettoye.columns:
                st.subheader(f"{var} vs {cible}")
                cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                fig = px.bar(cross_tab, barmode="group", text_auto=".2f",
                             title=f"{var} vs {cible}")
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Temporel
    # ============================
    with onglets[2]:
        st.header("3️⃣ Analyse temporelle")
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

        # Diagnostics par mois
        if "Mois" in df_nettoye.columns and "Diagnostic Catégorisé" in df_nettoye.columns:
            diag_mois = df_nettoye.groupby(["Mois","Diagnostic Catégorisé"]).size().reset_index(name="Nombre")
            diag_mois["Mois"] = pd.Categorical(diag_mois["Mois"], categories=mois_ordre, ordered=True)
            diag_mois = diag_mois.sort_values("Mois")
            fig_diag = px.line(diag_mois, x="Mois", y="Nombre", color="Diagnostic Catégorisé",
                               markers=True, title="Diagnostics par mois")
            st.plotly_chart(fig_diag, use_container_width=True)

        # Consultations par mois
        if "Mois" in df_nettoye.columns:
            df_nettoye["Mois"] = pd.Categorical(df_nettoye["Mois"], categories=mois_ordre, ordered=True)
            mois_counts = df_nettoye.groupby("Mois").size().reset_index(name="Nombre")
            fig_mois = px.line(mois_counts, x="Mois", y="Nombre", markers=True,
                               title="Consultations totales par mois")
            st.plotly_chart(fig_mois, use_container_width=True)

        # Consultations par urgence
        urgences = [f"Urgence{i}" for i in range(1,7)]
        consultations_par_urgence = {urg: df_nettoye[urg].notna().sum() 
                                     for urg in urgences if urg in df_nettoye.columns}
        if consultations_par_urgence:
            fig_urg = px.bar(x=list(consultations_par_urgence.keys()),
                             y=list(consultations_par_urgence.values()),
                             text=list(consultations_par_urgence.values()),
                             title="Nombre de consultations par Urgence")
            st.plotly_chart(fig_urg, use_container_width=True)

    # ============================
    # Onglet Biomarqueurs
    # ============================
    with onglets[3]:
        st.header("4️⃣ Biomarqueurs - statistiques descriptives")
        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                    "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        bio_data = {}
        for col in bio_cols:
            if col in df_seg.columns:
                df_seg[col] = pd.to_numeric(df_seg[col], errors="coerce")
                bio_data[col] = {
                    "Moyenne": df_seg[col].mean(),
                    "Médiane": df_seg[col].median(),
                    "Min": df_seg[col].min(),
                    "Max": df_seg[col].max()
                }
        if bio_data:
            st.table(pd.DataFrame(bio_data).T.round(2))


