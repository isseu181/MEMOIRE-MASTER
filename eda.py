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

def concat_dates_urgences(feuilles):
    """Concatène toutes les dates des urgences dans une seule série."""
    toutes_dates = pd.Series(dtype='datetime64[ns]')
    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom]
            col_date_candidates = [col for col in df_urg.columns if 'date' in col.lower()]
            if col_date_candidates:
                col_date = col_date_candidates[0]
                dates = pd.to_datetime(df_urg[col_date], errors='coerce').dropna()
                toutes_dates = pd.concat([toutes_dates, dates])
    return toutes_dates

# ============================
# Page Streamlit
# ============================
def show_eda():
    st.title("📊 Analyse exploratoire des données")
    file_path = "fichier_nettoye.xlsx"

    try:
        df_nettoye = pd.read_excel(file_path)
        st.success("✅ Fichier chargé avec succès !")
    except:
        st.warning(f"⚠️ Fichier '{file_path}' introuvable ou illisible.")
        return

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
        if 'Sexe' in df_nettoye.columns:
            sexe_counts = df_nettoye['Sexe'].value_counts()
            fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                         title="Répartition par sexe", color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textinfo='percent+label', pull=0.05)
            st.plotly_chart(fig, use_container_width=True)

        # Origine géographique
        if 'Origine Géographique' in df_nettoye.columns:
            origine_counts = df_nettoye['Origine Géographique'].value_counts()
            fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                         title="Répartition par origine géographique", color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_traces(textinfo='percent+label', pull=0.05)
            st.plotly_chart(fig, use_container_width=True)

        # Scolarité
        if "Niveau d'instruction scolarité" in df_nettoye.columns:
            scolar_counts = df_nettoye["Niveau d'instruction scolarité"].value_counts()
            fig = px.pie(scolar_counts, names=scolar_counts.index, values=scolar_counts.values,
                         title="Répartition de la scolarisation", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textinfo='percent+label', pull=0.05)
            st.plotly_chart(fig, use_container_width=True)

        # Âge
        age_col = "Âge du debut d etude en mois (en janvier 2023)"
        if age_col in df_nettoye.columns:
            df_nettoye[age_col] = pd.to_numeric(df_nettoye[age_col], errors='coerce')
            fig = px.histogram(df_nettoye, x=age_col, nbins=15,
                               title="Répartition des âges à l’inclusion",
                               color_discrete_sequence=["#2E86C1"])
            fig.update_traces(texttemplate="%{y}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Clinique
    # ============================
    with onglets[1]:
        st.header("2️⃣ Données cliniques")

        # Type de drépanocytose
        if 'Type de drépanocytose' in df_nettoye.columns:
            type_counts = df_nettoye['Type de drépanocytose'].value_counts()
            fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                         title="Répartition des types de drépanocytose")
            st.plotly_chart(fig, use_container_width=True)

        # Analyse bivariée qualitatives vs Evolution
        cible = "Evolution"
        qualitative_vars = ["Sexe","Origine Géographique","Diagnostic Catégorisé"]
        for var in qualitative_vars:
            if var in df_nettoye.columns and cible in df_nettoye.columns:
                st.subheader(f"{var} vs {cible}")
                cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                st.dataframe(cross_tab.round(2))
                fig = px.bar(cross_tab, barmode="group", text_auto=".2f",
                             title=f"{var} vs {cible}")
                st.plotly_chart(fig, use_container_width=True)

        # Analyse bivariée quantitatives vs Evolution
        quantitative_vars = ["Âge du debut d etude en mois (en janvier 2023)", 
                             "GR (/mm3)","GB (/mm3)","HB (g/dl)"]
        for var in quantitative_vars:
            if var in df_nettoye.columns and cible in df_nettoye.columns:
                st.subheader(f"{var} vs {cible}")
                stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
                st.table(stats_group)

    # ============================
    # Onglet Temporel
    # ============================
    with onglets[2]:
        st.header("3️⃣ Analyse temporelle")
        urgences = [f"Urgence{i}" for i in range(1,7)]
        nombre_consultations = {}
        for u in urgences:
            if u in df_nettoye.columns:
                nombre_consultations[u] = df_nettoye[u].notna().sum()
        if nombre_consultations:
            temp_df = pd.DataFrame.from_dict(nombre_consultations, orient='index', columns=['Nombre de consultations'])
            fig = px.bar(temp_df, y='Nombre de consultations', x=temp_df.index,
                         title="Nombre de consultations par urgence",
                         color='Nombre de consultations', text='Nombre de consultations')
            st.plotly_chart(fig, use_container_width=True)

        # Répartition mensuelle si colonne 'Mois' existe
        if 'Mois' in df_nettoye.columns:
            mois_counts = df_nettoye['Mois'].value_counts().sort_index()
            fig = px.line(x=mois_counts.index, y=mois_counts.values,
                          labels={"x":"Mois","y":"Nombre de consultations"},
                          title="Répartition mensuelle des urgences", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Biomarqueurs
    # ============================
    with onglets[3]:
        st.header("4️⃣ Biomarqueurs - statistiques descriptives")
        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                    "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        bio_data = {}
        for col in bio_cols:
            if col in df_nettoye.columns:
                df_nettoye[col] = pd.to_numeric(df_nettoye[col], errors='coerce')
                bio_data[col] = {
                    "Moyenne": df_nettoye[col].mean(),
                    "Médiane": df_nettoye[col].median(),
                    "Min": df_nettoye[col].min(),
                    "Max": df_nettoye[col].max()
                }
        if bio_data:
            bio_df = pd.DataFrame(bio_data).T.round(2)
            st.table(bio_df)
