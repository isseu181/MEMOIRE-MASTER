# eda.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ============================
# Fonctions utilitaires
# ============================
def oui_non_vers_binaire(valeur):
    if isinstance(valeur, str) and valeur.strip().lower() in ["oui","o"]:
        return 1
    elif isinstance(valeur, str) and valeur.strip().lower() in ["non","n"]:
        return 0
    return valeur

def convertir_df_oui_non(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns or []
    for col in df.columns:
        if col not in exclude_columns and df[col].isin(
            ["Oui","Non","OUI","NON","oui","non","O","N"]
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
                dates = pd.to_datetime(df_urg[col_date_candidates[0]], errors='coerce').dropna()
                toutes_dates = pd.concat([toutes_dates, dates])
    return toutes_dates

# ============================
# Page Streamlit
# ============================
def show_eda():
    st.title("📊 Analyse exploratoire des données")
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"

    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
        st.success("✅ Fichier chargé avec succès !")
    except:
        st.error(f"❌ Fichier introuvable : {file_path}")
        return

    onglets = st.tabs(["Démographique","Clinique","Temporel","Biomarqueurs"])

    # ============================
    # Onglet Démographique
    # ============================
    with onglets[0]:
        if 'Identite' in feuilles:
            identite = feuilles['Identite']
            identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
            st.header("1️⃣ Identité des patients")
            st.write("Nombre total de patients:", len(identite))

            # Sexe
            if 'Sexe' in identite.columns:
                sexe_counts = identite['Sexe'].value_counts()
                fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                             title="Répartition par sexe", color_discrete_sequence=px.colors.sequential.RdBu)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Origine géographique
            if 'Origine Géographique' in identite.columns:
                origine_counts = identite['Origine Géographique'].value_counts()
                fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                             title="Répartition par origine géographique", color_discrete_sequence=px.colors.sequential.Viridis)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Scolarité
            if "Niveau d'instruction scolarité" in identite.columns:
                scolar_counts = identite["Niveau d'instruction scolarité"].value_counts()
                fig = px.pie(scolar_counts, names=scolar_counts.index, values=scolar_counts.values,
                             title="Répartition de la scolarisation", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Âge
            age_col = "Âge du debut d etude en mois (en janvier 2023)"
            if age_col in identite.columns:
                identite[age_col] = pd.to_numeric(identite[age_col], errors='coerce')
                fig = px.histogram(identite, x=age_col, nbins=15,
                                   title="Répartition des âges à l’inclusion",
                                   color_discrete_sequence=["#2E86C1"])
                fig.update_traces(texttemplate="%{y}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Clinique
    # ============================
    with onglets[1]:
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)

            st.header("2️⃣ Type de drépanocytose et paramètres biologiques")
            if 'Type de drépanocytose' in drepano.columns:
                type_counts = drepano['Type de drépanocytose'].value_counts()
                st.table(type_counts)

            # Analyse bivariée (qualitatives 3D)
            cible = "Evolution" if "Evolution" in drepano.columns else None
            if cible:
                variables = ["Type de drépanocytose","Sexe","Origine Géographique"]
                for var in variables:
                    if var in drepano.columns:
                        cross_tab = pd.crosstab(drepano[var], drepano[cible])
                        fig = px.scatter_3d(cross_tab.reset_index(), x=cross_tab.index, y=cross_tab.columns[0],
                                            z=cross_tab.columns[-1], size=cross_tab.sum(axis=1), color=cross_tab.index,
                                            title=f"{var} vs {cible} (3D)")
                        st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Temporel
    # ============================
    with onglets[2]:
        st.header("📅 Analyse temporelle des urgences")
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

        urgences = [f'Urgence{i}' for i in range(1,7)]
        consultations_par_urgence = {}

        for urg in urgences:
            if urg in feuilles:
                df_urg = feuilles[urg].copy()
                df_urg = convertir_df_oui_non(df_urg)
                date_col_candidates = [c for c in df_urg.columns if "date" in c.lower()]
                if date_col_candidates:
                    df_urg = df_urg[df_urg[date_col_candidates[0]].notna()]
                    df_urg['Mois_text'] = pd.to_datetime(df_urg[date_col_candidates[0]], errors='coerce').dt.month
                    df_urg['Mois_text'] = df_urg['Mois_text'].map(lambda x: mois_ordre[int(x)-1] if pd.notna(x) else None)
                    mois_counts = df_urg['Mois_text'].value_counts().reindex(mois_ordre).fillna(0)
                    consultations_par_urgence[urg] = mois_counts

        if consultations_par_urgence:
            temp_df = pd.DataFrame(consultations_par_urgence)
            temp_df.index.name = "Mois"
            st.subheader("Nombre de consultations par urgence (par mois)")
            st.dataframe(temp_df.astype(int))

            fig = px.line(temp_df, x=temp_df.index, y=temp_df.columns,
                          labels={"value":"Nombre de consultations","Mois":"Mois"},
                          title="Évolution mensuelle des consultations par Urgence",
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet Biomarqueurs
    # ============================
    with onglets[3]:
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)
            bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
            st.header("📌 Biomarqueurs (statistiques descriptives)")
            stats_data = {}
            for col in bio_cols:
                if col in drepano.columns:
                    drepano[col] = pd.to_numeric(drepano[col], errors='coerce')
                    stats_data[col] = {
                        "Moyenne": drepano[col].mean(),
                        "Médiane": drepano[col].median(),
                        "Min": drepano[col].min(),
                        "Max": drepano[col].max()
                    }
            if stats_data:
                stats_df = pd.DataFrame(stats_data).T.round(2)
                st.table(stats_df)
