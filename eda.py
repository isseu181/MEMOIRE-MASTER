# eda.py
import streamlit as st
import pandas as pd
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
        feuilles = pd.read_excel(file_path, sheet_name=None)
    except:
        st.warning(f"⚠️ Fichier '{file_path}' introuvable ou illisible.")
        return

    # Création des onglets
    onglets = st.tabs(["Démographique", "Clinique", "Temporel", "Biomarqueurs"])

    # ============================
    # Onglet 1 : Démographique
    # ============================
    with onglets[0]:
        st.header("Informations Démographiques")
        if 'Identite' in feuilles:
            identite = feuilles['Identite']
            identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
            st.write("Nombre total de patients :", len(identite))

            # Sexe
            if 'Sexe' in identite.columns:
                fig = px.pie(identite, names='Sexe', title="Répartition par sexe", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

            # Origine géographique
            if 'Origine Géographique' in identite.columns:
                fig = px.pie(identite, names='Origine Géographique', title="Répartition par origine", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

            # Niveau d'instruction
            if "Niveau d'instruction scolarité" in identite.columns:
                fig = px.pie(identite, names="Niveau d'instruction scolarité", title="Répartition de la scolarisation", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

            # Âge
            age_col = "Âge du debut d etude en mois (en janvier 2023)"
            if age_col in identite.columns:
                identite[age_col] = pd.to_numeric(identite[age_col], errors='coerce')
                fig = px.histogram(identite, x=age_col, nbins=15, title="Répartition des âges à l’inclusion")
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet 2 : Clinique
    # ============================
    with onglets[1]:
        st.header("Informations Cliniques")

        # Type de drépanocytose + graphique 3D
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)
            if 'Type de drépanocytose' in drepano.columns:
                st.subheader("Type de drépanocytose")
                st.table(drepano['Type de drépanocytose'].value_counts())
                if all(x in drepano.columns for x in ["% d'Hb F","% d'Hb S"]):
                    fig = px.scatter_3d(drepano, x="Type de drépanocytose", y="% d'Hb F", z="% d'Hb S", color="Type de drépanocytose")
                    st.plotly_chart(fig, use_container_width=True)

        # Nombre de consultations par urgence
        st.subheader("Nombre de consultations par urgence")
        urgences = {}
        for i in range(1,7):
            nom = f'Urgence{i}'
            if nom in feuilles:
                df_urg = feuilles[nom]
                df_urg = convertir_df_oui_non(df_urg)
                urgences[nom] = len(df_urg)
        if urgences:
            st.table(pd.DataFrame.from_dict(urgences, orient='index', columns=['Nombre de consultations']))

        # Analyse bivariée qualitative
        st.subheader("Analyse bivariée : Evolution vs variables qualitatives")
        try:
            df_nettoye = pd.read_excel(file_path)
            cible = "Evolution"
            qualitative_vars = ["Type de drépanocytose","Sexe","Origine Géographique","Prise en charge","Diagnostic Catégorisé"]
            for var in qualitative_vars:
                if var in df_nettoye.columns:
                    cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize='index')*100
                    st.write(f"{var} vs {cible}")
                    st.dataframe(cross_tab.round(2))
                    fig = px.bar(cross_tab, barmode="group", text_auto=".2f", title=f"{var} vs {cible}")
                    st.plotly_chart(fig, use_container_width=True)
        except:
            pass

        # Analyse bivariée quantitative
        st.subheader("Analyse bivariée : Evolution vs variables quantitatives")
        quantitative_vars = ["Taux d'Hb (g/dL)","% d'Hb F","% d'Hb S","GB (/mm3)"]
        for var in quantitative_vars:
            if var in df_nettoye.columns:
                stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
                st.write(f"{var} vs {cible}")
                st.table(stats_group)

    # ============================
    # Onglet 3 : Temporel
    # ============================
    with onglets[2]:
        st.header("Analyse Temporelle")
        toutes_dates = concat_dates_urgences(feuilles)
        if not toutes_dates.empty:
            repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
            mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                         7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
            repartition_df = pd.DataFrame({
                'Mois':[mois_noms[m] for m in repartition_mensuelle.index],
                'Nombre de consultations': repartition_mensuelle.values
            })
            fig = px.line(repartition_df, x='Mois', y='Nombre de consultations', markers=True,
                          title="Répartition mensuelle des urgences drépanocytaires")
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Onglet 4 : Biomarqueurs
    # ============================
    with onglets[3]:
        st.header("Biomarqueurs")
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)
            bio_cols = ["Taux d'Hb (g/dL)","% d'Hb F","% d'Hb S","% d'HB C",
                        "Nbre de GB (/mm3)","Nbre de PLT (/mm3)"]
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
