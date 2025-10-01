# eda.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

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

def show_eda():
    st.title("📊 Analyse exploratoire des données")

    try:
        feuilles = pd.read_excel("Base_de_donnees_USAD_URGENCES1.xlsx", sheet_name=None)
    except:
        st.warning("Fichier introuvable ou corrompu.")
        return

    onglets = st.tabs(["Démographique", "Clinique", "Temporel", "Biomarqueurs"])

    # ===========================
    # Onglet 1 : Démographique
    # ===========================
    with onglets[0]:
        if 'Identite' in feuilles:
            identite = feuilles['Identite']
            identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
            st.header("📌 Informations démographiques")
            st.write("Nombre total de patients:", len(identite))

            if 'Sexe' in identite.columns:
                sexe_counts = identite['Sexe'].value_counts()
                fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                             title="Répartition par sexe")
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            if 'Origine Géographique' in identite.columns:
                origine_counts = identite['Origine Géographique'].value_counts()
                fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                             title="Répartition par origine géographique")
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            if "Niveau d'instruction scolarité" in identite.columns:
                scolar_counts = identite["Niveau d'instruction scolarité"].value_counts()
                fig = px.pie(scolar_counts, names=scolar_counts.index, values=scolar_counts.values,
                             title="Répartition de la scolarisation")
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            age_col = "Âge du debut d etude en mois (en janvier 2023)"
            if age_col in identite.columns:
                identite[age_col] = pd.to_numeric(identite[age_col], errors='coerce')
                fig = px.histogram(identite, x=age_col, nbins=15,
                                   title="Répartition des âges à l’inclusion")
                st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # Onglet 2 : Clinique
    # ===========================
    with onglets[1]:
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)

            st.header("📌 Type de drépanocytose et paramètres cliniques")
            if 'Type de drépanocytose' in drepano.columns:
                type_counts = drepano['Type de drépanocytose'].value_counts()
                st.table(type_counts)

    # ===========================
    # Onglet 3 : Temporel
    # ===========================
    with onglets[2]:
        st.header("📌 Consultations d'urgence et répartition temporelle")
        toutes_dates = concat_dates_urgences(feuilles)
        if not toutes_dates.empty:
            repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
            mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                         7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
            repartition_df = pd.DataFrame({
                'Mois':[mois_noms[m] for m in repartition_mensuelle.index],
                'Nombre de consultations': repartition_mensuelle.values
            })
            fig = px.line(repartition_df, x='Mois', y='Nombre de consultations',
                          title="Répartition mensuelle des urgences drépanocytaires", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # Onglet 4 : Biomarqueurs et analyse bivariée
    # ===========================
    with onglets[3]:
        st.header("📌 Biomarqueurs et analyse bivariée")
        try:
            df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
            cible = "Evolution"

            # Tableau des biomarqueurs
            bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
            st.subheader("📌 Tableau des biomarqueurs")
            stats_data = {}
            for col in bio_cols:
                if col in df_nettoye.columns:
                    df_nettoye[col] = pd.to_numeric(df_nettoye[col], errors='coerce')
                    stats_data[col] = {
                        "Moyenne": df_nettoye[col].mean(),
                        "Médiane": df_nettoye[col].median(),
                        "Min": df_nettoye[col].min(),
                        "Max": df_nettoye[col].max()
                    }
            if stats_data:
                stats_df = pd.DataFrame(stats_data).T.round(2)
                st.table(stats_df)

            # Analyse bivariée
            if cible in df_nettoye.columns:
                st.subheader("📌 Analyse bivariée")

                variables_qualitatives = ["Type de drépanocytose","Sexe","Origine Géographique","Diagnostic Catégorisé"]
                variables_quantitatives = ["Âge du debut d etude en mois (en janvier 2023)", 
                                           "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S"]

                st.markdown("### Variables qualitatives")
                for var in variables_qualitatives:
                    if var in df_nettoye.columns:
                        cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                        fig = px.bar(cross_tab, barmode="group", text_auto=".2f", title=f"{var} vs {cible}")
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Variables quantitatives")
                for var in variables_quantitatives:
                    if var in df_nettoye.columns:
                        stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
                        st.table(stats_group)
        except:
            pass
