# eda.py
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ---------------------------
# Fonctions utilitaires
# ---------------------------
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

# ---------------------------
# Onglets Streamlit
# ---------------------------
def show_eda():
    st.title(" Analyse exploratoire des données")
    
    # Charger fichier
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"
    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
    except:
        st.warning("⚠️ Fichiers introuvables. Vérifiez 'Base_de_donnees_USAD_URGENCES1.xlsx' et 'fichier_nettoye.xlsx'.")
        return

    onglets = st.tabs(["Démographique", "Clinique", "Temporel", "Biomarqueurs"])
    
    # ---------------------------
    # Onglet 1 : Démographique
    # ---------------------------
    with onglets[0]:
        st.header("1️⃣ Données démographiques")
        if 'Identite' in feuilles:
            identite = feuilles['Identite']
            identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
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
                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Onglet 2 : Clinique
    # ---------------------------
    with onglets[1]:
        st.header("2️⃣ Données cliniques")
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)
            # Type drépanocytose
            if 'Type de drépanocytose' in drepano.columns:
                st.subheader("Type de drépanocytose")
                st.table(drepano['Type de drépanocytose'].value_counts())

            # Paramètres biologiques
            bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S",
                        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
            st.subheader("📌 Paramètres biologiques (statistiques descriptives)")
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

    # ---------------------------
    # Onglet 3 : Temporel
    # ---------------------------
    with onglets[2]:
        st.header("3️⃣ Analyse temporelle")
        st.subheader("Consultations d'urgence")
        symptomes = ['Douleur','Fièvre','Pâleur','Ictère','Toux']
        for i in range(1,7):
            nom = f'Urgence{i}'
            if nom in feuilles:
                df_urg = feuilles[nom]
                df_urg = convertir_df_oui_non(df_urg)
                date_col_candidates = [c for c in df_urg.columns if "date" in c.lower()]
                if date_col_candidates:
                    df_urg = df_urg[df_urg[date_col_candidates[0]].notna()]
                st.subheader(f"{nom} - Nombre de consultations : {len(df_urg)}")
                data_symptomes = {}
                for s in symptomes:
                    if s in df_urg.columns and not df_urg[s].dropna().empty:
                        counts = df_urg[s].value_counts().to_dict()
                        data_symptomes[s] = counts
                if data_symptomes:
                    st.table(pd.DataFrame(data_symptomes).fillna(0).astype(int))

        # Répartition mensuelle
        st.subheader("Répartition mensuelle des urgences")
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
                          title="Répartition mensuelle des urgences drépanocytaires",
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Onglet 4 : Biomarqueurs et Analyse bivariée
    # ---------------------------
    with onglets[3]:
        st.header("4️⃣ Biomarqueurs et Analyse bivariée")
        # Variables quantitatives
        variables_quantitatives = [
            "Âge de début des signes (en mois)",
            "Âge du debut d etude en mois (en janvier 2023)",
            "Taux d'Hb (g/dL)",
            "VGM (fl/u3)",
            "HB (g/dl)",
            "% d'Hb S",
            "% d'Hb F",
            "Nbre de GB (/mm3)",
            "PLT (/mm3)",
            "Nbre de PLT (/mm3)",
            "TCMH (g/dl)",
            "CRP Si positive (Valeur)",
            "Nbre de transfusion avant 2017",
            "Nbre de transfusion Entre 2017 et 2023"
        ]
        cible = "Evolution"
        st.subheader("📊 Tableau des variables quantitatives vs Evolution")
        stats_quant = {}
        for var in variables_quantitatives:
            if var in df_nettoye.columns:
                df_nettoye[var] = pd.to_numeric(df_nettoye[var], errors='coerce')
                stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"])
                stats_quant[var] = stats_group
        if stats_quant:
            stats_df_quant = pd.concat(stats_quant.values(), keys=stats_quant.keys())
            stats_df_quant.index.names = ["Variable", "Evolution"]
            st.table(stats_df_quant.round(2))

        # Analyse bivariée pour variables qualitatives
        st.subheader("Analyse bivariée : Variables qualitatives vs Evolution")
        variables_qualitatives = [
            "Type de drépanocytose",
            "Sexe",
            "Origine Géographique",
            "Diagnostic Catégorisé",
            "Prise en charge"
        ]
        for var in variables_qualitatives:
            if var in df_nettoye.columns:
                st.markdown(f"**{var} vs {cible}**")
                cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                st.dataframe(cross_tab.round(2))
                fig = px.bar(cross_tab, barmode="group", text_auto=".2f",
                             title=f"{var} vs {cible}",
                             labels={'value':'Pourcentage','index':var})
                st.plotly_chart(fig, use_container_width=True)
        st.success(" Analyse des biomarqueurs et bivariée terminée.")
