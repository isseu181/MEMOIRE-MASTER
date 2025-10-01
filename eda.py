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
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"

    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
    except Exception:
        st.info("Impossible de charger le fichier principal.")
        return

    # ============================
    # Onglets horizontaux
    # ============================
    onglets = st.tabs(["Démographique", "Clinique", "Temporel", "Biomarqueurs", "Analyse bivariée"])

    # ---------------- DEMOGRAPHIQUE ----------------
    with onglets[0]:
        if 'Identite' in feuilles:
            identite = convertir_df_oui_non(feuilles['Identite'], exclude_columns=["Niveau d'instruction scolarité"])
            st.header("Informations démographiques")
            st.write("Nombre total de patients:", len(identite))

            if 'Sexe' in identite.columns:
                fig = px.pie(identite, names='Sexe', title="Répartition par sexe")
                st.plotly_chart(fig, use_container_width=True)

            if 'Origine Géographique' in identite.columns:
                fig = px.pie(identite, names='Origine Géographique', title="Origine géographique")
                st.plotly_chart(fig, use_container_width=True)

            if "Niveau d'instruction scolarité" in identite.columns:
                fig = px.pie(identite, names="Niveau d'instruction scolarité", title="Niveau scolaire")
                st.plotly_chart(fig, use_container_width=True)

            age_col = "Âge du debut d etude en mois (en janvier 2023)"
            if age_col in identite.columns:
                identite[age_col] = pd.to_numeric(identite[age_col], errors='coerce')
                fig = px.histogram(identite, x=age_col, nbins=15, title="Distribution des âges (mois)")
                st.plotly_chart(fig, use_container_width=True)

    # ---------------- CLINIQUE ----------------
    with onglets[1]:
        st.header("Consultations cliniques")
        symptomes = ['Douleur','Fièvre','Pâleur','Ictère','Toux']
        for i in range(1,7):
            nom = f'Urgence{i}'
            if nom in feuilles:
                df_urg = convertir_df_oui_non(feuilles[nom])
                st.subheader(f"{nom} - {len(df_urg)} consultations")

                data_symptomes = {s: df_urg[s].value_counts().to_dict()
                                  for s in symptomes if s in df_urg.columns}
                if data_symptomes:
                    st.table(pd.DataFrame(data_symptomes).fillna(0).astype(int))

    # ---------------- TEMPOREL ----------------
    with onglets[2]:
        st.header("Répartition temporelle des urgences")
        toutes_dates = concat_dates_urgences(feuilles)
        if not toutes_dates.empty:
            repartition = toutes_dates.dt.month.value_counts().sort_index()
            mois_noms = {1:'Jan',2:'Fév',3:'Mars',4:'Avr',5:'Mai',6:'Juin',
                         7:'Juil',8:'Août',9:'Sept',10:'Oct',11:'Nov',12:'Déc'}
            df_mois = pd.DataFrame({"Mois":[mois_noms[m] for m in repartition.index],
                                    "Consultations":repartition.values})
            fig = px.line(df_mois, x="Mois", y="Consultations", markers=True,
                          title="Consultations mensuelles")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- BIOMARQUEURS ----------------
    with onglets[3]:
        st.header("Biomarqueurs")
        try:
            df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
            bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
            stats = {col: df_nettoye[col].astype(float).describe()[["mean","50%","min","max"]]
                     for col in bio_cols if col in df_nettoye}
            if stats:
                st.table(pd.DataFrame(stats).T.rename(columns={"50%":"Mediane"}).round(2))
        except Exception:
            pass

    # ---------------- ANALYSE BIVARIÉE ----------------
    with onglets[4]:
        st.header("Analyse bivariée : Evolution vs variables")
        try:
            df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
            cible = "Evolution"
            variables = ["Type de drépanocytose","Sexe",
                         "Âge du debut d etude en mois (en janvier 2023)",
                         "Origine Géographique","Prise en charge","Diagnostic Catégorisé"]

            for var in variables:
                if var not in df_nettoye.columns: 
                    continue
                st.subheader(f"{var} vs {cible}")

                if df_nettoye[var].dtype == "object":
                    cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                    fig = px.imshow(cross_tab, text_auto=True, aspect="auto",
                                    title=f"Heatmap {var} vs {cible}")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    fig = px.box(df_nettoye, x=cible, y=var,
                                 title=f"Boxplot {var} selon {cible}")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
