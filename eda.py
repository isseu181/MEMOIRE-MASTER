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
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"

    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
        st.success("✅ Fichier chargé avec succès !")
    except FileNotFoundError:
        st.error(f"❌ Fichier introuvable. Assurez-vous que '{file_path}' est à la racine du projet.")
        return

    # ============================
    # Menu latéral sous-sections
    # ============================
    section = st.sidebar.selectbox("Sélectionnez une section", 
                                   ["Démographie", "Clinique", "Temporel", "Biomarqueurs"])

    # ============================
    # 1️⃣ Démographie
    # ============================
    if section == "Démographie":
        st.header("👨‍👩‍👧‍👦 Données démographiques")
        if 'Identite' in feuilles:
            identite = feuilles['Identite']
            identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
            st.write(f"Nombre total de patients : {len(identite)}")

            # Sexe
            if 'Sexe' in identite.columns:
                st.subheader("Répartition par sexe")
                sexe_counts = identite['Sexe'].value_counts()
                fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                             title="Sexe", color_discrete_sequence=px.colors.sequential.RdBu)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Origine géographique
            if 'Origine Géographique' in identite.columns:
                st.subheader("Origine géographique")
                origine_counts = identite['Origine Géographique'].value_counts()
                fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                             title="Origine Géographique", color_discrete_sequence=px.colors.sequential.Viridis)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Scolarité
            if "Niveau d'instruction scolarité" in identite.columns:
                st.subheader("Scolarisation des enfants")
                sco_counts = identite["Niveau d'instruction scolarité"].value_counts()
                fig = px.pie(sco_counts, names=sco_counts.index, values=sco_counts.values,
                             title="Scolarisation", color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Statut des parents
            if "Parents Salariés" in identite.columns:
                st.subheader("Statut des parents")
                parent_counts = identite["Parents Salariés"].value_counts()
                fig = px.pie(parent_counts, names=parent_counts.index, values=parent_counts.values,
                             title="Parents Salariés", color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 2️⃣ Clinique
    # ============================
    elif section == "Clinique":
        st.header("🩺 Données cliniques")
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)

            # Type de drépanocytose
            if 'Type de drépanocytose' in drepano.columns:
                st.subheader("Type de drépanocytose")
                st.table(drepano['Type de drépanocytose'].value_counts())

            # Prise en charge
            if 'Prise en charge' in drepano.columns:
                st.subheader("Prise en charge")
                prise_counts = drepano['Prise en charge'].value_counts()
                fig = px.pie(prise_counts, names=prise_counts.index, values=prise_counts.values,
                             title="Prise en charge", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textinfo='percent+label', pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 3️⃣ Temporel
    # ============================
    elif section == "Temporel":
        st.header("⏱️ Données temporelles")
        toutes_dates = concat_dates_urgences(feuilles)
        if not toutes_dates.empty:
            repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
            mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                         7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}

            repartition_df = pd.DataFrame({
                'Mois':[mois_noms[m] for m in repartition_mensuelle.index],
                'Nombre de consultations': repartition_mensuelle.values
            })
            st.subheader("Répartition mensuelle des urgences")
            fig = px.line(repartition_df, x='Mois', y='Nombre de consultations',
                          markers=True, title="Consultations mensuelles")
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 4️⃣ Biomarqueurs
    # ============================
    elif section == "Biomarqueurs":
        st.header("🧬 Paramètres biologiques")
        if 'Drépano' in feuilles:
            drepano = feuilles['Drépano']
            drepano = convertir_df_oui_non(drepano)
            bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                        "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]

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
