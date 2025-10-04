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
        df_seg = pd.read_excel("segmentation.xlsx")
    except:
        st.error("Impossible de charger segmentation.xlsx")
        return

    # ============================
    # Onglets horizontaux
    # ============================
    onglets = st.tabs(["Démographique", "Clinique", "Temporel", "Biomarqueurs"])

    # ============================
    # Classification des variables par onglet
    # ============================
    demographiques = ["Sexe","Origine Géographique","Statut des parents (Vivants/Décédés)",
                      "Parents Salariés","Prise en charge","Scolarité","Niveau d'instruction scolarité"]
    cliniques = ["Type de drépanocytose","Taux d'Hb (g/dL)","% d'Hb F","% d'Hb S","% d'HB C",
                 "Nbre de GB (/mm3)","% d'HB A2","Nbre de PLT (/mm3)","GsRh",
                 "Âge de début des signes (en mois)","Âge de découverte de la drépanocytose (en mois)",
                "Âge début de suivi du traitement (en mois)",
                 "L'hydroxyurée","Echange transfusionnelle","Prophylaxie à la pénicilline",
                 "Nbre d'hospitalisations avant 2017","Nbre d'hospitalisations entre 2017 et 2023",
                 "HDJ","CVO","Anémie","AVC","STA","Priapisme","Infections",
                 "Nbre de transfusion avant 2017","Nbre de transfusion Entre 2017 et 2023","Ictère"]
    temporelles = ["Date d'inclusion"]
    biomarqueurs = ["Taux d'Hb (g/dL)","% d'Hb F","% d'Hb S","% d'HB C","Nbre de GB (/mm3)","Nbre de PLT (/mm3)"]

    onglet_dict = {
        0: demographiques,
        1: cliniques,
        2: temporelles,
        3: biomarqueurs
    }

    # ============================
    # Boucle sur les onglets
    # ============================
    for i, onglet in enumerate(onglets):
        with onglet:
            st.header(f"Variables : {['Démographique','Clinique','Temporel','Biomarqueurs'][i]}")
            variables = onglet_dict[i]
            variables = [v for v in variables if v in df_seg.columns]

            # Choix variable
            var_choisie = st.selectbox("Choisissez une variable à afficher", variables)

            if var_choisie:
                # Si variable qualitative
                if df_seg[var_choisie].dtype == 'object' or df_seg[var_choisie].nunique() < 10:
                    counts = df_seg[var_choisie].value_counts()
                    fig = px.pie(counts, names=counts.index, values=counts.values,
                                 title=f"Répartition de {var_choisie}")
                    fig.update_traces(textinfo="percent+label", pull=0.05)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Variable quantitative
                    fig = px.histogram(df_seg, x=var_choisie, nbins=20, title=f"Distribution de {var_choisie}")
                    st.plotly_chart(fig, use_container_width=True)
