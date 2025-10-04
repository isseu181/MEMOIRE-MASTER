i# eda.py
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

    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
    except:
        st.warning("Impossible de charger fichier_nettoye.xlsx")
        df_nettoye = pd.DataFrame()  # DataFrame vide pour éviter les erreurs

    # Conversion Oui/Non en binaire
    df_seg = convertir_df_oui_non(df_seg)
    if not df_nettoye.empty:
        df_nettoye = convertir_df_oui_non(df_nettoye)

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
                 "Circonstances Courants","Autres (nommer)","Âge début de suivi du traitement (en mois)",
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
                # Univariée : Qualitative
                if df_seg[var_choisie].dtype == 'object' or df_seg[var_choisie].nunique() < 10:
                    counts = df_seg[var_choisie].value_counts()
                    fig = px.pie(counts, names=counts.index, values=counts.values,
                                 title=f"Répartition de {var_choisie}")
                    fig.update_traces(textinfo="percent+label", pull=0.05)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Univariée : Quantitative
                    fig = px.histogram(df_seg, x=var_choisie, nbins=20, title=f"Distribution de {var_choisie}")
                    st.plotly_chart(fig, use_container_width=True)

            # Bivariée : Clinique vs Evolution
            if i == 1 and not df_nettoye.empty and "Evolution" in df_nettoye.columns:
                st.subheader(f"Analyse bivariée : {var_choisie} vs Evolution")
                if var_choisie in df_nettoye.columns:
                    if df_nettoye[var_choisie].dtype == 'object' or df_nettoye[var_choisie].nunique() < 10:
                        pivot = pd.crosstab(df_nettoye[var_choisie], df_nettoye["Evolution"])
                        fig_bi = px.bar(pivot, barmode="group", title=f"{var_choisie} vs Evolution")
                        st.plotly_chart(fig_bi, use_container_width=True)

    # ============================
    # Onglet Temporel
    # ============================
    with onglets[2]:
        st.header("3️⃣ Analyse temporelle")
        mois_ordre = ["Janvier","Février","Mars","Avril","Mai","Juin",
                      "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

        if not df_nettoye.empty:
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

            # Consultations par NiveauUrgence
            if "NiveauUrgence" in df_nettoye.columns:
                urgence_counts = df_nettoye["NiveauUrgence"].value_counts().reset_index()
                urgence_counts.columns = ["NiveauUrgence", "Nombre"]
                fig_urgence = px.bar(urgence_counts, x="NiveauUrgence", y="Nombre",
                                     title="Consultations par Niveau d'Urgence")
                st.plotly_chart(fig_urgence, use_container_width=True)

