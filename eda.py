# utils/eda.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def style_table(df):
    """Applique un style joli aux DataFrames affichés dans Streamlit."""
    return df.style.set_properties(**{
        'background-color': '#f9f9f9',
        'color': '#2c3e50',
        'border-color': 'white'
    }).set_table_styles(
        [{'selector': 'th', 'props': [('background-color', '#2E86C1'),
                                      ('color', 'white'),
                                      ('font-weight', 'bold')]}]
    )

# ============================
# Page Streamlit
# ============================
def show_eda():
    st.markdown("<h1 style='text-align:center;color:#2E86C1;'>📊 Analyse exploratoire des données</h1>", unsafe_allow_html=True)
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"

    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
    except FileNotFoundError:
        st.warning("⚠️ Impossible de charger la base principale. Vérifie le fichier.")
        return

    # ============================
    # Identité
    # ============================
    if 'Identite' in feuilles:
        identite = convertir_df_oui_non(feuilles['Identite'], exclude_columns=["Niveau d'instruction scolarité"])
        st.markdown("## 👤 Identité des patients")

        # Sexe
        if 'Sexe' in identite.columns:
            sexe_counts = identite['Sexe'].value_counts()
            fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                         title="Répartition par sexe", color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textinfo='percent+label', pull=0.05, hoverinfo="label+percent+value")
            st.plotly_chart(fig, use_container_width=True)

        # Origine géographique
        if 'Origine Géographique' in identite.columns:
            origine_counts = identite['Origine Géographique'].value_counts()
            fig = px.pie(origine_counts, names=origine_counts.index, values=origine_counts.values,
                         title="Répartition par origine géographique", color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_traces(textinfo='percent+label', pull=0.05, hoverinfo="label+percent+value")
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Drépanocytose
    # ============================
    if 'Drépano' in feuilles:
        drepano = convertir_df_oui_non(feuilles['Drépano'])
        st.markdown("## 🧬 Drépanocytose")

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
            st.write("### 📌 Paramètres biologiques (statistiques descriptives)")
            stats_df = pd.DataFrame(stats_data).T.round(2)
            st.dataframe(style_table(stats_df))

    # ============================
    # Analyse binaire
    # ============================
    st.markdown("## ⚖️ Analyse binaire (Evolution vs autres variables)")

    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
        cible = "Evolution"

        if cible in df_nettoye.columns:
            variables_interessantes = [
                "Type de drépanocytose",
                "Sexe",
                "Âge du debut d etude en mois (en janvier 2023)",
                "Origine Géographique",
                "Taux d'Hb (g/dL)",
                "% d'Hb F",
                "% d'Hb S",
                "% d'HB C",
                "Nbre de GB (/mm3)",
                "Nbre de PLT (/mm3)"
            ]

            for var in variables_interessantes:
                if var in df_nettoye.columns:
                    st.subheader(f"📌 {var} vs Evolution")

                    # Cas 1 : variable catégorielle
                    if df_nettoye[var].dtype == "object":
                        cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index") * 100
                        st.dataframe(style_table(cross_tab.round(2)))

                        fig = px.bar(
                            cross_tab,
                            barmode="group",
                            title=f"Répartition de Evolution selon {var}",
                            text_auto=".2f",
                            labels={"value": "Pourcentage (%)", "index": var, "Evolution": "Evolution"},
                            color_discrete_sequence=px.colors.sequential.Blues
                        )
                        fig.update_traces(hovertemplate='%{x}<br>%{y:.2f}%')
                        st.plotly_chart(fig, use_container_width=True)

                    # Cas 2 : variable numérique
                    else:
                        df_nettoye[var] = pd.to_numeric(df_nettoye[var], errors='coerce')
                        stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
                        st.dataframe(style_table(stats_group))

    except FileNotFoundError:
        st.info("ℹ️ Le fichier 'fichier_nettoye.xlsx' n’a pas été trouvé. Place-le à la racine du projet.")
