# eda.py
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
            ["Oui", "Non", "OUI", "NON", "oui", "non", "O", "N"]
        ).any():
            df[col] = df[col].apply(oui_non_vers_binaire)
    return df


def concat_dates_urgences(feuilles):
    """Concatène toutes les dates des urgences dans une seule série."""
    toutes_dates = pd.Series(dtype="datetime64[ns]")
    for i in range(1, 7):
        nom = f"Urgence{i}"
        if nom in feuilles:
            df_urg = feuilles[nom]
            col_date_candidates = [col for col in df_urg.columns if "date" in col.lower()]
            if col_date_candidates:
                col_date = col_date_candidates[0]
                dates = pd.to_datetime(df_urg[col_date], errors="coerce").dropna()
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
    except FileNotFoundError:
        return

    # ============================
    # Création des onglets
    # ============================
    onglets = st.tabs(
        ["Démographique", "Clinique", "Temporel", "Biomarqueurs", "Analyse bivariée"]
    )

    # ---------------- DÉMOGRAPHIQUE ----------------
    with onglets[0]:
        if "Identite" in feuilles:
            identite = feuilles["Identite"]
            identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])

            st.header("1️⃣ Identité des patients")
            st.write("Nombre total de patients :", len(identite))

            # Sexe
            if "Sexe" in identite.columns:
                fig = px.pie(
                    identite,
                    names="Sexe",
                    title="Répartition par sexe",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                )
                fig.update_traces(textinfo="percent+label", pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Origine géographique
            if "Origine Géographique" in identite.columns:
                fig = px.pie(
                    identite,
                    names="Origine Géographique",
                    title="Répartition par origine géographique",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                )
                fig.update_traces(textinfo="percent+label", pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Scolarité
            if "Niveau d'instruction scolarité" in identite.columns:
                fig = px.pie(
                    identite,
                    names="Niveau d'instruction scolarité",
                    title="Répartition de la scolarisation",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig.update_traces(textinfo="percent+label", pull=0.05)
                st.plotly_chart(fig, use_container_width=True)

            # Âge
            age_col = "Âge du debut d etude en mois (en janvier 2023)"
            if age_col in identite.columns:
                identite[age_col] = pd.to_numeric(identite[age_col], errors="coerce")
                fig = px.histogram(
                    identite,
                    x=age_col,
                    nbins=15,
                    title="Répartition des âges à l’inclusion",
                    color_discrete_sequence=["#2E86C1"],
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---------------- CLINIQUE ----------------
    with onglets[1]:
        if "Drépano" in feuilles:
            drepano = feuilles["Drépano"]
            drepano = convertir_df_oui_non(drepano)

            st.header("2️⃣ Données cliniques")

            if "Type de drépanocytose" in drepano.columns:
                type_counts = drepano["Type de drépanocytose"].value_counts()
                st.table(type_counts)

            symptomes = ["Douleur", "Fièvre", "Pâleur", "Ictère", "Toux"]
            for i in range(1, 7):
                nom = f"Urgence{i}"
                if nom in feuilles:
                    df_urg = feuilles[nom]
                    df_urg = convertir_df_oui_non(df_urg)
                    st.subheader(f"{nom} - Nombre de consultations : {len(df_urg)}")
                    data_symptomes = {}
                    for s in symptomes:
                        if s in df_urg.columns:
                            counts = df_urg[s].value_counts().to_dict()
                            data_symptomes[s] = counts
                    if data_symptomes:
                        st.table(pd.DataFrame(data_symptomes).fillna(0).astype(int))

    # ---------------- TEMPOREL ----------------
    with onglets[2]:
        st.header("3️⃣ Répartition temporelle des urgences")
        toutes_dates = concat_dates_urgences(feuilles)
        if not toutes_dates.empty:
            repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
            mois_noms = {
                1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
                5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
                9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
            }
            repartition_df = pd.DataFrame({
                "Mois": [mois_noms[m] for m in repartition_mensuelle.index],
                "Nombre de consultations": repartition_mensuelle.values,
            })
            fig = px.line(
                repartition_df,
                x="Mois",
                y="Nombre de consultations",
                title="Répartition mensuelle des urgences drépanocytaires",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- BIOMARQUEURS ----------------
    with onglets[3]:
        if "Drépano" in feuilles:
            drepano = feuilles["Drépano"]
            drepano = convertir_df_oui_non(drepano)

            st.header("4️⃣ Biomarqueurs")
            bio_cols = [
                "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
            ]

            for col in bio_cols:
                if col in drepano.columns:
                    drepano[col] = pd.to_numeric(drepano[col], errors="coerce")
                    fig = px.histogram(
                        drepano,
                        x=col,
                        nbins=15,
                        title=f"Distribution de {col}",
                        color_discrete_sequence=["#16A085"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------- ANALYSE BIVARIÉE ----------------
    with onglets[4]:
        st.header("5️⃣ Analyse bivariée : Evolution vs variables")
        try:
            df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
            cible = "Evolution"

            # Variables qualitatives → 3D
            variables_qualitatives = [
                "Type de drépanocytose", "Sexe",
                "Origine Géographique", "Prise en charge",
                "Diagnostic Catégorisé"
            ]
            for var in variables_qualitatives:
                if var in df_nettoye.columns:
                    cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index") * 100
                    x, y, z = [], [], []
                    for i, cat_var in enumerate(cross_tab.index):
                        for j, cat_cible in enumerate(cross_tab.columns):
                            x.append(cat_var)
                            y.append(cat_cible)
                            z.append(cross_tab.loc[cat_var, cat_cible])
                    fig = go.Figure(data=[go.Bar3d(x=x, y=y, z=z, opacity=0.9)])
                    fig.update_layout(scene=dict(
                        xaxis_title=var, yaxis_title=cible, zaxis_title="Pourcentage"
                    ))
                    st.plotly_chart(fig, use_container_width=True)

            # Variables quantitatives → Boxplots
            variables_quantitatives = [
                "Âge du debut d etude en mois (en janvier 2023)",
                "Taux d'Hb (g/dL)", "% d'Hb F", "Nbre de GB (/mm3)"
            ]
            for var in variables_quantitatives:
                if var in df_nettoye.columns:
                    df_nettoye[var] = pd.to_numeric(df_nettoye[var], errors="coerce")
                    fig = px.box(df_nettoye, x=cible, y=var, points="all",
                                 title=f"Distribution de {var} selon {cible}")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
