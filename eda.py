# utils/eda.py
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
    # 1️⃣ Identité
    # ============================
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
    # 2️⃣ Drépanocytose
    # ============================
    if 'Drépano' in feuilles:
        drepano = feuilles['Drépano']
        drepano = convertir_df_oui_non(drepano)

        if 'Type de drépanocytose' in drepano.columns:
            st.header("2️⃣ Type de drépanocytose et paramètres biologiques")
            type_counts = drepano['Type de drépanocytose'].value_counts()
            st.table(type_counts)

        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                    "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        st.subheader("Paramètres biologiques")
        for col in bio_cols:
            if col in drepano.columns:
                drepano[col] = pd.to_numeric(drepano[col], errors='coerce')
                fig = px.histogram(drepano, x=col, nbins=15,
                                   title=col,
                                   color_discrete_sequence=["#E74C3C"])
                fig.update_traces(texttemplate="%{y}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 4️⃣ Consultations d'urgence
    # ============================
    st.header("4️⃣ Consultations d'urgence")
    symptomes = ['Douleur','Fièvre','Pâleur','Ictère','Toux']

    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom]
            df_urg = convertir_df_oui_non(df_urg)

            # Filtre seulement les patients venus (date non vide)
            date_col_candidates = [c for c in df_urg.columns if "date" in c.lower()]
            if date_col_candidates:
                df_urg = df_urg[df_urg[date_col_candidates[0]].notna()]

            st.subheader(f"{nom} - Nombre de consultations : {len(df_urg)}")

            # Affiche les symptômes (tableau seulement, pas de graphiques)
            data_symptomes = {}
            for s in symptomes:
                if s in df_urg.columns and not df_urg[s].dropna().empty:
                    counts = df_urg[s].value_counts().to_dict()
                    data_symptomes[s] = counts
            if data_symptomes:
                st.table(pd.DataFrame(data_symptomes).fillna(0).astype(int))

    # ============================
    # 5️⃣ Répartition mensuelle des urgences
    # ============================
    st.header("5️⃣ Répartition mensuelle des urgences")
    toutes_dates = concat_dates_urgences(feuilles)
    if not toutes_dates.empty:
        repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
        mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                     7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}

        repartition_df = pd.DataFrame({
            'Mois':[mois_noms[m] for m in repartition_mensuelle.index],
            'Nombre de consultations': repartition_mensuelle.values
        })
        repartition_df['Pourcentage (%)'] = (repartition_df['Nombre de consultations'] /
                                            repartition_df['Nombre de consultations'].sum()*100).round(2)
        st.table(repartition_df)

        fig = px.bar(repartition_df, x='Mois', y='Nombre de consultations',
                     title="Répartition mensuelle des urgences drépanocytaires",
                     text='Nombre de consultations',
                     color_discrete_sequence=["#2E86C1"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 6️⃣ Analyse binaire (Evolution vs autres variables)
    # ============================
    st.header("6️⃣ Analyse binaire : Evolution vs autres variables")

    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
        st.success("✅ Fichier 'fichier_nettoye.xlsx' chargé avec succès pour l'analyse binaire !")

        if "Evolution" in df_nettoye.columns:
            cible = "Evolution"

            # Variables explicatives intéressantes
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
                    st.subheader(f"📌 {var} vs {cible}")

                    # Cas 1 : variable catégorielle
                    if df_nettoye[var].dtype == "object":
                        cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index") * 100
                        st.write("Tableau croisé (%)")
                        st.dataframe(cross_tab.round(2))

                        fig = px.bar(
                            cross_tab,
                            barmode="group",
                            title=f"Répartition de {cible} selon {var}",
                            text_auto=".2f",
                            labels={"value": "Pourcentage (%)", "index": var, "Evolution": "Evolution"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Cas 2 : variable numérique
                    else:
                        fig = px.box(
                            df_nettoye,
                            x=cible,
                            y=var,
                            points="all",
                            title=f"Distribution de {var} selon {cible}",
                            labels={var: var, "Evolution": "Evolution"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("⚠️ La variable cible 'Evolution' est introuvable dans fichier_nettoye.xlsx.")

    except FileNotFoundError:
        st.warning("⚠️ Le fichier 'fichier_nettoye.xlsx' est introuvable. Place-le à la racine du projet.")
