# utils/eda.py
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

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
    toutes_dates = pd.Series(dtype='datetime64[ns]')
    for i in range(1, 7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom]
            col_date_candidates = [c for c in df_urg.columns if 'date' in c.lower()]
            if col_date_candidates:
                col_date = col_date_candidates[0]
                dates = pd.to_datetime(df_urg[col_date], errors='coerce').dropna()
                toutes_dates = pd.concat([toutes_dates, dates])
    return toutes_dates

def show_eda():
    # ========== Style de page ==========
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .section {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            color: #2E86C1;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Dashboard : Analyse USAD Drépanocytose")
    st.markdown("Bienvenue dans la section d'exploration des données")

    # Charger les feuilles Excel
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"
    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        st.warning("⚠️ Impossible de charger la base de données.")
        return

    # — Section : Identité —
    if 'Identite' in feuilles:
        st.markdown("## Identité des patients")
        identite = convertir_df_oui_non(feuilles['Identite'], exclude_columns=["Niveau d'instruction scolarité"])

        cols = st.columns(2)
        with cols[0]:
            if 'Sexe' in identite.columns:
                counts = identite['Sexe'].value_counts()
                fig = px.pie(counts, names=counts.index, values=counts.values,
                             title="Répartition par sexe")
                fig.update_traces(textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            if 'Origine Géographique' in identite.columns:
                counts2 = identite['Origine Géographique'].value_counts()
                fig2 = px.pie(counts2, names=counts2.index, values=counts2.values,
                              title="Origine géographique")
                fig2.update_traces(textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)

    # — Section : Paramètres biologiques —
    if 'Drépano' in feuilles:
        st.markdown("## Paramètres biologiques")
        drepano = convertir_df_oui_non(feuilles['Drépano'])
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

    # — Section : Urgences et symptômes —
    st.markdown("## Consultations d'urgence")
    symptomes = ['Douleur', 'Fièvre', 'Pâleur', 'Ictère', 'Toux']
    for i in range(1, 7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = convertir_df_oui_non(feuilles[nom])
            date_cols = [c for c in df_urg.columns if "date" in c.lower()]
            if date_cols:
                df_urg = df_urg[df_urg[date_cols[0]].notna()]

            st.markdown(f"### {nom} — {len(df_urg)} visites")
            data_sym = {}
            for s in symptomes:
                if s in df_urg.columns and not df_urg[s].dropna().empty:
                    data_sym[s] = df_urg[s].value_counts().to_dict()
            if data_sym:
                st.table(pd.DataFrame(data_sym).fillna(0).astype(int))

    # — Section : Répartition mensuelle des urgences —
    st.markdown("## Répartition mensuelle des urgences")
    toutes_dates = concat_dates_urgences(feuilles)
    if not toutes_dates.empty:
        rep = toutes_dates.dt.month.value_counts().sort_index()
        mois = {
            1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
            7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'
        }
        df_rep = pd.DataFrame({
            'Mois': [mois[m] for m in rep.index],
            'Nombre': rep.values
        })
        figm = px.bar(df_rep, x='Mois', y='Nombre', text='Nombre',
                      title="Répartition mensuelle des urgences")
        figm.update_traces(textposition="outside")
        st.plotly_chart(figm, use_container_width=True)

    # — Section : Analyse binaire (Evolution) —
    st.markdown("## Analyse binaire : Evolution")
    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
        cible = "Evolution"
        if cible in df_nettoye.columns:
            variables_interessantes = [
                "Type de drépanocytose", "Sexe",
                "Âge du debut d etude en mois (en janvier 2023)",
                "Origine Géographique", "Taux d'Hb (g/dL)",
                "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
            ]
            for var in variables_interessantes:
                if var in df_nettoye.columns:
                    st.markdown(f"### {var} vs Evolution")
                    # catégorielle
                    if df_nettoye[var].dtype == "object":
                        ctab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index") * 100
                        st.dataframe(ctab.round(2))
                        fig = px.bar(ctab, barmode="group", text_auto=".2f",
                                     labels={"value":"%","index":var, "Evolution":"Evolution"})
                        fig.update_traces(hovertemplate='%{x}<br>%{y:.2f}%')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        df_nettoye[var] = pd.to_numeric(df_nettoye[var], errors='coerce')
                        stats_group = df_nettoye.groupby(cible)[var].agg(["mean","median","min","max"]).round(2)
                        st.table(stats_group)
    except FileNotFoundError:
        pass
