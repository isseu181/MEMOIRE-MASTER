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
    """Concatène toutes les dates des urgences dans une seule DataFrame."""
    toutes_dates = pd.DataFrame()
    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom].copy()
            date_cols = [c for c in df_urg.columns if "date" in c.lower()]
            if date_cols:
                col_date = date_cols[0]
                df_urg[col_date] = pd.to_datetime(df_urg[col_date], errors='coerce')
                df_urg = df_urg.dropna(subset=[col_date])
                df_urg['Mois'] = df_urg[col_date].dt.month
                df_urg['Diagnostic'] = df_urg.get('Type de drépanocytose', "Non défini")
                toutes_dates = pd.concat([toutes_dates, df_urg[['Mois','Diagnostic']]], ignore_index=True)
    return toutes_dates

# ============================
# Page Streamlit principale
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
    # Menu principal
    # ============================
    menu_principal = st.sidebar.radio("Sélectionnez une catégorie", 
                                      ["Démographie", "Clinique", "Temporel", "Biomarqueurs", "Diagnostic vs Evolution"])

    # ============================
    # Démographie
    # ============================
    if menu_principal == "Démographie":
        identite = feuilles['Identite'].copy()
        identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
        sous_menu = st.sidebar.radio("Sous-menu", ["Vue générale", "Par sexe", "Par origine géographique", "Scolarité"])
        
        if sous_menu == "Vue générale":
            st.header("👥 Vue générale")
            st.write(f"Nombre total de patients : {len(identite)}")
        
        elif sous_menu == "Par sexe":
            st.header("🧑‍🤝‍🧑 Répartition par sexe")
            fig = px.pie(identite, names='Sexe', title="Répartition par sexe", color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        
        elif sous_menu == "Par origine géographique":
            st.header("🌍 Répartition par origine géographique")
            fig = px.pie(identite, names='Origine Géographique', title="Origine géographique", color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
        
        elif sous_menu == "Scolarité":
            st.header("🏫 Scolarité des enfants")
            if "Niveau d'instruction scolarité" in identite.columns:
                fig = px.pie(identite, names="Niveau d'instruction scolarité", title="Répartition par scolarité", color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Clinique
    # ============================
    elif menu_principal == "Clinique":
        drepano = feuilles['Drépano'].copy()
        drepano = convertir_df_oui_non(drepano)
        sous_menu = st.sidebar.radio("Sous-menu", ["Type de drépanocytose", "Prise en charge"])
        
        if sous_menu == "Type de drépanocytose":
            st.header("🧬 Type de drépanocytose")
            fig = px.histogram(drepano, x='Type de drépanocytose', title="Répartition des types de drépanocytose", color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig, use_container_width=True)
        
        elif sous_menu == "Prise en charge":
            st.header("💊 Prise en charge")
            prise_cols = ["Prise en charge","Prophylaxie à la pénicilline","L'hydroxyurée","Echange transfusionnelle"]
            df_pris = drepano[prise_cols].copy()
            for col in prise_cols:
                st.subheader(col)
                fig = px.pie(df_pris, names=col, title=f"{col}", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Temporel
    # ============================
    elif menu_principal == "Temporel":
        toutes_dates = concat_dates_urgences(feuilles)
        if toutes_dates.empty:
            st.warning("Aucune date d'urgence disponible.")
        else:
            sous_menu = st.sidebar.radio("Sous-menu", ["Par mois", "Par type de diagnostic"])
            if sous_menu == "Par mois":
                st.header("📅 Répartition mensuelle des urgences")
                repartition = toutes_dates['Mois'].value_counts().sort_index()
                mois_dict = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                             7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
                df_plot = pd.DataFrame({'Mois':[mois_dict[m] for m in repartition.index], 'Consultations': repartition.values})
                fig = px.bar(df_plot, x='Mois', y='Consultations', text='Consultations', title="Répartition mensuelle")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            elif sous_menu == "Par type de diagnostic":
                st.header("🧾 Répartition par type de diagnostic")
                diag_counts = toutes_dates.groupby(['Diagnostic','Mois']).size().reset_index(name='Counts')
                diag_counts['Mois'] = diag_counts['Mois'].map({1:'Janv',2:'Fév',3:'Mars',4:'Avr',5:'Mai',6:'Juin',
                                                               7:'Juil',8:'Août',9:'Sept',10:'Oct',11:'Nov',12:'Déc'})
                fig = px.line(diag_counts, x='Mois', y='Counts', color='Diagnostic', markers=True, title="Évolution des diagnostics par mois")
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Biomarqueurs
    # ============================
    elif menu_principal == "Biomarqueurs":
        drepano = feuilles['Drépano'].copy()
        drepano = convertir_df_oui_non(drepano)
        sous_menu = st.sidebar.radio("Sous-menu", ["Paramètres biologiques", "Évolution"])
        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]

        if sous_menu == "Paramètres biologiques":
            st.header("🧪 Paramètres biologiques")
            stats = drepano[bio_cols].agg(["mean","median","min","max"]).round(2)
            st.table(stats)
        
        elif sous_menu == "Évolution":
            st.header("📈 Évolution par biomarqueurs")
            cible = "Evolution"
            try:
                df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
                if cible in df_nettoye.columns:
                    for col in bio_cols:
                        if col in df_nettoye.columns:
                            fig = px.box(df_nettoye, x=cible, y=col, points="all", title=f"{col} vs {cible}")
                            st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError:
                st.warning("Fichier 'fichier_nettoye.xlsx' introuvable.")

    # ============================
    # Diagnostic vs Evolution
    # ============================
    elif menu_principal == "Diagnostic vs Evolution":
        st.header("🧬 Diagnostic catégorisé vs Evolution")
        try:
            df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
            if "Evolution" in df_nettoye.columns and "Type de drépanocytose" in df_nettoye.columns:
                cross_tab = pd.crosstab(df_nettoye["Type de drépanocytose"], df_nettoye["Evolution"], normalize='index')*100
                st.dataframe(cross_tab.round(2))
                fig = px.bar(cross_tab, barmode="group", text_auto=".2f", title="Diagnostic vs Evolution")
                st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("Fichier 'fichier_nettoye.xlsx' introuvable.")
