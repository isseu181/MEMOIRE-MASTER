import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sickle Insight Engine", layout="wide")
st.title("Sickle Insight Engine - Résultats")

# --- Téléversement du fichier ---
fichier = st.file_uploader("Téléverser le fichier Excel", type=["xlsx"])
if fichier:
    feuilles = pd.read_excel(fichier, sheet_name=None)
    st.success("Fichier chargé avec succès")

    # --- Chargement des feuilles ---
    identité = feuilles['Identite']
    drépanocytose = feuilles['Drépano']
    antécédents = feuilles['Antéccédents']
    urgences = [feuilles[f'Urgence{i}'] for i in range(1,7)]

    # --- Fonctions utilitaires ---
    def oui_non_vers_binaire(valeur):
        if isinstance(valeur, str) and valeur.strip().lower() in ["oui","o"]:
            return 1
        elif isinstance(valeur, str) and valeur.strip().lower() in ["non","n"]:
            return 0
        return valeur

    def convertir_df_oui_non(df, exclude_columns=None):
        exclude_columns = exclude_columns or []
        df = df.copy()
        for col in df.columns:
            if col not in exclude_columns and df[col].isin(["Oui","Non","OUI","NON","oui","non","O","N"]).any():
                df[col] = df[col].apply(oui_non_vers_binaire)
        return df

    def binaire_vers_oui_non(valeur):
        if valeur == 1: return "Oui"
        elif valeur == 0: return "Non"
        return valeur

    # Conversion Oui/Non
    identité = convertir_df_oui_non(identité, exclude_columns=['Niveau d\'instruction scolarité'])
    drépanocytose = convertir_df_oui_non(drépanocytose)
    antécédents = convertir_df_oui_non(antécédents)
    urgences = [convertir_df_oui_non(u) for u in urgences]

    # --- Résumé global ---
    st.header("Résumé global")
    total_patients = len(identité)
    total_urgences = sum(len(u) for u in urgences)
    sexe_counts = identité['Sexe'].value_counts()
    origine_counts = identité['Origine Géographique'].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", total_patients)
    col2.metric("Total Consultations d'Urgence", total_urgences)
    col3.subheader("Répartition par Sexe")
    fig, ax = plt.subplots()
    ax.pie(sexe_counts, labels=sexe_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    col3.pyplot(fig)
    col4.subheader("Répartition par Origine")
    fig, ax = plt.subplots()
    ax.pie(origine_counts, labels=origine_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    col4.pyplot(fig)

    # --- Menu latéral ---
    section = st.sidebar.selectbox(
        "Choisir la section à afficher",
        ["Identité", "Statut vaccinal", "Type de drépanocytose", "Paramètres biologiques",
         "Antécédents médicaux", "Consultations d'urgence", "Répartition mensuelle"]
    )

    # --- Sections détaillées ---
    if section == "Identité":
        st.header("1. Identité du patient")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sexe")
            fig, ax = plt.subplots()
            ax.pie(sexe_counts, labels=sexe_counts.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)
        with col2:
            st.subheader("Origine Géographique")
            fig, ax = plt.subplots()
            ax.pie(origine_counts, labels=origine_counts.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

    elif section == "Statut vaccinal":
        st.header("2. Statut vaccinal")
        vaccins = ['PEV Complet', 'Vaccin contre pneumocoque', 'Vaccin contre méningocoque', 'Vaccin contre Les salmonelles']
        for vaccin in vaccins:
            if vaccin in identité.columns:
                st.subheader(vaccin)
                counts = identité[vaccin].value_counts()
                fig, ax = plt.subplots()
                ax.pie(counts, labels=[binaire_vers_oui_non(k) for k in counts.index], autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                st.pyplot(fig)

    elif section == "Type de drépanocytose":
        st.header("3. Type de drépanocytose")
        type_counts = drépanocytose['Type de drépanocytose'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

    elif section == "Paramètres biologiques":
        st.header("4. Paramètres biologiques")
        paramètres = ['Taux d\'Hb (g/dL)', '% d\'Hb F', '% d\'Hb S', '% d\'HB C', 'Nbre de GB (/mm3)', 'Nbre de PLT (/mm3)']
        fig, ax = plt.subplots(figsize=(12,6))
        moyennes = [drépanocytose[col].mean() for col in paramètres]
        écarts = [drépanocytose[col].std() for col in paramètres]
        ax.bar(paramètres, moyennes, yerr=écarts, capsize=5, color='#36A2EB')
        ax.set_ylabel('Valeur moyenne ± écart-type')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif section == "Antécédents médicaux":
        st.header("5. Antécédents médicaux")
        for col in [c for c in antécédents.columns if c != 'ID patient']:
            st.subheader(col)
            counts = antécédents[col].value_counts()
            if counts.empty:
                st.write("Aucune donnée disponible")
            else:
                fig, ax = plt.subplots()
                ax.pie(counts, labels=[binaire_vers_oui_non(k) for k in counts.index], autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                st.pyplot(fig)

    elif section == "Consultations d'urgence":
        st.header("6. Consultations d’urgence")
        symptômes = ['Douleur', 'Fièvre', 'Pâleur', 'Ictère', 'Toux']
        urgences_noms = [f'Urgence{i}' for i in range(1,7)]
        for symptôme in symptômes:
            counts = [u[symptôme].value_counts().get(1,0) if symptôme in u.columns else 0 for u in urgences]
            fig, ax = plt.subplots()
            ax.plot(urgences_noms, counts, marker='o', label=symptôme)
            ax.set_ylabel("Nombre de cas")
            ax.set_xlabel("Consultation d'urgence")
            ax.set_title(f"Évolution du symptôme : {symptôme}")
            ax.grid(True)
            st.pyplot(fig)

    elif section == "Répartition mensuelle":
        st.header("7. Répartition mensuelle des urgences")
        toutes_dates = pd.Series(dtype='datetime64[ns]')
        for u in urgences:
            col_date_candidates = [c for c in u.columns if "date" in c.lower()]
            if col_date_candidates:
                dates = pd.to_datetime(u[col_date_candidates[0]], errors='coerce').dropna()
                toutes_dates = pd.concat([toutes_dates, dates])
        if not toutes_dates.empty:
            répartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
            mois_noms = {1: 'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                         7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
            fig, ax = plt.subplots(figsize=(12,6))
            répartition_complète = répartition_mensuelle.reindex(range(1,13), fill_value=0)
            répartition_complète.plot(kind='bar', color='#36A2EB', ax=ax)
            ax.set_xticklabels([mois_noms[i] for i in range(1,13)], rotation=45)
            ax.set_ylabel("Nombre de consultations")
            ax.set_title("Répartition mensuelle des urgences")
            st.pyplot(fig)
        else:
            st.write("Aucune donnée de date disponible pour la répartition mensuelle")
