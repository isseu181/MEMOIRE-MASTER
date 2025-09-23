# utils/eda.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

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
    st.subheader("📊 Analyse exploratoire des données")

    # Chemin automatique du fichier Excel dans le même dossier que App.py
    file_path = os.path.join(os.path.dirname(__file__), "../Base_de_données_USAD_URGENCES1.xlsx")

    if not os.path.exists(file_path):
        st.error(f"Fichier introuvable : {file_path}")
        return

    # Lecture de toutes les feuilles
    feuilles = pd.read_excel(file_path, sheet_name=None)

    st.success("Fichier chargé avec succès !")

    # ----------------------------
    # Feuilles à afficher
    # ----------------------------
    feuilles_utiles = ['Identite', 'Drépano', 'Antéccédents'] + [f'Urgence{i}' for i in range(1,7)]

    for nom in feuilles_utiles:
        if nom not in feuilles:
            st.warning(f"Feuille manquante : {nom}")

    # ----------------------------
    # 1️⃣ Identité
    # ----------------------------
    if 'Identite' in feuilles:
        identite = feuilles['Identite']
        identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
        st.markdown("### 1️⃣ Identité des patients")
        st.write("Nombre total de patients:", len(identite))

        # Sexe
        if 'Sexe' in identite.columns:
            sexe_counts = identite['Sexe'].value_counts().to_dict()
            st.write("Répartition par sexe:", sexe_counts)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(sexe_counts.values()), labels=list(sexe_counts.keys()), autopct="%1.1f%%", startangle=140)
            ax.set_title("Répartition par sexe")
            st.pyplot(fig)

        # Origine géographique
        if 'Origine Géographique' in identite.columns:
            origine_counts = identite['Origine Géographique'].value_counts().to_dict()
            st.write("Répartition par origine géographique:", origine_counts)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(origine_counts.values()), labels=list(origine_counts.keys()), autopct="%1.1f%%", startangle=140)
            ax.set_title("Répartition par origine géographique")
            st.pyplot(fig)

        # Âge
        age_col = "Âge du debut d etude en mois (en janvier 2023)"
        if age_col in identite.columns:
            st.markdown("#### Âge à l’inclusion")
            fig, ax = plt.subplots(figsize=(8,6))
            identite[age_col].dropna().hist(bins=20, color="#36A2EB", edgecolor="white", ax=ax)
            ax.set_title("Répartition des âges à l’inclusion")
            ax.set_xlabel("Âge (mois)")
            ax.set_ylabel("Nombre de patients")
            st.pyplot(fig)

    # ----------------------------
    # 2️⃣ Drépanocytose
    # ----------------------------
    if 'Drépano' in feuilles:
        drepano = feuilles['Drépano']
        drepano = convertir_df_oui_non(drepano)

        # Type de drépanocytose
        if 'Type de drépanocytose' in drepano.columns:
            type_counts = drepano['Type de drépanocytose'].value_counts().to_dict()
            st.markdown("### 2️⃣ Type de drépanocytose")
            st.write(type_counts)

        # Paramètres biologiques
        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        st.markdown("#### Paramètres biologiques")
        for col in bio_cols:
            if col in drepano.columns:
                fig, ax = plt.subplots(figsize=(8,4))
                drepano[col].dropna().hist(bins=20, color="#36A2EB", edgecolor='white', ax=ax)
                ax.set_title(col)
                st.pyplot(fig)

    # ----------------------------
    # 3️⃣ Antécédents médicaux
    # ----------------------------
    if 'Antéccédents' in feuilles:
        antecedents = feuilles['Antéccédents']
        antecedents = convertir_df_oui_non(antecedents)
        bin_cols = [col for col in antecedents.columns if set(antecedents[col].dropna().unique()).issubset({0,1})]
        st.markdown("### 3️⃣ Antécédents médicaux")
        for col in bin_cols:
            counts = antecedents[col].value_counts().to_dict()
            st.write(f"{col}: {counts}")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(counts.keys(), counts.values(), color="#36A2EB")
            st.pyplot(fig)

    # ----------------------------
    # 4️⃣ Consultations d'urgence
    # ----------------------------
    st.markdown("### 4️⃣ Consultations d'urgence")
    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom]
            df_urg = convertir_df_oui_non(df_urg)
            st.markdown(f"#### {nom}")
            st.write("Nombre de consultations:", len(df_urg))

            symptomes = ['Douleur', 'Fièvre', 'Pâleur', 'Ictère', 'Toux']
            for s in symptomes:
                if s in df_urg.columns:
                    counts = df_urg[s].value_counts().to_dict()
                    st.write(f"{s}: {counts}")

    # ----------------------------
    # 5️⃣ Répartition mensuelle des urgences
    # ----------------------------
    st.markdown("### 📅 Répartition mensuelle des urgences")
    toutes_dates = concat_dates_urgences(feuilles)
    if not toutes_dates.empty:
        repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
        mois_noms = {
            1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
            7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
        }

        repartition_df = pd.DataFrame({
            'Mois': [mois_noms[m] for m in repartition_mensuelle.index],
            'Nombre de consultations': repartition_mensuelle.values
        })
        repartition_df['Pourcentage (%)'] = (repartition_df['Nombre de consultations'] / repartition_df['Nombre de consultations'].sum() * 100).round(2)
        st.write(repartition_df)

        fig, ax = plt.subplots(figsize=(10,5))
        repartition_df.set_index('Mois')['Nombre de consultations'].plot(kind='bar', color='#36A2EB', ax=ax)
        ax.set_ylabel("Nombre de consultations")
        ax.set_xlabel("Mois")
        ax.set_title("Répartition mensuelle des urgences drépanocytaires")
        st.pyplot(fig)
    else:
        st.write("Aucune donnée de date disponible pour les urgences.")
