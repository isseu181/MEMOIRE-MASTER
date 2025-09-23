# utils/eda.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    file_path = "Base_de_données_USAD_URGENCES1.xlsx"

    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
        st.success("Fichier chargé avec succès !")
    except FileNotFoundError:
        st.error(f"Fichier introuvable. Assurez-vous que '{file_path}' est à la racine du projet.")
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
            sexe_counts = identite['Sexe'].value_counts().to_dict()
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(sexe_counts.values()), labels=list(sexe_counts.keys()),
                   autopct="%1.1f%%", startangle=140)
            ax.set_title("Répartition par sexe")
            st.pyplot(fig)

        # Origine géographique
        if 'Origine Géographique' in identite.columns:
            origine_counts = identite['Origine Géographique'].value_counts().to_dict()
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(origine_counts.values()), labels=list(origine_counts.keys()),
                   autopct="%1.1f%%", startangle=140)
            ax.set_title("Répartition par origine géographique")
            st.pyplot(fig)

        # Âge
        age_col = "Âge du debut d etude en mois (en janvier 2023)"
        if age_col in identite.columns:
            fig, ax = plt.subplots(figsize=(8,6))
            identite[age_col] = pd.to_numeric(identite[age_col], errors='coerce')
            identite[age_col].dropna().hist(bins=20, color="#36A2EB", edgecolor="white", ax=ax)
            ax.set_title("Répartition des âges à l’inclusion")
            ax.set_xlabel("Âge (mois)")
            ax.set_ylabel("Nombre de patients")
            st.pyplot(fig)

    # ============================
    # 2️⃣ Drépanocytose
    # ============================
    if 'Drépano' in feuilles:
        drepano = feuilles['Drépano']
        drepano = convertir_df_oui_non(drepano)

        if 'Type de drépanocytose' in drepano.columns:
            st.header("2️⃣ Type de drépanocytose et paramètres biologiques")
            type_counts = drepano['Type de drépanocytose'].value_counts().to_dict()
            st.write("Type de drépanocytose:", type_counts)

        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                    "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        st.subheader("Paramètres biologiques")
        for col in bio_cols:
            if col in drepano.columns:
                fig, ax = plt.subplots(figsize=(8,4))
                drepano[col] = pd.to_numeric(drepano[col], errors='coerce')
                drepano[col].dropna().hist(bins=20, color="#FF6384", edgecolor="white", ax=ax)
                ax.set_title(col)
                st.pyplot(fig)

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

            # Affiche les symptômes
            for s in symptomes:
                if s in df_urg.columns and not df_urg[s].dropna().empty:
                    counts = df_urg[s].value_counts().to_dict()
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.bar(counts.keys(), counts.values(), color="#36A2EB")
                    ax.set_title(f"{s} (patients présents)")
                    ax.set_ylabel("Nombre de patients")
                    ax.set_xlabel(s)
                    st.pyplot(fig)

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
        st.write(repartition_df)

        fig, ax = plt.subplots(figsize=(10,5))
        repartition_df.set_index('Mois')['Nombre de consultations'].plot(kind='bar', color='#36A2EB', ax=ax)
        ax.set_ylabel("Nombre de consultations")
        ax.set_xlabel("Mois")
        ax.set_title("Répartition mensuelle des urgences drépanocytaires")
        st.pyplot(fig)
