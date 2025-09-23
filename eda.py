import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

sns.set_theme(style="whitegrid")  # thème seaborn pour jolis graphiques

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
        if col not in exclude_columns and df[col].isin(["Oui","Non","OUI","NON","oui","non","O","N"]).any():
            df[col] = df[col].apply(oui_non_vers_binaire)
    return df

def concat_dates_urgences(feuilles):
    toutes_dates = pd.Series(dtype='datetime64[ns]')
    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom]
            col_date_candidates = [col for col in df_urg.columns if 'date' in col.lower()]
            if col_date_candidates:
                col_date = col_date_candidates[0]
                dates = pd.to_datetime(df_urg[col_date], errors='coerce').dropna()  # uniquement dates valides
                toutes_dates = pd.concat([toutes_dates, dates])
    return toutes_dates

def show_eda():
    st.title("📊 Analyse exploratoire des données - Mémoire")
    file_path = "Base_de_donnees_USAD_URGENCES1.xlsx"
    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
        st.success("✅ Fichier chargé avec succès !")
    except FileNotFoundError:
        st.error(f"❌ Fichier introuvable : {file_path}")
        return

    # ============================
    # 1️⃣ Identité
    # ============================
    if 'Identite' in feuilles:
        identite = feuilles['Identite']
        identite = convertir_df_oui_non(identite, exclude_columns=["Niveau d'instruction scolarité"])
        st.header("1️⃣ Identité des patients")
        st.write(f"Nombre total de patients : {len(identite)}")

        # Sexe
        if 'Sexe' in identite.columns:
            fig, ax = plt.subplots(figsize=(6,6))
            identite['Sexe'].value_counts().plot.pie(
                autopct="%1.1f%%", startangle=140, colors=["#FF9999","#66B2FF"], ax=ax
            )
            ax.set_ylabel("")
            ax.set_title("Répartition par sexe")
            st.pyplot(fig)

        # Origine géographique
        if 'Origine Géographique' in identite.columns:
            fig, ax = plt.subplots(figsize=(6,6))
            identite['Origine Géographique'].value_counts().plot.pie(
                autopct="%1.1f%%", startangle=140, ax=ax
            )
            ax.set_ylabel("")
            ax.set_title("Répartition par origine géographique")
            st.pyplot(fig)

        # Âge
        age_col = "Âge du debut d etude en mois (en janvier 2023)"
        if age_col in identite.columns:
            fig, ax = plt.subplots(figsize=(8,6))
            identite[age_col] = pd.to_numeric(identite[age_col], errors='coerce')
            identite[age_col].dropna().plot.hist(bins=20, color="#36A2EB", edgecolor="white", ax=ax)
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
        st.header("2️⃣ Type de drépanocytose et paramètres biologiques")

        # Type
        if 'Type de drépanocytose' in drepano.columns:
            st.subheader("Type de drépanocytose")
            st.dataframe(drepano['Type de drépanocytose'].value_counts())

        # Paramètres biologiques
        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        st.subheader("Paramètres biologiques")
        for col in bio_cols:
            if col in drepano.columns:
                fig, ax = plt.subplots(figsize=(8,4))
                drepano[col] = pd.to_numeric(drepano[col], errors='coerce')
                sns.histplot(drepano[col].dropna(), bins=20, kde=True, color="#36A2EB", ax=ax)
                ax.set_title(col)
                st.pyplot(fig)

    # ============================
    # 4️⃣ Consultations d'urgence
    # ============================
    st.header("4️⃣ Consultations d'urgence")
    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = feuilles[nom]
            df_urg = convertir_df_oui_non(df_urg)
            st.subheader(nom)
            st.write(f"Nombre de consultations : {len(df_urg)}")
            # Symptômes uniquement si date remplie
            date_col = [c for c in df_urg.columns if "date" in c.lower()]
            if date_col:
                df_urg = df_urg[df_urg[date_col[0]].notna()]  # seulement patients venus
            symptomes = ['Douleur','Fièvre','Pâleur','Ictère','Toux']
            for s in symptomes:
                if s in df_urg.columns:
                    counts = df_urg[s].value_counts().to_dict()
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.bar(counts.keys(), counts.values(), color="#36A2EB")
                    ax.set_title(s)
                    st.pyplot(fig)

    # ============================
    # 5️⃣ Répartition mensuelle des urgences
    # ============================
    st.header("5️⃣ Répartition mensuelle des urgences")
    toutes_dates = concat_dates_urgences(feuilles)
    if not toutes_dates.empty:
        repartition = toutes_dates.dt.month.value_counts().sort_index()
        mois_noms = {1:'Jan',2:'Fév',3:'Mar',4:'Avr',5:'Mai',6:'Jun',7:'Jul',8:'Aoû',9:'Sep',10:'Oct',11:'Nov',12:'Déc'}
        df_repart = pd.DataFrame({'Mois':[mois_noms[m] for m in repartition.index], 'Consultations':repartition.values})
        df_repart['Pourcentage'] = (df_repart['Consultations']/df_repart['Consultations'].sum()*100).round(2)
        st.dataframe(df_repart)

        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='Mois', y='Consultations', data=df_repart, palette="Blues_d", ax=ax)
        ax.set_title("Répartition mensuelle des urgences")
        st.pyplot(fig)
