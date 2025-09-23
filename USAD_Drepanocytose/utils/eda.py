# utils/eda.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

def show_eda():
    st.subheader("📊 Analyse exploratoire des données")
    fichier = "data/Base_de_donnees_USAD_URGENCES1.xlsx"
    try:
        feuilles = pd.read_excel(fichier, sheet_name=None)
        st.success("Fichier Excel chargé avec succès ✅")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return

    # Exemple : afficher les feuilles
    st.write("Feuilles disponibles :", list(feuilles.keys()))
