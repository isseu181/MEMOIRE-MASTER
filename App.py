# app_modern.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sickle Insight Engine", layout="wide", initial_sidebar_state="expanded")

# ============================
# Fonctions utilitaires
# ============================
def oui_non_vers_binaire(valeur):
    if isinstance(valeur, str) and valeur.strip().lower() in ["oui","o"]:
        return 1
    elif isinstance(valeur, str) and valeur.strip().lower() in ["non","n"]:
        return 0
    return valeur

def convertir_df_oui_non(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns or []
    for col in df.columns:
        if col not in exclude_columns and df[col].isin(["Oui","Non","OUI","NON","oui","non","O","N"]).any():
            df[col] = df[col].apply(oui_non_vers_binaire)
    return df

# ============================
# 1️⃣ Analyse descriptive
# ============================
def show_eda_modern():
    st.header("📊 Analyse descriptive")
    try:
        feuilles = pd.read_excel("Base_de_donnees_USAD_URGENCES1.xlsx", sheet_name=None)
    except:
        st.error("Fichier EDA introuvable")
        return

    identite = convertir_df_oui_non(feuilles["Identite"], exclude_columns=["Niveau d'instruction scolarité"])
    drepano = convertir_df_oui_non(feuilles["Drépano"])
    
    # KPI en colonnes
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients totaux", len(identite))
    col2.metric("Consultations totales", sum([len(feuilles[f'Urgence{i}']) for i in range(1,7) if f'Urgence{i}' in feuilles]))
    col3.metric("Hb moyenne", round(drepano["Taux d'Hb (g/dL)"].mean(),2))
    col4.metric("Âge moyen", round(identite["Âge du debut d etude en mois (en janvier 2023)"].mean(),1))

    # Graphiques interactifs
    st.subheader("Répartition par sexe")
    sexe_counts = identite["Sexe"].value_counts()
    fig_sexe = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values, 
                      title="Sexe", color_discrete_sequence=px.colors.sequential.RdBu)
    fig_sexe.update_traces(textinfo='percent+label', pull=0.05)
    st.plotly_chart(fig_sexe, use_container_width=True)

    st.subheader("Âge des patients")
    fig_age = px.histogram(identite, x="Âge du debut d etude en mois (en janvier 2023)", nbins=15,
                           title="Distribution des âges", color_discrete_sequence=["#2E86C1"])
    fig_age.update_traces(texttemplate="%{y}", textposition="outside")
    st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Paramètres biologiques (résumé)")
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    bio_summary = drepano[bio_cols].agg(["mean","median","max"]).round(2)
    st.dataframe(bio_summary)

# ============================
# 2️⃣ Clustering non supervisé
# ============================
def show_clustering_modern():
    st.header("📊 Clustering")
    df = pd.read_excel("segmentation.xlsx").applymap(lambda x: x.strip() if isinstance(x, str) else x)
    variables = ["Âge du debut d etude en mois (en janvier 2023)",
                 "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                 "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    df_selected = df[variables].copy()
    scaler = StandardScaler()
    df_selected[variables] = scaler.fit_transform(df_selected)

    # Méthode du coude
    inertia = [KMeans(n_clusters=k, random_state=42).fit(df_selected).inertia_ for k in range(1,11)]
    fig_coude = px.line(x=list(range(1,11)), y=inertia, markers=True, labels={"x":"k","y":"Inertia"}, title="Méthode du coude")
    st.plotly_chart(fig_coude, use_container_width=True)

    k_optimal = st.slider("Choisir k pour KMeans", 2, 10, 3)
    df_selected["Cluster"] = KMeans(n_clusters=k_optimal, random_state=42).fit_predict(df_selected)

    st.subheader("Résumé clusters")
    st.dataframe(df_selected.groupby("Cluster")[variables].agg(["mean","median","max"]).round(2))

    # PCA interactive
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_selected[variables])
    df_selected["PCA1"] = components[:,0]
    df_selected["PCA2"] = components[:,1]
    fig_pca = px.scatter(df_selected, x="PCA1", y="PCA2", color="Cluster", hover_data=variables,
                         title="Visualisation PCA des clusters", color_continuous_scale=px.colors.qualitative.Bold)
    st.plotly_chart(fig_pca, use_container_width=True)

# ============================
# 3️⃣ Analyse prédictive
# ============================
def show_prediction_modern():
    st.header("📊 Analyse binaire : Evolution")
    df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
    cible = "Evolution"
    variables_interessantes = ["Type de drépanocytose","Sexe","Âge du debut d etude en mois (en janvier 2023)"]

    for var in variables_interessantes:
        st.subheader(f"{var} vs {cible}")
        if df_nettoye[var].dtype=="object":
            cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
            st.dataframe(cross_tab.round(2))
            fig = px.bar(cross_tab, barmode="group", text_auto=".2f", title=f"{var} vs {cible}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(df_nettoye, x=cible, y=var, points="all", title=f"{var} selon {cible}")
            st.plotly_chart(fig, use_container_width=True)

# ============================
# Dashboard principal
# ============================
st.title("🏥 Sickle Insight Engine")
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller à", ["Analyse descriptive", "Clustering", "Analyse prédictive"])

if section=="Analyse descriptive":
    show_eda_modern()
elif section=="Clustering":
    show_clustering_modern()
else:
    show_prediction_modern()
