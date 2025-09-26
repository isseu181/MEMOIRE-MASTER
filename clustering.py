# clustering.py
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def show_clustering():
    st.set_page_config(page_title="Segmentation Patients", layout="wide")
    st.title("Segmentation de Patients - KMeans Clustering")

    # ================================
    # Chargement automatique
    # ================================
    st.info("Chargement automatique de la base de données : segmentation.xlsx")
    df = pd.read_excel("segmentation.xlsx")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # ================================
    # Variables sélectionnées
    # ================================
    variables_selected = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)",
        "Âge de début des signes (en mois)",
        "Âge de découverte de la drépanocytose (en mois)",
        "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        "Nbre de transfusion avant 2017",
        "Nbre de transfusion Entre 2017 et 2023",
        "HDJ",
        "Sexe", "Origine Géographique", "Parents Salariés",
        "PEV Complet", "Vaccin contre pneumocoque", "Vaccin contre méningocoque", 
        "Vaccin contre Les salmonelles", "L'hydroxyurée", "Echange transfusionnelle",  
        "Prise en charge", "Prophylaxie à la pénicilline", "CVO", "Anémie", "AVC", 
        "STA", "Priapisme", "Infections", "Ictère", "Type de drépanocytose",
    ]
    df_selected = df[variables_selected].copy()

    # ================================
    # Encodage qualitatif
    # ================================
    binary_mappings = {
        "Sexe": {"Masculin": 1, "Féminin": 0, "M": 1, "F": 0},
        "Parents Salariés": {"OUI": 1, "NON": 0},
        "PEV Complet": {"OUI": 1, "NON": 0},
        "Vaccin contre pneumocoque": {"OUI": 1, "NON": 0},
        "Vaccin contre méningocoque": {"OUI": 1, "NON": 0},
        "Vaccin contre Les salmonelles": {"OUI": 1, "NON": 0},
        "L'hydroxyurée": {"OUI": 1, "NON": 0},
        "Echange transfusionnelle": {"OUI": 1, "NON": 0},
        "Prophylaxie à la pénicilline": {"OUI": 1, "NON": 0},
        "CVO": {"OUI": 1, "NON": 0},
        "Anémie": {"OUI": 1, "NON": 0},
        "AVC": {"OUI": 1, "NON": 0},
        "STA": {"OUI": 1, "NON": 0},
        "Priapisme": {"OUI": 1, "NON": 0},
        "Infections": {"OUI": 1, "NON": 0},
        "Ictère": {"OUI": 1, "NON": 0},
    }
    df_selected.replace(binary_mappings, inplace=True)
    df_selected = pd.get_dummies(df_selected, columns=["Origine Géographique","Prise en charge","Type de drépanocytose"], drop_first=True)

    # ================================
    # Standardisation
    # ================================
    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)",
        "Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
        "Nbre de GB (/mm3)", "% d'HB A2", "Nbre de PLT (/mm3)",
        "Âge de début des signes (en mois)",
        "Âge de découverte de la drépanocytose (en mois)",
        "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        "Nbre de transfusion avant 2017",
        "Nbre de transfusion Entre 2017 et 2023",
        "HDJ"
    ]
    scaler = StandardScaler()
    df_scaled = df_selected.copy()
    df_scaled[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

    # ================================
    # Clustering
    # ================================
    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # ================================
    # PCA
    # ================================
    pca = PCA(n_components=2)
    df[['PCA1','PCA2']] = pca.fit_transform(df_scaled)
    explained_var = pca.explained_variance_ratio_

    # ================================
    # Onglets
    # ================================
    tabs = st.tabs(["Vue d'ensemble", "Visualisation ACP", "Profil détaillé"])

    # --- Vue d'ensemble ---
    with tabs[0]:
        st.subheader("Méthodologie et Graphe du coude")
        inertia = [KMeans(n_clusters=k, random_state=42).fit(df_scaled).inertia_ for k in range(1,11)]
        fig, ax = plt.subplots()
        ax.plot(range(1,11), inertia, marker='o')
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_ylabel("Inertia (SSE)")
        ax.set_title("Graphe du coude KMeans")
        st.pyplot(fig)

    # --- PCA ---
    with tabs[1]:
        st.subheader("Visualisation ACP")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax)
        st.pyplot(fig)
        st.write(f"Variance expliquée PCA1 : {explained_var[0]:.2%}, PCA2 : {explained_var[1]:.2%}")

    # --- Profil détaillé automatisé ---
    with tabs[2]:
        st.subheader("Profil détaillé par cluster")
        cluster_summary = df.groupby('Cluster')[quantitative_vars].mean()
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        for cluster in sorted(df['Cluster'].unique()):
            st.markdown(f"### Cluster {cluster} - {cluster_counts[cluster]} patients")
            st.markdown("**Caractéristiques Cliniques principales** :")
            # Exemple automatique basé sur Hb et hospitalisations
            hbF = cluster_summary.loc[cluster,"% d'Hb F"]
            hbS = cluster_summary.loc[cluster,"% d'Hb S"]
            hbC = cluster_summary.loc[cluster,"% d'HB C"]
            hosp = cluster_summary.loc[cluster,"Nbre d'hospitalisations entre 2017 et 2023"]
            transf = cluster_summary.loc[cluster,"Nbre de transfusion Entre 2017 et 2023"]

            st.markdown(f"- Hb F : {hbF:.3f}")
            st.markdown(f"- Hb S : {hbS:.3f}")
            st.markdown(f"- Hb C : {hbC:.3f}")
            st.markdown(f"- Hospitalisations récentes : {hosp:.3f}")
            st.markdown(f"- Transfusions récentes : {transf:.3f}")
            st.markdown("---")

        st.markdown("**Implications cliniques (automatisées)** :")
        st.markdown("- Cluster faible : suivi standard")
        st.markdown("- Cluster modéré : interventions préventives ciblées")
        st.markdown("- Cluster sévère : prise en charge intensive")
