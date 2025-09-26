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
    # Chargement automatique de la base
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
    # Encodage des variables qualitatives
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
    df_selected = pd.get_dummies(df_selected, columns=["Origine Géographique"], drop_first=False)
    df_selected = pd.get_dummies(df_selected, columns=["Prise en charge"], drop_first=True)
    df_selected = pd.get_dummies(df_selected, columns=["Type de drépanocytose"], drop_first=False)

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
    # Clustering et PCA calculés une seule fois
    # ================================
    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    df['PCA1'] = components[:,0]
    df['PCA2'] = components[:,1]
    explained_var = pca.explained_variance_ratio_

    # ================================
    # Onglets horizontaux
    # ================================
    tabs = st.tabs(["Vue d'ensemble", "Visualisation ACP", "Profil détaillé"])

    # --- Onglet 1 : Vue d'ensemble ---
    with tabs[0]:
        st.subheader("Méthodologie et Graphe du coude")
        st.write("""
        - Préprocessing des données
        - Encodage des variables qualitatives
        - Standardisation des variables quantitatives
        - Clustering KMeans
        """)
        inertia = [KMeans(n_clusters=k, random_state=42).fit(df_scaled).inertia_ for k in range(1,11)]
        fig, ax = plt.subplots()
        ax.plot(range(1,11), inertia, marker='o')
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_ylabel("Inertia (SSE)")
        ax.set_title("Graphe du coude pour KMeans")
        st.pyplot(fig)

    # --- Onglet 2 : Visualisation ACP ---
    with tabs[1]:
        st.subheader("Visualisation des clusters via PCA 2D")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax)
        ax.set_title("Clusters visualisés sur les 2 premières composantes principales")
        st.pyplot(fig)
        st.write(f"Variance expliquée PCA1 : {explained_var[0]:.2%}, PCA2 : {explained_var[1]:.2%}")

    # --- Onglet 3 : Profil détaillé ---
    with tabs[2]:
        st.subheader("Profil détaillé des clusters")
        selected_clusters = st.multiselect(
            "Sélectionnez les clusters à afficher",
            options=sorted(df['Cluster'].unique()),
            default=sorted(df['Cluster'].unique())
        )
        df_filtered = df[df['Cluster'].isin(selected_clusters)]
        st.dataframe(df_filtered)

        st.write("Histogrammes des variables quantitatives par cluster filtré :")
        for var in quantitative_vars:
            fig, ax = plt.subplots()
            sns.histplot(data=df_filtered, x=var, hue='Cluster', multiple='stack', palette='tab10', ax=ax)
            ax.set_title(f"Distribution de {var} par cluster")
            st.pyplot(fig)

        st.write("Résumé des clusters filtrés :")
        cluster_summary = df_filtered.groupby('Cluster')[quantitative_vars].mean()
        st.dataframe(cluster_summary)
