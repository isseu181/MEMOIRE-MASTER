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
df_selected[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

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

    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42)
        kmeans_test.fit(df_selected)
        inertia.append(kmeans_test.inertia_)

    fig, ax = plt.subplots()
    ax.plot(list(K_range), inertia, marker='o')
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Inertia (SSE)")
    ax.set_title("Graphe du coude pour KMeans")
    st.pyplot(fig)

    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_selected)
    st.success("✅ Clustering effectué !")

# --- Onglet 2 : Visualisation ACP ---
with tabs[1]:
    st.subheader("Visualisation des clusters via PCA 2D")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_selected)
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df['Cluster']

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    ax.set_title("Clusters visualisés sur les 2 premières composantes principales")
    st.pyplot(fig)

# --- Onglet 3 : Profil détaillé ---
with tabs[2]:
    st.subheader("Profil détaillé des clusters")
    st.write("Tableau complet des données avec cluster :")
    st.dataframe(df)

    st.write("Histogrammes des variables quantitatives par cluster :")
    for var in quantitative_vars:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=var, hue='Cluster', multiple='stack', palette='tab10', ax=ax)
        ax.set_title(f"Distribution de {var} par cluster")
        st.pyplot(fig)
