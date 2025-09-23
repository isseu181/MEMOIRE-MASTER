import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def show_clustering():
    st.title("Chapitre 3 : Classification non supervisée (Clustering)")
    
    # ================================
    # 1. Chargement des données
    # ================================
    df = pd.read_excel("segmentation.xlsx")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # ================================
    # 2. Variables sélectionnées
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
        "Prise en charge", "Prophylaxie à la pénicilline",
        "CVO", "Anémie", "AVC", "STA", "Priapisme", "Infections", "Ictère",
        "Type de drépanocytose"
    ]
    df_selected = df[variables_selected].copy()

    # ================================
    # 3. Encodage des qualitatives
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
    # 4. Standardisation
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
    # 5. Graphe du coude
    # ================================
    st.subheader("Méthode du coude")
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42)
        kmeans_test.fit(df_selected)
        inertia.append(kmeans_test.inertia_)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(K_range, inertia, marker='o')
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Inertia (SSE)")
    ax.set_title("Graphe du coude pour KMeans")
    st.pyplot(fig)

    # ================================
    # 6. Choix utilisateur + clustering
    # ================================
    k_optimal = st.slider("Choisir le nombre de clusters :", 2, 10, 3)
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    df_selected['Cluster'] = kmeans.fit_predict(df_selected)

    # ================================
    # 7. Résumé des clusters
    # ================================
    st.subheader("Résumé des clusters (moyennes)")
    cluster_summary = df_selected.groupby("Cluster").mean()
    st.dataframe(cluster_summary)

    # ================================
    # 8. PCA + Visualisation
    # ================================
    st.subheader("Projection PCA")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_selected.drop("Cluster", axis=1))
    df_selected["PCA1"] = components[:,0]
    df_selected["PCA2"] = components[:,1]

    explained_var = pca.explained_variance_ratio_
    st.write(f"Variance expliquée par PCA1 : {explained_var[0]*100:.1f}%")
    st.write(f"Variance expliquée par PCA2 : {explained_var[1]*100:.1f}%")

    fig2, ax2 = plt.subplots(figsize=(6,5))
    for c in range(k_optimal):
        subset = df_selected[df_selected["Cluster"] == c]
        ax2.scatter(subset["PCA1"], subset["PCA2"], label=f"Cluster {c}", alpha=0.6)
    ax2.set_xlabel(f"PCA1 ({explained_var[0]*100:.1f}%)")
    ax2.set_ylabel(f"PCA2 ({explained_var[1]*100:.1f}%)")
    ax2.set_title("Visualisation des clusters KMeans sur PCA")
    ax2.legend()
    st.pyplot(fig2)

