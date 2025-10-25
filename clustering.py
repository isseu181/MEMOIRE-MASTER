# clustering.py
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def show_clustering():
    st.title("Segmentation de Patients - KMeans Clustering (Plotly interactif)")

    # ================================
    # 1. Chargement de la base
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
        "Prise en charge", "Prophylaxie à la pénicilline", "CVO", "Anémie", "AVC", 
        "STA", "Priapisme", "Infections", "Ictère", "Type de drépanocytose",
    ]
    df_selected = df[variables_selected].copy()

    # ================================
    # 3. Encodage des variables qualitatives
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
    df_selected = pd.get_dummies(df_selected, columns=["Origine Géographique", "Prise en charge", "Type de drépanocytose"])

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
    df_scaled = df_selected.copy()
    df_scaled[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

    # ================================
    # 5. Clustering
    # ================================
    st.subheader("Clustering KMeans interactif")
    n_clusters = st.slider("Sélectionner le nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    df_scaled["Cluster"] = df["Cluster"]
    st.success(f"Clustering effectué avec {n_clusters} clusters !")

    # ================================
    # 6. PCA + Biplot interactif
    # ================================
    st.subheader("Visualisation interactive PCA 2D avec flèches des variables")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled.drop("Cluster", axis=1))
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df['Cluster']

    # Variance expliquée
    explained_var = pca.explained_variance_ratio_
    st.write(f"**Variance expliquée par PC1 :** {explained_var[0]:.2%}")
    st.write(f"**Variance expliquée par PC2 :** {explained_var[1]:.2%}")
    st.write(f"**Variance totale expliquée par PC1 et PC2 :** {(explained_var[0]+explained_var[1]):.2%}")

    # Couleurs
    colors = px.colors.qualitative.Safe

    # Scatter interactif des patients
    fig = go.Figure()
    for c in sorted(df_pca['Cluster'].unique()):
        subset = df_pca[df_pca['Cluster']==c]
        fig.add_trace(go.Scatter(
            x=subset['PC1'], y=subset['PC2'], mode='markers',
            name=f'Cluster {c}', marker=dict(size=8, color=colors[c]),
            text=[f"Patient {i}" for i in subset.index], hoverinfo='text'
        ))

    # Flèches des variables (loadings PCA)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    cols = df_scaled.drop("Cluster", axis=1).columns
    max_arrow_length = 3.0
    scale = max_arrow_length / max(np.sqrt(loadings[:,0]**2 + loadings[:,1]**2))
    for i, col in enumerate(cols):
        x, y = loadings[i,0]*scale, loadings[i,1]*scale
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y], mode='lines+markers+text',
            marker=dict(size=1, color='red'),
            line=dict(color='red', width=2),
            text=[None, col], textposition="top center",
            showlegend=False
        ))

    fig.update_layout(title="PCA 2D - Clusters + Flèches des variables",
                      xaxis_title=f"PC1 ({explained_var[0]*100:.1f}%)",
                      yaxis_title=f"PC2 ({explained_var[1]*100:.1f}%)",
                      width=900, height=600)
    st.plotly_chart(fig)

    # ================================
    # 7. Profil détaillé
    # ================================
    st.subheader("Profil détaillé des clusters")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("### Nombre de patients par cluster")
    st.dataframe(cluster_counts.rename("Nombre de patients"))

    st.write("### Moyennes standardisées (Z-scores) par cluster")
    cluster_means = df_scaled.groupby("Cluster").mean().T
    interpretations = []
    for var, row in cluster_means.iterrows():
        max_cluster = row.idxmax()
        min_cluster = row.idxmin()
        if abs(row[max_cluster]) < 0.2 and abs(row[min_cluster]) < 0.2:
            interp = "Peu discriminant"
        else:
            interp = f"Plus élevé dans Cluster {max_cluster}, plus bas dans Cluster {min_cluster}"
        interpretations.append(interp)
    cluster_means["Interprétation"] = interpretations
    st.dataframe(cluster_means)
