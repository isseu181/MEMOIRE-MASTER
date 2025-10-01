# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================
# Fonctions utilitaires
# ============================
def load_data(file_path="fichier_nettoye.xlsx"):
    try:
        df = pd.read_excel(file_path)
        return df
    except:
        st.warning(f"⚠️ Fichier '{file_path}' introuvable ou illisible.")
        return None

# ============================
# Analyse exploratoire
# ============================
def eda_dashboard(df):
    st.header("📊 Analyse exploratoire")

    st.subheader("1️⃣ Données démographiques")
    if 'Sexe' in df.columns:
        sexe_counts = df['Sexe'].value_counts()
        fig = px.pie(sexe_counts, names=sexe_counts.index, values=sexe_counts.values,
                     title="Répartition par sexe")
        st.plotly_chart(fig, use_container_width=True)

    if "Niveau d'instruction scolarité" in df.columns:
        scolar_counts = df["Niveau d'instruction scolarité"].value_counts()
        fig = px.pie(scolar_counts, names=scolar_counts.index, values=scolar_counts.values,
                     title="Répartition de la scolarisation")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("2️⃣ Variables quantitatives clés")
    quantitative_vars = ["Âge du debut d etude en mois (en janvier 2023)", 
                         "GR (/mm3)", "GB (/mm3)", "HB (g/dl)"]
    for var in quantitative_vars:
        if var in df.columns:
            fig = px.histogram(df, x=var, nbins=15, title=f"Distribution de {var}")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("3️⃣ Analyse temporelle")
    if 'Mois' in df.columns:
        mois_counts = df['Mois'].value_counts().sort_index()
        fig = px.line(x=mois_counts.index, y=mois_counts.values,
                      labels={"x":"Mois","y":"Nombre de consultations"},
                      title="Nombre de consultations par mois", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("4️⃣ Biomarqueurs")
    bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C",
                "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
    bio_data = {}
    for col in bio_cols:
        if col in df.columns:
            bio_data[col] = {
                "Moyenne": df[col].mean(),
                "Médiane": df[col].median(),
                "Min": df[col].min(),
                "Max": df[col].max()
            }
    if bio_data:
        bio_df = pd.DataFrame(bio_data).T.round(2)
        st.table(bio_df)

# ============================
# Clustering
# ============================
def clustering_dashboard(df):
    st.header("🤖 Clustering - Segmentation des patients")

    quantitative_vars = [
        "Âge du debut d etude en mois (en janvier 2023)", "Taux d'Hb (g/dL)",
        "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"
    ]
    df_scaled = df[quantitative_vars].copy()
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_scaled), columns=quantitative_vars)

    n_clusters = st.slider("Sélection du nombre de clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df_scaled)

    st.subheader("Nombre de patients par cluster")
    st.dataframe(df['Cluster'].value_counts().sort_index().rename("Nombre de patients"))

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                     title="Clusters visualisés via PCA 2D")
    st.plotly_chart(fig, use_container_width=True)

# ============================
# Classification supervisée (résumé)
# ============================
def classification_dashboard():
    st.header("📈 Classification supervisée")
    st.info("⚠️ Les résultats du modèle supervisé seront affichés ici après exécution du fichier classification.py")
    st.markdown("""
    - Comparaison des modèles : Decision Tree, Random Forest, SVM, LightGBM  
    - Métriques : Accuracy, Precision, Recall, F1, AUC  
    - Analyse du meilleur modèle : matrice de confusion, courbe ROC, variables importantes  
    """)

# ============================
# Dashboard principal
# ============================
def show_dashboard():
    st.set_page_config(page_title="Tableau de bord USAD", layout="wide")
    st.sidebar.title("Navigation Dashboard")
    page = st.sidebar.radio("Aller à :", [
        "Analyse exploratoire", "Clustering", "Classification supervisée"
    ])

    df = load_data()

    if page == "Analyse exploratoire" and df is not None:
        eda_dashboard(df)
    elif page == "Clustering" and df is not None:
        clustering_dashboard(df)
    elif page == "Classification supervisée":
        classification_dashboard()

# ============================
# Exécution principale
# ============================
if __name__ == "__main__":
    show_dashboard()
