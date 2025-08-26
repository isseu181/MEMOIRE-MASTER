# app_usad.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------
# Titre de l'application
# ----------------------
st.title("Analyse et prédiction des urgences drépanocytaires à l'USAD")

# ----------------------
# Chargement des données
# ----------------------
st.sidebar.header("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Données chargées avec succès !")
    st.write(df.head())

    # ----------------------
    # Statistiques descriptives
    # ----------------------
    st.header("Statistiques descriptives")
    st.subheader("Variables quantitatives")
    st.write(df.describe())

    st.subheader("Variables qualitatives")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())

    # ----------------------
    # Graphiques interactifs
    # ----------------------
    st.header("Visualisation des urgences")
    
    # Exemple : répartition par type de drépanocytose
    st.subheader("Répartition par type de drépanocytose")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Type de drépanocytose', palette='Set2', order=df['Type de drépanocytose'].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Exemple : répartition des urgences par mois
    if 'Mois' in df.columns:
        st.subheader("Répartition des urgences par mois")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Mois', palette='Set1', order=df['Mois'].value_counts().index)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # ----------------------
    # Prédiction de l'évolution
    # ----------------------
    st.header("Prédiction de l'évolution des urgences")

    # Sélection des variables explicatives pour le modèle
    feature_cols = ['Parents Salariés', 'Vaccin contre pneumocoque', 'Vaccin contre méningocoque', 
                    'Pâleur', 'Splénomégalie', 'Diagnostic Catégorisé', 'Prise en charge Hospitalisation', 
                    'Évaluation de la douleur', 'HB', 'GB', 'VGM', 'TCMH', 'PLT']

    # Encodage des variables qualitatives
    df_model = df[feature_cols + ['Evolution']].copy()
    le_dict = {}
    for col in df_model.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    # Séparer les données
    X = df_model[feature_cols]
    y = df_model['Evolution']

    # Entraîner un modèle simple (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.subheader("Simulation de prédiction")
    st.write("Remplissez les informations ci-dessous pour prédire l'évolution du patient")
    
    user_input = {}
    for col in feature_cols:
        if df[col].dtype == 'object' or df[col].nunique() < 10:
            user_input[col] = st.selectbox(col, df[col].unique())
        else:
            user_input[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    
    # Encoder les inputs utilisateur
    user_df = pd.DataFrame([user_input])
    for col in user_df.select_dtypes(include='object').columns:
        le = le_dict[col]
        user_df[col] = le.transform(user_df[col].astype(str))
    
    if st.button("Prédire"):
        pred = model.predict(user_df)[0]
        st.success(f"L'évolution prédite pour ce patient est : **{le_dict['Evolution'].inverse_transform([pred])[0]}**")

else:
    st.warning("Veuillez uploader un fichier CSV pour continuer.")
