# ================================
# deploiement.py
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================================
# 1️⃣ Charger le modèle, le scaler et les features
# ================================
best_model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))  # Liste des colonnes du modèle

# ================================
# 2️⃣ Interface utilisateur
# ================================
st.title("Déploiement du modèle - Prédiction de l'évolution")
st.write("Entrez les valeurs des variables pour prédire l'évolution du patient:")

# Variables quantitatives
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]

# Variables binaires
binary_vars = [
    'Pâleur', 'Souffle systolique fonctionnel', 'Vaccin contre méningocoque', 
    'Splénomégalie', 'Prophylaxie à la pénicilline', 'Parents Salariés', 
    'Prise en charge Hospitalisation', 'Radiographie du thorax Oui ou Non', 
    'Douleur provoquée (Os.Abdomen)', 'Vaccin contre pneumocoque'
]

# Variables ordinales
ordinal_vars = {
    'NiveauUrgence': ['Urgence1','Urgence2','Urgence3','Urgence4','Urgence5','Urgence6'],
    "Niveau d'instruction scolarité": ['Maternelle ','Elémentaire ','Secondaire','Enseignement Supérieur ','NON']
}

# Variables catégorielles à One-Hot encoder
categorical_vars = {
    'Diagnostic Catégorisé': ['Type1','Type2','Type3'],  # Remplacer par vos catégories exactes
    'Mois': [str(i) for i in range(1,13)]  # Janvier=1, Février=2...
}

# Collecte des entrées
input_data = {}
for var in quantitative_vars:
    input_data[var] = st.number_input(var, value=0.0)

for var in binary_vars:
    input_data[var] = st.selectbox(var, ['OUI','NON'])

for var, options in ordinal_vars.items():
    input_data[var] = st.selectbox(var, options)

for var, options in categorical_vars.items():
    input_data[var] = st.selectbox(var, options)

# Bouton de prédiction
if st.button("Prédire l'évolution"):
    df_input = pd.DataFrame([input_data])
    
    # Encodage binaire
    binary_mapping = {'OUI':1, 'NON':0}
    for var in binary_vars:
        df_input[var] = df_input[var].map(binary_mapping)
    
    # Encodage ordinal
    df_input['NiveauUrgence'] = df_input['NiveauUrgence'].map({
        'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6
    })
    df_input["Niveau d'instruction scolarité"] = df_input["Niveau d'instruction scolarité"].map({
        'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0
    })
    
    # One-Hot encoding pour les catégories
    for var, options in categorical_vars.items():
        df_input = pd.get_dummies(df_input, columns=[var], prefix=[var])
    
    # Ajouter les colonnes manquantes pour correspondre au modèle
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[features]
    
    # Standardisation
    df_input[quantitative_vars] = scaler.transform(df_input[quantitative_vars])
    
    # Prédiction
    proba = best_model.predict_proba(df_input)[:,1][0]
    prediction = best_model.predict(df_input)[0]
    
    st.write(f"Probabilité de complications : {proba:.2f}")
    st.write(f"Prédiction finale : {'Complications' if prediction==1 else 'Favorable'}")
