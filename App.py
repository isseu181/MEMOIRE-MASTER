# ===============================
# STREAMLIT APP - URGENCES DRÉPANOCYTAIRES
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===============================
# Titre de l'application
# ===============================
st.title("Analyse et Prédiction des Urgences Drépanocytaires - USAD")
st.markdown("Application interactive pour explorer les données, analyser et prédire les niveaux d'urgence des patients.")

# ===============================
# Importation des données
# ===============================
st.header("Chargement des données")
uploaded_file = st.file_uploader("Choisir le fichier CSV des urgences", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Fichier chargé avec succès!")
    st.dataframe(df.head())
else:
    st.warning("Veuillez charger le fichier CSV pour continuer.")
    st.stop()

# ===============================
# Analyse descriptive univariée
# ===============================
st.header("Analyse descriptive univariée")

# Variables catégorielles
cat_cols = ['Sexe', 'Origine Géographique', 'Statut des parents (Vivants/Décédés)',
            'Parents Salariés', 'Scolarité', 'Type de drépanocytose', 'NiveauUrgence']

st.subheader("Distribution des variables catégorielles")
for col in cat_cols:
    st.write(f"**{col}**")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, palette="Set2", order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Variables numériques
num_cols = ['Âge du debut d etude en mois', 'HB (g/dl)', 'GB (/mm3)', 'PLT (/mm3)']
st.subheader("Statistiques descriptives des variables numériques")
st.write(df[num_cols].describe())

# ===============================
# Analyse bivariée
# ===============================
st.header("Analyse bivariée")

# Exemple : Type de drépanocytose vs NiveauUrgence
st.subheader("Type de drépanocytose vs Niveau d'urgence")
ct = pd.crosstab(df['Type de drépanocytose'], df['NiveauUrgence'])
fig, ax = plt.subplots()
ct.plot(kind='bar', stacked=True, ax=ax, colormap='Paired')
plt.ylabel("Nombre de consultations")
st.pyplot(fig)

# Répartition mensuelle des urgences
if 'Mois' in df.columns:
    st.subheader("Répartition mensuelle des urgences")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Mois', palette="coolwarm", order=df['Mois'].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ===============================
# Modélisation prédictive
# ===============================
st.header("Modélisation prédictive du Niveau d'Urgence")

# Variables explicatives et cible
features = ['Âge du debut d etude en mois', 'HB (g/dl)', 'GB (/mm3)', 'PLT (/mm3)',
            'Sexe', 'Type de drépanocytose', 'Origine Géographique', 'Statut des parents (Vivants/Décédés)']
target = 'NiveauUrgence'

df_model = df[features + [target]].dropna()

# Encodage des variables catégorielles
le_dict = {}
for col in ['Sexe', 'Type de drépanocytose', 'Origine Géographique', 'Statut des parents (Vivants/Décédés)']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

# Séparation Train/Test
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédictions
y_pred = rf_model.predict(X_test)

# Évaluation
st.subheader("Évaluation du modèle")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.text(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
ax.set_title("Matrice de confusion")
st.pyplot(fig_cm)

# Importance des variables
st.subheader("Importance des variables")
feat_imp = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
st.bar_chart(feat_imp.set_index('Feature'))

# ===============================
# Simulation d'un patient
# ===============================
st.header("Simulation d'un patient")
st.markdown("Entrez les caractéristiques du patient pour prédire le niveau d'urgence:")

sim_data = {}
for col in features:
    if col in le_dict:
        options = le_dict[col].classes_
        sim_data[col] = st.selectbox(col, options)
    else:
        sim_data[col] = st.number_input(col, value=int(df[col].median()))

if st.button("Prédire le Niveau d'Urgence"):
    sim_df = pd.DataFrame([sim_data])
    # Encodage
    for col in le_dict:
        sim_df[col] = le_dict[col].transform(sim_df[col])
    pred = rf_model.predict(sim_df)[0]
    st.success(f"Niveau d'Urgence prédit pour ce patient : {pred}")
