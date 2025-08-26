# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Analyse Urgences Drépanocytaires", layout="wide")

st.title("Analyse et Modélisation des Urgences Drépanocytaires")

# --- Téléchargement direct depuis GitHub ---
st.header("1. Chargement des données depuis GitHub")
github_raw_url = "https://raw.githubusercontent.com/isseu181/MEMOIRE-MASTER/main/fichier_nettoye.xlsx"

try:
    df = pd.read_excel(github_raw_url)
    st.success("Fichier chargé depuis GitHub avec succès !")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Erreur lors du téléchargement du fichier : {e}")
    st.stop()

# --- Analyse descriptive univariée ---
st.header("2. Analyse descriptive univariée")

categorical_vars = df.select_dtypes(include='object').columns.tolist()
numeric_vars = df.select_dtypes(include=np.number).columns.tolist()

st.subheader("Variables catégorielles")
for col in categorical_vars:
    st.write(f"### {col}")
    st.bar_chart(df[col].value_counts())

st.subheader("Variables numériques")
st.write(df[numeric_vars].describe())

st.subheader("Histogrammes interactifs")
selected_num = st.selectbox("Choisir une variable numérique à visualiser", numeric_vars)
fig, ax = plt.subplots()
sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
st.pyplot(fig)

# --- Analyse bivariée ---
st.header("3. Analyse bivariée")
target = "Evolution"

selected_cat = st.selectbox("Choisir une variable catégorielle pour comparer avec Evolution", categorical_vars)
fig, ax = plt.subplots(figsize=(8,4))
sns.countplot(x=selected_cat, hue=target, data=df, ax=ax)
st.pyplot(fig)

selected_num2 = st.selectbox("Choisir une variable numérique pour comparer avec Evolution", numeric_vars)
fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(x=target, y=selected_num2, data=df, ax=ax)
st.pyplot(fig)

# --- Préparation pour modélisation ---
st.header("4. Préparation pour la modélisation")
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_vars:
    if df_encoded[col].nunique() <= 20:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

feature_cols = st.multiselect("Choisir les variables explicatives pour la prédiction", df_encoded.columns.tolist(), default=numeric_vars + categorical_vars)
X = df_encoded[feature_cols]
y = le.fit_transform(df_encoded[target].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Modélisation : Random Forest ---
st.header("5. Modélisation prédictive (Random Forest)")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

st.subheader("Performance du modèle")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# --- Visualisation interactive ---
st.header("6. Visualisation interactive")
selected_x = st.selectbox("Choisir une variable pour l'axe X", df_encoded.columns.tolist())
selected_y = st.selectbox("Choisir une variable pour l'axe Y", df_encoded.columns.tolist())
fig, ax = plt.subplots()
sns.scatterplot(x=df_encoded[selected_x], y=df_encoded[selected_y], hue=df_encoded[target], ax=ax)
st.pyplot(fig)

st.success("Application prête ! Les données ont été téléchargées directement depuis GitHub.")
