# ================================
# 1️⃣ Imports
# ================================
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

# ================================
# 2️⃣ Chargement des données
# ================================
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    return df

df = load_data("fichier_nettoye.xlsx")

# ================================
# 3️⃣ Variables sélectionnées
# ================================
variables_selection = [
    'Âge de début des signes (en mois)', 'NiveauUrgence', 'GR (/mm3)', 'GB (/mm3)',
    "Nbre d'hospitalisations avant 2017", 'CRP Si positive (Valeur)', 'Pâleur',
    'Âge du debut d etude en mois (en janvier 2023)', 'Souffle systolique fonctionnel',
    'VGM (fl/u3)', 'HB (g/dl)', 'Vaccin contre méningocoque', 'Nbre de GB (/mm3)',
    "% d'Hb S", 'Âge de découverte de la drépanocytose (en mois)', 'Splénomégalie',
    'Prophylaxie à la pénicilline', "Taux d'Hb (g/dL)", 'Parents Salariés',
    'PLT (/mm3)', 'Diagnostic Catégorisé', 'Prise en charge Hospitalisation',
    'Nbre de PLT (/mm3)', 'TCMH (g/dl)', 'Nbre de transfusion avant 2017',
    'Radiographie du thorax Oui ou Non', "Niveau d'instruction scolarité",
    "Nbre d'hospitalisations entre 2017 et 2023", "% d'Hb F",
    'Douleur provoquée (Os.Abdomen)', 'Mois', 'Vaccin contre pneumocoque',
    'HDJ', 'Nbre de transfusion Entre 2017 et 2023', 'Evolution'
]
df_selected = df[variables_selection].copy()

# ================================
# 4️⃣ Encodage
# ================================
binary_mappings = {
    'Pâleur': {'OUI':1, 'NON':0},
    'Souffle systolique fonctionnel': {'OUI':1, 'NON':0},
    'Vaccin contre méningocoque': {'OUI':1, 'NON':0},
    'Splénomégalie': {'OUI':1, 'NON':0},
    'Prophylaxie à la pénicilline': {'OUI':1, 'NON':0},
    'Parents Salariés': {'OUI':1, 'NON':0},
    'Prise en charge Hospitalisation': {'OUI':1, 'NON':0},
    'Radiographie du thorax Oui ou Non': {'OUI':1, 'NON':0},
    'Douleur provoquée (Os.Abdomen)': {'OUI':1, 'NON':0},
    'Vaccin contre pneumocoque': {'OUI':1, 'NON':0},
}
df_selected.replace(binary_mappings, inplace=True)

ordinal_mappings = {
    'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
    "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
}
df_selected.replace(ordinal_mappings, inplace=True)
df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé', 'Mois'], drop_first=True)

# ================================
# 5️⃣ Standardisation
# ================================
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
    'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
    'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
    "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]
scaler = StandardScaler()
df_selected[quantitative_vars] = scaler.fit_transform(df_selected[quantitative_vars])

# ================================
# 6️⃣ Variable cible
# ================================
df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
X = df_selected.drop(['Evolution', 'Evolution_Cible'], axis=1)
y = df_selected['Evolution_Cible']

# ================================
# 7️⃣ SMOTETomek
# ================================
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X, y)

# ================================
# 8️⃣ Division train/val/test
# ================================
X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ================================
# 9️⃣ Définition des modèles
# ================================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
}

# ================================
# 🔟 Entraînement & évaluation
# ================================
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)[:,1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_test_proba = model.predict_proba(X_test)[:,1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    results[name] = {
        "Model": model,
        "Confusion Matrix": cm,
        "Classification Report": classification_report(y_test, y_test_pred, output_dict=True),
        "AUC-ROC": roc_auc_score(y_test, y_test_proba),
        "Optimal Threshold": optimal_threshold
    }

# ================================
# Streamlit Tabs
# ================================
tab1, tab2, tab3 = st.tabs(["Méthodologie", "Comparaison des modèles", "Meilleur modèle"])

with tab1:
    st.header("Méthodologie et étapes")
    st.write("""
    1. **Chargement et sélection des variables** : nous avons choisi les variables pertinentes pour la prédiction.
    2. **Encodage des variables** : les variables binaires et ordinales ont été codées numériquement et les variables catégorielles via one-hot encoding.
    3. **Standardisation** : les variables quantitatives ont été standardisées.
    4. **Équilibrage des classes** : SMOTETomek a été utilisé pour équilibrer la variable cible.
    5. **Division train/validation/test** : pour entraîner et évaluer les modèles correctement.
    6. **Entraînement des modèles** : Decision Tree, Random Forest, SVM et LightGBM.
    7. **Évaluation et comparaison** : calcul de toutes les métriques (Accuracy, Precision, Recall, F1-Score, AUC-ROC) et visualisation.
    """)

with tab2:
    st.header("Comparaison des modèles")
    summary_metrics = []
    for name, res in results.items():
        report = res['Classification Report']
        summary_metrics.append({
            "Modèle": name,
            "Accuracy": report['accuracy'],
            "Precision": report['macro avg']['precision'],
            "Recall": report['macro avg']['recall'],
            "F1-Score": report['macro avg']['f1-score'],
            "AUC-ROC": res['AUC-ROC']
        })
    summary_df = pd.DataFrame(summary_metrics)
    summary_df["Score_Global"] = summary_df[["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]].mean(axis=1)
    st.dataframe(summary_df.sort_values(by="Score_Global", ascending=False))

    # Graphique interactif
    fig = go.Figure()
    metrics_to_plot = ["Accuracy", "Precision", "AUC-ROC"]
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(x=summary_df["Modèle"], y=summary_df[metric], name=metric))
    fig.update_layout(barmode='group', title="Comparaison des modèles", yaxis=dict(range=[0,1]))
    st.plotly_chart(fig)

with tab3:
    st.header("Meilleur modèle")
    best_model_name = summary_df.sort_values(by="Score_Global", ascending=False).iloc[0]["Modèle"]
    best_model_res = results[best_model_name]
    st.subheader(f"Modèle sélectionné : {best_model_name}")
    st.write("Matrice de confusion :")
    st.write(best_model_res["Confusion Matrix"])

    # Courbe AUC
    y_test_proba = best_model_res["Model"].predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    fig_auc = go.Figure()
    fig_auc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig_auc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig_auc.update_layout(title="Courbe AUC-ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_auc)
