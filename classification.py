import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

# ================================
# 1 Chargement des données
# ================================
df = pd.read_excel("fichier_nettoye.xlsx")

# ================================
# 2 Variables sélectionnées
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
# 3 Encodage binaire
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

# ================================
# 4 Encodages à comparer
# ================================

# --- Option 1 : Ordinal pour instruction + One-hot pour mois
df_enc1 = df_selected.copy()
ordinal_mappings = {
    'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
    "Niveau d'instruction scolarité": {'NON':0, 'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4}
}
df_enc1.replace(ordinal_mappings, inplace=True)
df_enc1 = pd.get_dummies(df_enc1, columns=['Diagnostic Catégorisé','Mois'], drop_first=True)

# --- Option 2 : One-hot pour instruction + Encodage cyclique pour mois
df_enc2 = df_selected.copy()
df_enc2.replace({'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6}}, inplace=True)
df_enc2 = pd.get_dummies(df_enc2, columns=["Niveau d'instruction scolarité"], drop_first=True)
df_enc2['Mois_num'] = df_enc2['Mois'].str.extract('(\d+)').astype(int)
df_enc2['Mois_sin'] = np.sin(2 * np.pi * df_enc2['Mois_num']/12)
df_enc2['Mois_cos'] = np.cos(2 * np.pi * df_enc2['Mois_num']/12)
df_enc2.drop(columns=['Mois','Mois_num'], inplace=True)
df_enc2 = pd.get_dummies(df_enc2, columns=['Diagnostic Catégorisé'], drop_first=True)

# ================================
# 5 Fonction pipeline entraînement & évaluation
# ================================
def run_pipeline(df_enc, option_name):
    quantitative_vars = [
        'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
        'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)',
        'HB (g/dl)', 'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)',
        'TCMH (g/dl)', "Nbre d'hospitalisations avant 2017",
        "Nbre d'hospitalisations entre 2017 et 2023",
        'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
        'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
    ]

    df_enc['Evolution_Cible'] = df_enc['Evolution'].map({'Favorable':0, 'Complications':1})
    X = df_enc.drop(['Evolution','Evolution_Cible'], axis=1)
    y = df_enc['Evolution_Cible']

    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.4, stratify=y_res, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[quantitative_vars] = scaler.fit_transform(X_train[quantitative_vars])
    X_val_scaled[quantitative_vars] = scaler.transform(X_val[quantitative_vars])
    X_test_scaled[quantitative_vars] = scaler.transform(X_test[quantitative_vars])

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_val_proba = model.predict_proba(X_val_scaled)[:,1]
        fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]

        y_test_proba = model.predict_proba(X_test_scaled)[:,1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        report = classification_report(y_test, y_test_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_test_proba)

        results.append({
            "Option": option_name,
            "Modèle": name,
            "Accuracy": round(report['accuracy'],3),
            "Precision": round(report['macro avg']['precision'],3),
            "Recall": round(report['macro avg']['recall'],3),
            "F1-Score": round(report['macro avg']['f1-score'],3),
            "AUC-ROC": round(auc,3),
            "Seuil optimal": round(optimal_threshold,3)
        })

    return pd.DataFrame(results)

# ================================
# 6 Lancer les deux options
# ================================
summary1 = run_pipeline(df_enc1, "Ordinal + One-hot mois")
summary2 = run_pipeline(df_enc2, "One-hot instruction + Mois cyclique")
summary_df = pd.concat([summary1, summary2]).reset_index(drop=True)

# ================================
# 7 Résultats comparatifs
# ================================
print(summary_df)
plt.figure(figsize=(12,6))
sns.barplot(x="Modèle", y="AUC-ROC", hue="Option", data=summary_df)
plt.title("Comparaison AUC-ROC selon l'encodage")
plt.ylim(0,1)
plt.show()
