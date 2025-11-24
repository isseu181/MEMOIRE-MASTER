# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display, HTML # Pour l'affichage dans des environnements comme Jupyter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

from imblearn.combine import SMOTETomek

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV 

warnings.filterwarnings('ignore') # Ignorer les avertissements Scikit-learn/LGBM

# ==============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================
try:
    df = pd.read_excel("fichier_nettoye.xlsx")
except FileNotFoundError:
    print("Erreur: Le fichier 'fichier_nettoye.xlsx' n'a pas été trouvé. Veuillez vérifier le chemin.")
    exit()

# Définition des variables
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

# Encodage binaire
binary_mappings = {col: {'OUI':1, 'NON':0} for col in [
    'Pâleur', 'Souffle systolique fonctionnel', 'Vaccin contre méningocoque', 
    'Splénomégalie', 'Prophylaxie à la pénicilline', 'Parents Salariés', 
    'Prise en charge Hospitalisation', 'Radiographie du thorax Oui ou Non', 
    'Douleur provoquée (Os.Abdomen)', 'Vaccin contre pneumocoque'
]}
df_selected.replace(binary_mappings, inplace=True)

# Encodage ordinal
ordinal_mappings = {
    'NiveauUrgence': {'Urgence1':1, 'Urgence2':2, 'Urgence3':3, 'Urgence4':4, 'Urgence5':5, 'Urgence6':6},
    "Niveau d'instruction scolarité": {'Maternelle ':1, 'Elémentaire ':2, 'Secondaire':3, 'Enseignement Supérieur ':4, 'NON':0}
}
df_selected.replace(ordinal_mappings, inplace=True)

# One-Hot Encoding
df_selected = pd.get_dummies(df_selected, columns=['Diagnostic Catégorisé', 'Mois'])

# Séparation X et y
df_selected['Evolution_Cible'] = df_selected['Evolution'].map({'Favorable':0, 'Complications':1})
X = df_selected.drop(['Evolution', 'Evolution_Cible'], axis=1)
y = df_selected['Evolution_Cible']

# ==============================================================================
# 3. DIVISION, STANDARDISATION et SMOTETomek (Pipeline Correct)
# ==============================================================================

# Variables quantitatives pour la standardisation
quantitative_vars = [
    'Âge de début des signes (en mois)', 'GR (/mm3)', 'GB (/mm3)',
    'Âge du debut d etude en mois (en janvier 2023)', 'VGM (fl/u3)', 'HB (g/dl)',
    'Nbre de GB (/mm3)', 'PLT (/mm3)', 'Nbre de PLT (/mm3)', 'TCMH (g/dl)',
    "Nbre d'hospitalisations avant 2017", "Nbre d'hospitalisations entre 2017 et 2023",
    'Nbre de transfusion avant 2017', 'Nbre de transfusion Entre 2017 et 2023',
    'CRP Si positive (Valeur)', "Taux d'Hb (g/dL)", "% d'Hb S", "% d'Hb F"
]

# A. Division Train/Validation/Test (80% / 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Division Train/Validation (60% / 20%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# B. Standardisation (Fit uniquement sur X_train)
scaler = StandardScaler()
X_train[quantitative_vars] = scaler.fit_transform(X_train[quantitative_vars])
X_val[quantitative_vars] = scaler.transform(X_val[quantitative_vars])
X_test[quantitative_vars] = scaler.transform(X_test[quantitative_vars])

# C. SMOTETomek (Appliqué uniquement sur X_train)
print("--- Déséquilibre ---")
print("Avant SMOTETomek sur TRAIN :")
print(y_train.value_counts())

smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

print("\nAprès SMOTETomek sur TRAIN :")
print(pd.Series(y_train_res).value_counts())

X_train = X_train_res
y_train = y_train_res

# ==============================================================================
# 4. FONCTION D'ÉVALUATION ET DÉFINITION DES MODÈLES
# ==============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Entraîne et évalue un modèle sur X_val, renvoie les métriques et le seuil optimal."""

    # Gérer les modèles sans predict_proba (CalibratedClassifierCV pour SVM/SVC)
    if not hasattr(model, "predict_proba"):
        model_calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5, ensemble=False)
        model_calibrated.fit(X_train, y_train)
        model = model_calibrated
    else:
        model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de validation
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # AUC-ROC
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    
    # Seuil optimal (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Évaluation au seuil optimal
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    report = classification_report(y_val, y_pred_optimal, output_dict=True, zero_division=0)
    
    return {
        'Model': model,
        'AUC-ROC': auc_roc,
        'Optimal Threshold': optimal_threshold,
        'Classification Report': report,
    }

# Définition des modèles avec les PARAMÈTRES SPÉCIFIÉS
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
    
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=10, random_state=42),
    
    # SVC avec probability=False pour être cohérent avec CalibratedClassifierCV ci-dessus
    "SVM (SVC)": SVC(C=1.0, kernel='rbf', gamma='scale', probability=False, random_state=42), 
    
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
}

# Entraînement et évaluation sur l'ensemble de validation
results = {}
print("\n--- Entraînement et évaluation des modèles sur Validation Set ---")
for name, model in models.items():
    print(f" - Modèle {name}...")
    results[name] = evaluate_model(model, X_train, y_train, X_val, y_val)
print("Évaluation terminée.")

# ==============================================================================
# 5. RÉSULTATS RÉCAPITULATIFS (TABLEAU STYLISÉ)
# ==============================================================================

summary_metrics = []
for name, res in results.items():
    report = res['Classification Report']
    summary_metrics.append({
        "Modèle": name,
        "Accuracy": round(report['accuracy'],3),
        "Precision": round(report['weighted avg']['precision'],3),
        "Recall": round(report['weighted avg']['recall'],3),
        "F1-Score": round(report['weighted avg']['f1-score'],3),
        "AUC-ROC": round(res['AUC-ROC'],3),
        "Seuil optimal": round(res['Optimal Threshold'],3)
    })

summary_df = pd.DataFrame(summary_metrics).sort_values(by="AUC-ROC", ascending=False).reset_index(drop=True)

print("\n--- Tableau de synthèse des performances (Validation Set) ---")
summary_df_styled = summary_df.style.background_gradient(subset=["Accuracy","Precision","Recall","F1-Score","AUC-ROC"], cmap='Blues') \
                             .set_properties(**{'text-align': 'center'}) \
                             .set_caption(" Comparaison des modèles selon leurs métriques (Validation Set)")
display(summary_df_styled)

# ==============================================================================
# 6. ÉVALUATION FINALE SUR L'ENSEMBLE DE TEST
# ==============================================================================

print("\n" + "="*70)
print("             ÉVALUATION FINALE SUR L'ENSEMBLE DE TEST")
print("="*70)

# Sélection du meilleur modèle basé sur l'AUC-ROC sur l'ensemble de validation
best_model_name = summary_df.iloc[0]['Modèle']
best_result = results[best_model_name]
best_model = best_result['Model']
optimal_threshold = best_result['Optimal Threshold']

print(f"Meilleur Modèle retenu (selon AUC-ROC sur Validation) : **{best_model_name}**")
print(f"Seuil Optimal appliqué : **{optimal_threshold:.3f}**")

# Prédictions sur l'ensemble de TEST
# Le modèle (qui peut être CalibratedClassifierCV) doit avoir predict_proba
y_test_proba = best_model.predict_proba(X_test)[:, 1]

# Application du seuil optimal trouvé sur l'ensemble de validation
y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

# Calcul des métriques finales
test_auc = roc_auc_score(y_test, y_test_proba)
test_cm = confusion_matrix(y_test, y_test_pred_optimal)

# Affichage du Rapport et de la Matrice finale
print("\n--- Rapport de Classification Final (Ensemble de Test) ---")
print(classification_report(y_test, y_test_pred_optimal, zero_division=0))
print(f"AUC-ROC Final sur Test: {test_auc:.3f}")

plt.figure(figsize=(7, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Favorable (0)', 'Complications (1)'],
            yticklabels=['Favorable (0)', 'Complications (1)'],
            linewidths=.5, linecolor='black', annot_kws={"size": 14})
plt.title(f'Matrice de Confusion Finale: {best_model_name}\n(Test Set - Seuil: {optimal_threshold:.3f})', fontsize=14)
plt.xlabel('Prédiction', fontsize=12)
plt.ylabel('Valeur Réelle', fontsize=12)
plt.show()
