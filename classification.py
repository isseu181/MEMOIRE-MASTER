import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Chargement des fichiers
# ================================
df = pd.read_excel("fichier_nettoye.xlsx")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# ================================
# Préparation des données pour la prédiction
# ================================
df_selected = df[features + ['Evolution']].copy()
binary_cols = [
    'Pâleur','Souffle systolique fonctionnel','Vaccin contre méningocoque','Splénomégalie',
    'Prophylaxie à la pénicilline','Parents Salariés','Prise en charge Hospitalisation',
    'Radiographie du thorax Oui ou Non','Douleur provoquée (Os.Abdomen)','Vaccin contre pneumocoque'
]
for col in binary_cols:
    df_selected[col] = df_selected[col].map({'OUI':1,'NON':0})

quantitative_vars = [col for col in features if col not in binary_cols]
df_selected[quantitative_vars] = scaler.transform(df_selected[quantitative_vars])
X = df_selected[features]
y = df_selected['Evolution'].map({'Favorable':0,'Complications':1})

# ================================
# App Streamlit
# ================================
def show_classification():
    st.title("Classification de l'évolution des patients")
    tabs = st.tabs(["Performance", "Variables importantes", "Méthodologie", "Simulateur"])

    # --- Onglet 1 : Performance ---
    with tabs[0]:
        st.subheader("Performance des modèles")
        st.dataframe(results_df.style.format({
            "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
            "F1-Score":"{:.3f}","AUC-ROC":"{:.3f}","Seuil optimal":"{:.3f}"
        }))

        # Sélection du meilleur modèle par métrique composite (ici on choisit celui avec max Accuracy)
        best_model_row = results_df.loc[results_df["Accuracy"].idxmax()]
        st.markdown(f"### Modèle sélectionné : {best_model_row['Modèle']}")

        # Matrice de confusion et ROC
        optimal_threshold = best_model_row['Seuil optimal']
        y_pred_proba = rf_model.predict_proba(X)[:,1]
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        cm = confusion_matrix(y, y_pred)
        st.write("Matrice de confusion : 0=Favorable, 1=Complications")
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        st.pyplot(plt)

        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_row["Modèle"]} (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('1 - Spécificité (Faux positifs)')
        plt.ylabel('Sensibilité (Vrais positifs)')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        st.write(f"Accuracy : {best_model_row['Accuracy']:.3f}, Precision : {best_model_row['Precision']:.3f}, Recall : {best_model_row['Recall']:.3f}, F1-Score : {best_model_row['F1-Score']:.3f}, Seuil optimal : {optimal_threshold:.3f}")

