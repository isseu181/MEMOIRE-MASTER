import plotly.graph_objects as go
import plotly.express as px

# ---------------- ANALYSE BIVARIÉE ----------------
with onglets[4]:
    st.header("Analyse bivariée : Evolution vs variables")
    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
        cible = "Evolution"

        # ============================
        # Variables qualitatives -> graphe 3D
        # ============================
        variables_qualitatives = [
            "Type de drépanocytose",
            "Sexe",
            "Origine Géographique",
            "Prise en charge",
            "Diagnostic Catégorisé"
        ]

        for var in variables_qualitatives:
            if var not in df_nettoye.columns: 
                continue
            st.subheader(f"{var} vs {cible}")

            # Table croisée normalisée
            cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100

            # Préparer données pour Bar3D
            x, y, z = [], [], []
            for i, cat_var in enumerate(cross_tab.index):
                for j, cat_cible in enumerate(cross_tab.columns):
                    x.append(cat_var)
                    y.append(cat_cible)
                    z.append(cross_tab.loc[cat_var, cat_cible])

            fig = go.Figure(data=[go.Bar3d(
                x=x, y=y, z=z,
                text=[f"{val:.2f}%" for val in z],
                textposition="auto",
                opacity=0.9
            )])

            fig.update_layout(
                title=f"Répartition en 3D de {var} selon {cible}",
                scene=dict(
                    xaxis_title=var,
                    yaxis_title=cible,
                    zaxis_title="Pourcentage"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # ============================
        # Variables quantitatives -> Boxplots
        # ============================
        variables_quantitatives = [
            "Âge du debut d etude en mois (en janvier 2023)",
            "Taux d'Hb (g/dL)",
            "% d'Hb F",
            "Nbre de GB (/mm3)"
        ]

        for var in variables_quantitatives:
            if var in df_nettoye.columns:
                df_nettoye[var] = pd.to_numeric(df_nettoye[var], errors="coerce")
                st.subheader(f"{var} vs {cible}")
                fig = px.box(df_nettoye, x=cible, y=var, points="all",
                             title=f"Distribution de {var} selon {cible}")
                st.plotly_chart(fig, use_container_width=True)

    except Exception:
        pass
