import plotly.graph_objects as go

# ---------------- ANALYSE BIVARIÉE ----------------
with onglets[4]:
    st.header("Analyse bivariée : Evolution vs variables")
    try:
        df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
        cible = "Evolution"
        variables = ["Type de drépanocytose","Sexe",
                     "Origine Géographique","Prise en charge","Diagnostic Catégorisé"]

        for var in variables:
            if var not in df_nettoye.columns: 
                continue
            st.subheader(f"{var} vs {cible}")

            # Table croisée en pourcentage
            cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100

            # Coordonnées pour la 3D
            x = []
            y = []
            z = []
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

        # Variables quantitatives (toujours en boxplot)
        quant_vars = ["Âge du debut d etude en mois (en janvier 2023)"]
        for var in quant_vars:
            if var in df_nettoye.columns:
                st.subheader(f"{var} vs {cible}")
                fig = px.box(df_nettoye, x=cible, y=var,
                             title=f"Boxplot {var} selon {cible}")
                st.plotly_chart(fig, use_container_width=True)

    except Exception:
        pass
