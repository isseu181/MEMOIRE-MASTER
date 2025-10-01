    # ---------------- ANALYSE BIVARIÉE ----------------
    with onglets[4]:
        st.header("Analyse bivariée : Evolution vs variables")
        try:
            df_nettoye = pd.read_excel("fichier_nettoye.xlsx")
            cible = "Evolution"

            if cible not in df_nettoye.columns:
                st.warning("⚠️ La variable cible 'Evolution' est absente du fichier.")
            else:
                # QUALITATIVES → 3D
                variables_qualitatives = [
                    "Type de drépanocytose","Sexe","Origine Géographique",
                    "Prise en charge","Diagnostic Catégorisé"
                ]
                for var in variables_qualitatives:
                    if var in df_nettoye.columns:
                        st.subheader(f"{var} vs {cible}")
                        cross_tab = pd.crosstab(df_nettoye[var], df_nettoye[cible], normalize="index")*100
                        cross_tab = cross_tab.round(1)

                        if not cross_tab.empty:
                            # Tentative 3D
                            try:
                                x, y, z = [], [], []
                                for i, cat_var in enumerate(cross_tab.index):
                                    for j, cat_cible in enumerate(cross_tab.columns):
                                        x.append(cat_var)
                                        y.append(cat_cible)
                                        z.append(cross_tab.loc[cat_var, cat_cible])

                                fig = go.Figure(data=[go.Bar3d(
                                    x=x, y=y, z=z,
                                    text=[f"{val:.1f}%" for val in z],
                                    textposition="auto", opacity=0.9
                                )])
                                fig.update_layout(
                                    scene=dict(
                                        xaxis_title=var,
                                        yaxis_title=cible,
                                        zaxis_title="Pourcentage"
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                # Fallback 2D
                                fig = px.bar(cross_tab, barmode="group",
                                             title=f"{var} vs {cible}")
                                st.plotly_chart(fig, use_container_width=True)

                # QUANTITATIVES → boxplots
                variables_quantitatives = [
                    "Âge du debut d etude en mois (en janvier 2023)",
                    "Taux d'Hb (g/dL)", "% d'Hb F", "Nbre de GB (/mm3)"
                ]
                for var in variables_quantitatives:
                    if var in df_nettoye.columns:
                        st.subheader(f"{var} selon {cible}")
                        df_nettoye[var] = pd.to_numeric(df_nettoye[var], errors="coerce")
                        if df_nettoye[var].notna().sum() > 0:
                            fig = px.box(df_nettoye, x=cible, y=var, points="all",
                                         title=f"{var} selon {cible}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"⚠️ Pas de données exploitables pour {var}")
        except FileNotFoundError:
            st.warning("⚠️ Le fichier 'fichier_nettoye.xlsx' est introuvable.")
