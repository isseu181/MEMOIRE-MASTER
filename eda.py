def show_eda():
    st.subheader("📊 Analyse exploratoire des données")
    file_path = "Base_de_données_USAD_URGENCES1.xlsx"
    
    try:
        feuilles = pd.read_excel(file_path, sheet_name=None)
        st.success("Fichier chargé avec succès !")
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {file_path}")
        return

    # ----------------------------
    # 1️⃣ Identité
    # ----------------------------
    if 'Identite' in feuilles:
        identite = convertir_df_oui_non(feuilles['Identite'], exclude_columns=["Niveau d'instruction scolarité"])
        st.markdown("### 1️⃣ Identité des patients")
        st.write("Nombre total de patients:", len(identite))

        # Sexe
        if 'Sexe' in identite.columns:
            sexe_counts = identite['Sexe'].value_counts().to_dict()
            st.write("Répartition par sexe:", sexe_counts)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(sexe_counts.values()), labels=list(sexe_counts.keys()), autopct="%1.1f%%", startangle=140)
            ax.set_title("Répartition par sexe")
            st.pyplot(fig)

        # Origine géographique
        if 'Origine Géographique' in identite.columns:
            origine_counts = identite['Origine Géographique'].value_counts().to_dict()
            st.write("Répartition par origine géographique:", origine_counts)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(list(origine_counts.values()), labels=list(origine_counts.keys()), autopct="%1.1f%%", startangle=140)
            ax.set_title("Répartition par origine géographique")
            st.pyplot(fig)

        # Âge
        age_col = "Âge du debut d etude en mois (en janvier 2023)"
        if age_col in identite.columns:
            st.markdown("#### Âge à l’inclusion")
            fig, ax = plt.subplots(figsize=(8,6))
            identite[age_col].dropna().hist(bins=20, color="#36A2EB", edgecolor="white", ax=ax)
            ax.set_title("Répartition des âges à l’inclusion")
            ax.set_xlabel("Âge (mois)")
            ax.set_ylabel("Nombre de patients")
            st.pyplot(fig)

    # ----------------------------
    # 2️⃣ Drépanocytose
    # ----------------------------
    if 'Drépano' in feuilles:
        drepano = convertir_df_oui_non(feuilles['Drépano'])
        st.markdown("### 2️⃣ Type de drépanocytose et paramètres biologiques")

        if 'Type de drépanocytose' in drepano.columns:
            type_counts = drepano['Type de drépanocytose'].value_counts().to_dict()
            st.write("Type de drépanocytose:", type_counts)

        # Paramètres biologiques
        bio_cols = ["Taux d'Hb (g/dL)", "% d'Hb F", "% d'Hb S", "% d'HB C", "Nbre de GB (/mm3)", "Nbre de PLT (/mm3)"]
        st.markdown("#### Paramètres biologiques")
        for col in bio_cols:
            if col in drepano.columns:
                data = pd.to_numeric(drepano[col], errors='coerce').dropna()
                if not data.empty:
                    fig, ax = plt.subplots(figsize=(8,4))
                    data.hist(bins=20, color="#36A2EB", edgecolor='white', ax=ax)
                    ax.set_title(col)
                    st.pyplot(fig)

    # ----------------------------
    # 3️⃣ Antécédents médicaux
    # ----------------------------
    if 'Antéccédents' in feuilles:
        antecedents = convertir_df_oui_non(feuilles['Antéccédents'])
        st.markdown("### 3️⃣ Antécédents médicaux")
        bin_cols = [col for col in antecedents.columns if set(antecedents[col].dropna().unique()).issubset({0,1})]
        for col in bin_cols:
            counts = antecedents[col].value_counts().to_dict()
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(counts.keys(), counts.values(), color="#36A2EB")
            ax.set_title(col)
            st.pyplot(fig)

    # ----------------------------
    # 4️⃣ Consultations d'urgence
    # ----------------------------
    st.markdown("### 4️⃣ Consultations d'urgence")
    for i in range(1,7):
        nom = f'Urgence{i}'
        if nom in feuilles:
            df_urg = convertir_df_oui_non(feuilles[nom])
            st.markdown(f"#### {nom}")
            st.write("Nombre de consultations:", len(df_urg))

            symptomes = ['Douleur', 'Fièvre', 'Pâleur', 'Ictère', 'Toux']
            for s in symptomes:
                if s in df_urg.columns:
                    counts = df_urg[s].value_counts().to_dict()
                    st.write(f"{s}: {counts}")

    # ----------------------------
    # 5️⃣ Répartition mensuelle des urgences
    # ----------------------------
    st.markdown("### 📅 Répartition mensuelle des urgences")
    toutes_dates = concat_dates_urgences(feuilles)
    if not toutes_dates.empty:
        repartition_mensuelle = toutes_dates.dt.month.value_counts().sort_index()
        mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                     7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
        repartition_df = pd.DataFrame({
            'Mois':[mois_noms[m] for m in repartition_mensuelle.index],
            'Nombre de consultations':repartition_mensuelle.values
        })
        repartition_df['Pourcentage (%)'] = (repartition_df['Nombre de consultations']/repartition_df['Nombre de consultations'].sum()*100).round(2)
        st.write(repartition_df)

        fig, ax = plt.subplots(figsize=(10,5))
        repartition_df.set_index('Mois')['Nombre de consultations'].plot(kind='bar', color='#36A2EB', ax=ax)
        ax.set_ylabel("Nombre de consultations")
        ax.set_xlabel("Mois")
        ax.set_title("Répartition mensuelle des urgences drépanocytaires")
        st.pyplot(fig)
    else:
        st.write("Aucune donnée de date disponible pour les urgences.")
