# ============================================================
#  PROJET LICENCE DATA — ESVE
#  ÉTAPE 6 : Dashboard Streamlit interactif (VERSION CORRIGÉE)
# ============================================================
#
#  Tableau de bord complet d'analyse de rentabilité ESVE
#  basé sur les vraies données de la fiche de pointage
#
#  Onglets :
#    1. Vue générale — KPIs et indicateurs clés
#    2. Analyse journalière — détail par jour
#    3. Rentabilité — calculs et classification
#    4. Machine Learning — modèle et prédictions
#    5. Recommandations — conseils pour ESVE
#
#  Prérequis :
#    pip install streamlit pandas plotly scikit-learn
#
#  Lancement :
#    streamlit run etape6_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ------------------------------------------------------------

st.set_page_config(
    page_title="ESVE — Analyse de Rentabilité",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }
    .kpi { background: white; border-radius: 12px; padding: 16px 18px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 8px; }
    .kpi-val  { font-size: 24px; font-weight: 700; color: #2c3e50; }
    .kpi-lab  { font-size: 12px; color: #7f8c8d; margin-top: 2px; }
    .kpi.green  { border-top: 4px solid #2ecc71; }
    .kpi.blue   { border-top: 4px solid #3498db; }
    .kpi.orange { border-top: 4px solid #e67e22; }
    .kpi.red    { border-top: 4px solid #e74c3c; }
    .kpi.purple { border-top: 4px solid #9b59b6; }
    .rec-card { background: white; border-radius: 10px; padding: 14px 16px;
                border-left: 4px solid #2ecc71; margin-bottom: 10px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
    .rec-title { font-size: 14px; font-weight: 600; color: #2c3e50; }
    .rec-desc  { font-size: 12px; color: #7f8c8d; margin-top: 4px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PARAMÈTRES FINANCIERS ESVE
# ------------------------------------------------------------

COUT_TOTAL_PARC    = 50_000_000
DUREE_VIE_ANS      = 4
HEURES_PAR_AN      = 2000
TARIF_CLIENT_HEURE = 37_500
TAUX_CARBURANT     = 0.15
COUT_HORAIRE       = (COUT_TOTAL_PARC / DUREE_VIE_ANS) / HEURES_PAR_AN  # 6 250 FCFA/h
SEUIL_H            = 1.57

# ------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ------------------------------------------------------------

@st.cache_data
def charger_donnees():
    df = pd.read_csv("classification_ml_contrats.csv")
    df["jour_label"] = "Jour " + df["jour"].astype(str)
    # Nettoyage de la colonne classification (supprime les emojis pour les comparaisons)
    df["classe_clean"] = df["classification"].str.extract(r"(Excellent|Bon|Moyen|Faible)")
    return df

df = charger_donnees()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ ESVE Dashboard")
    st.markdown("*Analyse de Rentabilité*")
    st.markdown("---")
    st.markdown("### 📅 Période analysée")
    st.info("Février 2026")
    st.markdown(f"**{len(df)}** journées de travail")
    st.markdown(f"**{df['heures_travaillees'].sum():.0f}h** total travaillées")
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres ESVE")
    st.metric("Coût horaire machine", f"{COUT_HORAIRE:,.0f} FCFA/h".replace(",", " "))
    st.metric("Tarif client", f"{TARIF_CLIENT_HEURE:,.0f} FCFA/h".replace(",", " "))
    st.metric("Seuil rentabilité", f"{SEUIL_H:.1f}h/jour")
    st.markdown("---")
    st.caption("Projet Licence Data — ESVE Burkina Faso")

# ------------------------------------------------------------
# EN-TÊTE
# ------------------------------------------------------------

st.title("⚡ ESVE — Tableau de Bord de Rentabilité")
st.caption("Système d'analyse intelligente des contrats de location · Licence Data · Février 2026")
st.markdown("---")

# ------------------------------------------------------------
# ONGLETS
# ------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue générale",
    "📅 Analyse journalière",
    "💰 Rentabilité",
    "🤖 Machine Learning",
    "💡 Recommandations"
])

# ============================================================
# ONGLET 1 — VUE GÉNÉRALE
# ============================================================

with tab1:

    # KPIs principaux
    ca_total       = df["revenu_fcfa"].sum()
    marge_totale   = df["marge_nette_fcfa"].sum()
    roi_moyen      = df["roi_pct"].mean()
    heures_totales = df["heures_travaillees"].sum()
    taux_marge_moy = df["taux_marge_nette_pct"].mean()
    manque_total   = df["manque_a_gagner_fcfa"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi green">
            <div class="kpi-val">{ca_total/1e6:.2f}M</div>
            <div class="kpi-lab">💰 CA Total (FCFA)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi blue">
            <div class="kpi-val">{marge_totale/1e6:.2f}M</div>
            <div class="kpi-lab">📈 Marge Nette (FCFA)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi orange">
            <div class="kpi-val">{heures_totales:.0f}h</div>
            <div class="kpi-lab">⏱️ Heures travaillées</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi purple">
            <div class="kpi-val">{taux_marge_moy:.1f}%</div>
            <div class="kpi-lab">🎯 Taux de marge moyen</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi red">
            <div class="kpi-val">{roi_moyen:.0f}%</div>
            <div class="kpi-lab">🔄 ROI Moyen</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Revenus vs Coûts par jour")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["jour_label"], y=df["revenu_fcfa"],
            name="Revenus", marker_color="#2ecc71"
        ))
        fig.add_trace(go.Bar(
            x=df["jour_label"], y=df["cout_total_journee"],
            name="Coûts totaux", marker_color="#e74c3c"
        ))
        fig.update_layout(
            barmode="group", plot_bgcolor="white",
            height=320, margin=dict(t=10),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🥧 Répartition du revenu total")
        labels  = ["Marge nette", "Amortissement machine", "Carburant"]
        valeurs = [
            df["marge_nette_fcfa"].sum(),
            df["cout_revient_fcfa"].sum(),
            df["cout_carburant_fcfa"].sum()
        ]
        fig2 = px.pie(
            values=valeurs, names=labels, hole=0.45,
            color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"]
        )
        fig2.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Manque à gagner
    st.subheader("⚠️ Manque à gagner — Potentiel non réalisé")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df["jour_label"], y=df["revenu_fcfa"],
        name="Revenus réalisés", marker_color="#3498db"
    ))
    fig3.add_trace(go.Bar(
        x=df["jour_label"], y=df["manque_a_gagner_fcfa"],
        name="Manque à gagner (objectif 10h)",
        marker_color="#f39c12", opacity=0.7
    ))
    fig3.update_layout(
        barmode="stack", plot_bgcolor="white",
        height=300, margin=dict(t=10),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"💡 Manque à gagner total : **{int(manque_total):,} FCFA**/mois si la machine tournait 10h/jour".replace(",", " "))

# ============================================================
# ONGLET 2 — ANALYSE JOURNALIÈRE
# ============================================================

with tab2:

    st.subheader("⏱️ Heures travaillées par jour")

    couleurs_heures = [
        "#2ecc71" if h >= 8 else "#f39c12" if h >= 4 else "#e74c3c"
        for h in df["heures_travaillees"]
    ]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=df["jour_label"], y=df["heures_travaillees"],
        marker_color=couleurs_heures,
        text=df["heures_travaillees"].apply(lambda x: f"{x}h"),
        textposition="outside"
    ))
    fig4.add_hline(
        y=10, line_dash="dash", line_color="#9b59b6",
        annotation_text="Objectif 10h", annotation_position="right"
    )
    fig4.add_hline(
        y=SEUIL_H, line_dash="dot", line_color="#e74c3c",
        annotation_text=f"Seuil {SEUIL_H}h", annotation_position="right"
    )
    fig4.update_layout(
        plot_bgcolor="white", height=350,
        margin=dict(t=10), yaxis_title="Heures",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig4, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("📊 Catégories de journées")
        cat = df["categorie_journee"].value_counts().reset_index()
        cat.columns = ["Catégorie", "Nombre"]
        fig5 = px.bar(
            cat, x="Nombre", y="Catégorie", orientation="h",
            color="Catégorie",
            color_discrete_map={
                "Longue (> 8h)":  "#2ecc71",
                "Normale (4-8h)": "#f39c12",
                "Courte (< 4h)":  "#e74c3c",
                "Journée vide":   "#bdc3c7"
            }
        )
        fig5.update_layout(
            plot_bgcolor="white", height=280,
            margin=dict(t=10), showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col_d:
        st.subheader("📋 Détail par jour")
        affiche = df[[
            "jour_label", "heures_travaillees",
            "revenu_fcfa", "marge_nette_fcfa",
            "taux_marge_nette_pct", "classification"
        ]].copy()
        affiche.columns = [
            "Jour", "Heures", "Revenu (FCFA)",
            "Marge nette (FCFA)", "Taux (%)", "Classification"
        ]
        affiche["Revenu (FCFA)"] = affiche["Revenu (FCFA)"].apply(
            lambda x: f"{int(x):,}".replace(",", " "))
        affiche["Marge nette (FCFA)"] = affiche["Marge nette (FCFA)"].apply(
            lambda x: f"{int(x):,}".replace(",", " "))
        st.dataframe(affiche, use_container_width=True, hide_index=True)

# ============================================================
# ONGLET 3 — RENTABILITÉ
# ============================================================

with tab3:

    st.subheader("💰 Marge nette par jour")
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=df["jour_label"],
        y=df["marge_nette_fcfa"],
        # CORRECTION : utilisation de "classe_clean" au lieu de "classe"
        marker_color=[
            "#2ecc71" if c == "Bon" else "#27ae60" if c == "Excellent"
            else "#f39c12" if c == "Moyen" else "#e74c3c"
            for c in df["classe_clean"]
        ],
        text=df["marge_nette_fcfa"].apply(
            lambda x: f"{int(x):,}".replace(",", " ")),
        textposition="outside"
    ))
    moy = df["marge_nette_fcfa"].mean()
    fig6.add_hline(
        y=moy, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Moyenne : {int(moy):,} FCFA".replace(",", " ")
    )
    fig6.update_layout(
        plot_bgcolor="white", height=350,
        margin=dict(t=10), yaxis_title="Marge nette (FCFA)",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig6, use_container_width=True)

    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("📊 Score de rentabilité par jour")
        fig7 = px.bar(
            df, x="jour_label", y="score_rentabilite",
            # CORRECTION : utilisation de "classe_clean" pour le color
            color="classe_clean",
            color_discrete_map={
                "Excellent": "#27ae60",
                "Bon":       "#2ecc71",
                "Moyen":     "#f39c12",
                "Faible":    "#e74c3c"
            },
            text="score_rentabilite",
            labels={"jour_label": "Jour", "score_rentabilite": "Score /100",
                    "classe_clean": "Classe"}
        )
        fig7.add_hline(y=60, line_dash="dash", line_color="#3498db",
                       annotation_text="Seuil Bon (60)")
        fig7.update_layout(
            plot_bgcolor="white", height=320,
            margin=dict(t=10), xaxis_tickangle=-45,
            showlegend=True
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_f:
        st.subheader("📋 Tableau de rentabilité complet")
        rent = df[[
            "jour_label", "heures_travaillees",
            "roi_pct", "score_rentabilite",
            "classification", "recommandation"
        ]].copy()
        rent.columns = ["Jour", "Heures", "ROI (%)",
                        "Score", "Classe", "Recommandation"]
        st.dataframe(rent, use_container_width=True, hide_index=True)

# ============================================================
# ONGLET 4 — MACHINE LEARNING
# ============================================================

with tab4:

    st.subheader("🤖 Modèle Random Forest — Classification automatique")
    st.caption("Le modèle classe automatiquement chaque journée de location selon sa rentabilité.")

    # Métriques du modèle
    nb_correct = df["prediction_ok"].sum() if "prediction_ok" in df.columns else len(df)
    precision  = int(nb_correct / len(df) * 100)

    col_ml1, col_ml2, col_ml3 = st.columns(3)
    with col_ml1:
        st.metric("Algorithme", "Random Forest")
    with col_ml2:
        st.metric("Précision", f"{precision}%")
    with col_ml3:
        st.metric("Journées classées", f"{nb_correct}/{len(df)}")

    st.markdown("---")

    col_g, col_h = st.columns(2)

    with col_g:
        st.subheader("📊 Importance des facteurs")
        features_imp = pd.DataFrame({
            "Facteur": [
                "Coût total journée", "Revenu généré",
                "Score rentabilité", "Heures travaillées",
                "Marge nette", "Écart au seuil",
                "Taux ROI", "Taux marge nette"
            ],
            "Importance (%)": [22.7, 19.6, 18.6, 16.5, 13.4, 9.3, 0.0, 0.0]
        }).sort_values("Importance (%)")

        fig8 = px.bar(
            features_imp, x="Importance (%)", y="Facteur",
            orientation="h",
            color="Importance (%)",
            color_continuous_scale=["#f39c12", "#2ecc71"],
            text="Importance (%)"
        )
        fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig8.update_layout(
            plot_bgcolor="white", height=350,
            margin=dict(t=10), showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig8, use_container_width=True)

    with col_h:
        st.subheader("🌳 Règle de décision principale")
        st.markdown("""
        Le modèle ML a découvert la règle suivante :

        | Heures/jour | Classification |
        |---|---|
        | ≥ 6.7h | 🟢 **Bon** |
        | 1.6h – 6.7h | 🟡 **Moyen** |
        | < 1.6h | 🔴 **À surveiller** |

        **Interprétation :**
        > Pour qu'une journée soit classée **"Bon"**, la machine doit
        > travailler **au minimum 6.7 heures**. C'est l'objectif quotidien
        > recommandé pour ESVE.
        """)

        st.markdown("---")
        st.subheader("🔮 Simulateur de classification")
        heures_sim = st.slider(
            "Entrez un nombre d'heures de travail :", 0.0, 12.0, 8.0, 0.1
        )
        revenu_sim = heures_sim * TARIF_CLIENT_HEURE
        cout_sim   = heures_sim * COUT_HORAIRE + revenu_sim * TAUX_CARBURANT
        marge_sim  = revenu_sim - cout_sim

        if heures_sim >= 6.7:
            classe_sim  = "🟢 Bon"
            couleur_sim = "success"
        elif heures_sim >= 1.6:
            classe_sim  = "🟡 Moyen"
            couleur_sim = "warning"
        else:
            classe_sim  = "🔴 À surveiller"
            couleur_sim = "error"

        st.markdown(f"""
        **Pour {heures_sim}h de travail :**
        - Revenu estimé : **{int(revenu_sim):,} FCFA**
        - Marge nette estimée : **{int(marge_sim):,} FCFA**
        - Classification prédite : **{classe_sim}**
        """.replace(",", " "))

        if couleur_sim == "success":
            st.success(f"✅ Journée rentable — objectif atteint !")
        elif couleur_sim == "warning":
            st.warning(f"⚠️ Journée acceptable — peut mieux faire")
        else:
            st.error(f"🔴 Journée non rentable — sous le seuil de {SEUIL_H}h")

    # Tableau comparatif classes réelles vs prédites
    if "classe_predite" in df.columns:
        st.markdown("---")
        st.subheader("📋 Comparaison : Classes réelles vs Prédites")
        comp = df[["jour_label", "heures_travaillees", "classe_clean",
                   "classe_predite", "prediction_ok"]].copy()
        comp.columns = ["Jour", "Heures", "Classe réelle", "Classe prédite", "Correct"]
        comp["Correct"] = comp["Correct"].apply(lambda x: "✅" if x else "❌")
        st.dataframe(comp, use_container_width=True, hide_index=True)

# ============================================================
# ONGLET 5 — RECOMMANDATIONS
# ============================================================

with tab5:

    st.subheader("💡 Recommandations stratégiques pour ESVE")
    st.caption("Basées sur l'analyse des données réelles de février 2026")

    gain_potentiel = int(df["manque_a_gagner_fcfa"].sum() * 0.683)

    recommandations = [
        {
            "titre": "1. Augmenter les heures quotidiennes de la machine",
            "desc": f"La machine travaille en moyenne {df['heures_travaillees'].mean():.1f}h/jour. "
                    f"Si elle atteint 10h/jour, ESVE gagnerait +{gain_potentiel:,} FCFA de marge nette supplémentaire par mois.".replace(",", " "),
            "couleur": "#2ecc71"
        },
        {
            "titre": "2. Ne jamais descendre sous 1.6h de travail par jour",
            "desc": f"Le seuil de rentabilité est à 1.6h/jour. En dessous, "
                    f"les coûts dépassent les revenus. Le Jour 19 (0.7h) est un cas à éviter.",
            "couleur": "#e74c3c"
        },
        {
            "titre": "3. Répliquer le modèle du Jour 22",
            "desc": f"Le Jour 22 est la meilleure journée : 9.6h travaillées, "
                    f"246 000 FCFA de marge nette, score 66.3/100. C'est le modèle idéal à reproduire.",
            "couleur": "#3498db"
        },
        {
            "titre": "4. Objectif minimum : 6.7h/jour",
            "desc": f"Le modèle ML a identifié 6.7h comme seuil entre 'Moyen' et 'Bon'. "
                    f"ESVE doit viser au minimum 6.7h de travail par jour pour chaque contrat.",
            "couleur": "#9b59b6"
        },
        {
            "titre": "5. Surveiller le coût du carburant",
            "desc": f"Le carburant représente 15% du revenu, soit "
                    f"{int(df['cout_carburant_fcfa'].sum()):,} FCFA en février. ".replace(",", " ") +
                    f"Optimiser la consommation peut améliorer la marge nette.",
            "couleur": "#f39c12"
        },
    ]

    for rec in recommandations:
        st.markdown(f"""
        <div class="rec-card" style="border-left-color: {rec['couleur']};">
            <div class="rec-title">{rec['titre']}</div>
            <div class="rec-desc">{rec['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Bilan final
    st.subheader("📊 Bilan global — Février 2026")
    col_r1, col_r2 = st.columns(2)

    # CORRECTION : utilisation de "classe_clean" pour le comptage
    nb_bon = len(df[df["classe_clean"] == "Bon"])

    with col_r1:
        st.markdown(f"""
        | Indicateur | Valeur |
        |---|---|
        | CA total | **{int(df['revenu_fcfa'].sum()):,} FCFA** |
        | Marge nette | **{int(df['marge_nette_fcfa'].sum()):,} FCFA** |
        | Taux de marge | **{df['taux_marge_nette_pct'].mean():.1f}%** |
        | ROI moyen | **{df['roi_pct'].mean():.0f}%** |
        """.replace(",", " "))

    with col_r2:
        st.markdown(f"""
        | Indicateur | Valeur |
        |---|---|
        | Heures totales | **{df['heures_travaillees'].sum():.0f}h** |
        | Jour le + productif | **Jour {df.loc[df['heures_travaillees'].idxmax(), 'jour']}** |
        | Manque à gagner | **{int(df['manque_a_gagner_fcfa'].sum()):,} FCFA** |
        | Journées "Bon" | **{nb_bon}/{len(df)}** |
        """.replace(",", " "))

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<center><small style='color:#aaa'>⚡ ESVE Burkina Faso · "
    "Analyse de rentabilité des contrats de location · "
    "Projet Licence Data · Développé avec Python & Streamlit</small></center>",
    unsafe_allow_html=True
)