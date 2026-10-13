"""Application Streamlit : prédiction de la qualité d'un vin avec le SVM final.

Lancement :  streamlit run app.py
Fichiers nécessaires (créés par la dernière cellule du notebook), dans le même dossier :
  - modele_svm.joblib   : le pipeline SVM avec son seuil optimisé
  - donnees_app.joblib  : les scores du jeu de test et le profil du vin moyen (explication et onglet 2)
"""
import io

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Qualité du vin", page_icon="🍷", layout="centered")

BLEU, ROUGE, ORANGE, VERT_EAU = "#2a78d6", "#e34948", "#eb6834", "#1baf7a"


def pleine_largeur(fonction, *args, **kwargs):
    """Affiche un élément sur toute la largeur, quelle que soit la version de Streamlit
    (les versions récentes utilisent width="stretch", les anciennes use_container_width=True)."""
    try:
        return fonction(*args, width="stretch", **kwargs)
    except TypeError:
        return fonction(*args, use_container_width=True, **kwargs)

VARIABLES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
             "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
# (min, max, pas, format) d'après le jeu UCI Wine Quality
BORNES = {
    "fixed acidity": (3.8, 15.9, 0.1, "%.1f"), "volatile acidity": (0.08, 1.58, 0.01, "%.2f"),
    "citric acid": (0.0, 1.66, 0.01, "%.2f"), "residual sugar": (0.6, 65.8, 0.1, "%.1f"),
    "chlorides": (0.009, 0.611, 0.001, "%.3f"), "free sulfur dioxide": (1.0, 289.0, 1.0, "%.0f"),
    "total sulfur dioxide": (6.0, 440.0, 1.0, "%.0f"), "density": (0.98711, 1.03898, 0.0001, "%.4f"),
    "pH": (2.72, 4.01, 0.01, "%.2f"), "sulphates": (0.22, 2.0, 0.01, "%.2f"),
    "alcohol": (8.0, 14.9, 0.1, "%.1f"),
}
NOMS_FR = {
    "fixed acidity": "Acidité fixe (g/L)", "volatile acidity": "Acidité volatile (g/L)",
    "citric acid": "Acide citrique (g/L)", "residual sugar": "Sucre résiduel (g/L)",
    "chlorides": "Chlorures (g/L)", "free sulfur dioxide": "SO₂ libre (mg/L)",
    "total sulfur dioxide": "SO₂ total (mg/L)", "density": "Densité (g/cm³)", "pH": "pH",
    "sulphates": "Sulfates (g/L)", "alcohol": "Alcool (% vol.)",
}
# Exemples : médianes de chaque type (entraînement), et deux vrais vins du jeu notés 7 et 8
EXEMPLES = {
    "Vin rouge moyen": ("rouge", [7.9, 0.52, 0.26, 2.2, 0.079, 13, 38, 0.99672, 3.31, 0.62, 10.1]),
    "Vin blanc moyen": ("blanc", [6.8, 0.27, 0.32, 4.75, 0.0425, 33, 132, 0.9935, 3.18, 0.48, 10.4]),
    "Bon rouge (note 7)": ("rouge", [9.1, 0.29, 0.33, 2.05, 0.063, 13, 27, 0.99516, 3.26, 0.84, 11.7]),
    "Bon blanc (note 8)": ("blanc", [7.0, 0.23, 0.36, 7.1, 0.028, 31, 104, 0.9922, 3.35, 0.47, 12.1]),
}


# ------------------------------------------------------------------ chargement
@st.cache_resource
def charger_modele():
    return joblib.load("modele_svm.joblib")


@st.cache_resource
def charger_donnees():
    try:
        return joblib.load("donnees_app.joblib")
    except FileNotFoundError:
        return None


try:
    modele = charger_modele()
except FileNotFoundError:
    st.error("Fichier `modele_svm.joblib` introuvable. Exécutez d'abord la cellule de sauvegarde "
             "à la fin du notebook, puis relancez l'application.")
    st.stop()

donnees = charger_donnees()
SEUIL = float(modele.best_threshold_)
COLONNES = list(getattr(modele, "feature_names_in_", VARIABLES + ["type_white"]))


def tableau(valeurs: dict, type_vin: str) -> pd.DataFrame:
    """Un vin -> DataFrame d'une ligne, colonnes dans l'ordre de l'entraînement."""
    ligne = dict(valeurs, type_white=1 if type_vin == "blanc" else 0)
    return pd.DataFrame([ligne])[COLONNES]


def score(x: pd.DataFrame) -> np.ndarray:
    return modele.decision_function(x[COLONNES])


# ------------------------------------------------------------------ état des champs
def remplir(nom_exemple: str):
    type_vin, vals = EXEMPLES[nom_exemple]
    st.session_state["type_vin"] = type_vin
    for var, v in zip(VARIABLES, vals):
        st.session_state[f"in_{var}"] = float(v)


def changer_type():
    remplir("Vin rouge moyen" if st.session_state["type_vin"] == "rouge" else "Vin blanc moyen")


if "type_vin" not in st.session_state:
    remplir("Vin rouge moyen")

# ------------------------------------------------------------------ interface
st.title("🍷 Ce vin est-il bon ?")
st.write("Le modèle (SVM à noyau RBF, seuil de décision optimisé) prédit si un vin obtiendrait une "
         "note **≥ 7 / 10** à partir de ses mesures physico-chimiques.")

onglet_pred, onglet_seuil, onglet_fichier, onglet_info = st.tabs(
    ["Prédiction", "Seuil et performance", "Prédire un fichier", "À propos"])

# ================================================================== 1. prédiction
with onglet_pred:
    st.caption("Remplir avec un exemple :")
    cols = st.columns(len(EXEMPLES))
    for c, nom in zip(cols, EXEMPLES):
        pleine_largeur(c.button, nom, on_click=remplir, args=(nom,))

    st.radio("Type de vin", ["rouge", "blanc"], horizontal=True, key="type_vin", on_change=changer_type)

    c1, c2 = st.columns(2)
    for i, var in enumerate(VARIABLES):
        vmin, vmax, pas, fmt = BORNES[var]
        (c1 if i % 2 == 0 else c2).number_input(NOMS_FR[var], min_value=vmin, max_value=vmax,
                                                 step=pas, format=fmt, key=f"in_{var}")

    type_vin = st.session_state["type_vin"]
    valeurs = {v: st.session_state[f"in_{v}"] for v in VARIABLES}
    x = tableau(valeurs, type_vin)
    s = float(score(x)[0])
    bon = s >= SEUIL

    st.divider()
    if bon:
        st.success("### ✅ Bon vin : note ≥ 7 prédite")
    else:
        st.error("### ❌ Vin ordinaire : note < 7 prédite")
    m1, m2 = st.columns(2)
    m1.metric("Score du SVM", f"{s:+.2f}")
    m2.metric("Écart au seuil", f"{s - SEUIL:+.2f}", help=f"Seuil optimisé par validation croisée : {SEUIL:.3f}")
    st.caption(f"Le vin est classé « bon » si son score dépasse **{SEUIL:.3f}**. "
               "Plus le score est loin du seuil, plus le modèle est sûr de lui.")

    # ---- explication : effet de chaque variable par rapport à un vin moyen du même type
    if donnees is not None and "profil_moyen" in donnees:
        st.subheader("Pourquoi cette prédiction ?")
        nom_type = "blancs" if type_vin == "blanc" else "rouges"
        ref = donnees["profil_moyen"].loc[1 if type_vin == "blanc" else 0]
        lignes = []
        for var in VARIABLES:
            x2 = x.copy()
            x2[var] = ref[var]
            effet = s - float(score(x2)[0])   # > 0 : cette valeur rend le vin meilleur qu'un vin moyen
            lignes.append({"Variable": NOMS_FR[var], "Effet": effet,
                           "Valeur saisie": valeurs[var], "Vin moyen": float(ref[var]),
                           "Sens": "Rend le vin meilleur" if effet >= 0 else "Rend le vin moins bon"})
        expl = pd.DataFrame(lignes)
        expl = expl.reindex(expl["Effet"].abs().sort_values(ascending=False).index)
        if expl["Effet"].abs().max() < 0.05:
            st.info(f"Ce vin est très proche du vin moyen des {nom_type} : aucune variable ne le distingue. "
                    "Modifiez une mesure (par exemple l'alcool) pour voir son effet.")
        else:
            graphe = alt.Chart(expl).mark_bar(cornerRadiusEnd=4, height=14).encode(
                x=alt.X("Effet:Q", title="Effet sur le score du SVM (par rapport à un vin moyen)"),
                y=alt.Y("Variable:N", sort=list(expl["Variable"]), title=None,
                        axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                color=alt.Color("Sens:N", scale=alt.Scale(domain=["Rend le vin meilleur", "Rend le vin moins bon"],
                                                          range=[BLEU, ROUGE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("Variable:N"), alt.Tooltip("Valeur saisie:Q", format=".4~f"),
                         alt.Tooltip("Vin moyen:Q", format=".4~f"), alt.Tooltip("Effet:Q", format="+.2f")],
            ).properties(height=380)
            regle = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888").encode(x="x:Q")
            pleine_largeur(st.altair_chart, graphe + regle)
            haut, bas_ = expl.sort_values("Effet").iloc[-1], expl.sort_values("Effet").iloc[0]
            phrases = []
            if haut["Effet"] > 0.05:
                phrases.append(f"Ce qui aide le plus : **{haut['Variable']}** "
                               f"({haut['Valeur saisie']:.4g} contre {haut['Vin moyen']:.4g} en moyenne).")
            if bas_["Effet"] < -0.05:
                phrases.append(f"Ce qui pénalise le plus : **{bas_['Variable']}** "
                               f"({bas_['Valeur saisie']:.4g} contre {bas_['Vin moyen']:.4g} en moyenne).")
            st.info(" ".join(phrases))
        st.caption(f"Méthode : pour chaque variable, on la remplace par sa médiane chez les {nom_type} de "
                   "l'entraînement et on mesure de combien le score change. Bleu : la valeur saisie rapproche "
                   "le vin d'un « bon » ; rouge : elle l'en éloigne.")

# ================================================================== 2. seuil
with onglet_seuil:
    if donnees is None:
        st.info("Fichier `donnees_app.joblib` introuvable : exécutez la cellule de sauvegarde du notebook.")
    else:
        y_t, s_t = np.asarray(donnees["y_test"]), np.asarray(donnees["scores_test"])
        st.write("Le modèle donne un **score** à chaque vin. Le **seuil** décide à partir de quel score on "
                 "annonce « bon ». Déplacez-le pour voir le compromis entre précision et rappel sur le "
                 f"jeu de test ({len(y_t)} vins, dont {int(y_t.sum())} bons).")
        bas, haut = float(np.floor(np.percentile(s_t, 2) * 10) / 10), float(np.ceil(np.percentile(s_t, 99) * 10) / 10)
        seuil = st.slider("Seuil de décision", bas, haut, value=round(SEUIL, 2), step=0.01)

        def metriques(t):
            p = s_t >= t
            vp, fp = int((p & (y_t == 1)).sum()), int((p & (y_t == 0)).sum())
            fn, vn = int((~p & (y_t == 1)).sum()), int((~p & (y_t == 0)).sum())
            prec = vp / (vp + fp) if vp + fp else 0.0
            rap = vp / (vp + fn)
            f1 = 2 * prec * rap / (prec + rap) if prec + rap else 0.0
            return prec, rap, f1, vp, fp, fn, vn

        prec, rap, f1, vp, fp, fn, vn = metriques(seuil)
        a, b, c = st.columns(3)
        a.metric("Précision", f"{prec:.2f}", help="Parmi les vins annoncés « bons », part de vrais bons vins")
        b.metric("Rappel", f"{rap:.2f}", help="Parmi les vrais bons vins, part retrouvée")
        c.metric("F1", f"{f1:.2f}")

        st.markdown("**Matrice de confusion**")
        pleine_largeur(st.dataframe, pd.DataFrame({"Prédit ordinaire": [vn, fn], "Prédit bon": [fp, vp]},
                                                  index=["Vraiment ordinaire", "Vraiment bon"]))
        st.caption(f"{fn} bons vins ratés (faux négatifs) · {fp} vins ordinaires annoncés bons (faux positifs)")

        grille = np.linspace(bas, haut, 120)
        courbes = pd.DataFrame([{"Seuil": t, "Métrique": nom, "Valeur": v}
                                for t in grille for nom, v in zip(["Précision", "Rappel", "F1"], metriques(t)[:3])])
        lignes = alt.Chart(courbes).mark_line(strokeWidth=2).encode(
            x=alt.X("Seuil:Q", title="Seuil de décision"),
            y=alt.Y("Valeur:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Métrique:N", scale=alt.Scale(domain=["Précision", "Rappel", "F1"],
                                                          range=[BLEU, ORANGE, VERT_EAU]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Seuil:Q", format=".2f"), "Métrique:N", alt.Tooltip("Valeur:Q", format=".3f")])
        actuel = alt.Chart(pd.DataFrame({"Seuil": [seuil]})).mark_rule(color="#555", strokeDash=[4, 3]).encode(x="Seuil:Q")
        pleine_largeur(st.altair_chart, (lignes + actuel).properties(height=300))
        st.caption(f"Ligne pointillée : seuil choisi. Le seuil du modèle ({SEUIL:.3f}) a été fixé par "
                   "validation croisée sur l'entraînement, **pas** sur ce jeu de test, qui ne sert ici "
                   "qu'à illustrer le compromis.")

# ================================================================== 3. fichier
with onglet_fichier:
    st.write("Déposez un fichier CSV avec les 11 mesures (mêmes noms de colonnes que le jeu UCI) et une "
             "colonne `type` contenant `red`/`white` (ou `rouge`/`blanc`).")
    modele_csv = pd.DataFrame([dict(zip(VARIABLES, v), type="red" if t == "rouge" else "white")
                               for t, v in EXEMPLES.values()])
    st.download_button("Télécharger un fichier d'exemple", modele_csv.to_csv(index=False).encode(),
                       "exemple_vins.csv", "text/csv")
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is not None:
        brut = fichier.getvalue().decode("utf-8", errors="replace")
        lu = pd.read_csv(io.StringIO(brut), sep=None, engine="python")
        lu.columns = [c.strip().strip('"') for c in lu.columns]
        manquantes = [v for v in VARIABLES if v not in lu.columns]
        if manquantes or "type" not in lu.columns:
            st.error("Colonnes manquantes : " + ", ".join(manquantes + ([] if "type" in lu.columns else ["type"])))
        else:
            t = lu["type"].astype(str).str.lower().str.strip()
            lu["type_white"] = t.isin(["white", "blanc"]).astype(int)
            sc = score(lu)
            res = lu[["type"] + VARIABLES].copy()
            res.insert(0, "score", sc.round(3))
            res.insert(0, "prédiction", np.where(sc >= SEUIL, "✅ bon", "❌ ordinaire"))
            st.success(f"{len(res)} vins analysés : {int((sc >= SEUIL).sum())} prédits bons.")
            pleine_largeur(st.dataframe, res)
            st.download_button("Télécharger les prédictions", res.to_csv(index=False).encode(),
                               "predictions_vins.csv", "text/csv")

# ================================================================== 4. à propos
with onglet_info:
    st.markdown(
        "- **Données** : UCI Wine Quality (Cortez et al., 2009), vins rouges et blancs, doublons supprimés.\n"
        "- **Cible** : « bon vin » si la note est ≥ 7 (19 % des vins).\n"
        "- **Modèle** : pipeline standardisation → SelectKBest → SVM à noyau RBF "
        "(C = 1, γ = 0,1, classes pondérées).\n"
        f"- **Seuil** : {SEUIL:.3f}, optimisé par validation croisée (`TunedThresholdClassifierCV`, critère F1).\n"
        "- **Performance sur le test** : précision ≈ 0,46, rappel ≈ 0,76, F1 ≈ 0,57, AUC ≈ 0,85.\n"
        "- **Limite** : les mesures chimiques n'expliquent qu'une partie de la note, qui reste en partie "
        "subjective. La prédiction est une aide, pas un verdict."
    )