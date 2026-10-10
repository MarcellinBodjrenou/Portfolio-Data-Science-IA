"""
pipeline.py
Fonction de creation des variables explicatives (feature engineering) a partir de
donnees journalieres (Date, Pluie, Debit), pour alimenter le modele de prevision
du debit a J+1 / J+2 / J+3.

La fonction `pipeline` reproduit exactement les variables construites dans le
notebook 03-Analyse_Feature_engening.ipynb :
    - Pluie, Debit du jour
    - Cum_Pluie_15/30/45/60/75/90   : cumul de pluie sur les N derniers jours (incl. aujourd'hui)
    - Debit-j-1 a Debit-j-5         : debit des 5 jours precedant aujourd'hui
    - Cum_Pluie_110/130/150/170/180/200 : cumul de pluie sur des fenetres plus longues
    - Mois, Jour                    : composantes calendaires du jour

Colonnes de sortie, dans le MEME ORDRE que celui utilise pour entrainer le modele
(cf. X_train dans 04-Compare_model.ipynb) :
    Pluie, Debit,
    Cum_Pluie_15, Cum_Pluie_30, Cum_Pluie_45, Cum_Pluie_60, Cum_Pluie_75, Cum_Pluie_90,
    Debit-j-1, Debit-j-2, Debit-j-3, Debit-j-4, Debit-j-5,
    Cum_Pluie_110, Cum_Pluie_130, Cum_Pluie_150, Cum_Pluie_170, Cum_Pluie_180, Cum_Pluie_200,
    Mois, Jour
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FENETRES_COURTES = [15, 30, 45, 60, 75, 90]
FENETRES_LONGUES = [110, 130, 150, 170, 180, 200]

COLONNES_SORTIE = (
    ["Pluie", "Debit"]
    + [f"Cum_Pluie_{n}" for n in FENETRES_COURTES]
    + [f"Debit-j-{j}" for j in range(1, 6)]
    + [f"Cum_Pluie_{n}" for n in FENETRES_LONGUES]
    + ["Mois", "Jour"]
)

# Nombre de jours d'historique minimum requis avant "aujourd'hui"
# (fixe par la plus longue fenetre de cumul de pluie)
NB_JOURS_HISTORIQUE_MIN = max(FENETRES_LONGUES)


def pipeline(historique: pd.DataFrame, date_jour, pluie_jour: float, debit_jour: float) -> pd.DataFrame:
    """
    Construit le vecteur de variables explicatives pour une nouvelle journee
    ("aujourd'hui"), a partir de l'historique recent et des mesures du jour.

    Parameters
    ----------
    historique : pd.DataFrame
        Historique journalier avec au moins les colonnes ["Date", "Pluie", "Debit"].
        Doit couvrir au moins les `NB_JOURS_HISTORIQUE_MIN` (200) jours precedant
        `date_jour`, SANS TROU (une ligne par jour calendaire, pas de jour manquant).
        Ne doit PAS contenir la ligne d'aujourd'hui (elle est fournie separement).
    date_jour : str ou datetime-like
        Date du jour a traiter (aujourd'hui).
    pluie_jour : float
        Pluie mesuree aujourd'hui.
    debit_jour : float
        Debit mesure aujourd'hui.

    Returns
    -------
    pd.DataFrame
        Une ligne unique contenant les variables explicatives, dans le meme ordre
        que celui utilise a l'entrainement du modele (voir COLONNES_SORTIE).
        Il suffit ensuite d'appliquer le scaler deja entraine puis `model.predict()`
        sur ce DataFrame pour obtenir la prevision du debit (J+1, J+2 ou J+3
        selon le modele utilise).

    Raises
    ------
    ValueError
        Si l'historique ne couvre pas assez de jours avant `date_jour`, si des
        dates sont manquantes (trous) sur la periode necessaire, ou si
        `date_jour` est deja present dans `historique`.
    """
    hist = historique.copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist[["Date", "Pluie", "Debit"]].sort_values("Date").reset_index(drop=True)

    date_jour = pd.to_datetime(date_jour)

    if (hist["Date"] == date_jour).any():
        raise ValueError(
            "`historique` contient deja une ligne pour date_jour ; "
            "il ne doit couvrir que le passe (avant aujourd'hui)."
        )

    # On ajoute la ligne du jour a l'historique pour calculer les cumuls / decalages
    nouvelle_ligne = pd.DataFrame({"Date": [date_jour], "Pluie": [pluie_jour], "Debit": [debit_jour]})
    complet = pd.concat([hist, nouvelle_ligne], ignore_index=True).sort_values("Date").reset_index(drop=True)

    idx_jour = complet.index[complet["Date"] == date_jour][0]

    # Verification qu'il y a assez d'historique et pas de trou de date
    nb_jours_dispo = idx_jour  # nombre de jours strictement avant aujourd'hui
    if nb_jours_dispo < NB_JOURS_HISTORIQUE_MIN:
        raise ValueError(
            f"Historique insuffisant : {nb_jours_dispo} jour(s) disponible(s) avant "
            f"{date_jour.date()}, {NB_JOURS_HISTORIQUE_MIN} jours minimum requis "
            "(fenetre de cumul de pluie la plus longue, Cum_Pluie_200)."
        )

    debut_controle = idx_jour - NB_JOURS_HISTORIQUE_MIN
    dates_attendues = pd.date_range(complet.loc[debut_controle, "Date"], date_jour, freq="D")
    dates_attendues = pd.Series(dates_attendues[~((dates_attendues.month == 2) & (dates_attendues.day == 29))])
    dates_reelles = complet.loc[debut_controle:idx_jour, "Date"].reset_index(drop=True)
    if not dates_reelles.equals(dates_attendues):
        raise ValueError(
        "L'historique contient des trous (dates manquantes) sur la periode "
        "necessaire au calcul des variables explicatives."
        )

    features = {"Pluie": pluie_jour, "Debit": debit_jour}

    # Cumuls de pluie sur les fenetres courtes (incluant aujourd'hui)
    for n in FENETRES_COURTES:
        features[f"Cum_Pluie_{n}"] = complet.loc[idx_jour - n : idx_jour, "Pluie"].sum()

    # Debit des 5 jours precedents
    for j in range(1, 6):
        features[f"Debit-j-{j}"] = complet.loc[idx_jour - j, "Debit"]

    # Cumuls de pluie sur les fenetres longues (incluant aujourd'hui)
    for n in FENETRES_LONGUES:
        features[f"Cum_Pluie_{n}"] = complet.loc[idx_jour - n : idx_jour, "Pluie"].sum()

    # Composantes calendaires
    features["Mois"] = date_jour.month
    features["Jour"] = date_jour.day

    return pd.DataFrame([features])[COLONNES_SORTIE]


def pipeline_predict(donnee, date_jour, pluie_jour, debit_jour, model, nom_sortie="Debit_predit"):
    """
    Raccourci pratique : construit les variables explicatives via `pipeline()`,
    applique le `scaler` deja entraine puis appelle `model.predict()`.

    Parameters
    ----------
    historique, date_jour, pluie_jour, debit_jour
        Voir `pipeline()`.
    scaler : objet sklearn deja entraine (ex: StandardScaler.fit sur X_train)
    model : modele sklearn deja entraine (ex: LinearRegression.fit sur X_train_sc, y_train)
    nom_sortie : str
        Nom donne a la prevision dans le dictionnaire retourne.

    Returns
    -------
    dict
        {"features": DataFrame des variables explicatives, nom_sortie: valeur predite}
    """
    df=donnee.copy()
    date_jour = pd.to_datetime(date_jour)
    historique=df[df["Date"]<date_jour]
    X = pipeline(historique, date_jour, pluie_jour, debit_jour)
    prediction = np.abs(model.predict(X))
    return  prediction
