"""
Pipeline de prévision du débit à j+1, j+2, j+3.

Entrée attendue : un DataFrame trié par date avec 3 colonnes brutes :
    - "Date"  : datetime
    - "Debit" : débit journalier mesuré
    - "Pluie" : pluie journalière mesurée

Sortie : un DataFrame avec les colonnes "Debit-j+1", "Debit-j+2", "Debit-j+3".

Tout le feature engineering (lags de débit, cumuls de pluie glissants,
variables calendaires) est fait à l'intérieur du pipeline : l'utilisateur
n'a jamais besoin de recalculer une feature à la main.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------------
# 1. BRIQUES DE FEATURE ENGINEERING (des transformers sklearn "maison")
# ----------------------------------------------------------------------
#
# Un transformer sklearn est une classe qui hérite de BaseEstimator et
# TransformerMixin et qui implémente deux méthodes :
#   - fit(X, y=None)      -> apprend des paramètres sur X (ici rien à
#                             apprendre, donc fit ne fait que "return self")
#   - transform(X)        -> renvoie X modifié
# TransformerMixin fournit gratuitement fit_transform() = fit().transform().
# Hériter de BaseEstimator donne get_params()/set_params(), utile pour le
# GridSearch et pour que Pipeline sache cloner l'objet.


class DateFeatures(BaseEstimator, TransformerMixin):
    """
    Extrait des variables calendaires depuis la colonne Date.

    Paramètres
    ----------
    include_year : bool, défaut True
        Si True, ajoute la colonne "Annee" (année civile).
        ATTENTION : une régression linéaire peut extrapoler un effet
        "tendance" au-delà des années vues à l'entraînement, ce qui est
        voulu ici (contrairement aux modèles à base d'arbres qui, eux,
        ne peuvent pas extrapoler). Mais si vous utilisez ce pipeline
        pendant plusieurs années, gardez un œil sur ce comportement :
        le modèle continuera à extrapoler linéairement indéfiniment.
        Mettez à False pour désactiver cette variable si vous préférez
        un modèle plus prudent (basé uniquement sur la saisonnalité).
    """

    def __init__(self, include_year: bool = True):
        self.include_year = include_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Mois"] = X["Date"].dt.month
        X["Jour"] = X["Date"].dt.day
        if self.include_year:
            X["Annee"] = X["Date"].dt.year
        return X


class LagFeatures(BaseEstimator, TransformerMixin):
    """
    Ajoute les valeurs passées (lags) d'une colonne : Debit-j-1, Debit-j-2, ...

    Théorie : le débit d'un cours d'eau est un processus autocorrélé
    (le débit de demain dépend fortement du débit d'aujourd'hui et des
    jours précédents -> c'est la "mémoire" hydrologique du bassin,
    liée au temps de percolation/écoulement de l'eau vers la rivière).
    Donner ces lags en entrée permet au modèle de capter cette inertie.

    Paramètres
    ----------
    lags : tuple[int], défaut (1,2,3,4,5)
        Liste des décalages (en jours) à créer.
    column : str, défaut "Debit"
        Colonne source sur laquelle calculer les lags.

    Fonctionnement
    --------------
    pandas.Series.shift(n) décale la série de n lignes vers le bas :
    la valeur à l'indice i devient celle qui était à l'indice i-n.
    C'est l'équivalent vectorisé de la boucle "data.loc[i-j]" utilisée
    dans le notebook d'exploration, mais sans boucle Python (donc plus
    rapide) et sans risque d'erreur d'indexation.
    """

    def __init__(self, lags=(1, 2, 3, 4, 5), column: str = "Debit"):
        self.lags = lags
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for lag in self.lags:
            X[f"{self.column}-j-{lag}"] = X[self.column].shift(lag)
        return X


class RollingCumFeatures(BaseEstimator, TransformerMixin):
    """
    Ajoute des cumuls de pluie sur des fenêtres glissantes passées
    (jour courant inclus) : Cum_Pluie_15, Cum_Pluie_30, ...

    Théorie : la pluie d'un seul jour ne suffit pas à expliquer le débit
    (le sol absorbe une partie de l'eau, les nappes se rechargent
    progressivement). Le cumul de pluie sur les N derniers jours est un
    proxy classique en hydrologie de l'humidité du sol / de l'état de
    saturation du bassin versant ("antecedent precipitation index").
    Les fenêtres courtes (15-90 j) captent la réponse rapide du bassin,
    les fenêtres longues (110-200 j) captent l'effet saisonnier lent
    (recharge des nappes sur toute une saison des pluies).

    Paramètres
    ----------
    windows : tuple[int]
        Tailles de fenêtres en jours (le jour courant + les N jours
        précédents, donc une fenêtre "15" utilise 16 valeurs).
    column : str, défaut "Pluie"

    Fonctionnement
    --------------
    pandas.Series.rolling(window=w, min_periods=w).sum() calcule la
    somme glissante sur une fenêtre de w observations consécutives.
    min_periods=w force à obtenir NaN tant qu'on n'a pas w observations
    complètes (par ex. sur les tout premiers jours de l'historique),
    ce qui évite de calculer un cumul biaisé avec moins de jours que
    prévu.
    """

    def __init__(
        self,
        windows=(15, 30, 45, 60, 75, 90, 110, 130, 150, 170, 180, 200),
        column: str = "Pluie",
    ):
        self.windows = windows
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for w in self.windows:
            # +1 car on inclut le jour courant, comme dans le notebook
            # d'origine (data.loc[i-w : i, "Pluie"].sum()).
            X[f"Cum_Pluie_{w}"] = X[self.column].rolling(
                window=w + 1, min_periods=w + 1
            ).sum()
        return X


# ----------------------------------------------------------------------
# 2. CONSTRUCTION DES CIBLES (y)
# ----------------------------------------------------------------------
#
# Les cibles ne sont pas des "features" : on ne les met donc pas dans le
# Pipeline sklearn (dont le rôle est de transformer X). On les construit
# à part avec un simple décalage négatif.


def make_targets(df: pd.DataFrame, horizons=(1, 2, 3), column: str = "Debit") -> pd.DataFrame:
    """
    Construit les cibles Debit-j+1, Debit-j+2, Debit-j+3.

    shift(-h) décale vers le haut : la ligne i récupère la valeur qui
    était à la ligne i+h, donc "le débit dans h jours".
    """
    return pd.DataFrame(
        {f"{column}-j+{h}": df[column].shift(-h) for h in horizons},
        index=df.index,
    )


# ----------------------------------------------------------------------
# 3. LE PIPELINE COMPLET : feature engineering + modèle, en un seul objet
# ----------------------------------------------------------------------
#
# Pourquoi ne pas tout mettre dans un unique sklearn.pipeline.Pipeline ?
# Parce qu'un Pipeline sklearn suppose que chaque transformer garde le
# même nombre de lignes entre l'entrée et la sortie (il ne touche jamais
# à y). Or ici, les lags/cumuls glissants créent des NaN sur les
# premières lignes (pas assez d'historique) qu'il faut retirer, et il
# faut retirer les MÊMES lignes de y. Un Pipeline sklearn classique ne
# gère pas ça proprement.
#
# La solution "production" standard : un objet englobant (ici
# DebitForecastModel) qui, en interne, utilise un vrai Pipeline sklearn
# pour la partie feature engineering (chaînage propre, réutilisable,
# clonable), gère l'alignement X/y et le nettoyage des NaN, puis entraîne
# le modèle final. C'est cet objet que vous utiliserez au quotidien.


class DebitForecastModel(BaseEstimator):
    """
    Pipeline complet : Date, Debit, Pluie (bruts) -> Debit-j+1, j+2, j+3.

    Paramètres
    ----------
    horizons : tuple[int], défaut (1, 2, 3)
        Horizons de prévision en jours.
    lags : tuple[int], défaut (1, 2, 3, 4, 5)
        Lags de débit utilisés comme features.
    windows : tuple[int]
        Fenêtres de cumul de pluie utilisées comme features.
    use_year_feature : bool, défaut True
        Voir DateFeatures.
    model : estimateur sklearn, défaut LinearRegression()
        Le modèle final. LinearRegression gère nativement le
        multi-output : en lui donnant un y à 3 colonnes (j+1, j+2, j+3),
        elle résout en une seule fois 3 systèmes de moindres carrés
        (un jeu de coefficients par colonne cible), ce qui est
        strictement équivalent à entraîner 3 LinearRegression séparées,
        mais en un seul appel .fit().
    """

    def __init__(
        self,
        horizons=(1, 2, 3),
        lags=(1, 2, 3, 4, 5),
        windows=(15, 30, 45, 60, 75, 90, 110, 130, 150, 170, 180, 200),
        use_year_feature: bool = True,
        model=None,
    ):
        self.horizons = horizons
        self.lags = lags
        self.windows = windows
        self.use_year_feature = use_year_feature
        self.model = model if model is not None else LinearRegression()

        # Le Pipeline sklearn "interne" ne gère que la partie feature
        # engineering (pas de perte de lignes gérée ici volontairement).
        self.feature_pipeline_ = Pipeline(
            steps=[
                ("dates", DateFeatures(include_year=self.use_year_feature)),
                ("lags", LagFeatures(lags=self.lags, column="Debit")),
                ("rolling", RollingCumFeatures(windows=self.windows, column="Pluie")),
            ]
        )

        self.feature_names_: list[str] | None = None

    # -- interne : calcule les features à partir des données brutes -----
    def _build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        raw_df = raw_df.sort_values("Date").reset_index(drop=True)
        # fit_transform() alors que rien n'est "appris" dans ces
        # transformers : c'est juste l'enchaînement fit() puis
        # transform() fourni gratuitement par TransformerMixin.
        feats = self.feature_pipeline_.fit_transform(raw_df)
        feature_cols = [c for c in feats.columns if c != "Date"]
        return feats, feature_cols

    # -- entraînement -----------------------------------------------------
    def fit(self, raw_df: pd.DataFrame):
        """
        raw_df : DataFrame avec les colonnes Date, Debit, Pluie
                 (historique complet servant à l'entraînement).
        """
        feats, feature_cols = self._build_features(raw_df)
        y = make_targets(feats, horizons=self.horizons, column="Debit")

        # On aligne X et y, puis on retire les lignes incomplètes
        # (début d'historique sans assez de recul, fin d'historique
        # sans assez de futur pour connaître Debit-j+3).
        combined = pd.concat([feats[feature_cols], y], axis=1).dropna()

        self.feature_names_ = feature_cols
        X = combined[feature_cols]
        Y = combined[list(y.columns)]

        self.model.fit(X, Y)
        self.target_names_ = list(y.columns)
        return self

    # -- prédiction ---------------------------------------------------
    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        raw_df : historique récent avec Date, Debit, Pluie. Doit contenir
                 au minimum max(windows)+1 jours consécutifs pour que les
                 cumuls de pluie longue durée soient calculables sur la
                 dernière ligne (par défaut : 201 jours).

        Retourne un DataFrame indexé comme les lignes valides de raw_df,
        avec les colonnes Debit-j+1, Debit-j+2, Debit-j+3 prédites.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Le modèle n'est pas encore entraîné : appelez fit() d'abord.")

        feats, _ = self._build_features(raw_df)
        X = feats[self.feature_names_]
        valid = X.dropna()
        if valid.empty:
            raise ValueError(
                "Historique insuffisant : impossible de calculer toutes les "
                "features (lags/cumuls) sur les données fournies."
            )
        preds = self.model.predict(valid)
        return pd.DataFrame(preds, columns=self.target_names_, index=valid.index)
