"""Modelos que dan P(sube). Todos con la misma interfaz: fit(X, y) y proba(X).

Dos referencias que hay que batir antes de creerse nada:

  · siempre_sube : la frecuencia de subidas del entrenamiento (en bolsa suele ser
                   ~53% en diario). Es lo que "sabe" quien no sabe nada.
  · momentum     : P(sube | la última barra subió) y P(sube | bajó), contadas en el
                   entrenamiento. Una regla de una línea.

Modelos, de más simple a más flexible:

  Lineales
  · lineal       : regresión logística casi sin regularizar.
  · ridge        : clasificador ridge (L2) + calibración de Platt.
  · lasso        : logística con L1: pone a cero las variables que no aportan.
  · elastic_net  : mezcla L1/L2: selecciona variables sin romperse con las correlacionadas.
  · lda          : análisis discriminante lineal con shrinkage (estable con pocas muestras).
  · svm_lineal   : máquina de vectores soporte lineal + calibración de Platt.
  Probabilísticos y de vecindad
  · naive_bayes  : Bayes ingenuo gaussiano (supone variables independientes: no lo son).
  · knn          : k vecinos sobre 6 variables fijas (con 30 dimensiones se pierde).
  No lineales
  · random_forest     : 300 árboles poco profundos y con hojas grandes.
  · extra_trees       : como el bosque, con cortes aleatorios (aún más varianza reducida).
  · gradient_boosting : boosting de histogramas, árboles de profundidad 3.
  · red_neuronal      : perceptrón multicapa pequeño (32-16) con parada temprana.
  Combinación
  · ensemble     : media de las probabilidades de ridge, lasso, random_forest y gradient_boosting.

Todos los hiperparámetros están FIJADOS de antemano, no ajustados mirando los
resultados: ajustarlos sería probar más configuraciones y habría que contarlas.
Con señal débil los modelos flexibles memorizan ruido, así que van regularizados
con fuerza (hojas de >=100 filas, profundidad baja, L2...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

BASELINES = ("siempre_sube", "momentum")
CLASSIC = ("lineal", "ridge", "knn")
EXTRA = ("lasso", "elastic_net", "lda", "svm_lineal", "naive_bayes",
         "random_forest", "extra_trees", "gradient_boosting", "red_neuronal", "ensemble")
MODELS = CLASSIC + EXTRA
DEEP = ("transformer",)  # opcional: necesita PyTorch y una ventana de barras (seq_len)
ALL_MODELS = BASELINES + MODELS + DEEP
ENSEMBLE_MEMBERS = ("ridge", "lasso", "random_forest", "gradient_boosting")


class PriorModel:
    name = "siempre_sube"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriorModel":
        self.p_ = float(np.clip(np.mean(y), 1e-3, 1 - 1e-3))
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p_)


class MomentumModel:
    name = "momentum"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MomentumModel":
        last_up = X["ret_1"].to_numpy() > 0
        y = np.asarray(y)
        prior = y.mean()
        # Suavizado hacia la frecuencia global: con pocos datos no se fía de 3 casos.
        self.p_up_ = (y[last_up].sum() + 10 * prior) / (last_up.sum() + 10)
        self.p_down_ = (y[~last_up].sum() + 10 * prior) / ((~last_up).sum() + 10)
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(X["ret_1"].to_numpy() > 0, self.p_up_, self.p_down_)


class SklearnModel:
    """Cualquier estimador de sklearn con predict_proba, opcionalmente sobre un subconjunto de columnas."""

    def __init__(self, name: str, factory, features=None):
        self.name = name
        self.factory = factory
        self.features = features

    def _cols(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.features:
            return X
        cols = [c for c in self.features if c in X.columns]
        return X[cols] if len(cols) >= 3 else X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        X = self._cols(X)
        ya = np.asarray(y).astype(int)
        self.constant_ = float(ya[0]) if len(np.unique(ya)) < 2 else None
        if self.constant_ is None:
            self.estimator_ = self.factory(len(X)).fit(X.to_numpy(), ya)
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        return self.estimator_.predict_proba(self._cols(X).to_numpy())[:, 1]


class PlattModel:
    """Clasificador sin probabilidades (ridge, SVM) + calibración de Platt.

    La calibración se ajusta sobre el último 20% del entrenamiento con un modelo
    entrenado en el 80% anterior: datos POSTERIORES a los que produjeron la
    puntuación, para que la probabilidad no salga optimista por verse a sí misma.
    """

    def __init__(self, name: str, factory):
        self.name = name
        self.factory = factory

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PlattModel":
        Xa, ya = X.to_numpy(), np.asarray(y).astype(int)
        cut = int(len(Xa) * 0.8)
        self.platt_ = None
        self.p_ = float(ya.mean())
        if len(np.unique(ya[:cut])) == 2 and len(np.unique(ya[cut:])) == 2:
            early = self.factory().fit(Xa[:cut], ya[:cut])
            scores = early.decision_function(Xa[cut:]).reshape(-1, 1)
            self.platt_ = LogisticRegression().fit(scores, ya[cut:])
            self.model_ = self.factory().fit(Xa, ya)
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.platt_ is None:
            return np.full(len(X), self.p_)
        scores = self.model_.decision_function(X.to_numpy()).reshape(-1, 1)
        return self.platt_.predict_proba(scores)[:, 1]


class EnsembleModel:
    name = "ensemble"

    def __init__(self, members=ENSEMBLE_MEMBERS):
        self.members = members

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        self.models_ = [make_model(m).fit(X, y) for m in self.members]
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.proba(X) for m in self.models_], axis=0)


KNN_FEATURES = ("ret_1", "ret_5", "ret_20", "vol_ratio", "rsi_14", "dist_ema50_atr")


def _knn(n: int):
    k = int(np.clip(np.sqrt(n), 25, 300))  # con más historia, más vecinos
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(k, max(1, n - 1))))


_FACTORIES = {
    "lineal": lambda: SklearnModel("lineal", lambda n: make_pipeline(
        StandardScaler(), LogisticRegression(C=1e4, max_iter=5000))),
    "ridge": lambda: PlattModel("ridge", lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))),
    "knn": lambda: SklearnModel("knn", _knn, features=KNN_FEATURES),
    "lasso": lambda: SklearnModel("lasso", lambda n: make_pipeline(
        StandardScaler(), LogisticRegression(l1_ratio=1.0, solver="saga", C=0.05, max_iter=3000, tol=1e-3))),
    "elastic_net": lambda: SklearnModel("elastic_net", lambda n: make_pipeline(
        StandardScaler(), LogisticRegression(l1_ratio=0.5, solver="saga", C=0.05, max_iter=3000, tol=1e-3))),
    "lda": lambda: SklearnModel("lda", lambda n: make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
    "svm_lineal": lambda: PlattModel("svm_lineal", lambda: make_pipeline(
        StandardScaler(), LinearSVC(C=0.01, max_iter=5000))),
    "naive_bayes": lambda: SklearnModel("naive_bayes", lambda n: make_pipeline(StandardScaler(), GaussianNB())),
    "random_forest": lambda: SklearnModel("random_forest", lambda n: RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=100, max_features="sqrt", n_jobs=-1, random_state=0)),
    "extra_trees": lambda: SklearnModel("extra_trees", lambda n: ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=100, max_features="sqrt", n_jobs=-1, random_state=0)),
    "gradient_boosting": lambda: SklearnModel("gradient_boosting", lambda n: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=False, random_state=0)),
    "red_neuronal": lambda: SklearnModel("red_neuronal", lambda n: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True,
                                        validation_fraction=0.15, max_iter=300, random_state=0))),
    "ensemble": lambda: EnsembleModel(),
}


def make_model(name: str):
    if name == "siempre_sube":
        return PriorModel()
    if name == "momentum":
        return MomentumModel()
    if name in _FACTORIES:
        return _FACTORIES[name]()
    if name == "transformer":
        from .transformer import TransformerModel
        return TransformerModel()
    raise ValueError(f"Modelo desconocido: {name!r}. Usa uno de {ALL_MODELS}.")
