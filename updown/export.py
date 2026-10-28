"""Modelos entrenados a JSON, para que el navegador calcule P(sube) sin servidor.

Solo se exportan modelos que se evalúan con unas líneas de JavaScript sin perder
precisión:

  · ridge             : escalado + producto escalar + calibración de Platt.
  · gradient_boosting : 150 árboles de profundidad 3 (umbrales y hojas tal cual).

La web (web/js/models.js) repite exactamente estas cuentas; tests/test_web.py
comprueba que las variables y las probabilidades coinciden con las de Python.

También se exporta el HMM de las etapas (medias, covarianzas, transiciones y nombres)
para que la web diga en qué etapa está cada mercado y con qué probabilidad
(web/js/stages.js), con el mismo filtro hacia delante que `updown/hmm.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .hmm import StageHMM
from .models import PlattModel, PriorModel, SklearnModel

WEB_MODELS = ("ridge", "gradient_boosting")


def export_model(model, columns: list[str]) -> dict:
    """Un modelo ya entrenado (de `make_model`) como diccionario serializable en JSON."""
    if isinstance(model, PriorModel):
        return {"kind": "constant", "p": model.p_}
    if isinstance(model, PlattModel):
        return _ridge(model, columns)
    if isinstance(model, SklearnModel) and model.constant_ is not None:
        return {"kind": "constant", "p": model.constant_}
    if isinstance(model, SklearnModel) and isinstance(model.estimator_, HistGradientBoostingClassifier):
        return _hist_gradient_boosting(model.estimator_, columns)
    raise TypeError(f"No se sabe exportar {getattr(model, 'name', model)!r}; exportables: {WEB_MODELS}")


def _ridge(model: PlattModel, columns: list[str]) -> dict:
    if model.platt_ is None:  # sin las dos clases en el entrenamiento: frecuencia constante
        return {"kind": "constant", "p": model.p_}
    scaler, clf = model.model_[0], model.model_[-1]
    return {
        "kind": "linear_platt",
        "features": list(columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": np.ravel(clf.coef_).tolist(),
        "intercept": float(np.ravel(clf.intercept_)[0]),
        "platt_a": float(model.platt_.coef_[0, 0]),
        "platt_b": float(model.platt_.intercept_[0]),
    }


def _hist_gradient_boosting(est: HistGradientBoostingClassifier, columns: list[str]) -> dict:
    """Árboles como listas de nodos [variable, umbral, izquierda, derecha, valor, nulo_a_la_izquierda, hoja].

    Las hojas ya llevan aplicada la tasa de aprendizaje, así que la suma en bruto es
    `base + suma de hojas` y la probabilidad, su sigmoide (lo mismo que predict_proba).
    """
    if est.n_features_in_ != len(columns):
        raise ValueError(f"El modelo tiene {est.n_features_in_} variables y se pasan {len(columns)} nombres.")
    trees = []
    for per_iteration in est._predictors:
        (pred,) = per_iteration  # clasificación binaria: un árbol por iteración
        nodes = pred.nodes
        if nodes["is_categorical"].any():
            raise ValueError("Variables categóricas no soportadas en la exportación.")
        trees.append([[int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]), int(n["right"]),
                       float(n["value"]), int(n["missing_go_to_left"]), int(n["is_leaf"])] for n in nodes])
    return {
        "kind": "trees",
        "features": list(columns),
        "base": float(np.ravel(est._baseline_prediction)[0]),
        "trees": trees,
    }


# Velas de 15 minutos: un día y 30 días, las ventanas de `regime_inputs` con datos continuos.
WEB_DAY, WEB_MONTH = 96, 2880


def export_stage_hmm(model: StageHMM) -> dict:
    """El HMM de etapas ya ajustado, con un nombre por estado (puede repetirse) y el umbral."""
    h = model.hmm_
    return {"names": list(model.names_), "threshold": model.threshold, "startprob": h.startprob_.tolist(),
            "transmat": h.transmat_.tolist(), "means": h.means_.tolist(), "covars": h.covars_.tolist()}


def web_hmm_observations(bars: pd.DataFrame, mu: float, sd: float, day: int = WEB_DAY,
                         month: int = WEB_MONTH) -> pd.DataFrame:
    """Las observaciones del HMM como las calcula la web: ventanas fijas y la referencia
    de volatilidad (`mu`, `sd`, de `hmm.vol_reference`) fija desde el entrenamiento."""
    c = bars["close"]
    vol = np.log(c).diff().rolling(day).std()
    logvol = np.log(vol.where(vol > 0))
    out = pd.DataFrame(index=bars.index)
    out["hmm_vol"] = ((logvol - mu) / sd).clip(-5, 5)
    out["hmm_tend"] = (np.log(c / c.shift(month)) / (vol * np.sqrt(month))).clip(-5, 5)
    return out.replace([np.inf, -np.inf], np.nan)


def tree_proba(spec: dict, X: np.ndarray) -> np.ndarray:
    """P(sube) recorriendo los árboles exportados: referencia para comprobar la exportación."""
    raw = np.full(len(X), spec["base"])
    for tree in spec["trees"]:
        for i, row in enumerate(X):
            node = tree[0]
            while not node[6]:
                x = row[node[0]]
                go_left = node[5] if np.isnan(x) else x <= node[1]
                node = tree[node[2] if go_left else node[3]]
            raw[i] += node[4]
    return 1.0 / (1.0 + np.exp(-raw))
