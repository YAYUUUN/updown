"""Acierto según confianza: ¿acierta más el modelo cuando está más seguro?

El AUC es la media de todas las velas, pero el modelo no está igual de seguro
siempre: a veces dice 50,5 % y a veces 58 %. Aquí se ordenan las predicciones
fuera de muestra por lo lejos que están del 50 % y se mide el acierto del 100 %,
50 %, 20 %, 10 % y 5 % más seguras. Si el acierto sube al estrechar, la
herramienta podría decir «esta vez no sé» en lugar de dar siempre un 51 %.

Regla para no inflar el resultado: el corte de «seguro» se decide SOLO con las
probabilidades (|p - 0,5|), nunca mirando si acertó. Los empates se resuelven
por orden de llegada, que tampoco sabe nada del resultado.

Dos versiones del corte:

  · confidence_curve      : percentil de TODAS las predicciones fuera de muestra.
                            Describe el modelo, pero usa la distribución de
                            probabilidades de todo el periodo, que en vivo no se
                            conoce de antemano.
  · confidence_curve_live : el corte de cada bloque se fija con las predicciones
                            del bloque ANTERIOR, como habría que hacerlo en vivo.
                            El primer bloque no tiene pasado y no cuenta. Se usa
                            solo el anterior porque los primeros modelos, con poco
                            entrenamiento, dan probabilidades más extremas: un
                            corte con todo el pasado selecciona mucho menos de lo
                            pedido.

El margen de error (IC 95 %) agrupa las velas por día: varios activos a la misma
hora y velas seguidas no son apuestas independientes, y contarlas como tales
estrecharía el intervalo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import se_by_day

COVERAGES = (1.0, 0.5, 0.2, 0.1, 0.05)


def ci95_by_day(correct: np.ndarray, times: pd.Index) -> tuple[float, float]:
    """Intervalo del acierto con error estándar agrupado por día (cluster-robusto)."""
    if len(correct) == 0:
        return float("nan"), float("nan")
    acc = float(np.mean(correct))
    half = 1.96 * se_by_day(correct, times)
    return acc - half, acc + half


def _summary(sel: pd.DataFrame) -> dict:
    p, y = sel["p"].to_numpy(), sel["up"].to_numpy()
    correct = ((p > 0.5) == (y == 1)).astype(float)
    lo, hi = ci95_by_day(correct, sel.index)
    return {
        "n": len(sel),
        "confianza_media": float(np.mean(np.maximum(p, 1 - p))) if len(sel) else float("nan"),
        "acierto": float(correct.mean()) if len(sel) else float("nan"),
        "ic95_bajo": lo,
        "ic95_alto": hi,
        "freq_sube": float(y.mean()) if len(sel) else float("nan"),
        "predice_sube": float((p > 0.5).mean()) if len(sel) else float("nan"),
    }


def confidence_curve(oos: pd.DataFrame, coverages=COVERAGES) -> pd.DataFrame:
    """Acierto de la fracción `cobertura` de predicciones más alejadas del 50 %.

    `oos` es la salida de `walk_forward` (columnas p, up). `umbral` es la menor
    |p - 0,5| que entra en la selección.
    """
    conf = (oos["p"] - 0.5).abs().to_numpy()
    order = np.argsort(-conf, kind="stable")
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(oos))))
        rows.append({"cobertura": c, "umbral": float(conf[order[k - 1]]), **_summary(oos.iloc[order[:k]])})
    return pd.DataFrame(rows).set_index("cobertura")


def confidence_curve_live(oos: pd.DataFrame, coverages=COVERAGES) -> pd.DataFrame:
    """Como `confidence_curve`, pero el corte de cada bloque sale del bloque anterior.

    `cobertura_real` es la parte de las velas evaluables (bloques 2.º en adelante)
    que pasó el corte: si se aleja de la objetivo, la confianza del modelo cambia
    con el tiempo y un umbral fijado de antemano no selecciona lo que se esperaba.
    """
    conf = (oos["p"] - 0.5).abs()
    folds = sorted(oos["fold"].unique())
    eligible = oos["fold"].isin(folds[1:])
    rows = []
    for c in coverages:
        parts, thresholds = [], []
        for prev, f in zip(folds[:-1], folds[1:]):
            past = conf[oos["fold"] == prev].to_numpy()
            thr = 0.0 if c >= 1 else float(np.quantile(past, 1 - c))
            now = oos["fold"] == f
            parts.append(oos[now & (conf >= thr)])
            thresholds.append(thr)
        sel = pd.concat(parts) if parts else oos.iloc[:0]
        rows.append({"cobertura": c, "cobertura_real": len(sel) / max(int(eligible.sum()), 1),
                     "umbral": float(np.mean(thresholds)) if thresholds else float("nan"), **_summary(sel)})
    return pd.DataFrame(rows).set_index("cobertura")


def curves(oos_by_model: dict[str, pd.DataFrame], live: bool = False, coverages=COVERAGES) -> pd.DataFrame:
    """Tabla larga (modelo, cobertura) para varios modelos a la vez."""
    fn = confidence_curve_live if live else confidence_curve
    parts = [fn(o, coverages).assign(modelo=m) for m, o in oos_by_model.items()]
    return pd.concat(parts).reset_index().set_index(["modelo", "cobertura"])
