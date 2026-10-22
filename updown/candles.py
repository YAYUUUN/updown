"""Evaluación vela a vela: ¿sube la 1.ª, 2.ª, 3.ª, 4.ª vela?, y ¿sube la hora completa?

Para cada objetivo se mide cada modelo fuera de muestra (walk-forward con purga)
y se compara con `siempre_sube` del mismo objetivo. El transformer se entrena una
sola vez por bloque con las 5 salidas a la vez (una red para todas las velas).

Los resultados se pueden desglosar por régimen de mercado (`by_regime`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluate import holm, scores, time_folds, walk_forward
from .features import model_columns
from .models import BASELINES, make_model
from .regimes import REGIME_LABELS, REGIMES


@dataclass
class CandleReport:
    targets: list[str]
    metrics: pd.DataFrame                      # filas (modelo, objetivo), columnas métricas
    oos: dict = field(repr=False, default_factory=dict)
    meta: dict = field(default_factory=dict)

    def pivot(self, metric: str) -> pd.DataFrame:
        """Tabla modelo x objetivo de una métrica (p. ej. 'auc' o 'acierto')."""
        return self.metrics[metric].unstack("objetivo")[self.targets]


def targets_of(data: pd.DataFrame) -> list[tuple[str, str, str, int]]:
    """(nombre, columna objetivo, columna retorno, purga) para cada vela y para el acumulado."""
    n = data.attrs.get("candles") or 0
    if not n:
        raise ValueError("Crea el dataset con candles=N para evaluar vela a vela.")
    out = [(f"vela_{k}", f"up_v{k}", f"fwd_v{k}", k) for k in range(1, n + 1)]
    h = data.attrs.get("horizon", n)
    out.append(("hora" if h == 4 else f"acumulado_{h}", "up", "fwd_ret", h))
    return out


def _multi_transformer(data: pd.DataFrame, targets, n_folds: int, min_train_frac: float) -> dict:
    cols_y = [t[1] for t in targets]
    model = make_model("transformer")
    cols = model_columns(data, model)
    labeled = data[data[cols_y].notna().all(axis=1)]
    keep = [c for c in ("symbol", "regimen") if c in labeled]
    purge = max(t[3] for t in targets)
    parts = {t[0]: [] for t in targets}
    for i, (train_t, test_t) in enumerate(time_folds(labeled.index, n_folds, min_train_frac, purge)):
        tr = labeled[labeled.index.isin(train_t)]
        te = labeled[labeled.index.isin(test_t)]
        P = make_model("transformer").fit(tr[cols], tr[cols_y]).proba(te[cols])
        for j, (name, ycol, rcol, _) in enumerate(targets):
            out = te[[ycol, rcol] + keep].rename(columns={ycol: "up", rcol: "fwd_ret"})
            out["p"] = np.clip(P[:, j], 1e-4, 1 - 1e-4)
            out["fold"] = i
            parts[name].append(out)
    return {name: pd.concat(v) for name, v in parts.items()}


def evaluate_candles(data: pd.DataFrame, models=("ridge", "random_forest", "transformer"),
                     n_folds: int = 6, min_train_frac: float = 0.4) -> CandleReport:
    targets = targets_of(data)
    names = list(BASELINES) + [m for m in models if m not in BASELINES]
    oos = {}
    for name, ycol, rcol, purge in targets:
        for m in names:
            if m == "transformer":
                continue
            oos[(m, name)] = walk_forward(data, m, n_folds, min_train_frac, purge, target=ycol, ret_col=rcol)
    if "transformer" in names:
        for name, df in _multi_transformer(data, targets, n_folds, min_train_frac).items():
            oos[("transformer", name)] = df

    rows = []
    tried = [m for m in names if m not in BASELINES]
    for name, _, _, purge in targets:
        se_h = 1 if name.startswith("vela_") else purge  # velas sueltas no se solapan; el acumulado sí
        base = oos[("siempre_sube", name)]
        base_fold = ((base["p"] - base["up"]) ** 2).groupby(base["fold"]).mean()
        block = {}
        for m in names:
            o = oos[(m, name)]
            s = scores(o, se_h)
            fb = ((o["p"] - o["up"]) ** 2).groupby(o["fold"]).mean()
            s["skill_vs_siempre_sube"] = 1 - s["brier"] / scores(base, se_h)["brier"]
            s["bloques_ganados"] = int((fb < base_fold.reindex(fb.index)).sum()) if m != "siempre_sube" else 0
            s["bloques"] = int(len(fb))
            block[m] = s
        t = pd.DataFrame(block).T
        t["p_auc_corregido"] = holm(t.loc[tried, "p_auc"]).reindex(t.index)
        t["ventaja"] = (t["p_auc_corregido"] < 0.05) & (t["skill_vs_siempre_sube"] > 0) & \
                       (t["bloques_ganados"] >= np.ceil(t["bloques"] * 2 / 3))
        t["objetivo"] = name
        rows.append(t.rename_axis("modelo").reset_index())
    metrics = pd.concat(rows).set_index(["modelo", "objetivo"])
    meta = {"desde": str(min(o.index.min() for o in oos.values())),
            "hasta": str(max(o.index.max() for o in oos.values())), "modelos": names}
    return CandleReport([t[0] for t in targets], metrics, oos, meta)


def by_regime(report: CandleReport, target: str, models=None) -> pd.DataFrame:
    """Métricas de cada modelo dentro de cada régimen, para un objetivo.

    La referencia `siempre_sube` usa su probabilidad fuera de muestra en esas mismas filas.
    """
    models = models or [m for m in report.meta["modelos"] if m != "siempre_sube"]
    base = report.oos[("siempre_sube", target)]
    if "regimen" not in base:
        raise ValueError("Crea el dataset con regimes=True para desglosar por régimen.")
    key = ["symbol"] if "symbol" in base else []
    b = base.rename_axis("time").reset_index().rename(columns={"p": "p_base"})[["time", *key, "p_base"]]
    rows = []
    for m in models:
        o = report.oos[(m, target)].rename_axis("time").reset_index().merge(b, on=["time", *key], how="inner")
        for reg, g in o.groupby("regimen"):
            if len(g) < 200 or g["up"].nunique() < 2:
                continue
            brier, brier_b = np.mean((g["p"] - g["up"]) ** 2), np.mean((g["p_base"] - g["up"]) ** 2)
            rows.append({"regimen": reg, "modelo": m, "n": len(g), "freq_sube": g["up"].mean(),
                         "acierto": ((g["p"] > 0.5) == (g["up"] == 1)).mean(),
                         "auc": roc_auc_score(g["up"], g["p"]), "skill_vs_siempre_sube": 1 - brier / brier_b})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["regimen_nombre"] = out["regimen"].map(REGIME_LABELS)
    order = {r: i for i, r in enumerate(REGIMES)}
    return out.sort_values(["regimen", "auc"], key=lambda s: s.map(order) if s.name == "regimen" else -s)
