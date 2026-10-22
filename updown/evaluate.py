"""Evaluación honesta: walk-forward, comparación con las referencias y veredicto.

Cómo se mide (y por qué):

  · Walk-forward con ventana creciente: cada bloque de test se predice con un
    modelo entrenado SOLO con el pasado. Nada de K-fold aleatorio, que entrena con
    el futuro.
  · Purga: entre el final del entrenamiento y el test se tiran `horizon` barras,
    porque el objetivo de la última barra de train mira dentro del test.
  · La métrica principal es el Brier (error cuadrático de la probabilidad) frente
    a `siempre_sube`. La accuracy engaña: si sube el 54% de los días, decir
    "siempre sube" ya acierta un 54% sin saber nada.
  · AUC con su p-valor, corregido por cuántos modelos se han probado (Holm, que
    controla lo mismo que Bonferroni con algo más de potencia). Probar 13 modelos
    y quedarse con el mejor sin corregir es hacer trampa: con 13 intentos, alguno
    sale "significativo" al 5% por puro azar casi la mitad de las veces.
  · Consistencia: el modelo tiene que ganar a la referencia en al menos 2/3 de
    los bloques, no solo en la media (una racha buena no es una señal).
  · Simulación con costes: aunque haya señal, puede no pagar las comisiones.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import model_columns
from .models import BASELINES, MODELS, make_model


@dataclass
class EvalReport:
    table: pd.DataFrame          # una fila por modelo
    folds: pd.DataFrame          # Brier por modelo y bloque
    calibration: pd.DataFrame    # probabilidad media vs frecuencia real por tramo
    equity: pd.DataFrame         # curvas acumuladas (log) de la simulación con costes
    oos: dict[str, pd.DataFrame] = field(repr=False, default_factory=dict)
    verdict: str = ""
    best: str | None = None
    meta: dict = field(default_factory=dict)


def time_folds(times: pd.Index, n_folds: int, min_train_frac: float, purge: int):
    """Bloques (train_times, test_times) sobre los instantes únicos, en orden."""
    uniq = pd.Index(times.unique()).sort_values()
    n = len(uniq)
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    for a, b in itertools.pairwise(edges):
        train_end = a - purge
        if train_end < 50 or b <= a:
            continue
        yield uniq[:train_end], uniq[a:b]


def walk_forward(data: pd.DataFrame, model_name: str, n_folds: int = 6,
                 min_train_frac: float = 0.4, horizon: int | None = None,
                 target: str = "up", ret_col: str = "fwd_ret") -> pd.DataFrame:
    """Predicciones fuera de muestra: columnas p, up, fwd_ret, fold (+ symbol, regimen si hay).

    `target`/`ret_col` permiten evaluar otro objetivo (p. ej. la vela 2) sin copiar la tabla;
    en la salida siempre se llaman `up` y `fwd_ret`. `horizon` fija la purga.
    """
    horizon = horizon or data.attrs.get("horizon", 1)
    cols = model_columns(data, make_model(model_name))
    labeled = data[data[target].notna()]
    keep = [c for c in ("symbol", "regimen") if c in labeled]
    parts = []
    for i, (train_t, test_t) in enumerate(time_folds(labeled.index, n_folds, min_train_frac, horizon)):
        tr = labeled[labeled.index.isin(train_t)]
        te = labeled[labeled.index.isin(test_t)]
        model = make_model(model_name).fit(tr[cols], tr[target])
        out = te[[target, ret_col, *keep]].rename(columns={target: "up", ret_col: "fwd_ret"})
        out["p"] = np.clip(model.proba(te[cols]), 1e-4, 1 - 1e-4)
        out["fold"] = i
        parts.append(out)
    if not parts:
        raise ValueError("No hay histórico suficiente para evaluar (hace falta más de ~150 barras).")
    return pd.concat(parts)


def scores(oos: pd.DataFrame, horizon: int = 1) -> dict:
    """Métricas fuera de muestra.

    Con horizonte h, predicciones consecutivas comparten h-1 barras de futuro: no son
    independientes. El error estándar del AUC se infla por sqrt(h) (n efectivo = n/h)
    para que el p-valor no salga optimista.
    """
    y, p = oos["up"].to_numpy(), oos["p"].to_numpy()
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc = roc_auc_score(y, p) if n1 and n0 else float("nan")
    se0 = math.sqrt(horizon * (n1 + n0 + 1) / (12.0 * n1 * n0)) if n1 and n0 else float("nan")
    z = (auc - 0.5) / se0 if not math.isnan(se0) else float("nan")
    return {
        "n": len(y),
        "acierto": float(((p > 0.5) == (y == 1)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "auc": float(auc),
        "p_auc": float(0.5 * math.erfc(z / math.sqrt(2))) if not math.isnan(z) else float("nan"),
        "freq_sube": float(y.mean()),
    }


def se_by_day(values: np.ndarray, times: pd.Index) -> float:
    """Error estándar de la media agrupando por día (cluster-robusto).

    Varios activos a la misma hora y velas seguidas no son observaciones
    independientes; tratarlas como tales daría un error estándar demasiado pequeño.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    days = pd.DatetimeIndex(times).floor("D") if isinstance(times, pd.DatetimeIndex) else np.arange(n)
    resid = pd.Series(values - values.mean()).groupby(np.asarray(days)).sum()
    c = len(resid)
    if c < 2:
        return float("nan")
    return float(np.sqrt(c / (c - 1) * (resid ** 2).sum()) / n)


def holm(p: pd.Series) -> pd.Series:
    """p-valores ajustados por Holm-Bonferroni (controla la tasa de falsos positivos familiar)."""
    p = p.astype(float)
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, name in enumerate(order):
        val = p[name]
        running = max(running, min(1.0, (m - i) * val)) if pd.notna(val) else running
        adj[name] = running if pd.notna(val) else float("nan")
    return pd.Series(adj).reindex(p.index)


def calibration_table(oos: pd.DataFrame, bins=(0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0)) -> pd.DataFrame:
    cut = pd.cut(oos["p"], bins=list(bins), include_lowest=True)
    g = oos.groupby(cut, observed=True)
    return pd.DataFrame({"p_media": g["p"].mean(), "sube_real": g["up"].mean(), "n": g["up"].size()})


def bars_per_year(times: pd.Index) -> float:
    uniq = pd.Index(times.unique()).sort_values()
    years = (uniq[-1] - uniq[0]).total_seconds() / (365.25 * 86400)
    return len(uniq) / years if years > 0 else float("nan")


def simulate(oos: pd.DataFrame, horizon: int, cost_bps: float, band: float) -> pd.Series:
    """Estrategia larga/fuera: dentro si P(sube) > 0.5 + band. Retornos log netos por paso.

    Con horizonte h se usa una predicción cada h barras para no solapar operaciones.
    Con varios activos, cartera equiponderada entre los que tengan señal en ese instante.
    """
    sims = []
    groups = oos.groupby("symbol") if "symbol" in oos else [(None, oos)]
    for _, g in groups:
        g = g.sort_index().iloc[::horizon]
        pos = (g["p"] > 0.5 + band).astype(float)
        trades = pos.diff().abs().fillna(pos.iloc[0] if len(pos) else 0.0)
        sims.append(pos * g["fwd_ret"] - trades * cost_bps / 1e4)
    return pd.concat(sims).groupby(level=0).mean().sort_index()


def evaluate(data: pd.DataFrame, models=MODELS, n_folds: int = 6, cost_bps: float = 5.0,
             band: float = 0.02, min_train_frac: float = 0.4) -> EvalReport:
    horizon = data.attrs.get("horizon", 1)
    names = list(BASELINES) + [m for m in models if m not in BASELINES]
    oos = {m: walk_forward(data, m, n_folds, min_train_frac, horizon) for m in names}

    base = oos["siempre_sube"]
    base_fold_brier = base.assign(se=(base["p"] - base["up"]) ** 2).groupby("fold")["se"].mean()
    tried = [m for m in names if m not in BASELINES]
    rows, fold_rows = [], {}
    bpy = bars_per_year(base.index) / horizon
    equity = {}
    for m in names:
        s = scores(oos[m], horizon)
        fb = oos[m].assign(se=(oos[m]["p"] - oos[m]["up"]) ** 2).groupby("fold")["se"].mean()
        fold_rows[m] = fb
        s["skill_vs_siempre_sube"] = 1 - s["brier"] / scores(base, horizon)["brier"]
        s["bloques_ganados"] = int((fb < base_fold_brier).sum()) if m != "siempre_sube" else 0
        s["bloques"] = len(fb)
        strat = simulate(oos[m], horizon, cost_bps, band)
        equity[m] = strat.cumsum()
        s["ret_anual_estrategia"] = float(np.expm1(strat.mean() * bpy)) if len(strat) else float("nan")
        rows.append(pd.Series(s, name=m))
    bh = simulate(base.assign(p=1.0), horizon, 0.0, 0.0)
    equity["comprar_y_mantener"] = bh.cumsum()

    table = pd.DataFrame(rows)
    table["p_auc_corregido"] = holm(table.loc[tried, "p_auc"]).reindex(table.index)
    table["ret_anual_comprar_y_mantener"] = float(np.expm1(bh.mean() * bpy))
    best, verdict = _verdict(table, tried, cost_bps)
    meta = {
        "horizonte": horizon, "coste_bps": cost_bps, "banda": band, "modelos_probados": len(tried),
        "datos_desde": str(data.index.min()), "desde": str(base.index.min()), "hasta": str(base.index.max()),
    }
    calib = calibration_table(oos[best]) if best else calibration_table(oos[tried[0]]) if tried else pd.DataFrame()
    return EvalReport(table=table, folds=pd.DataFrame(fold_rows), calibration=calib,
                      equity=pd.DataFrame(equity).ffill(), oos=oos, verdict=verdict, best=best, meta=meta)


def _verdict(table: pd.DataFrame, tried: list[str], cost_bps: float = 5.0) -> tuple[str | None, str]:
    """Severo a propósito: lo normal en mercados líquidos es que no haya ventaja."""
    if not tried:
        return None, "No se ha probado ningún modelo."
    t = table.loc[tried]
    passing = t[(t["p_auc_corregido"] < 0.05) & (t["skill_vs_siempre_sube"] > 0)
                & (t["bloques_ganados"] >= np.ceil(t["bloques"] * 2 / 3))]
    if passing.empty:
        best = t["skill_vs_siempre_sube"].idxmax()
        return best, ("SIN VENTAJA DEMOSTRADA. Ningún modelo predice mejor que 'siempre_sube' de forma "
                      "significativa y consistente. Sus probabilidades no son una señal: no las uses para operar.")
    best = passing["skill_vs_siempre_sube"].idxmax()
    r = table.loc[best]
    pays = r["ret_anual_estrategia"] > r["ret_anual_comprar_y_mantener"]
    simple = ""
    if "momentum" in table.index and table.loc["momentum", "skill_vs_siempre_sube"] >= r["skill_vs_siempre_sube"]:
        simple = " Ojo: la regla de una línea 'momentum' predice igual o mejor; el modelo no aporta nada sobre ella."
    return best, (f"POSIBLE VENTAJA ESTADÍSTICA en '{best}' (AUC {r['auc']:.3f}, p corregido "
                  f"{r['p_auc_corregido']:.3g}, gana en {int(r['bloques_ganados'])}/{int(r['bloques'])} bloques). "
                  + (f"La simulación {'sin costes' if cost_bps == 0 else 'con costes'} "
                     f"{'supera' if pays else 'NO supera'} a comprar y mantener, ")
                  + "y nada de esto garantiza que siga funcionando. Valídalo en vivo antes de creértelo." + simple)
