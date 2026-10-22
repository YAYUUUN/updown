"""Auditoría del evaluador con datos sintéticos de verdad conocida.

Antes de creerse un número sobre datos reales hay que comprobar que el
instrumento mide bien (es la misma idea que los tripwires de qbt):

  1. Paseo aleatorio  -> no hay nada que predecir: el veredicto DEBE ser "sin ventaja".
  2. Señal plantada   -> hay una autocorrelación real: algún modelo DEBE detectarla.
  3. Canario de fuga  -> se cuela el futuro como variable: el AUC DEBE dispararse.
     Si no lo hace, el evaluador está ciego y descartaría señales reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import evaluate
from .features import dataset


def synthetic_bars(n: int = 3000, rho: float = 0.0, seed: int = 0, vol: float = 0.01) -> pd.DataFrame:
    """Barras diarias con retornos AR(1) de coeficiente `rho` (0 = paseo aleatorio)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, vol, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = rho * r[t - 1] + eps[t]
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(12, 0.3, n)
    idx = pd.bdate_range("2010-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def stage_bars(n: int = 3000, seed: int = 0, stay: float = 0.99, vol=(0.005, 0.025), drift=(0.0004, -0.002),
               rho=(0.3, -0.3)) -> tuple[pd.DataFrame, pd.Series]:
    """Barras diarias que alternan dos etapas ocultas (cadena de Markov, duración media 1/(1-stay)).

    Por defecto:
      · calma   : volatilidad 0,5 %, deriva al alza, retornos con inercia (rho = +0,3).
      · agitada : volatilidad 2,5 %, deriva a la baja, retornos que se revierten (rho = -0,3).

    La regla cambia de signo entre etapas: un modelo por etapa puede aprender cada
    una. Con volatilidades distintas, un modelo lineal único ya imita buena parte del
    cambio (las variables sin normalizar pesan más en la etapa agitada); con la misma
    volatilidad, no. Devuelve las barras y la etapa verdadera de cada una.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.uniform() < stay else 1 - state[t - 1]
    vol, drift, rho = np.asarray(vol, dtype=float), np.asarray(drift, dtype=float), np.asarray(rho, dtype=float)
    r = np.zeros(n)
    for t in range(1, n):
        s = state[t]
        r[t] = drift[s] + rho[s] * (r[t - 1] - drift[state[t - 1]]) + rng.normal(0, vol[s])
    close = 100 * np.exp(np.cumsum(r))
    v = vol[state]
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, v / 4))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, v / 2)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, v / 2)))
    idx = pd.bdate_range("2010-01-01", periods=n, tz="UTC")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.lognormal(12, 0.3, n)}, index=idx)
    return bars, pd.Series(np.where(state == 0, "calma", "agitada"), index=idx, name="etapa_real")


def run(verbose: bool = True) -> dict[str, bool]:
    checks = {}

    rw = evaluate(dataset(synthetic_bars(rho=0.0, seed=1), "1d"))
    checks["paseo_aleatorio_sin_ventaja"] = rw.verdict.startswith("SIN VENTAJA")

    planted = evaluate(dataset(synthetic_bars(rho=0.15, seed=2), "1d"))
    checks["senal_plantada_detectada"] = planted.verdict.startswith("POSIBLE VENTAJA")

    leak = dataset(synthetic_bars(rho=0.0, seed=3), "1d")
    leak["fuga"] = leak["fwd_ret"].fillna(0.0)
    leak.attrs["features"] = leak.attrs["features"] + ["fuga"]
    leak.attrs["horizon"] = 1
    canary = evaluate(leak, models=("lineal",))
    checks["canario_de_fuga_visto"] = canary.table.loc["lineal", "auc"] > 0.9

    if verbose:
        for name, rep in (("Paseo aleatorio (rho=0)", rw), ("Señal plantada (rho=0.15)", planted)):
            print(f"\n== {name}")
            print(rep.table[["acierto", "auc", "p_auc_corregido", "skill_vs_siempre_sube", "bloques_ganados"]]
                  .round(4).to_string())
            print("Veredicto:", rep.verdict)
        print(f"\n== Canario de fuga: AUC del modelo lineal = {canary.table.loc['lineal', 'auc']:.3f}")
        print()
        for k, ok in checks.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {k}")
    return checks
