"""Predicción en vivo y registro verificable.

Flujo: descargar barras -> quitar la barra abierta -> construir variables ->
entrenar con todo el histórico etiquetado -> dar P(sube) para la última barra
cerrada, es decir, para el cierre de dentro de `horizon` barras.

Cada predicción se apunta en un CSV ANTES de conocer el resultado, y luego se
completa con lo que pasó. Ese registro es la única prueba que vale: un backtest
se puede retocar sin querer; un registro escrito por adelantado, no.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import INTERVAL_SECONDS, Market, default_context, drop_unclosed, fetch
from .features import dataset, model_columns
from .models import ALL_MODELS, MODELS, make_model


def load_bars(market: Market, now: pd.Timestamp | None = None, bars: int | None = None) -> pd.DataFrame:
    return drop_unclosed(fetch(market, bars), market.interval, now)


def load_dataset(market: Market, horizon: int = 1, context: list[Market] | None = None,
                 now: pd.Timestamp | None = None, bars: int | None = None, seq_len: int = 0,
                 candles: int = 0, regimes: bool = False) -> pd.DataFrame:
    n_bars = bars
    bars = load_bars(market, now, n_bars)
    ctx_markets = default_context(market) if context is None else context
    ctx = {}
    for cm in ctx_markets:
        try:
            ctx[cm.symbol] = load_bars(cm, now, n_bars)
        except Exception as exc:  # una referencia caída no debe tumbar la predicción
            print(f"[aviso] sin datos de contexto {cm.symbol}: {exc}")
    data = dataset(bars, market.interval, horizon, ctx, seq_len, candles, regimes)
    data.attrs["last_close"] = float(bars["close"].iloc[-1])
    return data


def load_pooled(markets: list[Market], horizon: int = 1, context: list[Market] | None = None,
                now: pd.Timestamp | None = None, bars: int | None = None, seq_len: int = 0,
                candles: int = 0, regimes: bool = False) -> pd.DataFrame:
    """Varios activos apilados en una sola tabla (columna `symbol`): más datos, mismo modelo."""
    parts, feats, seq = [], None, None
    for m in markets:
        d = load_dataset(m, horizon, context, now, bars, seq_len, candles, regimes)
        cols = d.attrs["features"]
        own_seq = d.attrs.get("seq_features", [])
        feats = cols if feats is None else [c for c in feats if c in cols]
        seq = own_seq if seq is None else [c for c in seq if c in own_seq]
        parts.append(d.assign(symbol=m.symbol))
    data = pd.concat(parts).sort_index()
    extra = [c for c in data.columns if c.startswith(("up_v", "fwd_v")) or c == "regimen"]
    data = data[feats + seq + extra + ["up", "fwd_ret", "symbol"]].dropna(subset=feats + seq)
    data.attrs.update({"features": feats, "seq_features": seq, "seq_len": seq_len, "horizon": horizon,
                       "candles": candles,
                       "interval": markets[0].interval})
    return data


@dataclass
class Prediction:
    symbol: str
    interval: str
    horizon: int
    bar_time: pd.Timestamp       # inicio de la última barra cerrada
    close: float
    target_time: pd.Timestamp    # cuándo se sabrá el resultado (fin de la barra objetivo)
    probs: dict[str, float]


def predict_now(market: Market, horizon: int = 1, models=MODELS, context: list[Market] | None = None,
                data: pd.DataFrame | None = None, now: pd.Timestamp | None = None) -> Prediction:
    data = load_dataset(market, horizon, context, now) if data is None else data
    train = data[data["up"].notna()]
    last = data.iloc[[-1]]
    probs = {}
    for name in ["siempre_sube", *[m for m in models if m != "siempre_sube"]]:
        if name not in ALL_MODELS:
            raise ValueError(f"Modelo desconocido: {name}")
        model = make_model(name)
        cols = model_columns(data, model)
        probs[name] = float(model.fit(train[cols], train["up"]).proba(last[cols])[0])
    bar_time = last.index[0]
    step = pd.Timedelta(seconds=INTERVAL_SECONDS[market.interval])
    return Prediction(market.symbol, market.interval, horizon, bar_time,
                      float(data.attrs.get("last_close", np.nan)), bar_time + step * (horizon + 1), probs)


def target_label(pred: Prediction) -> str:
    """Cuándo se resuelve la predicción, dicho como lo diría una persona."""
    if pred.interval in ("1d", "1wk", "1w"):
        unit = "sesión" if pred.interval == "1d" else "semana"
        return f"al cierre de la {unit} siguiente" if pred.horizon == 1 else f"dentro de {pred.horizon} {unit}es"
    return f"hacia las {pred.target_time.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC"


def log_path(folder: str, market: Market, horizon: int) -> str:
    safe = market.symbol.replace("^", "").replace("/", "-").replace("=", "")
    return os.path.join(folder, f"{market.source}_{safe}_{market.interval}_h{horizon}.csv")


def log_prediction(path: str, pred: Prediction) -> bool:
    """Añade la predicción si esa barra no estaba ya apuntada. Devuelve True si la añade."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"bar_time": pred.bar_time.isoformat(), "close": pred.close, "target_time": pred.target_time.isoformat(),
           "logged_at": pd.Timestamp.now(tz="UTC").isoformat(), **{f"p_{k}": round(v, 5) for k, v in pred.probs.items()},
           "close_real": np.nan, "sube_real": np.nan}
    if os.path.exists(path):
        old = pd.read_csv(path)
        if pred.bar_time.isoformat() in set(old["bar_time"]):
            return False
        pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)
    return True


def update_outcomes(path: str, bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rellena el resultado real de las predicciones cuyo horizonte ya ha pasado."""
    if not os.path.exists(path):
        return pd.DataFrame()
    log = pd.read_csv(path)
    closes = bars["close"]
    pos = {t.isoformat(): i for i, t in enumerate(closes.index)}
    for i, r in log[log["sube_real"].isna()].iterrows():
        j = pos.get(r["bar_time"])
        if j is not None and j + horizon < len(closes):
            future = float(closes.iloc[j + horizon])
            log.at[i, "close_real"] = future
            log.at[i, "sube_real"] = float(future > r["close"])
    log.to_csv(path, index=False)
    return log


def live_scoreboard(log: pd.DataFrame) -> pd.DataFrame:
    """Acierto y Brier de cada modelo SOLO con predicciones apuntadas por adelantado."""
    done = log[log["sube_real"].notna()]
    rows = {}
    for c in [c for c in log.columns if c.startswith("p_")]:
        p, y = done[c].to_numpy(), done["sube_real"].to_numpy()
        rows[c[2:]] = {"n": len(done), "acierto": float(((p > 0.5) == (y == 1)).mean()) if len(done) else np.nan,
                       "brier": float(np.mean((p - y) ** 2)) if len(done) else np.nan}
    return pd.DataFrame(rows).T
