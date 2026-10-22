"""Variables (features) y objetivo, sin mirar al futuro.

Regla de oro: cada fila t usa SOLO información de barras cerradas hasta t. El
único sitio del paquete que mira hacia delante es `build_target` (un shift
negativo), y está aislado a propósito: es donde nacen casi todas las fugas.

Todas las variables son adimensionales (retornos, ratios, z-scores), así que un
mismo modelo puede entrenarse con varios activos a la vez (`pooled`): es la forma
honesta de tener "mucha información" sin inventarla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INTERVAL_SECONDS, ORDER_FLOW
from .indicators import atr, ema, rsi

RETURN_WINDOWS = (1, 2, 3, 5, 10, 20, 60)


def build_features(bars: pd.DataFrame, interval: str,
                   context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    c, o, h, low, v = (bars[k] for k in ("close", "open", "high", "low", "volume"))
    logret = np.log(c).diff()
    a = atr(h, low, c, 14)
    f = pd.DataFrame(index=bars.index)

    for n in RETURN_WINDOWS:
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["vol_5"] = logret.rolling(5).std()
    f["vol_20"] = logret.rolling(20).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]
    f["atr_pct"] = a / c
    f["rsi_14"] = (rsi(c, 14) - 50.0) / 50.0
    f["dist_ema10_atr"] = (c - ema(c, 10)) / a
    f["dist_ema50_atr"] = (c - ema(c, 50)) / a
    macd = ema(c, 12) - ema(c, 26)
    f["macd_hist"] = (macd - ema(macd, 9)) / c
    f["boll_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    hi20, lo20 = h.rolling(20).max(), low.rolling(20).min()
    f["range_pos_20"] = (c - lo20) / (hi20 - lo20)
    f["gap"] = np.log(o / c.shift(1))
    f["body_atr"] = (c - o) / a

    if (v > 0).mean() > 0.9:  # índices y divisas no traen volumen
        logv = np.log1p(v)
        f["volume_z"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()

    _add_order_flow(f, bars, interval)
    _add_calendar(f, interval)
    for name, ctx in (context or {}).items():
        _add_context(f, name, ctx)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f.loc[:, f.notna().mean() > 0.5]


def taker_ratio(bars: pd.DataFrame) -> pd.Series:
    """Parte del volumen de cada vela que fueron compras agresivas (0,5 = equilibrio; sin volumen, 0,5)."""
    v = bars["volume"]
    return (bars["taker_buy"] / v).where(v > 0, 0.5)


def _add_order_flow(f: pd.DataFrame, bars: pd.DataFrame, interval: str) -> None:
    """Flujo de órdenes (solo velas de Binance): quién empuja y cuánta actividad hay.

      · taker_ratio    : parte del volumen de la vela que fueron compras agresivas.
      · taker_ratio_1h : lo mismo con todo el volumen de la última hora (si la vela dura menos).
      · taker_ratio_z  : si esa proporción es inusual frente al último día (z-score).
      · trades_z       : si el número de operaciones es inusual frente al último día.
    """
    if not set(ORDER_FLOW) <= set(bars.columns):
        return
    step = INTERVAL_SECONDS.get(interval, 86400)
    hour = round(3600 / step)
    day = max(round(86400 / step), 20)
    ratio = taker_ratio(bars)
    f["taker_ratio"] = ratio
    if hour >= 2:
        vol_h = bars["volume"].rolling(hour).sum()
        f["taker_ratio_1h"] = (bars["taker_buy"].rolling(hour).sum() / vol_h).where(vol_h > 0, 0.5)
    f["taker_ratio_z"] = (ratio - ratio.rolling(day).mean()) / ratio.rolling(day).std()
    logn = np.log1p(bars["trades"])
    f["trades_z"] = (logn - logn.rolling(day).mean()) / logn.rolling(day).std()


def _add_calendar(f: pd.DataFrame, interval: str) -> None:
    idx = f.index
    dow = idx.day_of_week
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    if interval not in ("1d", "1wk", "1w"):
        hour = idx.hour + idx.minute / 60.0
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)


def _add_context(f: pd.DataFrame, name: str, ctx: pd.DataFrame) -> None:
    """Retornos del mercado de referencia, alineados con la última barra suya YA cerrada."""
    cc = ctx["close"]
    cf = pd.DataFrame({
        f"{name}_ret_1": np.log(cc / cc.shift(1)),
        f"{name}_ret_5": np.log(cc / cc.shift(5)),
        f"{name}_ret_20": np.log(cc / cc.shift(20)),
    })
    if name.startswith("^VIX") or name.upper() == "VIX":
        cf[f"{name}_level_z"] = (cc - cc.rolling(60).mean()) / cc.rolling(60).std()
    left = pd.DataFrame(index=f.index.tz_convert("UTC")).reset_index(names="t")
    right = cf.set_index(cf.index.tz_convert("UTC")).reset_index(names="t").sort_values("t")
    merged = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    for col in cf.columns:
        f[col] = merged[col].to_numpy()


def build_target(bars: pd.DataFrame, horizon: int = 1) -> tuple[pd.Series, pd.Series]:
    """y=1 si el cierre dentro de `horizon` barras es mayor que el actual; y el retorno log.

    Las últimas `horizon` filas quedan en NaN: su futuro aún no existe.
    """
    c = bars["close"]
    fwd = np.log(c.shift(-horizon) / c)
    y = (fwd > 0).astype(float).where(fwd.notna())
    return y.rename("up"), fwd.rename("fwd_ret")


SEQ_CHANNELS = ("r", "hl", "body", "vz", "tb")


def sequence_features(bars: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """Las últimas `seq_len` barras como columnas seq_<canal>_<retardo> (retardo 0 = barra actual).

    Canales por barra: retorno log, rango log(high/low), cuerpo log(close/open),
    volumen relativo (z-score sobre 96 barras, recortado) y, si la fuente trae flujo
    de órdenes (Binance), la proporción de compras agresivas centrada en 0. Solo usan el pasado.
    """
    c, o, h, low, v = (bars[k] for k in ("close", "open", "high", "low", "volume"))
    logv = np.log1p(v)
    base = {
        "r": np.log(c).diff(),
        "hl": np.log(h / low),
        "body": np.log(c / o),
        "vz": ((logv - logv.rolling(96, min_periods=20).mean()) / logv.rolling(96, min_periods=20).std()).clip(-5, 5),
    }
    if set(ORDER_FLOW) <= set(bars.columns):
        base["tb"] = taker_ratio(bars) - 0.5
    cols = {}
    for ch in [ch for ch in SEQ_CHANNELS if ch in base]:
        series = base[ch].replace([np.inf, -np.inf], np.nan)
        for lag in range(seq_len):
            cols[f"seq_{ch}_{lag}"] = series.shift(lag)
    return pd.DataFrame(cols, index=bars.index)


def build_candle_targets(bars: pd.DataFrame, candles: int) -> pd.DataFrame:
    """Dirección de cada una de las próximas `candles` velas por separado.

    up_v{k}=1 si la vela k (la que cierra k barras después de t) cierra por encima
    del cierre de la vela anterior. Es "¿sube la 1.ª, la 2.ª... vela?", no el acumulado.
    """
    c = np.log(bars["close"])
    out = {}
    for k in range(1, candles + 1):
        r = c.shift(-k) - c.shift(-(k - 1))
        out[f"up_v{k}"] = (r > 0).astype(float).where(r.notna())
        out[f"fwd_v{k}"] = r
    return pd.DataFrame(out, index=bars.index)


def dataset(bars: pd.DataFrame, interval: str, horizon: int = 1,
            context: dict[str, pd.DataFrame] | None = None, seq_len: int = 0,
            candles: int = 0, regimes: bool = False) -> pd.DataFrame:
    """Tabla lista para modelar: features + `up` + `fwd_ret`, sin filas de calentamiento.

    Con `seq_len` > 0 añade la ventana de barras para los modelos secuenciales
    (transformer). Esas columnas no las ven los demás modelos. Con `regimes` añade
    el régimen por reglas (`regimen`) y las observaciones del HMM (`hmm_vol`,
    `hmm_tend`); tampoco son variables de los modelos: sirven para repartir velas.
    """
    X = build_features(bars, interval, context)
    y, fwd = build_target(bars, horizon)
    parts = [X]
    S = None
    if seq_len:
        S = sequence_features(bars, seq_len)
        if (bars["volume"] > 0).mean() < 0.9:
            S = S.drop(columns=[c for c in S.columns if c.startswith("seq_vz_")])
        parts.append(S)
    extra = []
    if candles:
        extra.append(build_candle_targets(bars, candles))
    if regimes:
        from .hmm import hmm_observations
        from .regimes import detect_regimes, regime_inputs
        inputs = regime_inputs(bars)
        extra += [detect_regimes(bars, inputs).to_frame(), hmm_observations(bars, inputs)]
    data = pd.concat(parts + extra, axis=1).join([y, fwd])
    ok = X.notna().all(axis=1) if S is None else X.notna().all(axis=1) & S.notna().all(axis=1)
    data = data[ok]
    data.attrs["features"] = list(X.columns)
    data.attrs["seq_features"] = [] if S is None else list(S.columns)
    data.attrs["seq_len"] = seq_len
    data.attrs["candles"] = candles
    data.attrs["horizon"] = horizon
    data.attrs["interval"] = interval
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    return data.attrs.get("features") or [c for c in data.columns
                                          if c not in ("up", "fwd_ret", "symbol", "regimen")
                                          and not c.startswith(("seq_", "up_v", "fwd_v", "hmm_"))]


def model_columns(data: pd.DataFrame, model) -> list[str]:
    """Columnas que recibe un modelo: las tabulares, más la ventana si es secuencial."""
    cols = feature_columns(data)
    if getattr(model, "uses_sequence", False):
        seq = data.attrs.get("seq_features") or [c for c in data.columns if c.startswith("seq_")]
        if not seq:
            raise ValueError("Este modelo necesita la ventana de barras: crea el dataset con seq_len > 0.")
        cols = cols + seq
    return cols
