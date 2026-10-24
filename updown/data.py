"""Datos de mercado sin bróker y sin claves.

Dos fuentes públicas:

  · Yahoo Finance (vía yfinance): acciones, ETF, índices, divisas y cripto
    (p. ej. AAPL, SPY, ^GSPC, EURUSD=X, BTC-USD). Gratis y sin cuenta.
  · Binance (API REST pública): cripto en tiempo real (p. ej. BTCUSDT).

Límites de Yahoo que conviene tener presentes (son suyos, no de este código):

    1m           -> solo los últimos 7 días
    2m..30m      -> últimos 60 días
    1h           -> últimos 730 días
    1d / 1wk     -> todo el histórico

La última barra que devuelven las dos fuentes suele estar ABIERTA (el precio se
sigue moviendo). Predecir sobre una barra abierta es mirar el futuro a medias,
así que todo el sistema trabaja solo con barras cerradas: `drop_unclosed`.

Las velas de Binance traen además el flujo de órdenes: cuántas operaciones hubo
(`trades`) y qué volumen fue de compras agresivas, es decir, órdenes que compran
al precio de venta en ese momento (`taker_buy`). Yahoo no lo da.
"""
from __future__ import annotations

import os
import pickle
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume"]
ORDER_FLOW = ["trades", "taker_buy"]  # solo Binance: nº de operaciones y volumen de compras agresivas

# Sube cuando cambian las columnas guardadas: una caché de otra versión se descarta
# entera en vez de mezclar velas viejas (sin flujo de órdenes) con las nuevas.
CACHE_VERSION = 2

YAHOO_MAX_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "1h": "730d", "1d": "max", "1wk": "max",
}

# UPDOWN_BINANCE_URL permite usar el espejo de solo datos de mercado
# (https://data-api.binance.vision/api/v3/klines), p. ej. desde servidores en EE. UU.
BINANCE_URL = os.environ.get("UPDOWN_BINANCE_URL", "https://api.binance.com/api/v3/klines")
BINANCE_DEFAULT_BARS = 20000  # 1h -> ~2,3 años; 1m -> ~2 semanas
BINANCE_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}

INTERVAL_SECONDS = {
    "1m": 60, "2m": 120, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600, "1h": 3600,
    "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, "1wk": 604800,
    "1w": 604800,
}

SOURCES = ("yahoo", "binance")

# Margen tras el cierre de sesión para que Yahoo publique el precio de cierre definitivo.
SESSION_BUFFER = pd.Timedelta(minutes=20)


class DataError(RuntimeError):
    """La fuente no devolvió datos utilizables."""


@dataclass(frozen=True)
class Market:
    """Qué serie se pide: fuente, símbolo e intervalo de barra."""

    source: str
    symbol: str
    interval: str

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"Fuente desconocida: {self.source!r}. Usa una de {SOURCES}.")
        valid = YAHOO_MAX_PERIOD if self.source == "yahoo" else BINANCE_INTERVALS
        if self.interval not in valid:
            raise ValueError(f"{self.source} no sirve el intervalo {self.interval!r}. Válidos: {sorted(valid)}")


def cache_dir() -> str:
    """Carpeta de caché: $UPDOWN_CACHE o ~/.cache/updown."""
    return os.environ.get("UPDOWN_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "updown")


def _cache_path(market: Market) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", market.symbol)
    return os.path.join(cache_dir(), f"{market.source}_{safe}_{market.interval}.pkl")


def _load_cache(path: str):
    try:
        with open(path, "rb") as fh:
            version, df, attrs, fetched_at = pickle.load(fh)  # el formato sin versión no encaja: ValueError
        if version != CACHE_VERSION:
            return None, None
        df.attrs = attrs
        return df, fetched_at
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, AttributeError, ImportError):
        return None, None


def _save_cache(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump((CACHE_VERSION, df, dict(df.attrs), time.time()), fh)
    os.replace(tmp, path)


def fetch(market: Market, bars: int | None = None, cache: bool = True, session=None) -> pd.DataFrame:
    """Barras OHLCV de la fuente indicada (incluye la última, que puede estar abierta).

    Con `cache` (por defecto) guarda las barras en disco:
      · Binance: solo se piden a la API las barras posteriores a la última guardada.
      · Yahoo: se reutiliza la descarga si tiene menos de un intervalo (máx. 1 h) de antigüedad.
    """
    if not cache:
        if market.source == "yahoo":
            return fetch_yahoo(market.symbol, market.interval)
        return fetch_binance(market.symbol, market.interval, bars=bars or BINANCE_DEFAULT_BARS, session=session)

    path = _cache_path(market)
    cached, fetched_at = _load_cache(path)
    if market.source == "yahoo":
        ttl = min(INTERVAL_SECONDS[market.interval], 3600)
        if cached is not None and time.time() - fetched_at < ttl:
            return cached
        df = fetch_yahoo(market.symbol, market.interval)
        _save_cache(path, df)
        return df

    want = bars or BINANCE_DEFAULT_BARS
    if cached is not None and len(cached) >= want:
        start = int(cached.index[-1].timestamp() * 1000)  # la última guardada puede estar abierta: se repide
        fresh = fetch_binance_since(market.symbol, market.interval, start, session=session)
        df = pd.concat([cached.iloc[:-1], fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = fetch_binance(market.symbol, market.interval, bars=want, session=session)
    _save_cache(path, df)
    return df.iloc[-want:]


def fetch_binance_since(symbol: str, interval: str, start_ms: int, session=None) -> pd.DataFrame:
    """Klines desde `start_ms` (incluida) hasta ahora, paginando hacia delante."""
    http = session or requests
    rows: list[list] = []
    while True:
        resp = http.get(BINANCE_URL, params={"symbol": symbol.upper(), "interval": interval,
                                             "startTime": start_ms, "limit": 1000}, timeout=15)
        if resp.status_code == 400:
            raise DataError(f"Binance rechazó {symbol!r}: {resp.text[:200]}")
        resp.raise_for_status()
        page = resp.json()
        rows += page
        if len(page) < 1000:
            break
        start_ms = int(page[-1][0]) + 1
    if not rows:
        raise DataError(f"Binance no devolvió datos nuevos para {symbol!r} ({interval}).")
    return parse_binance_klines(rows)


def fetch_yahoo(symbol: str, interval: str = "1d", period: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    period = period or YAHOO_MAX_PERIOD[interval]
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise DataError(f"Yahoo no devolvió datos para {symbol!r} ({interval}). ¿Existe el símbolo?")
    df = df.rename(columns=str.lower)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise DataError(f"Yahoo devolvió columnas inesperadas para {symbol!r}: faltan {sorted(missing)}")
    out = _clean(df[COLUMNS])
    try:  # hora de cierre de la sesión regular de hoy: permite usar la barra diaria en cuanto cierra
        end = ticker.history_metadata["currentTradingPeriod"]["regular"]["end"]
        out.attrs["session_end"] = pd.Timestamp(int(end), unit="s", tz="UTC")
    except (KeyError, TypeError, ValueError):
        pass
    return out


def fetch_binance(symbol: str, interval: str = "1h", bars: int = BINANCE_DEFAULT_BARS, session=None) -> pd.DataFrame:
    """Klines públicas de Binance, paginando hacia atrás de 1000 en 1000."""
    http = session or requests
    rows: list[list] = []
    end_time = None
    while len(rows) < bars:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": min(1000, bars - len(rows))}
        if end_time is not None:
            params["endTime"] = end_time
        resp = http.get(BINANCE_URL, params=params, timeout=15)
        if resp.status_code == 400:
            raise DataError(f"Binance rechazó {symbol!r}: {resp.text[:200]}")
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        rows = page + rows
        end_time = int(page[0][0]) - 1
        if len(page) < params["limit"]:
            break
    if not rows:
        raise DataError(f"Binance no devolvió datos para {symbol!r} ({interval}).")
    return parse_binance_klines(rows)


def parse_binance_klines(rows: list[list]) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    raw = pd.DataFrame(rows, columns=cols)
    out = raw[COLUMNS].astype(float)
    out["trades"] = raw["trades"].astype(float)
    out["taker_buy"] = raw["taker_base"].astype(float)
    out.index = pd.to_datetime(raw["open_time"].astype("int64"), unit="ms", utc=True)
    return _clean(out)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    return out[(out["close"] > 0)]


def drop_unclosed(bars: pd.DataFrame, interval: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Quita las barras que aún no han cerrado: inicio + duración > ahora.

    El índice de cada barra es su hora de INICIO (así lo dan Yahoo y Binance). En
    barras diarias de bolsa, la de hoy cuenta como cerrada SESSION_BUFFER después
    del cierre de la sesión regular (dato de Yahoo); sin ese dato, a medianoche.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")
    end = bars.index + pd.Timedelta(seconds=INTERVAL_SECONDS[interval])
    closed = np.asarray(end <= now)
    session_end = bars.attrs.get("session_end")
    if interval == "1d" and session_end is not None and len(bars) and not closed[-1]:
        same_day = bars.index[-1].date() == session_end.tz_convert(bars.index.tz).date()
        closed[-1] = same_day and now >= session_end + SESSION_BUFFER
    out = bars[closed]
    out.attrs = dict(bars.attrs)
    return out


def default_context(market: Market) -> list[Market]:
    """Mercados de referencia que se añaden como variables (el 'clima' general)."""
    if market.source == "binance":
        return [] if market.symbol.upper() == "BTCUSDT" else [Market("binance", "BTCUSDT", market.interval)]
    sym = market.symbol.upper()
    if sym.endswith("-USD"):  # cripto en Yahoo
        return [] if sym == "BTC-USD" else [Market("yahoo", "BTC-USD", market.interval)]
    context = [] if sym == "SPY" else [Market("yahoo", "SPY", market.interval)]
    if market.interval in ("1d", "1wk"):
        context.append(Market("yahoo", "^VIX", market.interval))
    return context
