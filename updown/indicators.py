"""Indicadores técnicos como funciones puras sobre pandas (vienen del bot qbt3).

RSI y ATR usan suavizado de Wilder (EWM con alpha=1/period), la definición
estándar que replican TradingView y casi todas las plataformas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder. Barra totalmente plana (0/0) -> 50 por convención."""
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0.0), period)
    avg_loss = wilder_smooth((-delta).clip(lower=0.0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)
