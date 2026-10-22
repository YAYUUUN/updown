"""Regímenes de mercado: pocos casos que cubren casi todo, detectados SOLO con el pasado.

No hace falta leer noticias para saber en qué situación está un mercado: las
noticias dejan huella en el precio y el volumen, y esa huella se ve en cuanto
cierra la barra. Cada barra t recibe UNA etiqueta, en este orden de prioridad:

  1. shock    (noticia / sobresalto): en las últimas barras hubo un movimiento de
              más de 4 desviaciones típicas o un volumen 3 desviaciones por encima
              de lo normal. Es la huella de una noticia inesperada.
  2. crisis   : volatilidad en el 20% más alto de su último año Y precio cayendo
              (por debajo de su media de ~30 días, con retorno de 30 días negativo).
  3. calma    (sin noticias): volatilidad en el 30% más bajo de su último año y sin
              tendencia clara. Mercado "aburrido".
  4. alcista  : precio por encima de su media de ~30 días y retorno de 30 días positivo.
  5. bajista  : lo demás con tendencia negativa (cae, pero sin pánico).
  (lateral    : lo que no encaja en nada de lo anterior.)

Los umbrales se calculan con ventanas que miran hacia atrás (percentiles del
último año desplazados una barra), así que la etiqueta de t nunca usa datos
posteriores a t. Lo que esto NO puede hacer: anticipar el régimen. Se detecta
cuando ya ha empezado, con algo de retraso, y los cambios bruscos se ven tarde.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("shock", "crisis", "calma", "alcista", "bajista", "lateral")
REGIME_LABELS = {
    "shock": "Noticia / sobresalto",
    "crisis": "Crisis (cae con pánico)",
    "calma": "Calma, sin noticias",
    "alcista": "Mercado alcista",
    "bajista": "Mercado bajista",
    "lateral": "Lateral",
}


def bars_per_day(index: pd.DatetimeIndex) -> float:
    days = (index[-1] - index[0]).total_seconds() / 86400
    return max(len(index) / days, 1.0) if days > 0 else 1.0


def regime_inputs(bars: pd.DataFrame) -> pd.DataFrame:
    """Las magnitudes con las que se decide el régimen (útiles para explicar la etiqueta)."""
    c, v = bars["close"], bars["volume"]
    bpd = bars_per_day(bars.index)
    day = max(int(round(bpd)), 5)            # ~1 día de barras (mínimo 5)
    month = max(int(round(30 * bpd)), 20)    # ~30 días
    year = max(int(round(365 * bpd)), 250)   # ~1 año
    r = np.log(c).diff()
    vol = r.rolling(day).std()
    out = pd.DataFrame(index=bars.index)
    out["vol"] = vol
    # Percentiles del último año, desplazados para no incluir la barra actual.
    out["vol_alta"] = vol.rolling(year, min_periods=month).quantile(0.8).shift(1)
    out["vol_baja"] = vol.rolling(year, min_periods=month).quantile(0.3).shift(1)
    out["tendencia"] = np.log(c / c.shift(month))
    out["sobre_media"] = c > c.rolling(month).mean()
    # Sobresalto: movimiento de >4 sigmas (sigma de la semana anterior) en las últimas barras.
    sigma = r.rolling(5 * day, min_periods=day).std().shift(1)
    big_move = (r.abs() > 4 * sigma).astype(float)
    shock = big_move.rolling(max(day // 4, 2), min_periods=1).max() > 0
    if (v > 0).mean() > 0.9:
        logv = np.log1p(v)
        vz = (logv - logv.rolling(5 * day, min_periods=day).mean().shift(1)) / \
            logv.rolling(5 * day, min_periods=day).std().shift(1)
        shock |= (vz > 3).astype(float).rolling(max(day // 4, 2), min_periods=1).max() > 0
    out["shock"] = shock
    # "Sin tendencia clara": movimiento de 30 días menor que una desviación típica de ese plazo.
    out["sin_tendencia"] = out["tendencia"].abs() < vol * np.sqrt(month)
    return out


def detect_regimes(bars: pd.DataFrame, inputs: pd.DataFrame | None = None) -> pd.Series:
    x = regime_inputs(bars) if inputs is None else inputs
    ready = x["vol_alta"].notna() & x["tendencia"].notna()
    lab = pd.Series("lateral", index=bars.index, dtype=object)
    bajista = (x["tendencia"] < 0) & ~x["sobre_media"]
    lab[bajista] = "bajista"
    lab[(x["tendencia"] > 0) & x["sobre_media"]] = "alcista"
    lab[(x["vol"] <= x["vol_baja"]) & x["sin_tendencia"]] = "calma"
    lab[(x["vol"] >= x["vol_alta"]) & bajista] = "crisis"
    lab[x["shock"]] = "shock"
    lab[~ready] = np.nan
    return lab.rename("regimen")


def regime_summary(labels: pd.Series) -> pd.DataFrame:
    """Qué parte del tiempo pasa el mercado en cada régimen."""
    s = labels.dropna()
    share = s.value_counts(normalize=True).reindex(REGIMES).fillna(0.0)
    return pd.DataFrame({"regimen": [REGIME_LABELS[r] for r in REGIMES], "tiempo": share.values}, index=REGIMES)
