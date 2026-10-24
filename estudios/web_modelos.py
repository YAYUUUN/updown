"""Modelos y su historial para la web: BTC, ETH y SOL en velas de 15 minutos.

    .venv\\Scripts\\python -m estudios.web_modelos

1. Descarga ~2 años de velas de 15 min de los tres activos (con caché).
2. Evalúa ridge y gradient_boosting fuera de muestra, vela a vela, con el mismo
   walk-forward purgado que `estudios.velas_regimenes`. Es el historial que la web
   enseña al lado de cada probabilidad.
3. Entrena los modelos finales con todo el histórico etiquetado y los exporta.

Escribe web/data/models.json y web/data/metrics.json. Con
UPDOWN_BINANCE_URL=https://data-api.binance.vision/api/v3/klines funciona también
donde api.binance.com no responde (los servidores de GitHub Actions están en EE. UU.).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import pandas as pd

from updown import Market, load_pooled
from updown.candles import evaluate_candles, targets_of
from updown.export import WEB_MODELS, export_model
from updown.features import feature_columns
from updown.models import make_model

warnings.filterwarnings("ignore")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "15m"
BARS = 70_000
OUT = os.path.join(os.path.dirname(__file__), "..", "web", "data")
METRICS = ("n", "acierto", "auc", "p_auc_corregido", "skill_vs_siempre_sube", "bloques_ganados", "bloques",
           "freq_sube", "ventaja")


def _plain(v):
    """Número de numpy/pandas a JSON: enteros como enteros, 6 cifras significativas, NaN como null."""
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return bool(v)
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return int(v) if v.is_integer() and abs(v) < 1e15 else float(f"{v:.6g}")
    return v


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bars", type=int, default=BARS, help="velas por activo (70.000 ≈ 2 años)")
    ap.add_argument("--out", default=OUT, help="carpeta de salida")
    args = ap.parse_args()

    t0 = time.time()
    markets = [Market("binance", s, INTERVAL) for s in SYMBOLS]
    data = load_pooled(markets, horizon=4, bars=args.bars, candles=4)
    cols = feature_columns(data)
    print(f"{len(data)} filas, {len(cols)} variables ({time.time() - t0:.0f} s)")

    t1 = time.time()
    rep = evaluate_candles(data, WEB_MODELS)
    print(f"evaluación {time.time() - t1:.0f} s")
    print(rep.pivot("auc").round(4).to_string())

    generated = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    targets = []
    metrics = {}
    t2 = time.time()
    for name, ycol, _, ahead in targets_of(data):
        labeled = data[data[ycol].notna()]
        specs = {}
        for m in ("siempre_sube", *WEB_MODELS):
            model = make_model(m).fit(labeled[cols], labeled[ycol])
            specs[m] = export_model(model, cols)
        targets.append({"name": name, "candles_ahead": ahead, "models": specs})
        metrics[name] = {m: {k: _plain(rep.metrics.loc[(m, name), k]) for k in METRICS}
                         for m in rep.meta["modelos"]}
    print(f"modelos finales {time.time() - t2:.0f} s")

    common = {"generated": generated, "interval": INTERVAL, "symbols": SYMBOLS, "rows": len(data)}
    _write(os.path.join(args.out, "models.json"), {
        **common, "features": cols,
        "trained_from": str(data.index.min()), "trained_to": str(data[data["up"].notna()].index.max()),
        "targets": targets,
    })
    _write(os.path.join(args.out, "metrics.json"), {
        **common, "oos_from": rep.meta["desde"], "oos_to": rep.meta["hasta"], "folds": 6, "targets": metrics,
    })
    print("Guardado en", os.path.abspath(args.out), f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
