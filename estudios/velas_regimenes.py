"""Vela a vela y por régimen: BTC, ETH y SOL en barras de 15 minutos.

    .venv\\Scripts\\python -m estudios.velas_regimenes

¿Qué vela de la próxima hora se predice mejor (1.ª, 2.ª, 3.ª, 4.ª) y en qué
régimen de mercado funciona cada modelo? Evaluación histórica fuera de muestra,
sin costes. Escribe estudios/resultados_velas_regimenes.md.
"""
from __future__ import annotations

import os
import time
import warnings

import pandas as pd

from estudios.comparativa import _md
from updown import Market, load_pooled
from updown.candles import by_regime, evaluate_candles

warnings.filterwarnings("ignore")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BARS = 70_000
MODELS = ("ridge", "random_forest", "gradient_boosting", "transformer")


def main() -> None:
    t0 = time.time()
    data = load_pooled([Market("binance", s, "15m") for s in SYMBOLS], horizon=4, bars=BARS,
                       seq_len=64, candles=4, regimes=True)
    print(f"{len(data)} filas ({time.time() - t0:.0f} s)")
    t1 = time.time()
    rep = evaluate_candles(data, MODELS)
    print(f"evaluación {time.time() - t1:.0f} s")
    auc, acc, adv = rep.pivot("auc"), rep.pivot("acierto"), rep.pivot("ventaja")
    print("AUC\n", auc.round(4).to_string(), "\nAcierto\n", acc.round(4).to_string(), "\nVentaja\n", adv.to_string())
    share = data.groupby("regimen").size() / data["regimen"].notna().sum()
    out = ["# Vela a vela y por régimen (15 min, BTC/ETH/SOL)", "",
           f"Generado {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC · {len(data)} filas · fuera de muestra "
           f"{rep.meta['desde'][:10]} → {rep.meta['hasta'][:10]} · sin costes · Holm por objetivo.", "",
           "## AUC por vela", "", _md(auc), "", "## Acierto por vela", "", _md(acc), "",
           "## ¿Ventaja demostrada? (Holm < 0.05, mejora el Brier y gana en ≥ 2/3 de bloques)", "",
           _md(adv.astype(int)), "", "## Tiempo en cada régimen", "", _md(share.to_frame("tiempo")), ""]
    for target in ("vela_1", "hora"):
        br = by_regime(rep, target).set_index(["regimen", "modelo"])[["n", "freq_sube", "acierto", "auc",
                                                                       "skill_vs_siempre_sube"]]
        print(f"\nPor régimen ({target})\n", br.round(4).to_string())
        flat = br.copy()
        flat.index = [f"{r} · {m}" for r, m in flat.index]
        out += [f"## Por régimen: {target}", "", _md(flat), ""]
    out += ["*Herramienta de investigación, no asesoramiento financiero.*"]
    path = os.path.join(os.path.dirname(__file__), "resultados_velas_regimenes.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    data.attrs.clear()
    print("Guardado en", path)


if __name__ == "__main__":
    main()
