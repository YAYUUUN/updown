"""Transformer vs. modelos clásicos: ¿subirá dentro de 1 hora (4 barras de 15 min)?

    .venv\\Scripts\\python -m estudios.transformer_15m

BTC, ETH y SOL (Binance) agrupados, ~2 años de barras de 15 min cada uno, ventana
de 64 barras (16 h) para el transformer. Evaluación histórica fuera de muestra,
sin costes. Escribe estudios/resultados_transformer_15m.md.
"""
from __future__ import annotations

import os
import time
import warnings

import pandas as pd

from estudios.comparativa import _md
from updown import Market, evaluate, load_pooled

warnings.filterwarnings("ignore")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BARS = 70_000          # ~2 años de 15 min
HORIZON = 4            # 4 x 15 min = 1 hora
SEQ_LEN = 64           # 16 horas de contexto para el transformer
MODELS = ("ridge", "lasso", "random_forest", "gradient_boosting", "transformer")
COLS = ["acierto", "freq_sube", "auc", "p_auc_corregido", "skill_vs_siempre_sube", "bloques_ganados", "bloques"]


def main() -> None:
    t0 = time.time()
    data = load_pooled([Market("binance", s, "15m") for s in SYMBOLS], horizon=HORIZON, bars=BARS, seq_len=SEQ_LEN)
    print(f"{len(data)} filas, {len(data.attrs['features'])} variables + ventana {SEQ_LEN}x"
          f"{len(data.attrs['seq_features']) // SEQ_LEN} ({time.time() - t0:.0f} s)")
    t1 = time.time()
    rep = evaluate(data, MODELS, n_folds=6, cost_bps=0.0)
    print(f"evaluación: {time.time() - t1:.0f} s")
    table = rep.table[COLS].sort_values("skill_vs_siempre_sube", ascending=False)
    print(table.round(4).to_string())
    print("\nPor bloque (Brier; menor es mejor):")
    print(rep.folds.round(5).to_string())
    print("\nVEREDICTO:", rep.verdict)
    out = ["# Transformer vs. clásicos: P(sube en 1 hora), barras de 15 min", "",
           f"Generado {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. {', '.join(SYMBOLS)} agrupados, {len(data)} filas; "
           f"fuera de muestra de {rep.meta['desde'][:10]} a {rep.meta['hasta'][:10]}. Horizonte {HORIZON} barras, "
           f"ventana {SEQ_LEN} barras, 6 bloques walk-forward, Holm sobre {len(MODELS)} modelos, p-valores con "
           f"n efectivo = n/{HORIZON} por el solape de objetivos.", "",
           _md(table), "", "Brier por bloque:", "", _md(rep.folds), "", f"**Veredicto:** {rep.verdict}", "",
           "*Herramienta de investigación, no asesoramiento financiero.*"]
    path = os.path.join(os.path.dirname(__file__), "resultados_transformer_15m.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("Guardado en", path)


if __name__ == "__main__":
    main()
