"""Comparativa de los 13 modelos en cuatro escenarios con datos reales.

    python -m estudios.comparativa            (desde la raíz del proyecto)

Escribe estudios/resultados_comparativa.md con una tabla por escenario. Todo es
evaluación histórica fuera de muestra (walk-forward); no genera señales actuales.
"""
from __future__ import annotations

import os
import time
import warnings

import pandas as pd

from updown import Market, evaluate, load_dataset, load_pooled

warnings.filterwarnings("ignore")

LARGE_CAPS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "KO", "PEP",
              "WMT", "HD", "UNH", "BAC", "CVX", "DIS", "INTC", "CSCO", "MRK", "PFE"]
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "GLD", "TLT"]

SCENARIOS = [
    ("SPY diario (desde 1993)", lambda: load_dataset(Market("yahoo", "SPY", "1d"))),
    ("20 grandes empresas de EE. UU., diario, agrupadas", lambda: load_pooled(
        [Market("yahoo", s, "1d") for s in LARGE_CAPS])),
    ("10 ETF, 1 hora, agrupados (730 días)", lambda: load_pooled([Market("yahoo", s, "1h") for s in ETFS])),
    ("BTCUSDT, 1 hora, Binance (20.000 barras)", lambda: load_dataset(Market("binance", "BTCUSDT", "1h"))),
]

COLS = ["acierto", "freq_sube", "auc", "p_auc_corregido", "skill_vs_siempre_sube", "bloques_ganados",
        "ret_anual_estrategia", "ret_anual_comprar_y_mantener"]


def _md(table: pd.DataFrame, index: str = "modelo") -> str:
    """Tabla markdown sin depender de `tabulate`."""
    head = f"| {index} | " + " | ".join(map(str, table.columns)) + " |"
    sep = "|---" * (len(table.columns) + 1) + "|"
    def cell(v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return ""
        return f"{int(v)}" if float(v).is_integer() and abs(v) < 1e12 else f"{v:.4g}"

    body = ["| " + str(i) + " | " + " | ".join(cell(v) for v in row) + " |" for i, row in table.iterrows()]
    return "\n".join([head, sep, *body])


def main() -> None:
    out = ["# Comparativa de modelos", "",
           f"Generado {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. Walk-forward de 6 bloques, coste 5 pb por "
           "operación, p-valores corregidos por Holm sobre 13 modelos.", ""]
    for title, loader in SCENARIOS:
        t0 = time.time()
        data = loader()
        rep = evaluate(data)
        took = time.time() - t0
        print(f"\n== {title}  ({len(data)} filas, {took:.0f} s)")
        table = rep.table[COLS].sort_values("skill_vs_siempre_sube", ascending=False)
        print(table.round(4).to_string())
        print("VEREDICTO:", rep.verdict)
        out += [f"## {title}", "",
                f"{len(data)} filas, {len(data.attrs['features'])} variables; fuera de muestra de "
                f"{rep.meta['desde'][:10]} a {rep.meta['hasta'][:10]}.", "",
                _md(table), "", f"**Veredicto:** {rep.verdict}", ""]
    out += ["*Herramienta de investigación, no asesoramiento financiero.*"]
    path = os.path.join(os.path.dirname(__file__), "resultados_comparativa.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\nGuardado en", path)


if __name__ == "__main__":
    main()
