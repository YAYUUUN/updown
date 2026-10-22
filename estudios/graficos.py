"""Gráficos del README a partir de estudios/resultados_velas_regimenes.md y, si existe,
estudios/resultados_etapas_confianza.md.

    python -m estudios.graficos
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")
COLORS = {"random_forest": "#1f6f8b", "gradient_boosting": "#2a9d8f", "transformer": "#e76f51",
          "ridge": "#8d99ae", "momentum": "#b5b5b5", "siempre_sube": "#d9d9d9", "lasso": "#6d597a",
          "knn": "#c9a227"}


def read_table(md: str, title: str) -> pd.DataFrame:
    """La primera tabla markdown de la sección `## title`."""
    block = md.split(f"## {title}", 1)[1].split("\n## ", 1)[0]
    first = block[block.index("\n|"):].split("\n\n", 1)[0]
    lines = [ln for ln in first.splitlines() if ln.startswith("|") and not ln.startswith("|---")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    df = pd.DataFrame(rows[1:], columns=rows[0]).set_index(rows[0][0])
    return df.apply(pd.to_numeric, errors="coerce")


def main() -> None:
    with open(os.path.join(HERE, "resultados_velas_regimenes.md"), encoding="utf-8") as fh:
        md = fh.read()
    os.makedirs(DOCS, exist_ok=True)

    auc = read_table(md, "AUC por vela").drop(index="siempre_sube")
    labels = {"vela_1": "1st candle", "vela_2": "2nd", "vela_3": "3rd", "vela_4": "4th", "hora": "Full hour"}
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = range(len(auc.columns))
    for name, row in auc.sort_values("vela_1").iterrows():
        ax.plot(list(x), row.values, marker="o", lw=2 if name in ("random_forest", "transformer") else 1.2,
                color=COLORS.get(name, "#333"), label=name)
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.text(len(auc.columns) - 1, 0.5015, "chance", ha="right", color="#777", fontsize=8)
    ax.set_xticks(list(x), [labels.get(c, c) for c in auc.columns])
    ax.set_ylabel("Out-of-sample AUC")
    ax.set_title("Will the next 15-min candle close higher? (BTC, ETH, SOL; ~2 years)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "auc_per_candle.png"))

    vela1 = md.split("## Por régimen: vela_1", 1)[1].split("\n## ", 1)[0]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in vela1.splitlines()
            if ln.startswith("|") and not ln.startswith("|---")]
    t = pd.DataFrame(rows[1:], columns=rows[0])
    t[["regimen", "modelo_"]] = t[t.columns[0]].str.split(" · ", expand=True)
    t["auc"] = pd.to_numeric(t["auc"])
    piv = t.pivot(index="regimen", columns="modelo_", values="auc")
    order = [r for r in ("shock", "crisis", "calma", "alcista", "bajista", "lateral") if r in piv.index]
    names = {"shock": "News / shock", "crisis": "Crisis", "calma": "Calm", "alcista": "Bull", "bajista": "Bear",
             "lateral": "Sideways"}
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    models = [m for m in ("momentum", "ridge", "transformer", "gradient_boosting", "random_forest") if m in piv]
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar([j + i * w for j in range(len(piv))], piv[m] - 0.5, w, bottom=0.5, color=COLORS[m], label=m)
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(piv))], [names[r] for r in piv.index])
    ax.set_ylim(0.495, max(0.56, piv.max().max() + 0.005))
    ax.set_ylabel("AUC, 1st candle")
    ax.set_title("Same models, different market regimes (regime detected from past data only)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "auc_by_regime.png"))
    print("docs/auc_per_candle.png, docs/auc_by_regime.png")
    confidence_chart()


def confidence_chart(path: str = os.path.join(HERE, "resultados_etapas_confianza.md")) -> None:
    """Acierto según cobertura (las predicciones más seguras primero), vela 1."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        acc = read_table(fh.read(), "Acierto según confianza: vela_1").drop(index="velas")
    acc = acc.drop(index=[m for m in acc.index if m.startswith(("por_etapas", "general_validado"))])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = range(len(acc.columns))
    for name, row in acc.iterrows():
        ax.plot(list(x), row.values, marker="o", lw=2 if name in ("random_forest", "transformer") else 1.2,
                color=COLORS.get(name, "#333"), label=name)
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.set_xticks(list(x), [f"{c}" for c in acc.columns])
    ax.set_xlabel("Share of candles the model dares to predict (most confident first)")
    ax.set_ylabel("Out-of-sample hit rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Does the model hit more when it is more confident? (next 15-min candle)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "confidence_curve.png"))
    print("docs/confidence_curve.png")


if __name__ == "__main__":
    main()
