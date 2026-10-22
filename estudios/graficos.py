"""Gráficos del README a partir de estudios/resultados_velas_regimenes.md.

    python -m estudios.graficos
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")
COLORS = {"random_forest": "#1f6f8b", "gradient_boosting": "#2a9d8f", "transformer": "#e76f51",
          "ridge": "#8d99ae", "momentum": "#b5b5b5", "siempre_sube": "#d9d9d9"}


def read_table(md: str, title: str) -> pd.DataFrame:
    block = md.split(f"## {title}", 1)[1].split("\n## ", 1)[0]
    lines = [l for l in block.splitlines() if l.startswith("|") and not l.startswith("|---")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    df = pd.DataFrame(rows[1:], columns=rows[0]).set_index(rows[0][0])
    return df.apply(pd.to_numeric, errors="coerce")


def main() -> None:
    md = open(os.path.join(HERE, "resultados_velas_regimenes.md"), encoding="utf-8").read()
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
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in vela1.splitlines()
            if l.startswith("|") and not l.startswith("|---")]
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


if __name__ == "__main__":
    main()
