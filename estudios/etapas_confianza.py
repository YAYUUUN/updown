"""Acierto según confianza y modelos por etapa: BTC, ETH y SOL en velas de 15 minutos.

    .venv\\Scripts\\python -m estudios.etapas_confianza                  (sin transformer)
    .venv\\Scripts\\python -m estudios.etapas_confianza --transformer    (también como candidato general; GPU)

Tres preguntas, con evaluación histórica fuera de muestra y sin costes:

  1. ¿Acierta más el modelo cuando está más seguro? Acierto del 100/50/20/10/5 %
     de predicciones más alejadas del 50 %. El corte se decide solo con las
     probabilidades; otra versión lo fija con el bloque anterior, como en vivo.
  2. ¿Qué etapas encuentra el HMM, cuántas velas superan el 90 % de probabilidad
     y coinciden con las etapas por reglas?
  3. ¿Especializarse por etapa mejora o solo reparte los datos en trozos más
     pequeños? Modelo por etapas frente al mismo procedimiento sin etapas, al
     mejor modelo único y a las referencias, con los rankings de cada etapa.

Escribe estudios/resultados_etapas_confianza.md.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import pandas as pd

from estudios.comparativa import _md
from updown import Market, load_pooled
from updown.confidence import confidence_curve, confidence_curve_live
from updown.hmm import UNCERTAIN
from updown.regimes import REGIMES
from updown.stages import STAGE_MODELS, evaluate_stages

warnings.filterwarnings("ignore")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BARS = 70_000                                   # ~2 años de velas de 15 min por símbolo
TARGETS = {"vela_1": ("up_v1", "fwd_v1", 1), "hora": ("up", "fwd_ret", 4)}
HOURS_PER_BAR = 0.25
OUT = os.path.join(os.path.dirname(__file__), "resultados_etapas_confianza.md")


def _pct(c: float) -> str:
    return f"{c:.0%}"


def confidence_section(rep, target: str) -> list[str]:
    methods = [m for m in rep.metrics.index if m not in ("siempre_sube", "momentum")]
    curves = {m: confidence_curve(rep.oos[m]) for m in methods}
    acc = pd.DataFrame({m: c["acierto"] for m, c in curves.items()}).T
    acc.columns = [_pct(c) for c in acc.columns]
    n = next(iter(curves.values()))["n"]
    acc.loc["velas"] = n.to_numpy()
    best = rep.meta["mejor_unico"]
    detail = curves[best][["n", "confianza_media", "acierto", "ic95_bajo", "ic95_alto", "freq_sube"]].copy()
    live = confidence_curve_live(rep.oos[best])
    detail["acierto_corte_previo"] = live["acierto"]
    detail["cobertura_corte_previo"] = live["cobertura_real"]
    detail.index = [_pct(c) for c in detail.index]
    return [f"## Acierto según confianza: {target}", "",
            "Acierto de la fracción de predicciones más alejadas del 50 % (el corte mira solo la probabilidad, "
            "nunca el resultado). Columnas: cobertura.", "", _md(acc), "",
            f"Detalle de `{best}` (el mejor modelo único a posteriori). IC 95 % agrupado por día. "
            "`corte_previo`: el umbral de cada bloque se fija con las predicciones del bloque anterior, como en "
            "vivo; `cobertura_corte_previo` es la parte de las velas que realmente pasó ese corte.", "",
            _md(detail, "cobertura"), ""]


def hmm_section(rep) -> list[str]:
    states = rep.hmm_states.assign(duracion_horas=rep.hmm_states["duracion_velas"] * HOURS_PER_BAR)
    states.index = [f"{b} · {s}" for b, s in zip(states["bloque"], states["estado"], strict=False)]
    share = rep.stage_share["por_etapas_hmm"].rename("parte de las velas de prueba")
    share = share[share > 0].to_frame()
    cross = rep.hmm_vs_rules.reindex(columns=[r for r in REGIMES if r in rep.hmm_vs_rules.columns])
    return ["## Etapas del HMM", "",
            f"{rep.meta['estados_hmm']} estados ajustados en cada bloque solo con el entrenamiento; probabilidad "
            f"filtrada; la etapa cuenta si supera el {rep.meta['umbral_hmm']:.0%} y si no la vela es «{UNCERTAIN}». "
            "Volatilidad y tendencia medias de cada estado (en desviaciones típicas) y duración media.", "",
            _md(states[["volatilidad_z", "tendencia_z", "duracion_horas"]], "bloque · estado"), "",
            _md(share, "etapa"), "",
            "De las velas de cada etapa del HMM (filas), qué parte cae en cada régimen por reglas (columnas). "
            f"Coinciden en el {rep.meta['acuerdo_hmm_reglas']:.1%} de las velas con etapa segura y el mismo nombre.",
            "", _md(cross, "hmm \\ reglas"), ""]


def _verdict(row) -> str:
    if row["p_corregido"] != row["p_corregido"]:
        return "idéntico al rival"
    if row["p_corregido"] < 0.05:
        return "mejora" if row["mejora_brier"] > 0 else "empeora"
    return "no se distingue"


def stage_section(rep, target: str) -> list[str]:
    cols = ["n", "acierto", "auc", "p_auc_corregido", "skill_vs_siempre_sube", "bloques_ganados", "bloques",
            "ventaja"]
    comp = rep.comparison.assign(veredicto=rep.comparison.apply(_verdict, axis=1))
    comp.index = [f"{f} vs {r}" for f, r in zip(comp["fuente"], comp["rival"], strict=False)]
    lines = [f"## Modelo por etapas: {target}", "",
             "Fuera de muestra. `general_validado` es el mismo procedimiento sin etapas (un solo modelo elegido en "
             "validación); los p-valores del AUC se corrigen por Holm contando también los modelos por etapas.", "",
             _md(rep.metrics[cols].astype({"ventaja": int})), "",
             "¿Especializarse mejora? Test pareado de Brier (Diebold-Mariano, error agrupado por día, Holm). "
             "`mejora_brier` > 0: el modelo por etapas se equivoca menos.", "",
             _md(comp[["modelo_rival", "mejora_brier", "z", "p_valor", "p_corregido", "bloques_ganados", "bloques",
                       "veredicto"]], "comparación"), ""]
    if rep.agreement.empty:
        return lines
    agree = rep.agreement.copy()
    agree.index = [f"{f} · {s}" for f, s in zip(agree["fuente"], agree["etapa"], strict=False)]
    lines += ["Rankings por etapa: lo que eligió la validación frente a lo que pasó fuera de muestra. "
              "`correlacion_rangos`: correlación entre el orden de los candidatos en validación y en prueba "
              "(estable ≥ 0,5; ruidosa ≤ 0).", "",
              _md(agree[["bloques_con_eleccion", "bloques", "mas_elegido", "veces_elegido", "puesto_fuera_de_muestra",
                         "mejor_fuera_de_muestra", "correlacion_rangos", "estabilidad"]], "fuente · etapa"), ""]
    top = rep.oos_rank[rep.oos_rank["puesto"] <= 5].copy()
    top.index = [f"{f} · {s} · {c}" for f, s, c in zip(top["fuente"], top["etapa"], top["candidato"], strict=False)]
    lines += ["<details><summary>Los 5 mejores candidatos de cada etapa fuera de muestra</summary>", "",
              _md(top[["puesto", "n", "skill", "auc"]], "fuente · etapa · candidato"), "", "</details>", ""]
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transformer", action="store_true", help="el transformer compite como candidato general")
    ap.add_argument("--barras", type=int, default=BARS, help="velas por símbolo")
    ap.add_argument("--objetivos", default=",".join(TARGETS), help="vela_1, hora o ambos")
    ap.add_argument("--salida", default=OUT)
    args = ap.parse_args(argv)
    models = STAGE_MODELS + (("transformer",) if args.transformer else ())
    t0 = time.time()
    data = load_pooled([Market("binance", s, "15m") for s in SYMBOLS], horizon=4, bars=args.barras,
                       seq_len=64 if args.transformer else 0, candles=4, regimes=True)
    print(f"{len(data)} filas, {len(data.attrs['features'])} variables ({time.time() - t0:.0f} s)")
    body, meta, took = [], None, {}
    for i, name in enumerate(t.strip() for t in args.objetivos.split(",")):
        ycol, rcol, purge = TARGETS[name]
        t1 = time.time()
        print(f"\n== {name}")
        rep = evaluate_stages(data, models, target=ycol, ret_col=rcol, purge=purge, verbose=True)
        took[name] = time.time() - t1
        print(rep.metrics[["acierto", "auc", "p_auc_corregido", "skill_vs_siempre_sube", "bloques_ganados"]]
              .round(4).to_string())
        print(rep.comparison[["fuente", "rival", "mejora_brier", "p_corregido"]].round(5).to_string())
        body += confidence_section(rep, name)
        if i == 0:
            meta = rep.meta
            body += hmm_section(rep)
        body += stage_section(rep, name)
    head = ["# Acierto según confianza y modelos por etapa (15 min, BTC/ETH/SOL)", "",
            f"Generado {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC · {len(data)} filas · fuera de muestra "
            f"{meta['desde'][:10]} → {meta['hasta'][:10]} · {meta['bloques']} bloques walk-forward · sin costes. "
            f"Candidatos: {', '.join(models)}"
            + (f" ({', '.join(meta['solo_general'])} solo como general)" if meta["solo_general"] else "")
            + f". Una etapa elige submodelo si tiene al menos {meta['min_entrenamiento']} velas de entrenamiento y "
            f"{meta['min_validacion']} de validación. Tiempo: "
            + ", ".join(f"{k} {v / 60:.0f} min" for k, v in took.items()) + ".", ""]
    with open(args.salida, "w", encoding="utf-8") as f:
        f.write("\n".join(head + body + ["*Herramienta de investigación, no asesoramiento financiero.*"]) + "\n")
    print("\nGuardado en", args.salida)


if __name__ == "__main__":
    main()
