"""Línea de comandos.

    python -m updown auditoria
    python -m updown evaluar AAPL --intervalo 1d
    python -m updown evaluar SPY,QQQ,IWM,DIA --intervalo 1h --agrupado
    python -m updown predecir BTC-USD --intervalo 1h
    python -m updown vivo BTCUSDT --fuente binance --intervalo 1m
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import audit
from .confidence import confidence_curve, confidence_curve_live
from .data import INTERVAL_SECONDS, Market, drop_unclosed, fetch
from .evaluate import evaluate
from .models import MODELS
from .predict import (live_scoreboard, load_dataset, load_pooled, log_path, log_prediction, predict_now,
                      target_label, update_outcomes)

DISCLAIMER = "Herramienta de investigación, no asesoramiento financiero."


def _markets(args) -> list[Market]:
    return [Market(args.fuente, s.strip(), args.intervalo) for s in args.simbolos.split(",") if s.strip()]


def _context(args):
    if args.sin_contexto:
        return []
    if args.contexto:
        return [Market(args.fuente, s.strip(), args.intervalo) for s in args.contexto.split(",")]
    return None  # el de por defecto según el mercado


def _models(args):
    return tuple(m.strip() for m in args.modelos.split(",")) if args.modelos else MODELS


def cmd_evaluar(args) -> int:
    markets = _markets(args)
    if args.agrupado or len(markets) > 1:
        data = load_pooled(markets, args.horizonte, _context(args))
    else:
        data = load_dataset(markets[0], args.horizonte, _context(args))
    rep = evaluate(data, _models(args), n_folds=args.bloques, cost_bps=args.coste_bps)
    pd.set_option("display.width", 160)
    cols = ["n", "acierto", "freq_sube", "auc", "p_auc_corregido", "skill_vs_siempre_sube",
            "bloques_ganados", "bloques", "ret_anual_estrategia", "ret_anual_comprar_y_mantener"]
    print(f"\n{', '.join(m.symbol for m in markets)} · {args.intervalo} · horizonte {args.horizonte} barra(s) · "
          f"{len(data)} filas desde {rep.meta['datos_desde'][:10]} · {len(data.attrs['features'])} variables")
    print(f"Evaluado fuera de muestra de {rep.meta['desde'][:16]} a {rep.meta['hasta'][:16]} "
          f"(antes de eso solo se entrena). Coste {args.coste_bps:g} pb por operación.")
    print(rep.table[cols].round(4).to_string())
    print("\nCalibración del mejor modelo (¿cuando dice 55%, sube el 55% de las veces?):")
    print(rep.calibration.round(3).to_string())
    if rep.best:
        cc = confidence_curve(rep.oos[rep.best])
        live = confidence_curve_live(rep.oos[rep.best])
        cc["acierto_corte_previo"], cc["cobertura_corte_previo"] = live["acierto"], live["cobertura_real"]
        print(f"\nAcierto según confianza de '{rep.best}' (el corte mira solo la probabilidad, nunca el resultado;"
              " 'corte_previo' lo fija con el bloque anterior, como en vivo):")
        print(cc[["n", "confianza_media", "acierto", "ic95_bajo", "ic95_alto", "acierto_corte_previo",
                  "cobertura_corte_previo"]].round(4).to_string())
    print("\nVEREDICTO:", rep.verdict)
    print(DISCLAIMER)
    return 0


def _print_prediction(pred) -> None:
    print(f"\n{pred.symbol} · {pred.interval} · última barra cerrada {pred.bar_time} · cierre {pred.close:g}")
    print(f"Probabilidad de cerrar por encima de {pred.close:g} {target_label(pred)}:")
    for name, p in pred.probs.items():
        print(f"   {name:13s} {p:6.1%}")


def cmd_predecir(args) -> int:
    pred = predict_now(_markets(args)[0], args.horizonte, _models(args), _context(args))
    _print_prediction(pred)
    print("\nUna probabilidad no vale nada sin su historial: mira `evaluar` y el registro de `vivo`.")
    print(DISCLAIMER)
    return 0


def cmd_vivo(args) -> int:
    market = _markets(args)[0]
    path = log_path(args.registro, market, args.horizonte)
    last_bar = None
    print(f"Modo vivo: {market.symbol} {market.interval}. Registro en {path}. Ctrl+C para salir.")
    while True:
        try:
            bars = drop_unclosed(fetch(market), market.interval)
            if bars.index[-1] != last_bar:
                last_bar = bars.index[-1]
                pred = predict_now(market, args.horizonte, _models(args), _context(args))
                if log_prediction(path, pred):
                    _print_prediction(pred)
                log = update_outcomes(path, bars, args.horizonte)
                board = live_scoreboard(log) if len(log) else pd.DataFrame()
                if len(board) and board["n"].iloc[0] > 0:
                    print("Historial en vivo (apuntado antes de saber el resultado):")
                    print(board.round(3).to_string())
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # caídas de red: se reintenta en la siguiente vuelta
            print(f"[aviso] {type(exc).__name__}: {exc}")
        try:
            time.sleep(max(15, min(INTERVAL_SECONDS[market.interval] // 4, 300)))
        except KeyboardInterrupt:
            return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no es UTF-8 por defecto
    ap = argparse.ArgumentParser(prog="updown", description="¿Subirá? Probabilidades con su historial honesto.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auditoria", help="comprueba el evaluador con datos sintéticos")
    for name in ("evaluar", "predecir", "vivo"):
        p = sub.add_parser(name)
        p.add_argument("simbolos", help="p. ej. AAPL, BTC-USD, ^GSPC; varios separados por comas para evaluar")
        p.add_argument("--fuente", default="yahoo", choices=["yahoo", "binance"])
        p.add_argument("--intervalo", default="1d")
        p.add_argument("--horizonte", type=int, default=1, help="barras hacia delante (1 = la siguiente)")
        p.add_argument("--modelos", default="", help=f"de entre {','.join(MODELS)} (por defecto, los tres)")
        p.add_argument("--contexto", default="", help="mercados de referencia, p. ej. SPY,^VIX")
        p.add_argument("--sin-contexto", action="store_true")
        if name == "evaluar":
            p.add_argument("--agrupado", action="store_true", help="un solo modelo entrenado con todos los símbolos")
            p.add_argument("--bloques", type=int, default=6)
            p.add_argument("--coste-bps", type=float, default=5.0, help="coste por operación en puntos básicos")
        if name == "vivo":
            p.add_argument("--registro", default="registro")
    args = ap.parse_args(argv)
    if args.cmd == "auditoria":
        return 0 if all(audit.run().values()) else 1
    return {"evaluar": cmd_evaluar, "predecir": cmd_predecir, "vivo": cmd_vivo}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
