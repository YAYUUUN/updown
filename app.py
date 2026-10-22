"""Interfaz web: qué vela se predice mejor y qué modelo funciona en cada escenario.

    .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import importlib.util
import warnings

import pandas as pd
import streamlit as st

from updown import Market, load_pooled
from updown.candles import by_regime, evaluate_candles
from updown.regimes import REGIME_LABELS, REGIMES

warnings.filterwarnings("ignore")
st.set_page_config(page_title="updown", page_icon="📈", layout="wide")

HAS_TORCH = importlib.util.find_spec("torch") is not None
MODEL_CHOICES = ["ridge", "lasso", "random_forest", "gradient_boosting", "extra_trees"] + (["transformer"] if HAS_TORCH else [])
TARGET_NAMES = {"vela_1": "1.ª vela", "vela_2": "2.ª vela", "vela_3": "3.ª vela", "vela_4": "4.ª vela",
                "hora": "Hora completa"}
SHORT = {"shock": "Noticia", "crisis": "Crisis", "calma": "Calma", "alcista": "Alcista", "bajista": "Bajista",
         "lateral": "Lateral"}
REGIME_ICONS = {"shock": "⚡", "crisis": "🔥", "calma": "😴", "alcista": "📈", "bajista": "📉", "lateral": "↔️"}


@st.cache_data(show_spinner=False)
def run(source: str, symbols: tuple[str, ...], interval: str, bars: int, models: tuple[str, ...]):
    seq = 64 if "transformer" in models else 0
    data = load_pooled([Market(source, s, interval) for s in symbols], horizon=4, bars=bars,
                       seq_len=seq, candles=4, regimes=True)
    rep = evaluate_candles(data, models)
    regimes = {t: by_regime(rep, t) for t in rep.targets}
    share = (data["regimen"].value_counts(normalize=True).reindex(REGIMES).fillna(0.0))
    now = data.groupby("symbol")["regimen"].last()
    last_time = data.groupby("symbol").apply(lambda g: g.index.max())
    return {
        "auc": rep.pivot("auc"), "acierto": rep.pivot("acierto"), "ventaja": rep.pivot("ventaja"),
        "skill": rep.pivot("skill_vs_siempre_sube"), "por_regimen": regimes, "tiempo": share,
        "ahora": now, "ultima_barra": last_time, "filas": len(data), "meta": rep.meta, "targets": rep.targets,
    }


st.title("updown · ¿subirá la próxima vela?")
st.caption("Evaluación histórica fuera de muestra (walk-forward) con datos públicos. "
           "Herramienta de investigación, no asesoramiento financiero.")

with st.sidebar:
    st.header("Datos")
    source = st.radio("Fuente", ["binance", "yahoo"], horizontal=True)
    default = "BTCUSDT,ETHUSDT,SOLUSDT" if source == "binance" else "SPY,QQQ,IWM"
    symbols = tuple(s.strip() for s in st.text_input("Símbolos (separados por comas)", default).split(",") if s.strip())
    interval = st.selectbox("Vela", ["15m", "5m", "1h"] if source == "binance" else ["15m", "5m", "1h"], index=0)
    bars = st.slider("Barras por símbolo (Binance)", 5_000, 70_000, 20_000, step=5_000,
                     help="Yahoo solo da 60 días de velas de 15 min, sin elegir.")
    models = tuple(st.multiselect("Modelos", MODEL_CHOICES, default=[m for m in ("ridge", "random_forest", "transformer")
                                                                    if m in MODEL_CHOICES]))
    go = st.button("Evaluar", type="primary", use_container_width=True)
    if not HAS_TORCH:
        st.info("El transformer necesita PyTorch (pip install torch).")

if go:
    st.session_state["params"] = (source, symbols, interval, bars, models)
if "params" not in st.session_state:
    st.info("Elige mercados y modelos a la izquierda y pulsa **Evaluar**. "
            "La primera vez tarda unos minutos (descarga y entrena en cada bloque).")
    st.stop()

with st.spinner("Descargando datos y evaluando bloque a bloque…"):
    try:
        res = run(*st.session_state["params"])
    except Exception as exc:
        st.error(f"No se pudo evaluar: {exc}")
        st.stop()

# --- régimen actual -------------------------------------------------------------
st.subheader("¿En qué escenario está cada mercado ahora?")
cols = st.columns(max(len(res["ahora"]), 1))
for col, (sym, reg) in zip(cols, res["ahora"].items()):
    col.metric(sym, f"{REGIME_ICONS.get(reg, '')} {SHORT.get(reg, 'sin datos')}",
               help=f"{REGIME_LABELS.get(reg, '')}. Última vela cerrada: {pd.Timestamp(res['ultima_barra'][sym]).tz_convert('UTC'):%Y-%m-%d %H:%M} UTC")

with st.expander("¿Cómo se sabe en qué escenario estamos?"):
    st.markdown(
        "Cada vela recibe **una** etiqueta usando solo datos anteriores a ella, por este orden de prioridad:\n\n"
        "1. ⚡ **Noticia / sobresalto**: en las últimas velas hubo un movimiento de más de 4 desviaciones típicas "
        "o un volumen 3 desviaciones por encima de lo normal. Las noticias no se leen: se ve su huella en el precio.\n"
        "2. 🔥 **Crisis**: volatilidad en el 20 % más alto de su último año y precio cayendo.\n"
        "3. 😴 **Calma, sin noticias**: volatilidad en el 30 % más bajo de su último año y sin tendencia clara.\n"
        "4. 📈 **Alcista**: por encima de su media de ~30 días y subiendo en 30 días.\n"
        "5. 📉 **Bajista**: cae, pero sin pánico.\n\n"
        "Límite honesto: el escenario se detecta cuando ya ha empezado, nunca antes.")
    st.bar_chart(res["tiempo"].rename(index=REGIME_LABELS).rename("parte del tiempo"))

# --- el botón de escenarios ---------------------------------------------------------
st.subheader("Resultados")
choice = st.segmented_control("Escenario", ["todos", *REGIMES], default="todos",
                              format_func=lambda r: "Todos" if r == "todos" else f"{REGIME_ICONS[r]} {REGIME_LABELS[r]}")
choice = choice or "todos"
pretty = {t: TARGET_NAMES.get(t, t) for t in res["targets"]}

if choice == "todos":
    st.markdown(f"{res['filas']:,} filas · fuera de muestra {res['meta']['desde'][:10]} → {res['meta']['hasta'][:10]}. "
                "AUC: 0,5 es azar; cuanto más alto, mejor ordena el modelo qué velas suben.")
    st.markdown("**AUC por vela** (¿qué vela de la próxima hora se predice mejor?)")
    auc = res["auc"].rename(columns=pretty)
    st.bar_chart(auc.drop(index=["siempre_sube"], errors="ignore").T, stack=False)
    c1, c2 = st.columns(2)
    c1.markdown("**Acierto**")
    c1.dataframe(res["acierto"].rename(columns=pretty).style.format("{:.1%}"), use_container_width=True)
    c2.markdown("**¿Ventaja demostrada?** (corregido por número de modelos y estable en ≥ 2/3 de los bloques)")
    c2.dataframe(res["ventaja"].rename(columns=pretty).replace({True: "✅ sí", False: "—"}), use_container_width=True)
else:
    target = st.radio("¿Qué quieres predecir?", res["targets"], format_func=lambda t: pretty[t], horizontal=True)
    table = res["por_regimen"][target]
    sub = table[table["regimen"] == choice] if len(table) else table
    if sub.empty:
        st.warning("Hay muy pocas velas en este escenario para medir algo fiable (mínimo 200).")
    else:
        share = float(res["tiempo"].get(choice, 0.0))
        st.markdown(f"**{REGIME_ICONS[choice]} {REGIME_LABELS[choice]}**: {share:.1%} del tiempo · "
                    f"{int(sub['n'].iloc[0]):,} velas evaluadas · sube el {sub['freq_sube'].iloc[0]:.1%} de las veces.")
        view = sub.set_index("modelo")[["acierto", "auc", "skill_vs_siempre_sube"]].sort_values("auc", ascending=False)
        st.dataframe(view.style.format({"acierto": "{:.1%}", "auc": "{:.3f}", "skill_vs_siempre_sube": "{:+.4f}"}),
                     use_container_width=True)
        best = view.index[0]
        st.caption(f"En este escenario ordena mejor **{best}**. Por escenario no se corrige por número de pruebas: "
                   "tómalo como orientación, no como prueba.")

st.divider()
st.caption("Datos: Binance (API pública) y Yahoo Finance. Solo velas cerradas. Nada de esto garantiza resultados futuros.")
