"""Interfaz web: qué vela se predice mejor, qué modelo gana en cada etapa y cuánto acierta cuando está seguro.

    .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import importlib.util
import warnings

import altair as alt
import pandas as pd
import streamlit as st

from updown import Market, load_pooled
from updown.candles import by_regime, evaluate_candles
from updown.confidence import confidence_curve, confidence_curve_live
from updown.hmm import STATE_LABELS, UNCERTAIN, current_stages
from updown.regimes import REGIME_LABELS, REGIMES
from updown.stages import evaluate_stages

warnings.filterwarnings("ignore")
st.set_page_config(page_title="updown", page_icon="📈", layout="wide")

HAS_TORCH = importlib.util.find_spec("torch") is not None
MODEL_CHOICES = ["ridge", "lasso", "random_forest", "gradient_boosting", "extra_trees"]
if HAS_TORCH:
    MODEL_CHOICES.append("transformer")
TARGET_NAMES = {"vela_1": "1.ª vela", "vela_2": "2.ª vela", "vela_3": "3.ª vela", "vela_4": "4.ª vela",
                "hora": "Hora completa"}
TARGET_COLUMNS = {**{f"vela_{k}": (f"up_v{k}", f"fwd_v{k}", k) for k in range(1, 5)}, "hora": ("up", "fwd_ret", 4)}
SHORT = {"shock": "Noticia", "crisis": "Crisis", "calma": "Calma", "alcista": "Alcista", "bajista": "Bajista",
         "lateral": "Lateral"}
REGIME_ICONS = {"shock": "⚡", "crisis": "🔥", "calma": "😴", "alcista": "📈", "bajista": "📉", "lateral": "↔️"}
STAGE_ICONS = {**REGIME_ICONS, "volatil": "🌪️", UNCERTAIN: "❔"}
STAGE_NAMES = {**REGIME_LABELS, **STATE_LABELS}
STAGE_ORDER = [*list(REGIMES), "volatil"]
SOURCE_NAMES = {"reglas": "Reglas fijas", "hmm": "HMM (con probabilidad)"}


@st.cache_data(show_spinner=False)
def load(source: str, symbols: tuple[str, ...], interval: str, bars: int, seq: int) -> pd.DataFrame:
    return load_pooled([Market(source, s, interval) for s in symbols], horizon=4, bars=bars,
                       seq_len=seq, candles=4, regimes=True)


@st.cache_data(show_spinner=False)
def run(source: str, symbols: tuple[str, ...], interval: str, bars: int, models: tuple[str, ...]):
    data = load(source, symbols, interval, bars, 64 if "transformer" in models else 0)
    rep = evaluate_candles(data, models)
    regimes = {t: by_regime(rep, t) for t in rep.targets}
    share = (data["regimen"].value_counts(normalize=True).reindex(REGIMES).fillna(0.0))
    now = data.groupby("symbol")["regimen"].last()
    last_time = data.groupby("symbol").apply(lambda g: g.index.max())
    tried = [m for m in rep.meta["modelos"] if m not in ("siempre_sube", "momentum")]
    conf = {t: {m: (confidence_curve(rep.oos[(m, t)]), confidence_curve_live(rep.oos[(m, t)])) for m in tried}
            for t in rep.targets}
    try:
        now_hmm = current_stages(data)
    except ValueError:  # muy poco histórico para ajustar el HMM
        now_hmm = None
    return {
        "auc": rep.pivot("auc"), "acierto": rep.pivot("acierto"), "ventaja": rep.pivot("ventaja"),
        "skill": rep.pivot("skill_vs_siempre_sube"), "por_regimen": regimes, "tiempo": share,
        "ahora": now, "ahora_hmm": now_hmm, "ultima_barra": last_time, "filas": len(data), "meta": rep.meta,
        "targets": rep.targets, "confianza": conf,
    }


@st.cache_data(show_spinner=False)
def run_stages(source: str, symbols: tuple[str, ...], interval: str, bars: int, models: tuple[str, ...],
               target: str):
    data = load(source, symbols, interval, bars, 64 if "transformer" in models else 0)
    ycol, rcol, purge = TARGET_COLUMNS[target]
    rep = evaluate_stages(data, models, target=ycol, ret_col=rcol, purge=purge)
    return {"metrics": rep.metrics, "comparison": rep.comparison, "choices": rep.choices,
            "oos_rank": rep.oos_rank, "agreement": rep.agreement, "share": rep.stage_share, "meta": rep.meta}


def stage_label(s: str) -> str:
    return f"{STAGE_ICONS.get(s, '')} {STAGE_NAMES.get(s, s)}".strip()


def verdict(row: pd.Series) -> str:
    if row["p_corregido"] != row["p_corregido"]:
        return "da exactamente lo mismo (todas las etapas se quedaron con el general)"
    if row["p_corregido"] < 0.05:
        return ("**mejora** (se equivoca menos" if row["mejora_brier"] > 0 else "**empeora** (se equivoca más") + \
            f", p corregido {row['p_corregido']:.3g})"
    return f"**no se distingue** (p corregido {row['p_corregido']:.2f})"


st.title("updown · ¿subirá la próxima vela?")
st.caption("Evaluación histórica fuera de muestra (walk-forward) con datos públicos. "
           "Herramienta de investigación, no asesoramiento financiero.")

with st.sidebar:
    st.header("Datos")
    source = st.radio("Fuente", ["binance", "yahoo"], horizontal=True)
    default = "BTCUSDT,ETHUSDT,SOLUSDT" if source == "binance" else "SPY,QQQ,IWM"
    symbols = tuple(s.strip() for s in st.text_input("Símbolos (separados por comas)", default).split(",") if s.strip())
    interval = st.selectbox("Vela", ["15m", "5m", "1h"], index=0)
    bars = st.slider("Barras por símbolo (Binance)", 5_000, 70_000, 20_000, step=5_000,
                     help="Yahoo solo da 60 días de velas de 15 min, sin elegir.")
    preferred = ("ridge", "random_forest", "transformer")
    models = tuple(st.multiselect("Modelos", MODEL_CHOICES, default=[m for m in preferred
                                                                    if m in MODEL_CHOICES]))
    stages_on = st.checkbox("Ranking por etapa (más lento)", value=True,
                            help="Entrena un submodelo por modelo y etapa en cada bloque. El transformer solo "
                                 "compite como modelo general.")
    go = st.button("Evaluar", type="primary", width="stretch")
    if not HAS_TORCH:
        st.info("El transformer necesita PyTorch (pip install torch).")

if go:
    st.session_state["params"] = (source, symbols, interval, bars, models)
    st.session_state["stages_on"] = stages_on
if "params" not in st.session_state:
    st.info("Elige mercados y modelos a la izquierda y pulsa **Evaluar**. "
            "La primera vez tarda unos minutos (descarga y entrena en cada bloque).")
    st.stop()
params = st.session_state["params"]

with st.spinner("Descargando datos y evaluando bloque a bloque…"):
    try:
        res = run(*params)
    except Exception as exc:  # noqa: BLE001  (cualquier fallo se enseña en la interfaz)
        st.error(f"No se pudo evaluar: {exc}")
        st.stop()

# --- etapa actual ---------------------------------------------------------------
st.subheader("¿En qué etapa está cada mercado ahora?")
cols = st.columns(max(len(res["ahora"]), 1))
for col, (sym, reg) in zip(cols, res["ahora"].items(), strict=False):
    col.metric(sym, f"{REGIME_ICONS.get(reg, '')} {SHORT.get(reg, 'sin datos')}",
               help=f"{REGIME_LABELS.get(reg, '')} (reglas). Última vela cerrada: "
                    f"{pd.Timestamp(res['ultima_barra'][sym]).tz_convert('UTC'):%Y-%m-%d %H:%M} UTC")
    hmm = res["ahora_hmm"]
    if hmm is not None and sym in hmm.index:
        probs = hmm.loc[sym].drop(["etapa", "vela"]).astype(float).sort_values(ascending=False)
        col.caption("HMM: " + " · ".join(f"{STAGE_NAMES.get(s, s)} {p:.0%}" for s, p in probs.items() if p >= 0.01))
if res["ahora_hmm"] is not None:
    st.caption("Las probabilidades del HMM describen el momento; no son una señal ni una frecuencia comprobada.")

with st.expander("¿Cómo se sabe en qué etapa estamos?"):
    st.markdown(
        "**Reglas fijas.** Cada vela recibe **una** etiqueta usando solo datos anteriores a ella, por este orden:\n\n"
        "1. ⚡ **Noticia / sobresalto**: en las últimas velas hubo un movimiento de más de 4 desviaciones típicas "
        "o un volumen 3 desviaciones por encima de lo normal. Las noticias no se leen: se ve su huella en el precio.\n"
        "2. 🔥 **Crisis**: volatilidad en el 20 % más alto de su último año y precio cayendo.\n"
        "3. 😴 **Calma, sin noticias**: volatilidad en el 30 % más bajo de su último año y sin tendencia clara.\n"
        "4. 📈 **Alcista**: por encima de su media de ~30 días y subiendo en 30 días.\n"
        "5. 📉 **Bajista**: cae, pero sin pánico.\n\n"
        "**HMM (con probabilidad).** Un modelo oculto de Markov aprende de los datos 4 estados, cada uno con su "
        "volatilidad y su tendencia típicas, y la probabilidad de pasar de uno a otro. Cada estado se nombra por "
        "esas características. Da probabilidades filtradas (solo con la información hasta esa vela) y la etapa "
        "solo cuenta si supera el 90 %; si no, la vela es «incierta» y la predice el modelo general.\n\n"
        "Límite honesto: la etapa se detecta cuando ya ha empezado, nunca antes.")
    st.bar_chart(res["tiempo"].rename(index=REGIME_LABELS).rename("parte del tiempo"))

pretty = {t: TARGET_NAMES.get(t, t) for t in res["targets"]}
tab_candles, tab_stages, tab_conf = st.tabs(["Por vela", "Ranking por etapa", "Confianza"])

# --- por vela y escenario (reglas) ----------------------------------------------------
with tab_candles:
    choice = st.segmented_control("Escenario", ["todos", *REGIMES], default="todos", key="escenario",
                                  format_func=lambda r: "Todos" if r == "todos"
                                  else f"{REGIME_ICONS[r]} {REGIME_LABELS[r]}")
    choice = choice or "todos"
    if choice == "todos":
        st.markdown(f"{res['filas']:,} filas · fuera de muestra {res['meta']['desde'][:10]} → "
                    f"{res['meta']['hasta'][:10]}. AUC: 0,5 es azar; cuanto más alto, mejor ordena el modelo qué "
                    "velas suben.")
        st.markdown("**AUC por vela** (¿qué vela de la próxima hora se predice mejor?)")
        auc = res["auc"].drop(index=["siempre_sube"], errors="ignore").rename(columns=pretty)
        auc_long = auc.rename_axis("modelo").reset_index().melt("modelo", var_name="objetivo", value_name="auc")
        chart = alt.Chart(auc_long).mark_line(point=True).encode(
            x=alt.X("objetivo:N", sort=list(auc.columns), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("auc:Q", scale=alt.Scale(zero=False), title="AUC fuera de muestra"),
            color=alt.Color("modelo:N", title="Modelo"),
            tooltip=["modelo", "objetivo", alt.Tooltip("auc:Q", format=".4f")])
        chance = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart(chart + chance, width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Acierto**")
        c1.dataframe(res["acierto"].rename(columns=pretty).style.format("{:.1%}"), width="stretch")
        c2.markdown("**¿Ventaja demostrada?** (corregido por número de modelos y estable en ≥ 2/3 de los bloques)")
        c2.dataframe(res["ventaja"].rename(columns=pretty).replace({True: "✅ sí", False: "—"}),
                     width="stretch")
    else:
        target = st.radio("¿Qué quieres predecir?", res["targets"], format_func=lambda t: pretty[t], horizontal=True,
                          key="objetivo_velas")
        table = res["por_regimen"][target]
        sub = table[table["regimen"] == choice] if len(table) else table
        if sub.empty:
            st.warning("Hay muy pocas velas en este escenario para medir algo fiable (mínimo 200).")
        else:
            share = float(res["tiempo"].get(choice, 0.0))
            st.markdown(f"**{REGIME_ICONS[choice]} {REGIME_LABELS[choice]}**: {share:.1%} del tiempo · "
                        f"{int(sub['n'].iloc[0]):,} velas evaluadas · "
                        f"sube el {sub['freq_sube'].iloc[0]:.1%} de las veces.")
            view = sub.set_index("modelo")[["acierto", "auc", "skill_vs_siempre_sube"]]
            view = view.sort_values("auc", ascending=False)
            st.dataframe(view.style.format({"acierto": "{:.1%}", "auc": "{:.3f}", "skill_vs_siempre_sube": "{:+.4f}"}),
                         width="stretch")
            st.caption(f"En este escenario ordena mejor **{view.index[0]}**. Por escenario no se corrige por número de "
                       "pruebas: tómalo como orientación, no como prueba.")

# --- ranking por etapa ---------------------------------------------------------------
def show_stages(params: tuple) -> None:
    c1, c2 = st.columns(2)
    target = c1.radio("¿Qué quieres predecir?", ["vela_1", "hora"], format_func=lambda t: TARGET_NAMES[t],
                      horizontal=True, key="objetivo_etapas")
    src = c2.radio("Etapas", list(SOURCE_NAMES), format_func=SOURCE_NAMES.get, horizontal=True, key="fuente")
    with st.spinner("Entrenando un submodelo por modelo y etapa en cada bloque…"):
        try:
            sr = run_stages(*params, target)
        except Exception as exc:  # noqa: BLE001  (cualquier fallo se enseña en la interfaz)
            st.error(f"No se pudo evaluar por etapas: {exc}")
            return
    m, comp = sr["metrics"], sr["comparison"]
    staged, best = f"por_etapas_{src}", sr["meta"]["mejor_unico"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Modelo por etapas · AUC", f"{m.loc[staged, 'auc']:.4f}",
              f"{m.loc[staged, 'auc'] - m.loc['general_validado', 'auc']:+.4f} frente al general")
    k2.metric("Mismo procedimiento sin etapas · AUC", f"{m.loc['general_validado', 'auc']:.4f}",
              help="Un solo modelo, elegido en la misma validación: aísla el efecto de especializarse.")
    k3.metric(f"Mejor modelo único ({best}) · AUC", f"{m.loc[best, 'auc']:.4f}",
              help="Elegido mirando la prueba, lo que le favorece.")
    rows = comp[comp["fuente"] == src].set_index("rival")
    st.markdown(f"Frente al mismo procedimiento sin etapas, especializarse {verdict(rows.loc['general_validado'])}. "
                f"Frente al mejor modelo único, {verdict(rows.loc['mejor_unico_a_posteriori'])}.")

    ch = sr["choices"][sr["choices"]["fuente"] == src]
    stages = sorted(ch["etapa"].unique(), key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
    stage = st.segmented_control("Etapa", stages, default=stages[0] if stages else None, format_func=stage_label,
                                 key=f"etapa_{src}")
    if stage:
        g = ch[ch["etapa"] == stage]
        picks = g["ganador"].value_counts()
        st.markdown("**Elegido en validación** (por bloque): " +
                    ", ".join(f"`{w}` {n}/{len(g)}" for w, n in picks.items()) +
                    (f" · {int((g['motivo'] != 'validación').sum())} bloque(s) con pocas velas se quedaron el general"
                     if (g["motivo"] != "validación").any() else ""))
        ag = sr["agreement"]
        ag = ag[(ag["fuente"] == src) & (ag["etapa"] == stage)]
        if not ag.empty:
            rho = ag["correlacion_rangos"].iloc[0]
            st.markdown(f"**¿Coinciden validación y prueba?** {ag['estabilidad'].iloc[0].capitalize()}"
                        + (f" (correlación de rangos {rho:.2f})" if pd.notna(rho) else "")
                        + ". Si no coinciden, la etapa es demasiado ruidosa para fiarse de la elección.")
        rank = sr["oos_rank"]
        rank = rank[(rank["fuente"] == src) & (rank["etapa"] == stage)].set_index("candidato")
        st.markdown("**Fuera de muestra** en las velas de esta etapa (todos los bloques):")
        st.dataframe(rank[["puesto", "n", "skill", "auc"]].style.format({"n": "{:,.0f}", "skill": "{:+.4f}",
                                                                        "auc": "{:.3f}"}),
                     width="stretch")
        st.caption("`modelo@etapa`: entrenado solo con velas de esa etapa · `modelo@general`: con todas. "
                   "`skill`: mejora del error frente a «siempre sube» en esas mismas velas.")
    share = sr["share"][staged]
    st.caption("Parte de las velas de prueba en cada etapa: " +
               " · ".join(f"{stage_label(s)} {v:.0%}" for s, v in share[share > 0].items()))


with tab_stages:
    if st.session_state.get("stages_on"):
        show_stages(params)
    else:
        st.info("Activa **Ranking por etapa** en la barra lateral y pulsa **Evaluar**.")

# --- confianza --------------------------------------------------------------------------
with tab_conf:
    c1, c2 = st.columns(2)
    target = c1.radio("¿Qué quieres predecir?", res["targets"], format_func=lambda t: pretty[t], horizontal=True,
                      key="objetivo_confianza")
    curves = res["confianza"][target]
    model = c2.selectbox("Detalle del modelo", list(curves), key="modelo_confianza")
    long = pd.concat([c.assign(modelo=name).reset_index() for name, (c, _) in curves.items()])
    long["cobertura_txt"] = long["cobertura"].map(lambda c: f"{c:.0%}")
    order = [f"{c:.0%}" for c in sorted(long["cobertura"].unique(), reverse=True)]
    x = alt.X("cobertura_txt:N", sort=order, title="Parte de las velas en las que se atreve a predecir",
              axis=alt.Axis(labelAngle=0))
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=x, y=alt.Y("acierto:Q", scale=alt.Scale(zero=False), title="Acierto", axis=alt.Axis(format="%")),
        color=alt.Color("modelo:N", title="Modelo"), tooltip=["modelo", "cobertura_txt", "n",
                                                              alt.Tooltip("acierto:Q", format=".1%")])
    band = alt.Chart(long[long["modelo"] == model]).mark_area(opacity=0.15).encode(x=x, y="ic95_bajo:Q",
                                                                                     y2="ic95_alto:Q")
    chance = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart(band + lines + chance, width="stretch")
    detail, live = curves[model]
    view = detail[["n", "confianza_media", "acierto", "ic95_bajo", "ic95_alto", "freq_sube"]].copy()
    view["acierto_corte_previo"] = live["acierto"]
    view["cobertura_corte_previo"] = live["cobertura_real"]
    view.index = [f"{c:.0%}" for c in view.index]
    st.dataframe(view.style.format({"n": "{:,.0f}", "confianza_media": "{:.1%}", "acierto": "{:.1%}",
                                    "ic95_bajo": "{:.1%}", "ic95_alto": "{:.1%}", "freq_sube": "{:.1%}",
                                    "acierto_corte_previo": "{:.1%}", "cobertura_corte_previo": "{:.0%}"}),
                 width="stretch")
    st.caption("Se ordenan las predicciones por lo lejos que están del 50 % y se mide el acierto de las más "
               "seguras. El corte mira solo la probabilidad, nunca si acertó. La banda es el IC 95 % del modelo "
               "elegido, agrupando por día. `corte_previo`: el umbral se fija con el bloque anterior, como habría "
               "que hacerlo en vivo; si su cobertura se aleja de la pedida, la confianza del modelo "
               "cambia con el tiempo.")

st.divider()
st.caption("Datos: Binance (API pública) y Yahoo Finance. Solo velas cerradas. "
           "Nada de esto garantiza resultados futuros.")
