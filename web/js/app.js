// La página: resultados de los estudios (data/results.json), historial fuera de muestra de
// los modelos de la web (data/metrics.json) y P(sube) en directo con velas de Binance.
import { buildFeatures, dropUnclosed, featureRow, INTERVAL_MS, parseKlines } from "./features.js";
import { dataTo, history, hitSummary, modelSets } from "./history.js";
import { predict } from "./models.js";
import { filterStages, hmmObservations } from "./stages.js";

const API = ["https://data-api.binance.vision", "https://api.binance.com"];
const LIVE_MODELS = ["gradient_boosting", "ridge"];
const MODEL_NAMES = {
  gradient_boosting: "Gradient boosting", random_forest: "Random forest", ridge: "Ridge",
  transformer: "Transformer", momentum: "Momentum (one-line rule)", siempre_sube: "Always up",
  lineal: "Logistic regression", lasso: "Lasso", elastic_net: "Elastic net", lda: "LDA", svm_lineal: "Linear SVM",
  naive_bayes: "Naive Bayes", knn: "k-NN", extra_trees: "Extra trees", red_neuronal: "Neural network",
  ensemble: "Ensemble",
};
const CHART_MODELS = ["random_forest", "gradient_boosting", "transformer", "ridge", "momentum", "siempre_sube"];
const TARGETS = {
  vela_1: { button: "Next candle", ahead: 1, question: "Will the candle now forming close above the last close?" },
  vela_2: { button: "2nd", ahead: 2, question: "Will the 2nd candle from now close above the 1st?" },
  vela_3: { button: "3rd", ahead: 3, question: "Will the 3rd candle from now close above the 2nd?" },
  vela_4: { button: "4th", ahead: 4, question: "Will the 4th candle from now close above the 3rd?" },
  hora: { button: "Next hour", ahead: 4, question: "Will the price four candles from now be above the last close?" },
};
const REGIMES = {
  shock: "News / shock", crisis: "Crisis", calma: "Calm", alcista: "Bull", bajista: "Bear", lateral: "Sideways",
};
const MARKETS = [
  [/^SPY diario/, "SPY, daily since 1993"],
  [/^20 grandes/, "20 large US stocks, daily"],
  [/^10 ETF/, "10 ETFs, hourly (2 years)"],
  [/^BTCUSDT, 1 hora/, "BTC/USDT, hourly (Binance)"],
];

const WINDOWS = [[48, "12 h"], [96, "24 h"], [288, "3 days"]];
// La tendencia de la etapa mira 30 días atrás: hacen falta ~5.000 velas, no las 1.000 de una petición.
const HISTORY_BARS = 4800;
const STAGES = {
  calma: "Calm", alcista: "Bull", bajista: "Bear", lateral: "Sideways", crisis: "Crisis", volatil: "Volatile",
  uncertain: "Uncertain",
};
const stageKey = (s) => (s === "uncertain" ? "incierta" : s);  // así se llama en metrics.json

const state = {
  target: "vela_1", models: null, metrics: null, live: null, error: null, offset: 0, next: 0, api: 0,
  recent: { symbol: "BTCUSDT", model: "gradient_boosting", window: matchMedia("(max-width: 560px)").matches ? 48 : 96 },
};
const $ = (sel) => document.querySelector(sel);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const int = (x) => Math.round(x).toLocaleString("en-US");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const day = (s) => new Date(s).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
const symLabel = (s) => s.replace(/USDT$/, "/USDT");

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function binance(path) {
  let last;
  for (let k = 0; k < API.length; k++) {
    const i = (state.api + k) % API.length;
    try {
      const r = await fetch(API[i] + path);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      state.api = i;
      return await r.json();
    } catch (e) {
      last = e;
    }
  }
  throw last;
}

// ---------- historial y controles ----------

function renderTargets() {
  const box = $("#targets");
  box.innerHTML = "";
  for (const [name, t] of Object.entries(TARGETS)) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = t.button;
    b.setAttribute("aria-pressed", String(name === state.target));
    b.addEventListener("click", () => {
      state.target = name;
      box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderCards();
      renderRecord();
      renderRecent();
    });
    box.append(b);
  }
}

function renderRecord() {
  const m = state.metrics?.targets?.[state.target];
  const body = $("#record tbody");
  if (!m) {
    body.innerHTML = `<tr><td colspan="5">Track record unavailable.</td></tr>`;
    return;
  }
  $("#record-title").textContent = `Track record: ${TARGETS[state.target].question}`;
  const n = m.gradient_boosting?.n ?? m.siempre_sube.n;
  $("#record-note").textContent =
    `Out of sample: ${int(n)} predictions from ${day(state.metrics.oos_from)} to ${day(state.metrics.oos_to)}, ` +
    `BTC, ETH and SOL together, ${state.metrics.folds} walk-forward periods. Each model was trained only on data before ` +
    `the period it predicts.`;
  const rows = [...LIVE_MODELS, "momentum", "siempre_sube"].filter((k) => m[k]);
  body.innerHTML = rows.map((k) => {
    const r = m[k];
    const ref = !LIVE_MODELS.includes(k);
    const skill = k === "siempre_sube" ? "reference" : `${r.skill_vs_siempre_sube >= 0 ? "+" : ""}${pct(r.skill_vs_siempre_sube, 2)}`;
    const won = k === "siempre_sube" ? "–" : `${r.bloques_ganados}/${r.bloques}`;
    const tag = r.ventaja ? ` <span class="tag yes" title="Holm-corrected p &lt; 0.05, better Brier score and wins in at least 2/3 of the periods">edge</span>` : "";
    return `<tr class="${ref ? "ref" : ""}"><td><span class="model-name"><i class="dot" style="background:var(--m-${k})"></i>${esc(MODEL_NAMES[k])}</span>${tag}</td>` +
      `<td>${pct(r.acierto)}</td><td>${r.auc.toFixed(3)}</td><td>${skill}</td><td>${won}</td></tr>`;
  }).join("");
}

// ---------- en directo ----------

function meter(p, color) {
  const x = (v) => ((Math.min(Math.max(v, 0.4), 0.6) - 0.4) / 0.2) * 100;
  const a = x(Math.min(p, 0.5));
  const b = x(Math.max(p, 0.5));
  return `<div class="meter" role="img" aria-label="${pct(p)} probability of closing higher">` +
    `<i style="left:${a}%;width:${Math.max(b - a, 0.6)}%;background:${color}"></i></div>`;
}

function renderCards() {
  const box = $("#cards");
  const symbols = state.models?.symbols ?? ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
  const t = TARGETS[state.target];
  box.innerHTML = symbols.map((sym) => {
    const live = state.live?.[sym];
    const head = `<div class="card-head"><span class="sym">${symLabel(sym)}</span>` +
      `<span class="price">${live ? live.close.toLocaleString("en-US", { maximumFractionDigits: 2 }) : ""}</span></div>`;
    if (!live?.probs) {
      const msg = state.error ? "No live data." : live ? "Not enough recent candles." : "Loading…";
      return `<article class="card empty">${head}<p>${msg}</p></article>`;
    }
    const probs = live.probs[state.target];
    const rows = LIVE_MODELS.map((k) => `<div class="model-row"><div class="model-line">` +
      `<span class="model-name"><i class="dot" style="background:var(--m-${k})"></i>${MODEL_NAMES[k]}</span>` +
      `<span class="prob">${pct(probs[k])}<small>up</small></span></div>${meter(probs[k], `var(--m-${k})`)}</div>`).join("");
    const resolves = live.time + INTERVAL_MS * (t.ahead + 1);
    return `<article class="card">${head}${stageLine(live)}${rows}` +
      `<div class="meter-scale" aria-hidden="true"><span>40%</span><span>50%</span><span>60%</span></div>` +
      `<p class="baseline">“Always up” says ${pct(probs.siempre_sube)} · resolves at ${clock(resolves)}</p>` +
      `${stageRecord(live)}</article>`;
  }).join("");
}

/** La etapa de ahora con lo segura que está; si ninguna llega al umbral, «uncertain» y las dos más probables. */
function stageLine(live) {
  const st = stageAt(live, live.bars.time.length - 1);
  if (!st) {
    const why = live.stages?.length ? "not enough recent candles." : "shown after the next weekly model update.";
    return `<p class="stage muted">Market stage: ${why}</p>`;
  }
  const [first, second] = st.probs;
  const shown = st.probs.filter(([, p]) => p >= 0.005);   // el desglose solo si hay más de una etapa con peso
  const others = shown.length > 1 ? shown.map(([s, p]) => `${STAGES[s] ?? s} ${pct(p, 0)}`).join(" · ") : "";
  const main = st.label === "uncertain"
    ? `<b>Uncertain</b> <span class="muted">between ${STAGES[first[0]]} and ${STAGES[second?.[0]] ?? "–"}</span>`
    : `<b>${STAGES[st.label] ?? st.label}</b> <span class="muted">· ${pct(first[1], 0)} sure</span>`;
  return `<div class="stage"><p><span class="stage-k">Market stage</span> ${main}</p>` +
    (others ? `<p class="stage-probs">${others}</p>` : "") + "</div>";
}

/** Cuánto acertó cada modelo, fuera de muestra, en las velas de esta misma etapa. */
function stageRecord(live) {
  const st = stageAt(live, live.bars.time.length - 1);
  const rec = state.metrics?.by_stage?.targets?.[state.target];
  if (!st || !rec) return "";
  const key = stageKey(st.label);
  const parts = LIVE_MODELS.filter((m) => rec[m]?.[key]).map((m) => `${MODEL_NAMES[m].toLowerCase()} ${pct(rec[m][key].acierto)}`);
  const n = rec[LIVE_MODELS[0]]?.[key]?.n;
  if (!parts.length || !n) return "";
  const where = st.label === "uncertain" ? "when the stage was uncertain" : `in ${STAGES[st.label].toLowerCase()} markets`;
  return `<p class="baseline">Out of sample ${where}: ${parts.join(", ")} right (${int(n)} candles)</p>`;
}

function renderStatus() {
  const el = $("#status");
  if (state.error) {
    el.className = "status warn";
    el.textContent = "Live data unavailable: Binance did not answer from your network (it is blocked in some " +
      "countries). The track record and results are not affected.";
    return;
  }
  if (!state.live) return;
  const times = Object.values(state.live).map((l) => l.time);
  const last = Math.max(...times);
  const now = Date.now() + state.offset;
  const left = Math.max(0, Math.round((state.next - now) / 1000));
  const mmss = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
  const stale = now - (last + INTERVAL_MS) > 2 * INTERVAL_MS;
  el.className = stale ? "status warn" : "status";
  el.textContent = `Based on the candle that closed at ${clock(last + INTERVAL_MS)} (your time).` +
    (stale ? " Binance's latest candle is older than expected." : ` Next update in ${mmss}.`);
}

/** Las últimas `total` velas: la primera vez paginando hacia atrás, luego solo lo nuevo. */
async function klines(sym, interval, total) {
  const base = `/api/v3/klines?symbol=${sym}&interval=${interval}`;
  let rows = state.rows?.[sym];
  if (rows?.length) {
    rows = rows.concat(await binance(`${base}&limit=1000`));   // parseKlines se queda con la última versión de cada vela
  } else {
    rows = [];
    let end = null;
    while (rows.length < total) {
      const page = await binance(`${base}&limit=1000${end ? `&endTime=${end}` : ""}`);
      if (!page.length) break;
      rows = page.concat(rows);
      end = page[0][0] - 1;
      if (page.length < 1000) break;
    }
  }
  rows = rows.slice(-total - 1000);
  (state.rows ??= {})[sym] = rows;
  return rows;
}

/** Etapa de cada vela con cada juego de modelos que traiga HMM (el actual y el de la semana anterior). */
function stagesFor(sym, bars) {
  return modelSets(state.models).filter((s) => s.stages?.vol_ref?.[sym]).map((s) => ({
    source: s.source, from: s.from,
    ...filterStages(s.stages.hmm, hmmObservations(bars, s.stages.vol_ref[sym], s.stages.day, s.stages.month)),
  }));
}

/** La etapa de la vela k según el HMM publicado entonces (el más antiguo para velas anteriores). */
function stageAt(live, k) {
  const sets = live.stages;
  if (!sets?.length) return null;
  const set = sets.find((s) => live.bars.time[k] >= s.from) ?? sets[sets.length - 1];
  const label = set.label[k];
  if (!label) return null;
  const probs = set.stages.map((s) => [s, set.probs[s][k]]).sort((a, b) => b[1] - a[1]);
  return { label, probs, source: set.source };
}

async function refresh() {
  clearTimeout(state.timer);
  const models = state.models;
  try {
    const { serverTime } = await binance("/api/v3/time");
    state.offset = serverTime - Date.now();
    const live = {};
    await Promise.all(models.symbols.map(async (sym) => {
      const rows = await klines(sym, models.interval, HISTORY_BARS);
      const bars = dropUnclosed(parseKlines(rows), serverTime);
      const i = bars.time.length - 1;
      if (i < 0) return;
      const features = buildFeatures(bars);
      const row = featureRow(features, models.features, i);
      let probs = null;
      if (row) {
        probs = {};
        for (const target of models.targets) {
          probs[target.name] = {};
          for (const [name, spec] of Object.entries(target.models)) probs[target.name][name] = predict(spec, row);
        }
      }
      live[sym] = { time: bars.time[i], close: bars.close[i], probs, bars, features, stages: stagesFor(sym, bars) };
    }));
    state.live = live;
    state.error = null;
    const last = Math.max(...Object.values(live).map((l) => l.time));
    state.next = last + 2 * INTERVAL_MS + 4000;  // cuando cierre la vela en curso, con margen
  } catch (e) {
    state.error = e;
    state.next = Date.now() + state.offset + 60_000;
  }
  renderCards();
  renderStatus();
  renderRecent();
  const wait = Math.max(state.next - (Date.now() + state.offset), 15_000);
  state.timer = setTimeout(refresh, wait);
}

// ---------- últimas velas: qué predijo y si acertó ----------

const priceFmt = (x) => x.toLocaleString("en-US", { maximumFractionDigits: x >= 1000 ? 0 : x >= 10 ? 2 : 4 });
const when = (ms) => new Date(ms).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

// Los botones se crean una vez (así no se pierde el foco del teclado al pulsar) y luego solo cambia cuál está pulsado.
function seg(box, items, current, onPick) {
  if (box.children.length !== items.length) {
    box.innerHTML = "";
    for (const [value, label] of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.dataset.value = String(value);
      b.addEventListener("click", () => onPick(value));
      box.append(b);
    }
  }
  for (const b of box.children) b.setAttribute("aria-pressed", String(b.dataset.value === String(current)));
}

function renderRecentControls() {
  const r = state.recent;
  const symbols = state.models?.symbols ?? ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
  seg($("#recent-symbols"), symbols.map((s) => [s, symLabel(s)]), r.symbol, (v) => { r.symbol = v; renderRecent(); });
  seg($("#recent-models"), LIVE_MODELS.map((m) => [m, MODEL_NAMES[m]]), r.model, (v) => { r.model = v; renderRecent(); });
  seg($("#recent-window"), WINDOWS, r.window, (v) => { r.window = v; renderRecent(); });
}

function recentSummary(sym, rows, firstShown) {
  const r = state.recent;
  const s = hitSummary(rows);
  const span = WINDOWS.find(([w]) => w === r.window)?.[1] ?? "";
  const who = `${MODEL_NAMES[r.model].toLowerCase()} on ${symLabel(sym)}`;
  if (!s.n) {
    return rows.length
      ? `No decided candles in the last ${span} yet for ${who}.`
      : `No candles to score yet for ${who}: the published model has seen everything up to ` +
        `${when(dataTo(state.models))}. Its predictions appear here as new candles close.`;
  }
  let text = `Last ${span}, ${who}: right on <b>${s.k} of ${s.n}</b> decided candles (${pct(s.rate)}; 95% interval ` +
    `${pct(s.lo, 0)}–${pct(s.hi, 0)}). ${s.ups} of them closed up, so “always up” would have scored ${pct(s.ups / s.n)}.`;
  if (s.lo <= 0.5 && s.hi >= 0.5) text += " With this few candles that is compatible with a coin flip; the long track record below is what counts.";
  if (firstShown > 0) text += ` The first ${firstShown} candles of the window are not scored: the model had already seen them.`;
  const live = state.live?.[sym];
  if (live?.stages?.length) {
    const by = new Map();
    for (const x of rows) {
      if (x.hit === null) continue;
      const st = stageAt(live, x.index)?.label;
      if (!st) continue;
      const c = by.get(st) ?? { k: 0, n: 0 };
      c.n += 1;
      c.k += x.hit ? 1 : 0;
      by.set(st, c);
    }
    const parts = [...by].sort((a, b) => b[1].n - a[1].n).map(([s, c]) => `${STAGES[s] ?? s} ${c.k} of ${c.n}`);
    if (parts.length) text += `<br><span class="muted">Right by stage when it predicted: ${parts.join(" · ")}.</span>`;
  }
  return text;
}

function recentChart(sym, live, rows) {
  const r = state.recent;
  const { bars } = live;
  const n = bars.time.length;
  const ahead = state.models.targets.find((t) => t.name === state.target).candles_ahead;
  const N = Math.min(r.window, n);
  const k0 = n - N;
  const slots = N + ahead;                 // velas cerradas + las que aún deciden predicciones pendientes
  const byTarget = new Map(rows.map((x) => [x.index + ahead, x]));
  // Al ancho real del recuadro: así el texto mide 12 px también en el móvil.
  const W = Math.max(300, Math.round(($("#chart-recent").clientWidth || 744) - 26));
  const L = 6, R = 66, T = 24, RIB = live.stages?.length ? 18 : 0, PH = 220, GAP = 16, LH = 58, B = 28;
  const PT = T + RIB + (RIB ? 8 : 0);       // arriba del panel de precios
  const H = PT + PH + GAP + LH + B;
  const slot = (W - L - R) / slots;
  const bw = Math.max(1, Math.min(slot * 0.62, 12));
  const cx = (s) => L + (s + 0.5) * slot;
  let lo = Infinity, hi = -Infinity;
  for (let k = k0; k < n; k++) { lo = Math.min(lo, bars.low[k]); hi = Math.max(hi, bars.high[k]); }
  const pad = (hi - lo) * 0.06 || hi * 0.001;
  lo -= pad; hi += pad;
  const y = (v) => PT + PH * (1 - (v - lo) / (hi - lo));
  let g = `<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">` +
    `<rect width="5" height="5" fill="var(--band)"/><line x1="0" y1="0" x2="0" y2="5" stroke="var(--muted)" stroke-width="1.2"/>` +
    `</pattern></defs>`;

  // etapa del mercado en cada vela: tramos seguidos con su nombre; rayado si es incierta
  if (RIB) {
    let s0 = 0;
    for (let s = 1; s <= N; s++) {
      const cur = stageAt(live, k0 + s0)?.label;
      if (s < N && stageAt(live, k0 + s)?.label === cur) continue;
      if (cur) {
        const x0 = L + s0 * slot, w = (s - s0) * slot;
        g += `<rect class="${cur === "uncertain" ? "rib unsure" : "rib"}" x="${(x0 + 1).toFixed(1)}" y="${T}" ` +
          `width="${Math.max(w - 2, 0.5).toFixed(1)}" height="${RIB}" rx="3"/>`;
        const name = STAGES[cur] ?? cur;
        if (w >= name.length * 7 + 12 && cur !== "uncertain") {
          g += `<text class="rib-label" x="${(x0 + w / 2).toFixed(1)}" y="${T + RIB / 2 + 4}" text-anchor="middle">${name}</text>`;
        }
      }
      s0 = s;
    }
    g += `<text x="${W - R + 8}" y="${T + RIB / 2 + 4}">stage</text>`;
  }

  // ejes: precios a la derecha, cuadrícula fina
  const rough = (hi - lo) / 4;
  const mag = 10 ** Math.floor(Math.log10(rough));
  const stepY = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => m >= rough);
  for (let v = Math.ceil(lo / stepY) * stepY; v <= hi; v += stepY) {
    const yy = y(v).toFixed(1);
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>`;
    g += `<text x="${W - R + 8}" y="${(+yy + 4).toFixed(1)}">${priceFmt(v)}</text>`;
  }
  // horas abajo, separadas al menos 76 px
  const tickH = [1, 2, 3, 4, 6, 12, 24].find((h) => h * 4 * slot >= 76) ?? 24;
  for (let s = 0; s < N; s++) {
    const d = new Date(bars.time[k0 + s]);
    if (d.getMinutes() !== 0 || d.getHours() % tickH !== 0) continue;
    const label = d.getHours() === 0 ? d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }) : clock(bars.time[k0 + s]);
    g += `<text x="${cx(s).toFixed(1)}" y="${H - 8}" text-anchor="middle">${label}</text>`;
  }

  // dónde empieza lo que el modelo no había visto, y el cambio de modelo si cae en la ventana
  const firstScored = rows.length ? rows[0].index + ahead : Infinity;
  const marks = [];
  if (firstScored > k0 && firstScored < n) marks.push([firstScored - k0, "not seen by the model from here →"]);
  const sw = rows.find((x, i) => i > 0 && x.source !== rows[i - 1].source);
  if (sw && sw.index + ahead > k0 && sw.index + ahead < n) marks.push([sw.index + ahead - k0, "model retrained"]);
  let lastLabelX = -Infinity;
  for (const [s, label] of marks) {
    const xx = (L + s * slot).toFixed(1);
    g += `<line class="edge" x1="${xx}" x2="${xx}" y1="${T - 6}" y2="${PT + PH + GAP + LH}"/>`;
    if (+xx - lastLabelX > 170 && +xx < W - R - 60) {
      g += `<text class="note" x="${+xx + 5}" y="${T - 10}">${label}</text>`;
      lastLabelX = +xx;
    }
  }

  // velas: huecas si cierran arriba, rellenas si cierran abajo
  for (let s = 0; s < N; s++) {
    const k = k0 + s;
    const o = bars.open[k], c = bars.close[k];
    const x = cx(s);
    const top = y(Math.max(o, c)), bot = y(Math.min(o, c));
    const up = c >= o;
    g += `<line class="${up ? "wick" : "wick down"}" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${y(bars.high[k]).toFixed(1)}" y2="${y(bars.low[k]).toFixed(1)}"/>`;
    g += `<rect class="${up ? "body" : "body down"}" x="${(x - bw / 2).toFixed(1)}" y="${top.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(bot - top, 1).toFixed(1)}"/>`;
  }

  // franja de predicciones: arriba = dijo que subía, abajo = que bajaba; largo = distancia al 50 %
  const y0 = PT + PH + GAP + LH / 2;
  g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y0}" y2="${y0}"/>`;
  g += `<text x="${W - R + 8}" y="${PT + PH + GAP + 9}">60%</text><text x="${W - R + 8}" y="${y0 + 4}">50%</text>` +
    `<text x="${W - R + 8}" y="${PT + PH + GAP + LH}">40%</text>`;
  for (let s = 0; s < slots; s++) {
    const x = byTarget.get(k0 + s);
    if (!x) continue;
    const h = Math.max(2, (Math.min(Math.abs(x.p - 0.5), 0.1) / 0.1) * (LH / 2 - 3));
    const yy = x.p > 0.5 ? y0 - h : y0;
    const cls = x.hit === null ? "pending" : x.hit ? "hit" : bw >= 3 ? "miss" : "miss thin";
    const inset = cls === "hit" || cls === "miss thin" ? 0 : 0.75;
    g += `<rect class="${cls}" x="${(cx(s) - bw / 2 + inset).toFixed(1)}" y="${(yy + inset).toFixed(1)}" ` +
      `width="${Math.max(bw - 2 * inset, 0.8).toFixed(1)}" height="${Math.max(h - 2 * inset, 1).toFixed(1)}" rx="1"/>`;
  }
  g += `<rect class="cursor" id="recent-cursor" x="0" y="${T - 6}" width="${slot.toFixed(2)}" ` +
    `height="${PT - T + PH + GAP + LH + 6}" visibility="hidden"/>`;
  const label = `${symLabel(sym)} 15-minute candles for the last ${N} periods, with the prediction made for each one and whether it was right`;
  return { svg: svg(W, H, g, label), geom: { W, L, slot, slots, k0, n, ahead, byTarget } };
}

function tipHTML(sym, live, geom, s) {
  const { bars } = live;
  const k = geom.k0 + s;
  const x = geom.byTarget.get(k);
  const closed = k < geom.n;
  const start = closed ? bars.time[k] : bars.time[geom.n - 1] + (k - geom.n + 1) * INTERVAL_MS;
  let h = `<b>${symLabel(sym)} · ${clock(start)}–${clock(start + INTERVAL_MS)}</b>`;
  if (closed) h += `<br>Open ${priceFmt(bars.open[k])} · close ${priceFmt(bars.close[k])}`;
  else h += "<br>Candle not closed yet";
  const st = closed ? stageAt(live, k) : null;
  if (st) {
    const top = st.probs.slice(0, st.label === "uncertain" ? 2 : 1).map(([s, p]) => `${STAGES[s] ?? s} ${pct(p, 0)}`).join(", ");
    h += `<br>Stage: <b>${STAGES[st.label] ?? st.label}</b> (${top})`;
  }
  if (!x) return h + "<br><span class=\"muted\">Not scored: the model had already seen this candle.</span>";
  const said = x.p > 0.5 ? "up" : "down";
  h += `<br>${MODEL_NAMES[state.recent.model]} said <b>${said}</b> (P(up) ${pct(x.p)}) when the ${clock(x.time)} candle closed`;
  h += x.hit === null ? "<br>Not decided yet" : x.hit ? "<br><b>✓ Right</b>" : "<br><b>✗ Wrong</b>";
  h += `<br><span class="muted">${x.source === "current" ? "Current model" : "Model published the week before"}</span>`;
  return h;
}

function wireRecentHover(sym, live, geom) {
  const box = $("#chart-recent");
  const svgEl = box.querySelector("svg");
  const tip = $("#recent-tip");
  const cursor = svgEl.querySelector("#recent-cursor");
  let current = -1;
  const show = (s, clientX, clientY) => {
    if (s < 0 || s >= geom.slots) { hide(); return; }
    current = s;
    cursor.setAttribute("x", (geom.L + s * geom.slot).toFixed(2));
    cursor.setAttribute("visibility", "visible");
    tip.innerHTML = tipHTML(sym, live, geom, s);
    tip.hidden = false;
    const fig = box.parentElement.getBoundingClientRect();
    const rect = svgEl.getBoundingClientRect();
    const px = clientX ?? rect.left + ((geom.L + (s + 0.5) * geom.slot) / geom.W) * rect.width;
    const py = clientY ?? rect.top + rect.height * 0.3;
    let left = px - fig.left + 14;
    if (left + tip.offsetWidth > fig.width) left = Math.max(px - fig.left - tip.offsetWidth - 14, 0);
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(py - fig.top - tip.offsetHeight - 10, 0)}px`;
  };
  const hide = () => { current = -1; tip.hidden = true; cursor.setAttribute("visibility", "hidden"); };
  const slotAt = (clientX) => {
    const rect = svgEl.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * geom.W;
    return Math.floor((x - geom.L) / geom.slot);
  };
  svgEl.addEventListener("pointermove", (e) => show(slotAt(e.clientX), e.clientX, e.clientY));
  svgEl.addEventListener("pointerleave", hide);
  svgEl.setAttribute("tabindex", "0");
  svgEl.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const s = current < 0 ? geom.slots - 1 : current + (e.key === "ArrowRight" ? 1 : -1);
      show(Math.min(Math.max(s, 0), geom.slots - 1));
    } else if (e.key === "Escape") hide();
  });
  svgEl.addEventListener("blur", hide);
}

function recentTable(rows, live) {
  const { bars } = live;
  const body = $("#recent-table tbody");
  body.innerHTML = rows.slice().reverse().map((x) => {
    const k = x.index + (x.target - x.time) / INTERVAL_MS;
    const close = k < bars.time.length ? priceFmt(bars.close[k]) : "–";
    const result = x.hit === null ? "not decided" : x.hit ? "✓ right" : "✗ wrong";
    return `<tr><td>${when(x.target)}</td><td>${close}</td><td>${pct(x.p)}</td><td>${x.p > 0.5 ? "up" : "down"}</td>` +
      `<td class="text">${result}</td></tr>`;
  }).join("");
}

function recentAll() {
  const body = $("#recent-all tbody");
  const symbols = state.models.symbols;
  let since = Infinity, most = 0;
  body.innerHTML = symbols.map((sym) => {
    const live = state.live?.[sym];
    if (!live?.bars) return `<tr><td>${symLabel(sym)}</td><td colspan="3" class="text">No live data.</td></tr>`;
    let ups = "–";
    const cells = LIVE_MODELS.map((m) => {
      const rows = history(live.bars, live.features, state.models, state.target, m);
      if (rows.length) since = Math.min(since, rows[0].target);
      const s = hitSummary(rows);
      if (s.n) ups = pct(s.ups / s.n);
      most = Math.max(most, s.n);
      return s.n ? `${pct(s.rate)} <small>of ${int(s.n)}</small>` : "–";
    });
    return `<tr><td>${symLabel(sym)}</td>${cells.map((c) => `<td>${c}</td>`).join("")}<td>${ups}</td></tr>`;
  }).join("");
  $("#recent-all-note").textContent = Number.isFinite(since)
    ? `Hit rate on every scored candle your browser has (since ${when(since)}), for this target. With ${int(most)} ` +
      `candles the 95% interval is about ±${(196 * Math.sqrt(0.25 / most)).toFixed(0)} points, so small differences ` +
      "between models mean nothing here."
    : "";
}

function renderRecent() {
  if (!state.models) return;
  renderRecentControls();
  const r = state.recent;
  const live = state.live?.[r.symbol];
  const summary = $("#recent-summary");
  $("#recent-note").textContent = `${TARGETS[state.target].question} Each candle shows the prediction that the model ` +
    "published at the time made for it. Only candles that model had not seen while training are scored.";
  if (state.error || !live?.bars) {
    summary.textContent = state.error ? "Live data unavailable: Binance did not answer from your network." : "Loading the latest candles…";
    $("#chart-recent").innerHTML = "";
    return;
  }
  const rows = history(live.bars, live.features, state.models, state.target, r.model);
  const ahead = state.models.targets.find((t) => t.name === state.target).candles_ahead;
  const k0 = live.bars.time.length - Math.min(r.window, live.bars.time.length);
  const inWindow = rows.filter((x) => x.index + ahead >= k0);
  const firstScored = rows.length ? rows[0].index + ahead : live.bars.time.length;
  summary.innerHTML = recentSummary(r.symbol, inWindow, Math.max(0, Math.min(firstScored, live.bars.time.length) - k0));
  const { svg: chart, geom } = recentChart(r.symbol, live, rows);
  $("#chart-recent").innerHTML = chart;
  wireRecentHover(r.symbol, live, geom);
  recentTable(inWindow, live);
  recentAll();
}

// ---------- resultados de los estudios ----------

const section = (doc, title) => doc.sections.find((s) => s.title.startsWith(title));
const rowsOf = (sec) => Object.fromEntries(sec.rows.map((r) => [r[0], r.slice(1)]));

function svg(w, h, inner, label) {
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`;
}

function chartCandles(doc) {
  const sec = section(doc, "AUC por vela");
  const rows = rowsOf(sec);
  const cols = sec.columns.slice(1);
  const models = CHART_MODELS.filter((m) => rows[m]);
  const W = 720, H = 320, L = 56, R = 24, T = 18, B = 40;
  const values = models.flatMap((m) => rows[m]);
  const lo = Math.min(0.495, Math.floor(Math.min(...values) * 200) / 200);
  const hi = Math.ceil(Math.max(...values) * 200) / 200 + 0.0025;
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const step = (W - L - R) / (cols.length - 0.4);
  const x = (j) => L + step * (j + 0.3) + (j === cols.length - 1 ? step * 0.25 : 0);
  let g = "";
  for (let v = Math.ceil(lo * 100) / 100; v <= hi + 1e-9; v += 0.01) {
    const yy = y(v).toFixed(1);
    g += `<line class="${Math.abs(v - 0.5) < 1e-9 ? "chance" : "grid"}" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>`;
    g += `<text x="${L - 8}" y="${(+yy + 4).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  const names = { vela_1: "1st candle", vela_2: "2nd", vela_3: "3rd", vela_4: "4th", hora: "whole hour" };
  cols.forEach((c, j) => {
    g += `<text class="label" x="${x(j).toFixed(1)}" y="${H - 14}" text-anchor="middle">${names[c] ?? c}</text>`;
  });
  for (const m of models) {
    const pts = rows[m].map((v, j) => [x(j), y(v)]);
    const path = pts.slice(0, 4).map((p, j) => `${j ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    g += `<path d="${path}" fill="none" stroke="var(--m-${m})" stroke-width="2.2"/>`;
    for (const [px, py] of pts) g += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="4.2" fill="var(--m-${m})"/>`;
  }
  $("#chart-candles").innerHTML = svg(W, H, g, "Out-of-sample AUC for each of the next four candles and the whole hour, by model");
  $("#legend-candles").innerHTML = models.map((m) =>
    `<li><i class="dot" style="background:var(--m-${m})"></i>${esc(MODEL_NAMES[m])}</li>`).join("");
}

function chartRegimes(doc) {
  const sec = section(doc, "Por régimen: vela_1");
  const share = rowsOf(section(doc, "Tiempo en cada régimen"));
  const col = sec.columns.indexOf("auc") - 1;
  const by = {};
  for (const r of sec.rows) {
    const [reg, model] = r[0].split(" · ");
    (by[reg] ??= {})[model] = r[1 + col];
  }
  const regs = Object.keys(REGIMES).filter((r) => by[r]);
  const W = 720, L = 190, R = 24, T = 26, rowH = 42, H = T + rowH * regs.length + 30;
  const values = regs.flatMap((r) => Object.values(by[r]));
  const lo = 0.5, hi = Math.ceil(Math.max(...values) * 100) / 100 + 0.002;
  const x = (v) => L + (W - L - R) * (v - lo) / (hi - lo);
  let g = "";
  for (let v = lo; v <= hi + 1e-9; v += 0.01) {
    const xx = x(v).toFixed(1);
    g += `<line class="${v === lo ? "chance" : "grid"}" x1="${xx}" x2="${xx}" y1="${T - 8}" y2="${H - 28}"/>`;
    g += `<text x="${xx}" y="${H - 10}" text-anchor="middle">${v.toFixed(2)}</text>`;
  }
  regs.forEach((reg, i) => {
    const yy = T + rowH * i + rowH / 2;
    const s = share[reg]?.[0];
    g += `<text class="label" x="0" y="${yy + 4}">${REGIMES[reg]}</text>`;
    if (s != null) g += `<text x="${L - 14}" y="${yy + 4}" text-anchor="end">${Math.round(s * 100)}% of time</text>`;
    const vals = Object.values(by[reg]);
    g += `<line class="grid" x1="${x(Math.min(...vals)).toFixed(1)}" x2="${x(Math.max(...vals)).toFixed(1)}" y1="${yy}" y2="${yy}" stroke-width="3"/>`;
    for (const m of CHART_MODELS) {
      if (by[reg][m] == null) continue;
      g += `<circle cx="${x(by[reg][m]).toFixed(1)}" cy="${yy}" r="5" fill="var(--m-${m})"><title>${MODEL_NAMES[m]}: ${by[reg][m].toFixed(3)}</title></circle>`;
    }
  });
  $("#chart-regimes").innerHTML = svg(W, H, g, "Out-of-sample AUC on the next candle in each market regime, by model");
}

function marketsTable(doc) {
  const out = [];
  for (const sec of doc.sections) {
    const name = MARKETS.find(([re]) => re.test(sec.title))?.[1] ?? sec.title;
    const col = (c) => sec.columns.indexOf(c);
    const rows = sec.rows.filter((r) => r[0] !== "siempre_sube" && r[0] !== "momentum");
    const base = sec.rows.find((r) => r[0] === "siempre_sube");
    if (!rows.length || !base) continue;
    const best = rows.reduce((a, b) => (b[col("skill_vs_siempre_sube")] > a[col("skill_vs_siempre_sube")] ? b : a));
    const verdictLine = sec.notes.find((n) => n.startsWith("**Veredicto:**")) ?? "";
    let verdict = "No demonstrated edge";
    if (verdictLine.includes("POSIBLE VENTAJA")) {
      verdict = verdictLine.includes("NO supera")
        ? "Small statistical edge; loses to buy-and-hold after costs"
        : "Statistical edge";
    }
    const info = sec.notes[0]?.match(/^(\d+) filas.*?de (\d{4}-\d{2}-\d{2}) a (\d{4}-\d{2}-\d{2})/);
    const sub = info ? `<br><small>${int(+info[1])} rows · ${info[2].slice(0, 4)}–${info[3].slice(0, 4)}</small>` : "";
    out.push(`<tr><td>${esc(name)}${sub}</td><td class="text">${esc(MODEL_NAMES[best[0]] ?? best[0].replace(/_/g, " "))}</td>` +
      `<td>${best[col("auc")].toFixed(3)}</td><td>${pct(best[col("acierto")])} (${pct(base[col("acierto")])})</td>` +
      `<td class="text">${verdict}</td></tr>`);
  }
  $("#markets tbody").innerHTML = out.join("");
}

function fillFromResults(doc) {
  const rows = rowsOf(section(doc, "AUC por vela"));
  const models = Object.keys(rows).filter((m) => m !== "siempre_sube" && m !== "momentum");
  const v1 = Math.max(...models.map((m) => rows[m][0]));
  const v4 = models.map((m) => rows[m][3]);
  const [a, b] = [Math.min(...v4).toFixed(2), Math.max(...v4).toFixed(2)];
  document.querySelectorAll('[data-fill="best-auc-1"]').forEach((el) => { el.textContent = v1.toFixed(2); });
  document.querySelectorAll('[data-fill="auc-4"]').forEach((el) => { el.textContent = a === b ? a : `${a}–${b}`; });
  const intro = doc.intro.match(/(\d+) filas.*?fuera de muestra (\S+) → (\S+)/);
  if (intro) {
    $("#results-intro").textContent = `Purged walk-forward evaluation on BTC, ETH and SOL, 15-minute candles: ` +
      `${int(+intro[1])} rows, out of sample from ${day(intro[2])} to ${day(intro[3])}, no costs. ` +
      `Random forest and the transformer are only in the Python package.`;
  }
}

function setupCopy() {
  document.querySelectorAll(".copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const pre = document.getElementById(btn.dataset.copy);
      try {
        await navigator.clipboard.writeText(pre.innerText);
        btn.textContent = "Copied";
      } catch {
        const range = document.createRange();
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = "Selected";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 1600);
    });
  });
}

async function main() {
  setupCopy();
  renderTargets();
  renderCards();
  const [models, metrics, results] = await Promise.allSettled(
    ["data/models.json", "data/metrics.json", "data/results.json"].map(getJSON));
  if (results.status === "fulfilled") {
    for (const fn of [fillFromResults, chartCandles, chartRegimes]) {
      try { fn(results.value.candles); } catch (e) { console.error(e); }
    }
    try { marketsTable(results.value.comparison); } catch (e) { console.error(e); }
  }
  if (metrics.status === "fulfilled") {
    state.metrics = metrics.value;
    renderRecord();
  }
  if (models.status !== "fulfilled") {
    state.error = models.reason;
    renderCards();
    renderStatus();
    return;
  }
  state.models = models.value;
  document.querySelectorAll('[data-fill="n-features"]').forEach((el) => { el.textContent = models.value.features.length; });
  document.querySelectorAll('[data-fill="trained"]').forEach((el) => {
    el.textContent = `this version uses data up to ${day(models.value.trained_to)}`;
  });
  $("#generated").textContent = `Models trained ${day(models.value.generated)}`;
  renderRecent();
  let width = innerWidth, pending;
  addEventListener("resize", () => {
    clearTimeout(pending);
    pending = setTimeout(() => { if (innerWidth !== width) { width = innerWidth; renderRecent(); } }, 200);
  });
  setInterval(renderStatus, 1000);
  refresh();
}

main();
