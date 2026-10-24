// La página: resultados de los estudios (data/results.json), historial fuera de muestra de
// los modelos de la web (data/metrics.json) y P(sube) en directo con velas de Binance.
import { buildFeatures, dropUnclosed, featureRow, INTERVAL_MS, parseKlines } from "./features.js";
import { predict } from "./models.js";

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

const state = { target: "vela_1", models: null, metrics: null, live: null, error: null, offset: 0, next: 0, api: 0 };
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
    return `<article class="card">${head}${rows}` +
      `<div class="meter-scale" aria-hidden="true"><span>40%</span><span>50%</span><span>60%</span></div>` +
      `<p class="baseline">“Always up” says ${pct(probs.siempre_sube)} · resolves at ${clock(resolves)}</p></article>`;
  }).join("");
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

async function refresh() {
  clearTimeout(state.timer);
  const models = state.models;
  try {
    const { serverTime } = await binance("/api/v3/time");
    state.offset = serverTime - Date.now();
    const live = {};
    await Promise.all(models.symbols.map(async (sym) => {
      const rows = await binance(`/api/v3/klines?symbol=${sym}&interval=${models.interval}&limit=1000`);
      const bars = dropUnclosed(parseKlines(rows), serverTime);
      const i = bars.time.length - 1;
      if (i < 0) return;
      const row = featureRow(buildFeatures(bars), models.features, i);
      let probs = null;
      if (row) {
        probs = {};
        for (const target of models.targets) {
          probs[target.name] = {};
          for (const [name, spec] of Object.entries(target.models)) probs[target.name][name] = predict(spec, row);
        }
      }
      live[sym] = { time: bars.time[i], close: bars.close[i], probs };
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
  const wait = Math.max(state.next - (Date.now() + state.offset), 15_000);
  state.timer = setTimeout(refresh, wait);
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
  setInterval(renderStatus, 1000);
  refresh();
}

main();
