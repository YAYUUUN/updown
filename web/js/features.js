// Variables de updown/features.py para velas de Binance, operación por operación,
// para que el navegador dé la misma probabilidad que Python (lo comprueba tests/test_web.py).
// Las medias exponenciales repiten el algoritmo de pandas (ewm con adjust=False), y las
// ventanas móviles devuelven NaN si les falta algún dato, como rolling(n).

export const INTERVAL_MS = 15 * 60 * 1000;
const RETURN_WINDOWS = [1, 2, 3, 5, 10, 20, 60];
const HOUR = 4;   // velas de 15 min en una hora
const DAY = 96;   // velas de 15 min en un día

/** Klines de la API de Binance -> columnas numéricas, ordenadas y sin velas repetidas. */
export function parseKlines(rows) {
  const byTime = new Map();
  for (const r of rows) {
    const close = Number(r[4]);
    if (!(close > 0)) continue;
    byTime.set(Number(r[0]), r);
  }
  const times = [...byTime.keys()].sort((a, b) => a - b);
  const col = (k) => Float64Array.from(times, (t) => Number(byTime.get(t)[k]));
  return {
    time: times, open: col(1), high: col(2), low: col(3), close: col(4), volume: col(5),
    trades: col(8), takerBuy: col(9),
  };
}

/** Quita las velas que aún no han cerrado: inicio + 15 min > ahora (hora del servidor). */
export function dropUnclosed(bars, nowMs) {
  let n = bars.time.length;
  while (n > 0 && bars.time[n - 1] + INTERVAL_MS > nowMs) n--;
  const out = {};
  for (const [k, v] of Object.entries(bars)) out[k] = v.slice(0, n);
  return out;
}

const empty = (n) => new Float64Array(n).fill(NaN);
const map = (a, fn) => Float64Array.from(a, fn);
const zip = (a, b, fn) => Float64Array.from(a, (x, i) => fn(x, b[i]));

function shift(a, k) {
  const out = empty(a.length);
  for (let i = k; i < a.length; i++) out[i] = a[i - k];
  return out;
}

function diff(a) {
  const out = empty(a.length);
  for (let i = 1; i < a.length; i++) out[i] = a[i] - a[i - 1];
  return out;
}

function rolling(a, n, fn) {
  const out = empty(a.length);
  for (let i = n - 1; i < a.length; i++) {
    const w = a.subarray(i - n + 1, i + 1);
    if (!w.some(Number.isNaN)) out[i] = fn(w);
  }
  return out;
}

const sum = (w) => w.reduce((s, x) => s + x, 0);
const mean = (w) => sum(w) / w.length;
function std(w) {  // desviación típica muestral (ddof=1), como pandas
  const m = mean(w);
  let ss = 0;
  for (const x of w) ss += (x - m) * (x - m);
  return Math.sqrt(ss / (w.length - 1));
}
const maxOf = (w) => Math.max(...w);
const minOf = (w) => Math.min(...w);

/** pandas ewm(...).mean() con adjust=False e ignore_na=False. */
function ewmMean(x, { span, alpha }, minPeriods) {
  const com = span !== undefined ? (span - 1) / 2.0 : (1.0 - alpha) / alpha;
  const a = 1.0 / (1.0 + com);
  const oldWtFactor = 1.0 - a;
  const out = empty(x.length);
  if (!x.length) return out;
  let weighted = x[0];
  let nobs = Number.isNaN(weighted) ? 0 : 1;
  let oldWt = 1.0;
  if (nobs >= minPeriods) out[0] = weighted;
  for (let i = 1; i < x.length; i++) {
    const cur = x[i];
    const isObs = !Number.isNaN(cur);
    nobs += isObs ? 1 : 0;
    if (!Number.isNaN(weighted)) {
      oldWt *= oldWtFactor;
      if (isObs) {
        if (weighted !== cur) {
          weighted = oldWt * weighted + a * cur;
          weighted /= oldWt + a;
        }
        oldWt = 1.0;
      }
    } else if (isObs) {
      weighted = cur;
    }
    out[i] = nobs >= minPeriods ? weighted : NaN;
  }
  return out;
}

const ema = (s, span) => ewmMean(s, { span }, span);
const wilder = (s, period) => ewmMean(s, { alpha: 1.0 / period }, period);
const clipLower0 = (d) => (d < 0 ? 0 : d);  // NaN y -0.0 se quedan como están

function rsi(close, period) {
  const delta = diff(close);
  const gain = wilder(map(delta, clipLower0), period);
  const loss = wilder(map(delta, (d) => clipLower0(-d)), period);
  return Float64Array.from(gain, (g, i) => {
    const l = loss[i];
    if (g === 0 && l === 0) return 50.0;
    return 100.0 - 100.0 / (1.0 + g / l);
  });
}

function atr(high, low, close, period) {
  const prev = shift(close, 1);
  const tr = Float64Array.from(high, (h, i) => {
    const parts = [h - low[i], Math.abs(h - prev[i]), Math.abs(low[i] - prev[i])].filter((v) => !Number.isNaN(v));
    return parts.length ? Math.max(...parts) : NaN;
  });
  return wilder(tr, period);
}

const zscore = (s, n) => {
  const m = rolling(s, n, mean);
  const sd = rolling(s, n, std);
  return Float64Array.from(s, (x, i) => (x - m[i]) / sd[i]);
};

/** Las mismas columnas que build_features(bars, "15m") para velas de Binance (sin mercado de referencia). */
export function buildFeatures(bars) {
  const { open: o, high: h, low, close: c, volume: v } = bars;
  const n = c.length;
  const f = new Map();
  const logret = diff(map(c, Math.log));
  const a = atr(h, low, c, 14);

  for (const k of RETURN_WINDOWS) {
    const prev = shift(c, k);
    f.set(`ret_${k}`, zip(c, prev, (x, p) => Math.log(x / p)));
  }
  const vol5 = rolling(logret, 5, std);
  const vol20 = rolling(logret, 20, std);
  f.set("vol_5", vol5);
  f.set("vol_20", vol20);
  f.set("vol_ratio", zip(vol5, vol20, (x, y) => x / y));
  f.set("atr_pct", zip(a, c, (x, y) => x / y));
  f.set("rsi_14", map(rsi(c, 14), (r) => (r - 50.0) / 50.0));
  const ema10 = ema(c, 10);
  const ema50 = ema(c, 50);
  f.set("dist_ema10_atr", Float64Array.from(c, (x, i) => (x - ema10[i]) / a[i]));
  f.set("dist_ema50_atr", Float64Array.from(c, (x, i) => (x - ema50[i]) / a[i]));
  const ema12 = ema(c, 12);
  const ema26 = ema(c, 26);
  const macd = zip(ema12, ema26, (x, y) => x - y);
  const signal = ema(macd, 9);
  f.set("macd_hist", Float64Array.from(macd, (x, i) => (x - signal[i]) / c[i]));
  f.set("boll_z", zscore(c, 20));
  const hi20 = rolling(h, 20, maxOf);
  const lo20 = rolling(low, 20, minOf);
  f.set("range_pos_20", Float64Array.from(c, (x, i) => (x - lo20[i]) / (hi20[i] - lo20[i])));
  const prevClose = shift(c, 1);
  f.set("gap", zip(o, prevClose, (x, p) => Math.log(x / p)));
  f.set("body_atr", Float64Array.from(c, (x, i) => (x - o[i]) / a[i]));

  let positive = 0;
  for (const x of v) if (x > 0) positive++;
  if (positive / n > 0.9) f.set("volume_z", zscore(map(v, Math.log1p), 20));

  // flujo de órdenes
  const ratio = Float64Array.from(bars.takerBuy, (t, i) => (v[i] > 0 ? t / v[i] : 0.5));
  f.set("taker_ratio", ratio);
  const volH = rolling(v, HOUR, sum);
  const takerH = rolling(bars.takerBuy, HOUR, sum);
  f.set("taker_ratio_1h", Float64Array.from(volH, (x, i) => (x > 0 ? takerH[i] / x : 0.5)));
  f.set("taker_ratio_z", zscore(ratio, DAY));
  f.set("trades_z", zscore(map(bars.trades, Math.log1p), DAY));

  // calendario (hora de inicio de la vela, en UTC; lunes = 0)
  const dow = Float64Array.from(bars.time, (t) => (new Date(t).getUTCDay() + 6) % 7);
  const hour = Float64Array.from(bars.time, (t) => {
    const d = new Date(t);
    return d.getUTCHours() + d.getUTCMinutes() / 60.0;
  });
  f.set("dow_sin", map(dow, (d) => Math.sin(2 * Math.PI * d / 7)));
  f.set("dow_cos", map(dow, (d) => Math.cos(2 * Math.PI * d / 7)));
  f.set("hour_sin", map(hour, (x) => Math.sin(2 * Math.PI * x / 24)));
  f.set("hour_cos", map(hour, (x) => Math.cos(2 * Math.PI * x / 24)));

  for (const [k, col] of f) f.set(k, map(col, (x) => (Number.isFinite(x) ? x : NaN)));
  return f;
}

/** Fila `i` con las columnas en el orden que espera el modelo; null si falta alguna. */
export function featureRow(features, names, i) {
  const row = new Float64Array(names.length);
  for (let j = 0; j < names.length; j++) {
    const col = features.get(names[j]);
    if (!col || Number.isNaN(col[i])) return null;
    row[j] = col[i];
  }
  return row;
}
