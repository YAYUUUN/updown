// Etapa del mercado con su probabilidad: el HMM de updown/hmm.py exportado por
// updown/export.py, con el mismo filtro hacia delante (cada vela solo ve su pasado).
// tests/test_web.py comprueba que las probabilidades coinciden con las de Python.

/** Observaciones del HMM (hmm_vol, hmm_tend) como web_hmm_observations de updown/export.py:
 *  volatilidad de un día frente a la referencia del último año fijada al entrenar, y retorno
 *  de 30 días dividido por la volatilidad de ese plazo. NaN mientras falten velas. */
export function hmmObservations(bars, ref, day, month) {
  const c = bars.close;
  const n = c.length;
  const vol = new Float64Array(n).fill(NaN);
  const r = new Float64Array(n).fill(NaN);
  for (let i = 1; i < n; i++) r[i] = Math.log(c[i]) - Math.log(c[i - 1]);
  for (let i = day; i < n; i++) {            // desviación típica muestral de los últimos `day` retornos
    let m = 0;
    for (let k = i - day + 1; k <= i; k++) m += r[k];
    m /= day;
    let ss = 0;
    for (let k = i - day + 1; k <= i; k++) ss += (r[k] - m) * (r[k] - m);
    vol[i] = Math.sqrt(ss / (day - 1));
  }
  const clip = (x) => (Number.isNaN(x) ? NaN : Math.min(Math.max(x, -5), 5));
  const [mu, sd] = ref;
  const obsVol = Float64Array.from(vol, (v) => (v > 0 ? clip((Math.log(v) - mu) / sd) : NaN));
  const obsTrend = Float64Array.from(c, (x, i) =>
    (i >= month ? clip(Math.log(x / c[i - month]) / (vol[i] * Math.sqrt(month))) : NaN));
  return [obsVol, obsTrend];
}

/** Probabilidad filtrada de cada etapa en cada vela y la etapa si llega al umbral («uncertain» si no). */
export function filterStages(spec, obs) {
  const { names, startprob, transmat, means, covars, threshold } = spec;
  const K = names.length;
  const inv = covars.map(([[a, b], [, d]]) => {
    const det = a * d - b * b;
    return { a: d / det, b: -b / det, d: a / det, logdet: Math.log(det) };
  });
  const LOG2PI = Math.log(2 * Math.PI);
  const stages = [...new Set(names)].sort();
  const [ov, ot] = obs;
  const n = ov.length;
  const probs = Object.fromEntries(stages.map((s) => [s, new Float64Array(n).fill(NaN)]));
  const label = new Array(n).fill(null);
  let alpha = null;
  const logb = new Float64Array(K);
  for (let t = 0; t < n; t++) {
    const seen = !Number.isNaN(ov[t]) && !Number.isNaN(ot[t]);
    let top = -Infinity;
    for (let k = 0; k < K; k++) {
      if (!seen) { logb[k] = 0; continue; }  // sin observación no hay información
      const dx = ov[t] - means[k][0], dy = ot[t] - means[k][1];
      const q = inv[k];
      logb[k] = -0.5 * (2 * LOG2PI + q.logdet + dx * dx * q.a + 2 * dx * dy * q.b + dy * dy * q.d);
      top = Math.max(top, logb[k]);
    }
    if (!seen) top = 0;
    const next = new Float64Array(K);
    let total = 0;
    for (let j = 0; j < K; j++) {
      let prior = 0;
      if (alpha === null) prior = startprob[j];
      else for (let i = 0; i < K; i++) prior += alpha[i] * transmat[i][j];
      next[j] = prior * Math.exp(logb[j] - top);
      total += next[j];
    }
    for (let j = 0; j < K; j++) next[j] /= total;
    alpha = next;
    if (!seen) continue;
    let best = null, bestP = -1;
    for (const s of stages) {
      let p = 0;
      for (let k = 0; k < K; k++) if (names[k] === s) p += alpha[k];
      probs[s][t] = p;
      if (p > bestP) { best = s; bestP = p; }
    }
    label[t] = bestP >= threshold ? best : "uncertain";
  }
  return { stages, probs, label };
}
