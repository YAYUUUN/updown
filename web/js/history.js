// Historial de las últimas velas: qué predijo en cada vela el modelo que estaba publicado
// entonces y si acertó. Solo cuentan las velas que ese modelo NO vio al entrenarse.
//
// models.json trae los modelos actuales (entrenados con velas hasta `data_to`) y, en
// `previous`, los de una semana antes. Una predicción hecha al cerrar la vela t usa:
//   · los actuales, si t >= data_to (ni esa fila ni las velas que predice estaban en su entrenamiento);
//   · los anteriores, si previous.data_to <= t < data_to;
//   · ninguno, si es más antigua: los dos modelos ya habían visto esas velas y la predicción saldría inflada.
import { featureRow, INTERVAL_MS } from "./features.js";
import { predict } from "./models.js";

// Las medias exponenciales (EMA, RSI, ATR) arrancan en la primera vela descargada; tras 300
// velas su arranque pesa menos de una millonésima y las variables coinciden con las de Python.
export const WARMUP = 300;

const parseTime = (s) => Date.parse(String(s).replace(" ", "T"));

/** Último instante de datos de un juego de modelos (ms). Los models.json antiguos solo traen
 *  `trained_to`, la última fila etiquetada para la hora entera: 4 velas antes de la última. */
export function dataTo(set) {
  if (set.data_to) return parseTime(set.data_to);
  return parseTime(set.trained_to) + 4 * INTERVAL_MS;
}

/** Los juegos de modelos disponibles, del más reciente al más antiguo. */
export function modelSets(models) {
  const sets = [{ source: "current", from: dataTo(models), targets: models.targets, stages: models.stages ?? null }];
  if (models.previous) {
    sets.push({ source: "previous", from: dataTo(models.previous), targets: models.previous.targets,
      stages: models.previous.stages ?? null });
  }
  return sets;
}

/**
 * Predicciones fuera de muestra de un modelo para un objetivo, en orden temporal.
 * Cada elemento: { index, time, target, p, up, hit, source }
 *   time   : inicio de la vela tras cuyo cierre se predice.
 *   target : inicio de la vela que decide el resultado.
 *   up/hit : null mientras esa vela no haya cerrado.
 */
export function history(bars, features, models, targetName, modelName, { warmup = WARMUP } = {}) {
  const sets = modelSets(models);
  const ahead = models.targets.find((t) => t.name === targetName)?.candles_ahead;
  if (!ahead) return [];
  const whole = targetName === "hora";  // la hora entera: cierre dentro de 4 velas frente al actual
  const n = bars.time.length;
  const out = [];
  for (let i = warmup; i < n; i++) {
    const t = bars.time[i];
    const set = sets.find((s) => t >= s.from);
    if (!set) continue;
    const spec = set.targets.find((x) => x.name === targetName)?.models?.[modelName];
    if (!spec) continue;
    const row = featureRow(features, models.features, i);
    if (!row) continue;
    const p = predict(spec, row);
    const j = i + ahead;
    let up = null;
    if (j < n) up = whole ? bars.close[j] > bars.close[i] : bars.close[j] > bars.close[j - 1];
    out.push({
      index: i, time: t, target: t + ahead * INTERVAL_MS, p, up,
      hit: up === null ? null : (p > 0.5) === up, source: set.source,
    });
  }
  return out;
}

/** Aciertos con su intervalo de Wilson al 95 % (con pocas velas el azar da mucho margen). */
export function hitSummary(rows) {
  const done = rows.filter((r) => r.hit !== null);
  const n = done.length;
  const k = done.filter((r) => r.hit).length;
  const ups = done.filter((r) => r.up).length;
  if (!n) return { n, k, ups, rate: NaN, lo: NaN, hi: NaN };
  const z = 1.96;
  const phat = k / n;
  const den = 1 + (z * z) / n;
  const mid = (phat + (z * z) / (2 * n)) / den;
  const half = (z * Math.sqrt((phat * (1 - phat)) / n + (z * z) / (4 * n * n))) / den;
  return { n, k, ups, rate: phat, lo: mid - half, hi: mid + half };
}
