// Variables y probabilidades calculadas por la web (web/js) para unas klines dadas.
//   node tests/js/parity.mjs klines.json models.json  ->  JSON por stdout
// Lo llama tests/test_web.py para compararlo con Python.
import { readFileSync } from "node:fs";
import { buildFeatures, featureRow, parseKlines } from "../../web/js/features.js";
import { predict } from "../../web/js/models.js";

const [klinesPath, modelsPath] = process.argv.slice(2);
const bars = parseKlines(JSON.parse(readFileSync(klinesPath, "utf8")));
const spec = JSON.parse(readFileSync(modelsPath, "utf8"));
const features = buildFeatures(bars);

const out = { time: bars.time, features: {}, probs: {} };
for (const [name, col] of features) out.features[name] = Array.from(col, (x) => (Number.isNaN(x) ? null : x));
for (const target of spec.targets) {
  out.probs[target.name] = {};
  for (const [model, m] of Object.entries(target.models)) {
    out.probs[target.name][model] = bars.time.map((_, i) => {
      const row = featureRow(features, spec.features, i);
      return row ? predict(m, row) : null;
    });
  }
}
process.stdout.write(JSON.stringify(out));
