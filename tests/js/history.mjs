// Historial de la web (web/js/history.js) para unas klines y unos modelos dados.
//   node tests/js/history.mjs klines.json models.json  ->  JSON por stdout
// Lo llama tests/test_web.py para compararlo con Python.
import { readFileSync } from "node:fs";
import { buildFeatures, parseKlines } from "../../web/js/features.js";
import { history } from "../../web/js/history.js";

const [klinesPath, modelsPath] = process.argv.slice(2);
const bars = parseKlines(JSON.parse(readFileSync(klinesPath, "utf8")));
const models = JSON.parse(readFileSync(modelsPath, "utf8"));
const features = buildFeatures(bars);

const out = {};
for (const target of models.targets) {
  out[target.name] = {};
  for (const model of Object.keys(target.models)) out[target.name][model] = history(bars, features, models, target.name, model);
}
process.stdout.write(JSON.stringify(out));
