// Etapas calculadas por la web (web/js/stages.js) para unas klines y un HMM exportado.
//   node tests/js/stages.mjs klines.json stages.json  ->  JSON por stdout
// Lo llama tests/test_web.py para compararlo con Python.
import { readFileSync } from "node:fs";
import { parseKlines } from "../../web/js/features.js";
import { filterStages, hmmObservations } from "../../web/js/stages.js";

const [klinesPath, specPath] = process.argv.slice(2);
const bars = parseKlines(JSON.parse(readFileSync(klinesPath, "utf8")));
const spec = JSON.parse(readFileSync(specPath, "utf8"));
const obs = hmmObservations(bars, spec.ref, spec.day, spec.month);
const res = filterStages(spec.hmm, obs);
const plain = (a) => Array.from(a, (x) => (Number.isNaN(x) ? null : x));
const out = { time: bars.time, obs: obs.map(plain), label: res.label, probs: {} };
for (const s of res.stages) out.probs[s] = plain(res.probs[s]);
process.stdout.write(JSON.stringify(out));
