// P(sube) con los modelos exportados por updown/export.py: las mismas cuentas que
// predict_proba de scikit-learn, sin librerías.

const expit = (x) => 1.0 / (1.0 + Math.exp(-x));

function linearPlatt(spec, x) {
  let score = 0;
  for (let i = 0; i < x.length; i++) score += ((x[i] - spec.mean[i]) / spec.scale[i]) * spec.coef[i];
  score += spec.intercept;
  return expit(spec.platt_a * score + spec.platt_b);
}

// Nodo: [variable, umbral, izquierda, derecha, valor, nulo_a_la_izquierda, hoja]
function trees(spec, x) {
  let raw = spec.base;
  for (const tree of spec.trees) {
    let node = tree[0];
    while (!node[6]) {
      const v = x[node[0]];
      const left = Number.isNaN(v) ? node[5] === 1 : v <= node[1];
      node = tree[left ? node[2] : node[3]];
    }
    raw += node[4];
  }
  return expit(raw);
}

/** Probabilidad de que suba, para una fila con las columnas de `spec.features`. */
export function predict(spec, x) {
  switch (spec.kind) {
    case "constant": return spec.p;
    case "linear_platt": return linearPlatt(spec, x);
    case "trees": return trees(spec, x);
    default: throw new Error(`Unknown model kind: ${spec.kind}`);
  }
}
