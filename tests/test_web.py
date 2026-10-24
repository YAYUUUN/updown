"""La web calcula lo mismo que Python: variables, modelos exportados y probabilidades.

Si la versión en JavaScript de las variables se desviara un poco de updown/features.py,
la web daría probabilidades distintas de las que se evaluaron, sin ningún error visible.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from updown.data import parse_binance_klines
from updown.export import WEB_MODELS, export_model, tree_proba
from updown.features import build_features, dataset, feature_columns
from updown.models import make_model

ROOT = os.path.dirname(os.path.dirname(__file__))
NODE = shutil.which("node")


def synthetic_klines(n: int = 1500, seed: int = 5) -> list[list]:
    """Klines con el formato de la API de Binance (precios como texto), sin datos reales."""
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp("2026-01-05", tz="UTC").timestamp() * 1000)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.002, n)))
    volume = rng.lognormal(3, 0.6, n)
    volume[rng.random(n) < 0.01] = 0.0  # alguna vela sin volumen: la proporción de compras pasa a 0,5
    taker = volume * rng.uniform(0.3, 0.7, n)
    trades = rng.integers(50, 5000, n)
    rows = []
    for i in range(n):
        t = t0 + i * 900_000
        rows.append([t, f"{open_[i]:.8f}", f"{high[i]:.8f}", f"{low[i]:.8f}", f"{close[i]:.8f}",
                     f"{volume[i]:.8f}", t + 899_999, "0", int(trades[i]), f"{taker[i]:.8f}", "0", "0"])
    return rows


def test_exported_trees_match_sklearn():
    rows = synthetic_klines(900, seed=1)
    data = dataset(parse_binance_klines(rows), "15m")
    cols = feature_columns(data)
    train = data[data["up"].notna()]
    model = make_model("gradient_boosting").fit(train[cols], train["up"])
    spec = export_model(model, cols)
    np.testing.assert_allclose(tree_proba(spec, data[cols].to_numpy()), model.proba(data[cols]), rtol=0, atol=1e-12)


@pytest.mark.skipif(NODE is None, reason="hace falta Node.js para ejecutar el JavaScript de la web")
def test_web_matches_python(tmp_path):
    rows = synthetic_klines()
    bars = parse_binance_klines(rows)
    data = dataset(bars, "15m")
    cols = feature_columns(data)
    # Como en la web, se predicen velas POSTERIORES al entrenamiento. Sobre las propias filas de
    # entrenamiento la comparación no es justa: muchos umbrales de los árboles son exactamente un
    # valor de esas filas, y una diferencia en el último decimal (sumas en otro orden) cambia de rama.
    cut = int(len(data) * 0.6)
    train, test = data.iloc[:cut], data.iloc[cut:]
    models = {m: make_model(m).fit(train[cols], train["up"]) for m in ("siempre_sube", *WEB_MODELS)}
    spec = {"features": cols,
            "targets": [{"name": "vela_1", "models": {m: export_model(md, cols) for m, md in models.items()}}]}
    (tmp_path / "klines.json").write_text(json.dumps(rows))
    (tmp_path / "models.json").write_text(json.dumps(spec))

    res = subprocess.run([NODE, os.path.join(ROOT, "tests", "js", "parity.mjs"),
                          str(tmp_path / "klines.json"), str(tmp_path / "models.json")],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)

    expected = build_features(bars, "15m")
    assert list(js["features"]) == list(expected.columns)
    for name in expected.columns:
        got = np.array([np.nan if x is None else x for x in js["features"][name]], dtype=float)
        np.testing.assert_allclose(got, expected[name].to_numpy(), rtol=1e-9, atol=1e-12, err_msg=name)

    pos = {int(t): i for i, t in enumerate(js["time"])}
    rows_idx = [pos[int(t.timestamp() * 1000)] for t in test.index]
    for m, model in models.items():
        got = np.array(js["probs"]["vela_1"][m], dtype=float)[rows_idx]
        np.testing.assert_allclose(got, model.proba(test[cols]), rtol=0, atol=1e-9, err_msg=m)
