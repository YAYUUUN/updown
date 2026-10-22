import numpy as np
import pandas as pd
import pytest

from updown.confidence import ci95_by_day, confidence_curve, confidence_curve_live


def _oos(n=4000, seed=0, calibrated=True):
    """Predicciones de juguete en 4 bloques; si `calibrated`, sube con probabilidad p."""
    rng = np.random.default_rng(seed)
    p = np.clip(0.5 + rng.normal(0, 0.04, n), 0.01, 0.99)
    up = (rng.uniform(size=n) < (p if calibrated else 0.5)).astype(float)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"p": p, "up": up, "fold": np.repeat(np.arange(4), n // 4)}, index=idx)


@pytest.mark.parametrize("curve", [confidence_curve, confidence_curve_live])
def test_confidence_cut_ignores_outcomes(curve):
    oos = _oos()
    a = curve(oos)
    b = curve(oos.assign(up=1 - oos["up"]))   # mismas probabilidades, resultados al revés
    same = ["n", "umbral", "confianza_media", "predice_sube"]
    pd.testing.assert_frame_equal(a[same], b[same])
    assert np.allclose(a["acierto"] + b["acierto"], 1.0)


def test_top_slice_is_the_most_confident():
    oos = _oos()
    curve = confidence_curve(oos)
    k = int(round(0.05 * len(oos)))
    top = (oos["p"] - 0.5).abs().nlargest(k)
    assert curve.loc[0.05, "n"] == k
    assert curve.loc[0.05, "umbral"] == pytest.approx(top.min())
    assert curve.loc[1.0, "n"] == len(oos)


def test_confident_predictions_hit_more_when_there_is_signal():
    curve = confidence_curve(_oos(n=40000, seed=1))
    assert curve.loc[0.05, "acierto"] > curve.loc[1.0, "acierto"] + 0.03
    flat = confidence_curve(_oos(n=40000, seed=2, calibrated=False))
    assert abs(flat.loc[0.05, "acierto"] - 0.5) < 0.04          # sin señal, seguro no es acertar


def test_live_threshold_comes_from_previous_blocks_only():
    oos = _oos()
    live = confidence_curve_live(oos)
    assert live.loc[1.0, "n"] == int((oos["fold"] > 0).sum())  # el primer bloque no tiene pasado
    changed = oos.copy()
    changed.loc[changed["fold"] == 3, "p"] = 0.5               # el último bloque no mueve su propio corte
    assert confidence_curve_live(changed).loc[0.1, "umbral"] == pytest.approx(live.loc[0.1, "umbral"])
    assert confidence_curve_live(changed).loc[0.1, "n"] < live.loc[0.1, "n"]


def test_ci_is_wider_when_rows_are_not_independent():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-01-01", periods=3000, freq="1D", tz="UTC")
    correct = (rng.uniform(size=3000) < 0.53).astype(float)
    lo1, hi1 = ci95_by_day(correct, idx)
    lo3, hi3 = ci95_by_day(np.repeat(correct, 3), idx.repeat(3))  # 3 activos que se mueven igual
    assert hi1 - lo1 == pytest.approx(2 * 1.96 * np.sqrt(0.53 * 0.47 / 3000), rel=0.05)
    assert hi3 - lo3 == pytest.approx(hi1 - lo1, rel=0.01)       # triplicar filas no da más certeza
