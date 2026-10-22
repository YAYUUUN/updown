import numpy as np
import pandas as pd
import pytest

from updown import audit
from updown.data import drop_unclosed, parse_binance_klines
from updown.evaluate import holm, time_folds, walk_forward
from updown.features import build_features, build_target, dataset, model_columns, sequence_features
from updown.models import BASELINES, MODELS, make_model


@pytest.fixture(scope="module")
def bars():
    return audit.synthetic_bars(n=600, rho=0.1, seed=7)


def test_features_do_not_look_ahead(bars):
    t = 400
    original = build_features(bars, "1d").iloc[: t + 1]
    tampered = bars.copy()
    tampered.iloc[t + 1:] *= 3.0  # cambiar el futuro no puede mover el pasado
    changed = build_features(tampered, "1d").iloc[: t + 1]
    pd.testing.assert_frame_equal(original, changed)


def test_target_is_next_close_direction(bars):
    y, fwd = build_target(bars, horizon=1)
    c = bars["close"]
    assert y.iloc[10] == float(c.iloc[11] > c.iloc[10])
    assert np.isclose(fwd.iloc[10], np.log(c.iloc[11] / c.iloc[10]))
    assert np.isnan(y.iloc[-1]) and np.isnan(fwd.iloc[-1])


def test_folds_are_chronological_and_purged(bars):
    times = bars.index
    for train_t, test_t in time_folds(times, n_folds=5, min_train_frac=0.4, purge=3):
        gap = times.get_loc(test_t.min()) - times.get_loc(train_t.max())
        assert gap > 3


@pytest.mark.parametrize("name", BASELINES + MODELS)
def test_every_model_returns_probabilities(name, bars):
    data = dataset(bars, "1d")
    cols = data.attrs["features"]
    train = data[data["up"].notna()].iloc[:400]
    test = data.iloc[400:450]
    p = make_model(name).fit(train[cols], train["up"]).proba(test[cols])
    assert p.shape == (len(test),)
    assert np.all((p >= 0) & (p <= 1))


def test_walk_forward_only_predicts_after_training(bars):
    data = dataset(bars, "1d")
    oos = walk_forward(data, "ridge", n_folds=4)
    assert oos.index.min() > data.index[int(len(data[data['up'].notna()]) * 0.4) - 1]
    assert set(oos.columns) >= {"p", "up", "fwd_ret", "fold"}


def test_audit_tripwires_pass():
    assert all(audit.run(verbose=False).values())


def test_holm_matches_textbook_example():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adj = holm(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monótono: nunca menor que el anterior


def test_drop_unclosed_intraday_removes_open_bar():
    idx = pd.date_range("2026-01-05 14:00", periods=4, freq="1h", tz="UTC")
    b = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    out = drop_unclosed(b, "1h", now=pd.Timestamp("2026-01-05 17:30", tz="UTC"))
    assert out.index[-1] == pd.Timestamp("2026-01-05 16:00", tz="UTC")


def test_drop_unclosed_daily_uses_session_close():
    idx = pd.date_range("2026-01-05", periods=3, freq="D", tz="America/New_York")
    b = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    b.attrs["session_end"] = pd.Timestamp("2026-01-07 16:00", tz="America/New_York").tz_convert("UTC")
    before = drop_unclosed(b, "1d", now=pd.Timestamp("2026-01-07 15:00", tz="America/New_York"))
    after = drop_unclosed(b, "1d", now=pd.Timestamp("2026-01-07 16:30", tz="America/New_York"))
    assert before.index[-1].day == 6
    assert after.index[-1].day == 7


def test_parse_binance_klines():
    rows = [[1767225600000, "100", "110", "90", "105", "12.5", 1767229199999, "0", 10, "7.5", "0", "0"],
            [1767229200000, "105", "106", "100", "101", "8.0", 1767232799999, "0", 7, "2.0", "0", "0"]]
    df = parse_binance_klines(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "trades", "taker_buy"]
    assert df["close"].tolist() == [105.0, 101.0]
    assert df["trades"].tolist() == [10.0, 7.0] and df["taker_buy"].tolist() == [7.5, 2.0]
    assert str(df.index.tz) == "UTC"


def _with_order_flow(bars, seed=0):
    rng = np.random.default_rng(seed)
    out = bars.copy()
    out["taker_buy"] = out["volume"] * rng.uniform(0.3, 0.7, len(out))
    out["trades"] = rng.integers(100, 1000, len(out)).astype(float)
    return out


def test_order_flow_features(bars):
    b = _with_order_flow(audit.synthetic_bars(n=400, seed=3).set_axis(
        pd.date_range("2026-01-01", periods=400, freq="15min", tz="UTC")))
    b.iloc[10, b.columns.get_loc("volume")] = 0.0
    f = build_features(b, "15m")
    i = 300
    assert f["taker_ratio"].iloc[i] == pytest.approx(b["taker_buy"].iloc[i] / b["volume"].iloc[i])
    assert f["taker_ratio"].iloc[10] == 0.5                       # sin volumen: equilibrio
    hour = b.iloc[i - 3:i + 1]                                     # 4 velas de 15 min
    assert f["taker_ratio_1h"].iloc[i] == pytest.approx(hour["taker_buy"].sum() / hour["volume"].sum())
    day = np.log1p(b["trades"].iloc[i - 95:i + 1])
    assert f["trades_z"].iloc[i] == pytest.approx((day.iloc[-1] - day.mean()) / day.std())
    assert "taker_ratio" not in build_features(bars, "1d")         # Yahoo no trae flujo de órdenes
    tampered = b.copy()
    tampered.iloc[i + 1:, tampered.columns.get_loc("taker_buy")] *= 0.1
    pd.testing.assert_frame_equal(f.iloc[: i + 1], build_features(tampered, "15m").iloc[: i + 1])


def test_transformer_window_gets_taker_channel(bars):
    b = _with_order_flow(bars)
    s = sequence_features(b, 4)
    assert s["seq_tb_2"].iloc[100] == pytest.approx(b["taker_buy"].iloc[98] / b["volume"].iloc[98] - 0.5)
    assert not any(c.startswith("seq_tb_") for c in sequence_features(bars, 4).columns)


def test_sequence_window_does_not_look_ahead(bars):
    t = 300
    a = sequence_features(bars, 16).iloc[: t + 1]
    tampered = bars.copy()
    tampered.iloc[t + 1:] *= 2.0
    b = sequence_features(tampered, 16).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)
    assert a["seq_r_3"].iloc[t] == pytest.approx(np.log(bars["close"].iloc[t - 3] / bars["close"].iloc[t - 4]))


def test_transformer_returns_probabilities(bars):
    pytest.importorskip("torch")
    data = dataset(bars, "1d", seq_len=16)
    model = make_model("transformer")
    model.epochs = 2
    cols = model_columns(data, model)
    assert any(c.startswith("seq_") for c in cols)
    train = data[data["up"].notna()].iloc[:400]
    p = model.fit(train[cols], train["up"]).proba(data.iloc[400:450][cols])
    assert p.shape == (50,) and np.all((p > 0) & (p < 1))


def test_sequence_columns_hidden_from_tabular_models(bars):
    data = dataset(bars, "1d", seq_len=8)
    assert not any(c.startswith("seq_") for c in model_columns(data, make_model("ridge")))


def test_regimes_use_only_the_past(bars):
    from updown.regimes import detect_regimes
    long = audit.synthetic_bars(n=1500, rho=0.0, seed=9)
    t = 1200
    a = detect_regimes(long).iloc[: t + 1]
    tampered = long.copy()
    tampered.iloc[t + 1:] *= 0.3  # una "crisis" futura no puede cambiar etiquetas pasadas
    b = detect_regimes(tampered).iloc[: t + 1]
    pd.testing.assert_series_equal(a, b)
    assert set(a.dropna().unique()) <= {"shock", "crisis", "calma", "alcista", "bajista", "lateral"}


def test_shock_regime_fires_on_a_jump():
    from updown.regimes import detect_regimes
    b = audit.synthetic_bars(n=1500, rho=0.0, seed=10)
    b.iloc[1400:, :4] *= 1.15  # salto del 15% de golpe
    assert detect_regimes(b).iloc[1400] == "shock"


def test_candle_targets_are_single_candles(bars):
    data = dataset(bars, "1d", horizon=4, candles=4)
    c = np.log(bars["close"])
    t = data.index[100]
    i = bars.index.get_loc(t)
    assert data.loc[t, "fwd_v3"] == pytest.approx(c.iloc[i + 3] - c.iloc[i + 2])
    assert data.loc[t, "up"] == float(c.iloc[i + 4] > c.iloc[i])
    assert not any(col.startswith(("up_v", "fwd_v")) for col in data.attrs["features"])


def test_evaluate_candles_finds_planted_first_candle():
    from updown.candles import evaluate_candles
    d = dataset(audit.synthetic_bars(n=3000, rho=0.2, seed=5), "1d", horizon=4, candles=4, regimes=True)
    rep = evaluate_candles(d, ("ridge",), n_folds=4)
    auc = rep.pivot("auc")
    assert auc.loc["ridge", "vela_1"] > auc.loc["ridge", "vela_3"]


class _FakeBinance:
    """Imita /api/v3/klines con velas de 1 h desde t0; `now` marca hasta dónde existen."""

    def __init__(self, n_now):
        self.t0, self.step, self.n_now, self.calls = 1767225600000, 3_600_000, n_now, []

    def _row(self, i):
        t = self.t0 + i * self.step
        return [t, "1", "1", "1", str(100 + i), "1", t + self.step - 1, "0", 1, "0", "0", "0"]

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        lim = params["limit"]
        if "startTime" in params:
            i0 = (params["startTime"] - self.t0) // self.step
            idx = range(i0, min(i0 + lim, self.n_now))
        else:
            end = self.n_now if "endTime" not in params else (params["endTime"] - self.t0) // self.step + 1
            idx = range(max(0, end - lim), end)
        rows = [self._row(i) for i in idx]

        class R:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self, rows=rows):
                return rows
        return R()


def test_binance_cache_only_downloads_new_bars(tmp_path, monkeypatch):
    from updown.data import Market, fetch
    monkeypatch.setenv("UPDOWN_CACHE", str(tmp_path))
    m = Market("binance", "BTCUSDT", "1h")
    api = _FakeBinance(n_now=1500)
    first = fetch(m, bars=1200, session=api)
    assert len(first) == 1200 and first["close"].iloc[-1] == 100 + 1499
    api.n_now, api.calls = 1510, []
    second = fetch(m, bars=1200, session=api)
    assert all("startTime" in c for c in api.calls)       # solo pide lo nuevo
    assert len(second) == 1200 and second["close"].iloc[-1] == 100 + 1509
    assert second.index.is_unique and second.index.is_monotonic_increasing


def test_old_cache_without_version_is_downloaded_again(tmp_path, monkeypatch):
    import pickle
    import time

    from updown.data import Market, _cache_path, fetch
    monkeypatch.setenv("UPDOWN_CACHE", str(tmp_path))
    m = Market("binance", "BTCUSDT", "1h")
    api = _FakeBinance(n_now=1500)
    old = fetch(m, bars=1200, session=api)[["open", "high", "low", "close", "volume"]]  # como las de antes
    with open(_cache_path(m), "wb") as fh:
        pickle.dump((old, {}, time.time()), fh)                                         # formato sin versión
    api.calls = []
    df = fetch(m, bars=1200, session=api)
    assert not any("startTime" in c for c in api.calls)    # descarga completa, no incremental
    assert {"trades", "taker_buy"} <= set(df.columns) and df["taker_buy"].notna().all()


@pytest.mark.parametrize(("name", "expected"), [("lasso", 1.0), ("elastic_net", 0.5)])
def test_sparse_models_really_use_l1(name, expected, bars):
    """En scikit-learn < 1.8, l1_ratio sin penalty se ignora y el modelo queda en L2 sin avisar."""
    data = dataset(bars, "1d")
    cols = data.attrs["features"]
    train = data[data["up"].notna()]
    est = make_model(name).fit(train[cols], train["up"]).estimator_[-1]
    if est.penalty == "l1":
        assert expected == 1.0
    else:  # "elasticnet" en scikit-learn < 1.8; en >= 1.8 manda l1_ratio
        assert est.penalty != "l2"
        assert est.l1_ratio == expected
    if name == "lasso":
        assert (est.coef_ == 0).any()  # L1 de verdad: descarta variables
