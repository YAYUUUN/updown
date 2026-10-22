import numpy as np
import pandas as pd
import pytest

from updown import audit
from updown.features import dataset, feature_columns
from updown.hmm import HMM_COLUMNS, UNCERTAIN, GaussianHMM, StageHMM, hmm_observations, name_states


def _simulated_hmm(T=6000, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.97, 0.02, 0.01], [0.02, 0.97, 0.01], [0.01, 0.02, 0.97]])
    mu = np.array([[-1.0, 0.0], [0.5, 1.0], [1.5, -1.0]])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.choice(3, p=A[s[t - 1]])
    return mu[s] + rng.normal(0, 0.35, (T, 2)), s, mu


def test_hmm_recovers_simulated_states():
    x, s, mu = _simulated_hmm()
    m = GaussianHMM(3).fit([x[:3000], x[3000:]])
    match = [int(np.argmin(((m.means_ - mu[k]) ** 2).sum(axis=1))) for k in range(3)]
    assert sorted(match) == [0, 1, 2]
    assert np.allclose(m.means_[match], mu, atol=0.1)
    P = np.vstack(m.filter([x[:3000], x[3000:]]))
    assert (np.array(match)[s] == P.argmax(axis=1)).mean() > 0.95
    assert np.allclose(P.sum(axis=1), 1.0)
    h = np.array(m.loglik_history_)
    assert h[-1] > h[0] and np.all(np.diff(h) > -1e-5 * abs(h[0]))  # EM no empeora (salvo la regularización)


def test_hmm_filter_does_not_look_ahead():
    x, _, _ = _simulated_hmm(T=3000, seed=1)
    m = GaussianHMM(3).fit([x])
    t = 2000
    tampered = x.copy()
    tampered[t + 1:] = tampered[t + 1:][::-1] * 3       # otro futuro
    a, b = m.filter([x])[0], m.filter([tampered])[0]
    np.testing.assert_allclose(a[: t + 1], b[: t + 1])
    assert not np.allclose(a[t + 1:], b[t + 1:])


def test_missing_observations_carry_no_information():
    x, _, _ = _simulated_hmm(T=500, seed=2)
    m = GaussianHMM(3).fit([x])
    gap = x.copy()
    gap[200] = np.nan
    p = m.filter([gap])[0]
    np.testing.assert_allclose(p[200], p[199] @ m.transmat_)   # sin dato: solo la transición


def test_state_names_follow_their_characteristics():
    means = np.array([[1.5, -1.0], [-1.0, 0.1], [0.2, 1.2], [0.3, -1.3], [2.0, 0.5], [0.0, 0.2], [-1.0, 1.5]])
    assert name_states(means) == ["crisis", "calma", "alcista", "bajista", "volatil", "lateral", "alcista"]
    quiet = np.array([[-1.6, 0.7], [-1.5, 2.2], [-1.7, 0.1], [-1.5, 0.2]])   # nada volátil: ningún «volatil»
    assert name_states(quiet) == ["calma", "alcista", "calma", "calma"]


def test_states_in_the_same_zone_add_up():
    bars, _ = audit.stage_bars(n=2500, seed=3)
    obs = hmm_observations(bars)
    model = StageHMM(n_states=4).fit(obs.iloc[:1500])
    model.names_ = ["calma", "calma", "crisis", "crisis"]                 # dos estados por zona
    model.stages_ = ["calma", "crisis"]
    states = np.vstack(model.hmm_.filter([obs.dropna().to_numpy()]))
    P = model.proba(obs.dropna())
    np.testing.assert_allclose(P["calma"], states[:, 0] + states[:, 1])
    assert set(model.labels(obs.dropna()).dropna()) <= {"calma", "crisis", UNCERTAIN}


def test_hmm_stages_use_only_the_past():
    bars, _ = audit.stage_bars(n=2500, seed=3)
    data = dataset(bars, "1d", regimes=True)
    assert set(HMM_COLUMNS) <= set(data.columns)
    assert not set(HMM_COLUMNS) & set(feature_columns(data))          # no son variables de los modelos
    cut, t = data.index[1200], data.index[1800]
    model = StageHMM().fit(data[data.index <= cut])                    # parámetros: solo entrenamiento
    labels = model.labels(data)

    tampered = bars.copy()
    tampered.loc[tampered.index > t, ["open", "high", "low", "close"]] *= np.linspace(1, 0.4, (bars.index > t).sum())[:, None]
    data2 = dataset(tampered, "1d", regimes=True)
    pd.testing.assert_frame_equal(data[list(HMM_COLUMNS)].loc[:t], data2[list(HMM_COLUMNS)].loc[:t])
    refit = StageHMM().fit(data2[data2.index <= cut])
    np.testing.assert_allclose(refit.hmm_.means_, model.hmm_.means_)
    labels2 = refit.labels(data2)
    pd.testing.assert_series_equal(labels.loc[:t], labels2.loc[:t])
    assert not labels.loc[t:].equals(labels2.loc[t:])


def test_hmm_finds_planted_calm_and_turbulent_stages():
    bars, real = audit.stage_bars(n=3000, seed=1)
    obs = hmm_observations(bars)
    model = StageHMM(n_states=2).fit(obs.iloc[:1800])
    labels = model.labels(obs).iloc[1800:]
    real = real.iloc[1800:]
    assert set(model.names_) == {"calma", "crisis"}                    # la agitada cae: «crisis»
    assert (labels[real == "calma"] == "calma").mean() > 0.9
    assert (labels[real == "agitada"] == "crisis").mean() > 0.75
    assert (labels == UNCERTAIN).mean() < 0.15
    probs = model.proba(obs).iloc[1800:]
    assert np.allclose(probs.dropna().sum(axis=1), 1.0)


def test_pooled_stage_hmm_filters_each_symbol_separately():
    a, _ = audit.stage_bars(n=1500, seed=4)
    b, _ = audit.stage_bars(n=1500, seed=5)
    pooled = pd.concat([hmm_observations(a).assign(symbol="A"), hmm_observations(b).assign(symbol="B")]).sort_index()
    model = StageHMM(n_states=2).fit(pooled)
    alone = model.proba(pooled[pooled["symbol"] == "A"])
    together = model.proba(pooled)[pooled["symbol"].to_numpy() == "A"]
    pd.testing.assert_frame_equal(alone, together)


def test_inner_validation_is_the_purged_end_of_training():
    from updown.stages import inner_split
    times = pd.date_range("2026-01-01", periods=1000, freq="15min", tz="UTC")
    fit_t, val_t = inner_split(times, 0.2, purge=4)
    assert len(val_t) == 200 and val_t.min() == times[800] and val_t.max() == times[-1]
    assert fit_t.max() == times[795]                                     # 4 velas de purga


def test_stage_choice_ignores_the_test_period():
    from updown.evaluate import time_folds
    from updown.stages import evaluate_stages
    bars, _ = audit.stage_bars(n=2500, seed=8)
    data = dataset(bars, "1d", regimes=True)
    kw = dict(models=("ridge", "lasso"), n_folds=3, min_train=150, min_val=60)
    rep = evaluate_stages(data, **kw)
    train_t, _ = next(time_folds(data[data["up"].notna()].index, 3, 0.4, 1))

    tampered = data.copy()
    future = tampered.index > train_t.max()                              # prueba del 1.er bloque y después
    feats = tampered.attrs["features"]
    tampered.loc[future, "up"] = 1 - tampered.loc[future, "up"]          # otros resultados
    tampered.loc[future, "fwd_ret"] *= -1
    tampered.loc[future, feats] = tampered.loc[future, feats].to_numpy()[::-1]   # otras variables
    tampered.loc[future, ["hmm_vol", "hmm_tend"]] *= -1                  # otras etapas
    tampered.loc[future, "regimen"] = "calma"
    rep2 = evaluate_stages(tampered, **kw)

    first = lambda df: df[df["bloque"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(first(rep.choices), first(rep2.choices))
    pd.testing.assert_frame_equal(first(rep.val_rank), first(rep2.val_rank))
    assert not np.allclose(rep.oos["por_etapas_hmm"]["p"], rep2.oos["por_etapas_hmm"]["p"])


def test_stage_model_learns_a_rule_that_flips_between_stages():
    from updown.stages import evaluate_stages
    bars, real = audit.stage_bars(n=3000, seed=6, vol=(0.01, 0.01), drift=(0.0, 0.0), rho=(0.4, -0.4))
    data = dataset(bars, "1d", regimes=True)
    data["etapa_real"] = real.reindex(data.index)                        # etapa conocida: prueba la mecánica
    rep = evaluate_stages(data, ("ridge",), sources=("etapa_real",), n_folds=3, min_train=300, min_val=150)
    vs = rep.comparison.set_index("rival").loc["general_validado"]
    assert vs["mejora_brier"] > 0 and vs["p_valor"] < 0.01
    assert rep.metrics.loc["por_etapas_etapa_real", "auc"] > rep.metrics.loc["general_validado", "auc"] + 0.03
    assert set(rep.metrics.index) >= {"siempre_sube", "momentum", "ridge", "general_validado"}
    assert rep.metrics.loc["por_etapas_etapa_real", "p_auc_corregido"] >= rep.metrics.loc["por_etapas_etapa_real", "p_auc"]


def test_small_stages_fall_back_to_the_general_model():
    from updown.stages import evaluate_stages
    bars, _ = audit.stage_bars(n=2000, seed=9)
    data = dataset(bars, "1d", regimes=True)
    rep = evaluate_stages(data, ("ridge",), sources=("reglas",), n_folds=3, min_train=10 ** 9)
    assert (rep.choices["motivo"] != "validación").all()
    np.testing.assert_allclose(rep.oos["por_etapas_reglas"]["p"], rep.oos["general_validado"]["p"])
