"""Flujos completos sin red: predicción, registro en vivo y línea de comandos (con datos sintéticos)."""
import pandas as pd
import pytest

from updown import __main__ as cli
from updown import audit
from updown.data import DataError, Market
from updown.features import dataset
from updown.predict import live_scoreboard, log_path, log_prediction, predict_now, target_label, update_outcomes


@pytest.fixture(scope="module")
def bars():
    return audit.synthetic_bars(n=700, rho=0.1, seed=21)


def test_predict_now_returns_one_probability_per_model(bars):
    data = dataset(bars, "1d")
    pred = predict_now(Market("yahoo", "SYN", "1d"), 1, ("ridge", "random_forest"), data=data)
    assert set(pred.probs) == {"siempre_sube", "ridge", "random_forest"}
    assert all(0 < p < 1 for p in pred.probs.values())
    assert pred.bar_time == data.index[-1]
    assert target_label(pred) == "al cierre de la sesión siguiente"


def test_prediction_log_is_written_before_and_completed_after(bars, tmp_path):
    market = Market("yahoo", "SYN", "1d")
    path = log_path(str(tmp_path), market, 1)
    pred = predict_now(market, 1, ("ridge",), data=dataset(bars.iloc[:500], "1d"))
    assert log_prediction(path, pred)
    assert not log_prediction(path, pred)  # la misma vela no se apunta dos veces
    pending = pd.read_csv(path)
    assert pending["sube_real"].isna().all()  # antes de conocer el resultado

    log = update_outcomes(path, bars, horizon=1)
    i = bars.index.get_loc(pred.bar_time)
    assert log.loc[0, "close_real"] == pytest.approx(bars["close"].iloc[i + 1])
    assert log.loc[0, "sube_real"] == float(bars["close"].iloc[i + 1] > bars["close"].iloc[i])
    board = live_scoreboard(log)
    assert board.loc["ridge", "n"] == 1


def test_english_cli_alias_runs_an_evaluation(bars, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dataset", lambda market, horizon, context: dataset(bars, "1d", horizon))
    code = cli.main(["evaluate", "SYN", "--interval", "1d", "--models", "ridge", "--folds", "3", "--no-context"])
    out = capsys.readouterr().out
    assert code == 0
    assert "VEREDICTO" in out and "ridge" in out


def test_spanish_and_english_names_parse_the_same():
    es = cli.build_parser().parse_args(["evaluar", "BTCUSDT", "--fuente", "binance", "--intervalo", "15m",
                                        "--horizonte", "4", "--agrupado"])
    en = cli.build_parser().parse_args(["evaluate", "BTCUSDT", "--source", "binance", "--interval", "15m",
                                        "--horizon", "4", "--pooled"])
    for key in ("fuente", "intervalo", "horizonte", "agrupado", "simbolos"):
        assert getattr(es, key) == getattr(en, key)
    assert es.func is en.func


def test_download_errors_are_explained(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise DataError("Binance rechazó 'XYZ'")

    monkeypatch.setattr(cli, "load_dataset", fail)
    code = cli.main(["evaluar", "XYZ", "--fuente", "binance"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Binance no está disponible en todos los países" in err
