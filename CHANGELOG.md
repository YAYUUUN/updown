# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Web demo: "How it did on the latest candles", a 15-minute candlestick chart of BTC, ETH and SOL with the
  prediction made for each candle and whether it was right, a hover tooltip (also with the arrow keys), a table
  view and the hit rate of the last days per market and model, with its 95% interval. Each candle uses the model
  that was published at the time, and only candles that model had not seen while training are scored
  (`web/js/history.js`).
- `estudios/web_modelos.py` also exports the models that would have been published a week earlier
  (`previous` in `models.json`) and the last candle the models have seen (`data_to`), so the chart keeps a
  week of history right after the weekly retrain. Older `models.json` files still work.
- `tests/test_web.py` checks the page's history against Python: no prediction for a candle the model had seen,
  the right model for each period and the same probabilities and outcomes.
- Web demo: the market stage of BTC, ETH and SOL (calm, bull, bear, sideways, crisis or volatile) with how sure the
  hidden Markov model is, a stage ribbon over the latest candles (hatched when no stage reaches 90%), the hit rate
  by stage in the window and each model's out-of-sample hit rate in the current stage. The page now downloads
  about 5,000 candles per market (30 days of trend need them) and then only the new ones (`web/js/stages.js`).
- `updown/export.py` exports the stage HMM and computes its observations as the page does, with the past-year
  volatility reference fixed at training (`hmm.vol_reference`); `estudios/web_modelos.py` exports it for the
  current and the previous week and measures each model's out-of-sample hit rate by stage, with stages from an
  HMM fitted only on earlier data in each walk-forward block (`hmm.stage_labels_by_period`). A test checks the
  page's stage probabilities against Python.

## [0.2.0] - 2026-10-24

### Added
- Web demo on GitHub Pages (`web/`): BTC, ETH and SOL probabilities computed in the browser from Binance's
  public candles with `ridge` and `gradient_boosting`, each next to that model's out-of-sample track record,
  plus the study results as charts and a "run it on your computer" section. No server, no analytics.
- `updown/export.py`: exports trained `ridge` and `gradient_boosting` models as plain JSON (coefficients and
  trees) that give the same probabilities as scikit-learn.
- `estudios/web_modelos.py` (evaluate, retrain and export the web models) and `estudios/web_resultados.py`
  (study tables as JSON). A Pages workflow deploys the site and retrains the models every Monday.
- `tests/test_web.py`: runs the page's JavaScript with Node and checks its features and probabilities
  against Python. CI now sets up Node.
- `UPDOWN_BINANCE_URL` to use Binance's market-data mirror where `api.binance.com` does not answer.

## [0.1.1] - 2026-10-23

### Fixed
- `lasso` and `elastic_net` silently trained as L2 (ridge-like) models on scikit-learn older than 1.8, which is
  what pip installs on Python 3.10: `l1_ratio` was ignored without `penalty`. The penalty is now set explicitly
  on older versions, and a test checks that both models really use L1.

### Changed
- CI also tests Python 3.10, the minimum version the package declares.

## [0.1.0] - 2026-10-23

First public release.

### Added
- Market data from Yahoo Finance and the Binance public API, closed candles only, with an on-disk cache that
  fetches only new Binance candles.
- Scale-free features, including order flow (aggressive-buy share, trade count) on Binance data.
- 13 classical models, two naive baselines and a transformer over the last 64 candles that predicts each of the
  next four candles and the full hour at once.
- Purged walk-forward evaluation: Brier skill against "always up", Holm-corrected AUC, effective sample size
  for overlapping targets, block consistency and an optional cost simulation.
- Synthetic audit of the evaluator (random walk, planted signal, leakage canary), run in CI.
- Market stages: rule-based labels and a hidden Markov model filtered with past data only, per-stage
  submodels chosen on validation, and hit rate by confidence.
- Streamlit app with a stage selector, per-stage rankings and the confidence curve.
- Command line with Spanish and English names (`evaluar`/`evaluate`, `--intervalo`/`--interval`...), and a
  clear message when a download fails.
- Reproducible studies in `estudios/` with their results.

### Fixed
- The live prediction log compared the outcome with the close stored in the log; if that value was
  missing, every prediction was scored as "down". It now uses the close of the predicted candle.
