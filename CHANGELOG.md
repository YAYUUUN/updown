# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

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
