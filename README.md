# updown

**Will the next candle close higher?** `updown` estimates that probability with 13 classical models and a transformer, using free public market data, and puts an honest out-of-sample track record next to every number.

Most "price prediction" projects report a nice backtest. This one is built the other way round: first an evaluator that is hard to fool (and is tested to catch its own mistakes), then the models. In liquid markets the usual answer is "no demonstrated edge", and the tool says so.

> Research and educational tool. **Not financial advice**, not a trading system. It does not place orders and the web app shows historical evaluation only.

*Documentación en español: [README.es.md](README.es.md).*

![AUC per candle](docs/auc_per_candle.png)

## What it found

On BTC, ETH and SOL (Binance, 15-minute candles, about 2 years, ~210,000 rows, evaluated out of sample, no costs):

- **The next candle is slightly predictable, and the edge fades with distance.** Out-of-sample AUC is about 0.54 for the 1st candle, 0.52 for the 2nd and 0.51–0.52 for the 4th (0.50 is chance). Hit rate goes from ~53% to ~51%.
- **All models beat both naive baselines** ("always up" and one-line momentum) on every target, after Holm correction and in at least 4 of 6 walk-forward blocks.
- **Random forest and gradient boosting are best overall; the transformer ties ridge.** The transformer is the best model in the *crisis* regime.
- **Predictability depends on the market regime:** weakest right after news-like shocks (AUC ~0.53), highest in calm and sideways markets (~0.55).
- **On daily US equities (SPY since 1993) and hourly ETFs, nothing beats "always up".**

![AUC by regime](docs/auc_by_regime.png)

Full tables: [`estudios/resultados_velas_regimenes.md`](estudios/resultados_velas_regimenes.md), [`estudios/resultados_comparativa.md`](estudios/resultados_comparativa.md), [`estudios/resultados_transformer_15m.md`](estudios/resultados_transformer_15m.md).

## How it avoids fooling itself

| Safeguard | Why |
|---|---|
| Walk-forward with an expanding window and a purge of `horizon` bars | Every prediction comes from a model trained only on the past; the purge stops the last training target from peeking into the test block. |
| Only closed candles | The last candle a feed returns is usually still moving; using it leaks the future. |
| Brier score against "always up", not accuracy | If prices rise 55% of the time, saying "up" every time is already 55% accurate. |
| Holm-corrected AUC p-values | With 13 models, one of them looks "significant" at 5% by luck almost half the time. |
| Effective sample size n/h for overlapping targets | Consecutive 1-hour targets share 3 of their 4 candles; ignoring that makes p-values optimistic. |
| Consistency: beat the baseline in ≥ 2/3 of blocks | One lucky period is not a signal. |
| Synthetic audit (`updown auditoria`, run in CI) | On a random walk the verdict must be "no edge"; a planted signal must be found; an injected future value must make AUC jump. |
| Prediction log written before outcomes (`updown vivo`) | The only track record that cannot be tuned after the fact. |

## Install

```bash
git clone https://github.com/YAYUUUN/updown && cd updown
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[app,dev]"                             # add ,deep for the transformer (PyTorch)
python -m pytest -q
```

For a CUDA build of PyTorch, follow [pytorch.org](https://pytorch.org/get-started/locally/) (e.g. `--index-url https://download.pytorch.org/whl/cu128`).

## Use

The command names are in Spanish:

```bash
updown auditoria                                            # check the evaluator on synthetic data
updown evaluar SPY --intervalo 1d                           # 13 models, walk-forward
updown evaluar BTCUSDT,ETHUSDT --fuente binance --intervalo 15m --horizonte 4
python -m estudios.velas_regimenes                          # per-candle and per-regime study
streamlit run app.py                                        # web app
```

The **web app** shows which regime each market is in right now and, with a regime selector, which model predicts each of the next four candles best in that regime.

## Models

Baselines: `siempre_sube` (training up-frequency) and `momentum` (P(up | last candle up/down)).

| Family | Models |
|---|---|
| Linear | `lineal` (logistic), `ridge`, `lasso`, `elastic_net`, `lda`, `svm_lineal` |
| Probabilistic / neighbours | `naive_bayes`, `knn` (6 fixed features) |
| Non-linear | `random_forest`, `extra_trees`, `gradient_boosting`, `red_neuronal` (MLP) |
| Ensemble | `ensemble` (mean of ridge, lasso, random forest, gradient boosting) |
| Deep | `transformer`: 2-layer encoder over the last 64 candles (return, range, body, relative volume) plus the tabular features; one network predicts all four candles and the full hour. |

Hyperparameters are fixed in advance; tuning them on results would be more trials to correct for. About 26 scale-free features (multi-horizon returns, volatility, RSI, MACD, Bollinger, ATR, range position, gap, relative volume, calendar, and a reference market such as SPY or BTC), so several assets can train a single model.

## Market regimes

Every candle gets one label computed from past data only. News is not read; its footprint in price and volume is.

| Regime | Rule |
|---|---|
| News / shock | a move above 4σ or volume 3σ above normal in the last few candles |
| Crisis | volatility in the top 20% of its past year and price falling |
| Calm | volatility in the bottom 30% of its past year and no clear trend |
| Bull / Bear | 30-day trend and position versus the 30-day mean |
| Sideways | everything else |

A regime is detected once it has started, never before.

## Data

| Source | What | Depth |
|---|---|---|
| Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) | stocks, ETFs, indices, FX, crypto | 1m: 7 days · 5–30m: 60 days · 1h: 730 days · 1d: full history |
| Binance public REST API | crypto, real time | as far back as requested (paginated) |

Downloads are cached in `~/.cache/updown` (or `$UPDOWN_CACHE`); Binance updates fetch only new candles. `yfinance` is an unofficial client intended for personal and research use: check Yahoo's terms before anything else. Binance is not available in every country. No market data is shipped with this repository.

## Limitations

- Statistically real is not economically useful: a 53% hit rate on 15-minute candles is small next to trading costs, which this study deliberately ignores.
- The large-cap stock universe in the comparison is today's large caps (survivorship bias).
- Two years of crypto data cover a limited set of market conditions; nothing here guarantees the next regime behaves the same.

## Project layout

```
updown/       data.py · features.py · regimes.py · models.py · transformer.py
              evaluate.py · candles.py · predict.py · audit.py · __main__.py (CLI)
app.py        Streamlit app
estudios/     reproducible studies and their results
tests/        offline test suite (synthetic data, mocked API)
```

## License

MIT © Jiajun Wang
