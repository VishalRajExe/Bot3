# Backtesting methodology

A backtest here is a *measurement instrument*, not a marketing artifact. The engine is
deliberately conservative: when a choice exists between a flattering and a pessimistic
assumption, it takes the pessimistic one.

## 1. Execution model

| Step | Rule |
|---|---|
| Decision time | Close of bar *i*. All indicator inputs use bars ≤ *i* only. |
| Order creation | The strategy's signal is converted to a market order for the **next** bar. |
| Fill time | Open of bar *i+1* (never the close that produced the signal). |
| Fill price | Open ± (half-spread + slippage), adverse to the trade direction. |
| Fees | 10 bps taker fee on the filled notional, both sides. |
| Stops | Checked against bar *i+1…*'s high/low before new signals; a gap through the stop fills at the open. |
| End of window | Position is closed at the final close only for backtests; this fill is flagged `end_of_period` and marked as an anomaly in the trade record. Paper replay seeds set `close_at_end=false` so a held position is carried into live trading instead of being fake-closed. |
| Rejections | Min notional, quantity step, insufficient cash, no valid stop — the order is stored with a reason, never silently dropped. |

The cost model (`costs` in `config.json`):

```
round-trip drag = 2 × 10 bps fee + 2 × (1 bp half-spread + 5 bps slippage) ≈ 32 bps
```

Buy & hold is charged the same entry costs, so the benchmark is not artificially strong.

## 2. Look-ahead controls

1. **Causal indicators.** EMA is recursive (`adjust=False`); rolling σ uses the trailing window;
   ATR/RSI/ADX use trailing windows only.
2. **Shifted channels.** Donchian highs/lows, breakouts and Bollinger bands used for entry are
   shifted by one bar so the current bar's own high cannot trigger its own breakout.
3. **Regime labels** are computed from data up to and including the current bar, never centred
   or forward-filled.
4. **No future fills.** Signals produce orders for the next bar; a strategy cannot see the bar
   it will be filled in.
5. **Warm-up.** Each strategy declares `warmup_bars`; evaluation starts only after the slowest
   indicator has a full window (min 65 bars for the EMA-65 baseline, 202 for a 200-EMA filter).
6. **Tests.** `test_no_lookahead_signal_matches_truncated_history` re-runs every strategy on a
   truncated frame and asserts the signals on the common prefix are identical, and
   `test_engine_does_not_use_signal_bar_price` checks that the fill cannot be taken at the bar
   whose close produced the signal.

## 3. Evaluation windows

| Purpose | Window |
|---|---|
| Full-period research | 2017-10-23 → 2026-10-20 (daily), 2017-08-28 → 2026-10-20 (4h) |
| Out-of-sample verdicts | 2024-01-19 → 2026-10-20 (last ~20% of bars), minimum 20 OOS trades |
| ML validation | expanding walk-forward, 700-bar minimum train, 120-bar test blocks, 5-bar purge + 5-bar embargo |

Rules:

* A variant is developed on the in-sample window only. The OOS window is evaluated once per
  variant and the verdict is written to `reports/experiments.md` before interpretation.
* An experiment is **accepted** only if OOS net return beats the baseline with no material
  drawdown regression and at least 20 OOS trades. Fewer trades ⇒ **inconclusive**, regardless
  of the return number.
* No repeated re-optimisation against the same OOS window. If the window is reused for a new
  round of variants, that is a new study and must be stated as such.

## 4. Metrics computed

Initial/final equity, net return, CAGR, max drawdown and its duration, volatility, Sharpe,
Sortino, Calmar, trade count, win rate, profit factor, average win/loss, expectancy per trade,
average R-multiple, average holding time, total fees, total slippage, exposure, turnover, period
length, monthly returns, per-regime breakdown, and benchmark comparisons (BTC buy & hold, cash).

Ratio caveats: Sharpe/Sortino are computed from bar returns and annualised; with fewer than ~20
trades they are indicative at best. Win rate is never reported without profit factor and
expectancy.

## 5. Benchmarks

* **BTC buy & hold** — one entry at the first bar open, same fee/spread/slippage on entry,
  no rebalancing, no exit. Its drawdown over the full window is −83.9%.
* **Cash** — uninvested quote currency, 0% (no risk-free yield modelled).
* **The project baseline** — `ema_std` with its default parameters, always available for
  comparison.
* **A simple technical strategy** — `trend_ema` (50/200 cross) is the "boring" reference.

## 6. Reproducibility

Every run stores, in `backtest_runs` and `reports/backtests/<run_id>/metrics.json`:

```
run_id, strategy, version, params, period.start/end, dataset_hash, dataset_bars,
costs (fee/spread/slippage/min notional/step/tick), risk limits,
metrics, benchmark, notes, code_version
```

Re-running the same configuration on the same dataset hash reproduces the same metrics; the
suite includes a reproducibility test that runs a backtest twice and compares metrics.

Artifacts per run:

```
SUMMARY.md       human-readable summary with params, costs, metrics, benchmarks
metrics.json     machine-readable, includes monthly returns and regime breakdown
trades.csv       one row per round trip (entry/exit price+time, reason, fees, R, MFE/MAE, regime)
equity.csv       per-bar equity, cash, position, price, drawdown
chart.html       candlesticks with EMA/σ bands and entry/exit markers
performance.html equity curve and drawdown
```

`reports/backtests/comparison.md` aggregates the latest run of each strategy; duplicates in the
file come from repeated comparison runs and are harmless.

## 7. Known limitations

1. **Fills are optimistic about liquidity.** A market order at the next open with 5 bps slippage
   is fine for small size on BTC/USDT, not for large size or stressed markets.
2. **No partial fills, no queue position, no exchange downtime** inside the simulation; outages
   appear as gaps in the data and are handled by refusing to trade stale bars.
3. **Single asset, one cycle-heavy sample.** Nine years is two BTC cycles; results are
   conditional on that history.
4. **Parameter search is limited on purpose** (≤ 10 variants per family) to avoid the "best of
   many noise fits" failure mode, but the multiple-comparison risk is not zero.
5. **Regime attribution is approximate** (rolling labels, not ex-post regimes).
6. **Survivorship of the venue** — Binance is the only validated data source.

## 8. How to reproduce the published numbers

```bash
./.venv/Scripts/python.exe -m btcbot.cli fetch
./.venv/Scripts/python.exe -m btcbot.cli compare
./.venv/Scripts/python.exe -m btcbot.cli experiments
./.venv/Scripts/python.exe -m btcbot.cli report
```

`compare` writes the strategy table; `experiments` writes `reports/experiments.md` (the OOS
verdicts); `report` writes per-run artifacts and `comparison.md`.
