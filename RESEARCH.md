# Research record

All findings below were gathered during this build (October 2026) from the repositories,
documentation and APIs cited. Numerical results are produced by this repository and are stored
under `reports/`; nothing here is copied from a vendor claim. Sources should be re-checked
before redistribution, because licences and APIs change.

---

## 1. Starting point

The workspace was empty: no `PRD.md`, no strategy notes, no Freqtrade checkout, no transcripts
and no git history (`git status` failed — not a repository). The brief's reference to "existing
project materials" describing a 65-period EMA / 65-period standard-deviation BTC strategy
therefore had no local source. That absence is itself a finding: **the baseline had to be
reconstructed**, and it is labelled as a reconstruction everywhere it appears.

## 2. Framework comparison

| | Freqtrade | Jesse | Hummingbot | Own engine (chosen) |
|---|---|---|---|---|
| Licence | GPL-3.0-or-later | MIT | Apache-2.0 | — |
| Source | github.com/freqtrade/freqtrade | github.com/jesse-ai/jesse | github.com/hummingbot/hummingbot | this repo |
| Paper trading on public data | yes (`dry_run`) | yes | yes (paper trade mode) | yes |
| Data download | strong (many exchanges/timeframes) | good | market-data connectors | Binance klines only |
| Backtest realism | fees, slippage, spread options | fees + slippage | fee models, order-book oriented | fee + spread + slippage per side, next-bar fills |
| Look-ahead safeguards | documented `startup_candle_count` rules | documented | n/a (live-first) | causal indicators + shifted channels + tests |
| Strategy interface | Python class with `populate_*` | Python class | script/controller | small `Strategy` protocol |
| UI | FreqUI | dashboard | Hummingbot client | custom dashboard |
| Docker required | recommended | no (needs Postgres) | recommended | no |
| Dependency weight | ~45 packages | large (Postgres, Redis-ish stack) | very large | 10 runtime packages |
| Maintenance | very active | active | very active | n/a |
| Python 3.13/3.14 | historically lags new CPython releases | unclear | unclear | runs on 3.14.6 |

**Decision.** Build a small engine rather than embed a framework:

1. **Licence.** Freqtrade is GPL-3.0-or-later. Importing or vendoring its engine would make the
   combined work GPL. This project is intended to stay permissively licensed and easy to audit,
   so no Freqtrade code is used. (Studying behaviour and re-implementing documented concepts is
   fine; copying code is not.)
2. **Scope.** The task is one pair, one position, long-only spot, daily/4h research. A
   45-dependency framework with a plugin ecosystem, Telegram, hyperopt and multi-exchange
   plumbing adds maintenance cost without adding evidence.
3. **Transparency.** The interesting part of this project is the *measurement*: cost model,
   next-bar fills, risk limits, OOS protocol. Those are ~1,500 lines and can be read in an
   afternoon; inside Freqtrade they would be framework behaviour the reader must trust.
4. **Reuse where it is safe.** Ideas and public conventions (fee/slippage modelling,
   `startup_candle_count`-style warm-up, walk-forward evaluation) were adopted; code was not.
   See `LICENSE_AUDIT.md` for the per-project record and `THIRD_PARTY_NOTICES.md` for the
   (small) set of third-party components actually shipped.

Jesse (MIT) was the closest alternative: its licence is compatible, but it requires a database
server and ships its own dashboard, which would have replaced the UI work rather than
complemented it. Hummingbot is built around market making and order-book connectors; it is the
wrong shape for a daily trend system.

## 3. The EMA-65 / σ-65 baseline

**What was found.** Web research surfaced only an unverified Reddit-style description of a
"z-score state machine" on BTC daily candles using a 65-period EMA and 65-period standard
deviation, with no reproducible rules, no band multipliers and no performance data. No
authoritative origin (paper, book, vendor page) was located.

**What is implemented** (`btcbot/strategies/ema_std.py`), explicitly as a reconstruction:

1. `z = (close − EMA65) / σ65`, population σ over the last 65 closes.
2. Bullish state when `z ≥ +0.75`.
3. Bearish state when `z ≤ −0.75`.
4. Between thresholds the previous state persists (hysteresis, no whipsaw at the mid-line).
5. Long when bullish, flat when bearish; long-only spot.
6. Protective stop at `entry − 3 × ATR14` (this project requires every strategy to define a
   hard stop). `use_stop=false` reproduces the pure state rule — which the risk layer refuses to
   size, because an unstoppable position cannot be risk-sized (recorded as a rejected experiment).

Because the original multiplier is unknown, it is a parameter, and the report shows a
**sensitivity grid** (`0.5, 1.0, 1.5, 2.0`) instead of claiming an optimum.

## 4. Strategy families tested

| ID | Strategy | Rules (summary) | Timeframe |
|---|---|---|---|
| D | `ema_std` | z-score state machine above, EMA65 ± σ65, ATR stop | 1d, 4h |
| A | `trend_ema` | EMA50/EMA200 golden cross while close > EMA200; death-cross exit; 3×ATR stop | 1d |
| B | `breakout_donchian` | close > prior 20-bar high with volume ≥ 20-bar average; exit close < prior 10-bar low; 2.5×ATR stop | 4h |
| C | `meanrev_bb_rsi` | close < lower Bollinger(20, 2σ) and RSI14 < 30 while close > EMA200; exit at mid-band or RSI > 55; 2×ATR stop | 4h |
| E | `regime_adaptive` | Donchian breakouts in `bull_trend`, Bollinger/RSI reversion in `range_low_vol`, **no entries** in bear or high-volatility regimes | 4h |
| F | `ml_gated` | enter when walk-forward P(positive 5-bar forward return net of costs) ≥ 0.55 and close > EMA200; exit on decay; 2.5×ATR stop | 1d |

Every strategy declares `warmup_bars`, its parameters, a human-readable reason for each signal,
and a version string that is stored with every order and trade.

## 5. Market data

| Item | Value |
|---|---|
| Source | Binance public REST, `GET /api/v3/klines` (no key) — https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data |
| Pair | BTC/USDT spot |
| Daily | 3,351 bars, 2017-08-17 → 2026-10-19, **0 gaps**, quality `ok` |
| 4-hour | 20,093 bars, 16 genuine exchange outages (exchange maintenance), reported not filled |
| Normalisation | UTC epoch seconds for bar open; `close_time` stored alongside |
| Provenance | `data_sources`, `data_quality_checks`, `candles.source`, `ingested_at` |

Live quotes come from `GET /api/v3/ticker/bookTicker` (best bid/ask) with a 10-second cache.
Binance is unavailable in some jurisdictions; the exchange layer is isolated behind
`ExchangeClient` so another venue can be added, but no second venue has been validated.

## 6. Cost and execution assumptions

| Assumption | Value |
|---|---|
| Taker/maker fee | 10 bps per side (0.10%) |
| Half-spread | 1 bp of the 2 bp quoted spread, paid on each fill |
| Slippage | 5 bps adverse to the trade, each side |
| Round-trip drag | ≈ 32 bps before market movement |
| Fill timing (backtest) | signal at close of bar *i*, fill at **open of bar i+1** |
| Fill timing (live paper) | signal at close of bar *i*, fill at the live quote when the poll runs |
| Stop | checked against bar high/low (backtest, gap-aware) or the live quote (paper) |
| Rejections | below min notional ($10), qty below step, insufficient cash, no stop distance |

## 7. Results (simulated, net of the costs above)

Full-period comparison, identical data and risk settings (`reports/backtests/comparison.md`):

| Strategy | TF | Net % | CAGR % | MaxDD % | Sharpe | Trades | Win % | Exposure % | Buy & hold % |
|---|---|---|---|---|---|---|---|---|---|
| ema_std | 1d | +32.00 | 3.14 | 5.28 | 0.81 | 26 | 42.3 | 53.7 | +1,261.35 |
| ema_std | 4h | +61.53 | 5.38 | 7.25 | 1.18 | 176 | 33.0 | 52.6 | +1,843.77 |
| trend_ema | 1d | +21.73 | 2.31 | 8.16 | 0.53 | 7 | 57.1 | 48.2 | +662.62 |
| breakout_donchian | 4h | +49.20 | 4.46 | 5.20 | 1.07 | 261 | 37.2 | 36.9 | +1,880.09 |
| meanrev_bb_rsi | 4h | +0.91 | 0.10 | 1.73 | 0.18 | 20 | 60.0 | 1.0 | +2,018.33 |
| regime_adaptive | 4h | +31.10 | 3.03 | 4.16 | 1.04 | 124 | 37.9 | 17.0 | +2,018.33 |

Buy & hold over the same 9.08-year window: **+2,018%**, CAGR 40.0%, **max drawdown −83.9%**,
Sharpe 0.842 (`reports/backtests/bt_20261021_000447_regime_adaptive_cdc3b2/metrics.json`).

**Reading this honestly**

* No strategy comes close to buy-and-hold on absolute return. The brief's rule — "a strategy
  should not be declared successful just because its absolute return is positive" — is the
  reason the table shows `excess_vs_bh_pct`: every value is negative (roughly −1,200 to −2,000
  percentage points over nine years).
* What the strategies do deliver is a *different risk profile*: drawdown 4–8% instead of 84%,
  with Sharpe roughly equal to or slightly above buy & hold (1.04–1.18 for the 4h systems,
  0.81 for the daily baseline). That is consistent with **risk-controlled exposure**, not with
  a demonstrated statistical edge.
* Exposure explains most of the return gap: risk sizing (0.5% per trade, 20% notional cap) keeps
  the book small and mostly in cash. The experiments `risk_budget_1.0pct` and `risk_budget_2.0pct`
  confirm that returns scale with position size — which is **not** evidence of edge.
* Win rates are uninformative alone (mean reversion wins 60% with almost no profit factor;
  trend systems win less often but win bigger).

## 8. Controlled experiments (`reports/experiments.md`)

Protocol: each variant is compared with the untouched baseline on the same data and costs, and
judged on an out-of-sample window (2024-01-19 → 2026-10-20) with a minimum of 20 OOS trades.
Verdicts are written before interpretation.

| Experiment | Verdict | OOS net % | OOS trades |
|---|---|---|---|
| band 0.5 / −0.5 | accepted (worth paper-trading) | 3.66 vs baseline 2.86 | 31 |
| band 1.0 | rejected | 2.36 | 41 |
| band 1.5 | rejected | 2.70 | 28 |
| band 2.0 | inconclusive (<20 trades) | 3.57 | 15 |
| asymmetric 1.0 / −1.0 | inconclusive | 2.32 | 10 |
| pure state, no stop | rejected | — | 0 (risk layer refuses to size) |
| trend filter EMA200 | inconclusive | 2.64 | 9 |
| wider stop 5×ATR | inconclusive | 1.58 | 10 |
| tighter stop 2×ATR | inconclusive | 4.66 | 10 |
| risk budget 1.0% / 2.0% | report only | 5.73 / 11.47 | 10 |

Of ten variants, one "passed" the pre-registered bar, six were inconclusive because the OOS
sample is too small to distinguish them from noise, two were rejected, and the no-stop variant
was refused by the risk layer. This is the expected outcome for a 9-year single-asset daily
sample: **the parameter surface is flat inside the noise.** No variant is promoted as an
improvement.

## 9. Machine learning (`docs/MODEL_VALIDATION.md`)

Gradient-boosted trees, 5-bar forward return label net of costs, purged walk-forward with
embargo, fixed hyper-parameters, out-of-fold predictions only.

* Out-of-fold **AUC 0.533** over 2,447 predictions in 21 folds (gate: ≥ 0.55) → **rejected**.
* The ML-gated strategy returned +7.69% versus +21.95% for the plain EMA-65 baseline on the same
  window, with more trades (102 vs 21) — worse after costs.
* The model is retained as an artifact and a negative result, and is **not** used by the running
  paper session. A rejected model cannot replace an accepted one automatically.

## 10. Sentiment

* Alternative.me Fear & Greed index (`https://api.alternative.me/fng/?limit=0`): 3,182 daily
  observations with timestamps; no key required.
* RSS headlines (CoinDesk, Cointelegraph, The Block) are stored as context only, timestamped by
  retrieval; they are not used as a trading input.
* Predictive test: adding sentiment features to the ML model changed out-of-fold AUC from 0.533
  to 0.5287 (Δ −0.0043). A sentiment-only contrarian rule returned **+0.04%** over nine years
  versus +32.0% for the EMA baseline and +1,604% for buy & hold.
* Verdict: **no reliable benefit found; sentiment remains disabled for trading** and is shown in
  the dashboard for context. This is a negative result, not a hidden failure.

## 11. Evidence on trend following in crypto

The peer-reviewed literature on simple daily trend/breakout rules in crypto is thin and
sensitive to sample period, costs and data quality; published Sharpe ratios frequently ignore
the 20–80% drawdowns that come with a fully invested BTC position, and many studies stop before
the 2022 drawdown. This project therefore treats its own backtests as **descriptive**, not as
confirmation of a durable premium, and reports drawdown, exposure and costs next to return.

## 12. Known threats to validity

1. **Single asset, single history.** Nine years is roughly two BTC cycles; regimes are highly
   correlated within the sample.
2. **Cost model is an approximation.** 10 bps + 2 bps spread + 5 bps slippage is reasonable for
   BTC/USDT but not a fill simulator; thin books or outages would be worse.
3. **Survivorship of the venue.** Binance-specific history; another exchange would differ.
4. **Regime labels are retrospective windows**, so per-regime attribution is approximate.
5. **Multiple comparisons.** Six strategies and ten variants were examined; the one "accepted"
   result is exactly the kind of outcome that can arise by chance. It is documented as
   "worth paper-trading", never as proven.
6. **Paper ≠ live.** Queue position, partial fills, rate limits and exchange outages are not
   modelled in the live paper path beyond quote staleness checks.

## Sources

* Freqtrade repository and docs — https://github.com/freqtrade/freqtrade , https://www.freqtrade.io
* Jesse — https://github.com/jesse-ai/jesse , https://docs.jesse.trade
* Hummingbot — https://github.com/hummingbot/hummingbot , https://hummingbot.org
* Binance Spot API — https://binance-docs.github.io/apidocs/spot/en/
* Alternative.me Fear & Greed — https://alternative.me/crypto/fear-and-greed-index/ ,
  `https://api.alternative.me/fng/?limit=0`
* TradingView Lightweight Charts — https://github.com/tradingview/lightweight-charts
* pandas / numpy / scikit-learn documentation for the indicator and CV implementations used
