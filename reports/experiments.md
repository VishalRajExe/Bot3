# Experiment log

Generated 2026-10-20T18:35:47+00:00

All numbers are simulated, net of fees/spread/slippage.

## Strategy suite

| Strategy | TF | Net % | CAGR % | MaxDD % | Sharpe | Trades | Win % | B&H % |
|---|---|---|---|---|---|---|---|---|
| ema_std | 1d | 31.997872 | 3.135834 | 5.277436 | 0.813887 | 26 | 42.307692 | 1261.354432 |
| ema_std | 4h | 61.531184 | 5.383319 | 7.247181 | 1.181896 | 176 | 32.954545 | 1843.767445 |
| trend_ema | 1d | 21.734509 | 2.308146 | 8.161729 | 0.531372 | 7 | 57.142857 | 662.617783 |
| breakout_donchian | 4h | 49.200385 | 4.462236 | 5.195728 | 1.072818 | 261 | 37.164751 | 1880.091734 |
| meanrev_bb_rsi | 4h | 0.912239 | 0.100025 | 1.729557 | 0.182735 | 20 | 60.0 | 2018.325859 |
| regime_adaptive | 4h | 31.103622 | 3.026391 | 4.161642 | 1.042201 | 124 | 37.903226 | 2018.325859 |

## Controlled changes vs the EMA-65 baseline

| Experiment | Verdict | Full net % | OOS net % | OOS MaxDD % | OOS trades | Note |
|---|---|---|---|---|---|---|
| band_0.5_0.5 | accepted | 30.391917 | 3.664178 | 1.341228 | 31 | OOS return 3.664178% vs baseline 2.858248%, drawdown within tolerance. Accepted for paper-trading comparison, not proven |
| band_1.0_1.0 | rejected | 18.834295 | 2.361274 | 2.240337 | 41 | out-of-sample return 2.361274% did not beat baseline 2.858248% |
| band_1.5_1.5 | rejected | 13.689789 | 2.701264 | 1.439833 | 28 | out-of-sample return 2.701264% did not beat baseline 2.858248% |
| band_2.0_2.0 | inconclusive | 10.425323 | 3.571133 | 0.40307 | 15 | out-of-sample trades 15 below the required 20; the sample is too small to judge, regardless of the return difference |
| band_1.0_-1.0 | inconclusive | 32.056722 | 2.321292 | 2.326249 | 10 | out-of-sample trades 10 below the required 20; the sample is too small to judge, regardless of the return difference |
| pure_state_no_stop | rejected | 0.0 | None | None | 0 | risk layer refuses to size an unstoppable position |
| trend_filter_200 | inconclusive | 25.743122 | 2.635421 | 1.866233 | 9 | out-of-sample trades 9 below the required 20; the sample is too small to judge, regardless of the return difference |
| wider_stop_5atr | inconclusive | 17.317946 | 1.576224 | 1.508632 | 10 | out-of-sample trades 10 below the required 20; the sample is too small to judge, regardless of the return difference |
| tighter_stop_2atr | inconclusive | 46.161176 | 4.65858 | 2.503686 | 10 | out-of-sample trades 10 below the required 20; the sample is too small to judge, regardless of the return difference |
| risk_budget_1.0pct | report_only | 15.73919 | 5.726893 | 4.055454 | 10 | return scales with size, not edge |
| risk_budget_2.0pct | report_only | 7.801802 | 11.466129 | 7.176385 | 10 | return scales with size, not edge |
| timeframe_4h | report_only | 61.531184 | 10.695712 | 4.806168 | 51 | not directly comparable |

## Machine learning

- Model `gbdt_1d_h5_20261020` status: **rejected** (out-of-fold AUC 0.533 below required 0.55)
- Out-of-fold AUC: 0.533 on 2447 predictions across 21 folds
- Strategy net return: 7.685752% vs EMA-65 21.947354% over the same window
- Verdict: **rejected** — model gate: out-of-fold AUC 0.533 below required 0.55. ML strategy OOS return 7.685752%.

## Sentiment

- Available: True (coverage 3182 days)
- Fear & Greed coverage: 3182 daily bars. Incremental OOF AUC from sentiment features: -0.0043 (base 0.533, with 0.5287). Sentiment-only rule net return 0.03913% vs EMA-65 31.997872% and buy&hold 1603.98365%. Sentiment is rejected as a trading input on this evidence and remains disabled.

## Interpretation

An experiment is only accepted when the out-of-sample window improves after costs with no material drawdown regression. 'Accepted' means 'worth paper-trading', never 'proven profitable'.
