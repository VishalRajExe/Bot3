# Per-coin meta-labeling models

Each model is trained on that coin's own history and only filters entries proposed by its primary strategy. A model is used for trading **only** when it passes purged walk-forward validation (AUC ≥ 0.55 and positive net improvement).

| Symbol | Strategy | Status | AUC | Brier | Threshold | OOS trades | Improvement | Artifact |
|---|---|---|---:|---:|---:|---:|---:|---|
| MATIC/USD | trend_pullback | **insufficient** | — | — | — | 0 | — pp | `` |
| DOGE/USD | trend_breakout | **rejected** | 0.405 | 0.358 | 0.50 | 20 | -0.94 pp | `DOGEUSD_trend_breakout_4h_202610220656.joblib` |
| LTC/USD | trend_breakout | **rejected** | 0.468 | 0.311 | 0.50 | 17 | -0.61 pp | `LTCUSD_trend_breakout_4h_202610220656.joblib` |
| AAVE/USD | trend_breakout | **rejected** | 0.528 | 0.251 | 0.50 | 7 | 1.46 pp | `AAVEUSD_trend_breakout_4h_202610220656.joblib` |
| LINK/USD | trend_breakout | **active** | 0.580 | 0.286 | 0.60 | 20 | 0.68 pp | `LINKUSD_trend_breakout_4h_202610220655.joblib` |
| AVAX/USD | trend_breakout | **rejected** | 0.426 | 0.317 | 0.50 | 12 | 0.20 pp | `AVAXUSD_trend_breakout_4h_202610220655.joblib` |
| SOL/USD | trend_breakout | **rejected** | 0.609 | 0.219 | 0.50 | 4 | 0.02 pp | `SOLUSD_trend_breakout_4h_202610220655.joblib` |
| ETH/USD | trend_breakout | **rejected** | 0.517 | 0.280 | 0.65 | 30 | 0.66 pp | `ETHUSD_trend_breakout_4h_202610220655.joblib` |
| BTC/USD | trend_breakout | **rejected** | 0.525 | 0.259 | 0.50 | 63 | -0.13 pp | `BTCUSD_trend_breakout_4h_202610220655.joblib` |
| MATIC/USD | trend_pullback | **insufficient** | — | — | — | 0 | — pp | `` |
| DOGE/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| LTC/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| XRP/USD | trend_pullback | **insufficient** | — | — | — | 0 | — pp | `` |
| AAVE/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| UNI/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| LINK/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| AVAX/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| SOL/USD | trend_breakout | **insufficient** | — | — | — | 0 | — pp | `` |
| ETH/USD | trend_breakout | **rejected** | 0.517 | 0.280 | 0.65 | 30 | 0.66 pp | `ETHUSD_trend_breakout_4h_202610220654.joblib` |
| BTC/USD | trend_breakout | **rejected** | 0.525 | 0.259 | 0.50 | 63 | -0.13 pp | `BTCUSD_trend_breakout_4h_202610220654.joblib` |

## What rejection means

9 model(s) failed validation and are **not** used to trade. They remain in the registry so the failure is reviewable rather than hidden. This is the expected outcome when a 10-candidate strategy grid is already near the information limit of 5 years of 4h bars.
