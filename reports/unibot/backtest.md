# Portfolio backtest — ratchet test

- run id: `uni_20261022_135221_1d8290`
- period: 2021-01-01 → 2026-10-22 (5.80 years)
- timeframe: 4h · capital $100.00

## Headline

| Metric | Strategy | Equal-weight hold | BTC hold |
|---|---:|---:|---:|
| Net return | -4.43% | +363.72% | +176.89% |
| Max drawdown | 12.76% | 90.09% | 77.07% |
| Final equity | $95.57 | $463.72 | $276.89 |

## Trading statistics

| Metric | Value |
|---|---:|
| Closed trades | 78 |
| Win rate | 34.62% |
| Profit factor | 0.85 |
| Expectancy / trade | $-0.06 |
| Average R | -0.00 |
| Average hold | 111.79 h |
| Exposure | 10.96% |
| Sharpe / Sortino | -0.14 / -0.07 |
| Fees paid | $5.20 |
| Slippage cost | $1.46 |
| Exit mix | `{"strategy": 20, "stop": 58}` |

## Per symbol

| symbol | trades | win | net | avg_ret | avg_r |
|---|---|---|---|---|---|
| BTC/USD | 29 | 44.83 | 1.16 | 0.69 | 0.08 |
| ETH/USD | 22 | 36.36 | -2.45 | 0.30 | -0.02 |
| SOL/USD | 0 | — | 0.00 | — | — |
| AVAX/USD | 4 | 25.00 | -1.02 | -2.66 | -0.29 |
| LINK/USD | 11 | 18.18 | -1.62 | -1.95 | -0.20 |
| UNI/USD | 0 | — | 0.00 | — | — |
| AAVE/USD | 4 | 25.00 | 1.00 | 1.85 | 0.58 |
| XRP/USD | 0 | — | 0.00 | — | — |
| LTC/USD | 8 | 25.00 | -1.49 | -2.63 | -0.17 |
| DOGE/USD | 0 | — | 0.00 | — | — |
| MATIC/USD | 0 | — | 0.00 | — | — |

## Per strategy

| strategy | trades | win | net | avg_ret |
|---|---|---|---|---|
| trend_breakout | 78 | 34.62 | -4.43 | -0.24 |

## Method notes

- Signals use closed bars; fills happen at the next bar's open.
- Stops fill at `min(open, stop)` so gaps cost more than the stop.
- Fees: 25 bps taker per side, plus half-spread and 5 bps slippage on every fill.
- One shared cash balance and the live risk manager govern all symbols.
- Long-only spot, no leverage, no shorting.

> Sample size is adequate for a first read; still not a live guarantee.

Generated 2026-10-22T13:53:13.230761+00:00
