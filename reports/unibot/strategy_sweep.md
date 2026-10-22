# Per-symbol strategy sweep

Selection uses the **held-out test slice** only (last 35% of history).
Train metrics are shown for context, not used to pick the winner.

| Symbol | Selected strategy | Params | Test return | Test DD | Test trades | Win | Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| BTC/USD | trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 1.36% | 6.51% | 25 | 28.00% | 0.16 |
| ETH/USD | trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 11.02% | 4.69% | 27 | 33.33% | 0.88 |
| SOL/USD | trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 1.94% | 2.57% | 5 | 60.00% | 0.71 |
| AVAX/USD | trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 6.38% | 2.93% | 10 | 50.00% | 0.65 |
| LINK/USD | trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 2.83% | 3.53% | 16 | 37.50% | 0.28 |
| UNI/USD | trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 6.40% | 1.92% | 5 | 80.00% | 1.74 |
| AAVE/USD | trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 2.47% | 5.53% | 13 | 30.77% | 0.33 |
| XRP/USD | trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 1.70% | 2.25% | 10 | 30.00% | 0.54 |
| LTC/USD | trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 4.11% | 6.05% | 21 | 42.86% | 0.38 |
| DOGE/USD | trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 2.64% | 4.71% | 8 | 25.00% | 0.58 |
| MATIC/USD | trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 0.00% | -0.00% | 0 | —% | — |

## Full grid

### BTC/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 1.36% | 6.51% | 25 | 0.209 | selected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | -0.45% | 7.16% | 22 | -1.063 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 0.91% | 5.74% | 22 | 0.158 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -1.05% | 5.33% | 12 | -1.198 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -6.19% | 10.37% | 29 | -1.597 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -4.41% | 6.91% | 17 | -1.637 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | 0.11% | 1.96% | 7 | 0.034 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.75% | 0.36% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | 0.72% | 3.72% | 11 | 0.193 | rejected |

### ETH/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 6.11% | 7.79% | 42 | 0.784 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 11.02% | 4.69% | 27 | 2.35 | selected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 10.62% | 5.85% | 21 | 1.815 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 7.43% | 1.91% | 9 | 2.228 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 2.94% | 4.73% | 27 | 0.622 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -2.01% | 4.61% | 12 | -1.436 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | -1.09% | 1.42% | 4 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -3.68% | 3.82% | 8 | -1.179 | rejected |

### SOL/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | -1.08% | 4.18% | 9 | -0.755 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 1.94% | 2.57% | 5 | 0.452 | selected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 1.63% | 2.41% | 4 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 2.32% | 1.88% | 3 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -0.45% | 1.93% | 6 | -0.736 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -1.90% | 2.78% | 5 | -1.01 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | 0.65% | 0.09% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -0.53% | 1.32% | 2 | -999.0 | rejected |

### AVAX/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 6.14% | 4.62% | 16 | 1.328 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 5.74% | 3.71% | 12 | 1.545 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 6.38% | 2.93% | 10 | 2.175 | selected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -1.63% | 2.76% | 7 | -0.955 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -2.60% | 5.05% | 27 | -1.514 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -1.69% | 2.98% | 9 | -0.939 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | -0.36% | 1.45% | 6 | -0.709 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.50% | 1.21% | 3 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -0.24% | 2.04% | 8 | -0.669 | rejected |

### LINK/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 5.36% | 7.81% | 31 | 0.687 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 3.47% | 5.38% | 23 | 0.644 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 2.83% | 3.53% | 16 | 0.802 | selected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -4.45% | 6.24% | 12 | -1.714 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -5.48% | 9.95% | 34 | -1.551 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | 2.99% | 4.30% | 14 | 0.696 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | 0.43% | 1.92% | 4 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 1.14% | 0.30% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -2.51% | 3.94% | 13 | -1.636 | rejected |

### UNI/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 5.42% | 2.90% | 8 | 1.124 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 6.40% | 1.92% | 5 | 1.92 | selected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 3.87% | 1.91% | 5 | 1.161 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 1.92% | 0.86% | 3 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 0.38% | 1.65% | 5 | 0.114 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -1.13% | 1.56% | 4 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | -1.11% | 1.13% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | -1.10% | 1.12% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -1.12% | 1.14% | 1 | -999.0 | rejected |

### AAVE/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | -7.21% | 8.62% | 26 | -1.836 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | -3.94% | 7.72% | 18 | -1.51 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 2.47% | 5.53% | 13 | 0.447 | selected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -2.65% | 4.14% | 12 | -1.641 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -4.78% | 7.07% | 28 | -1.676 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -5.25% | 5.52% | 14 | -1.951 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | -2.57% | 2.57% | 5 | -1.2 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | -1.81% | 1.81% | 2 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -1.50% | 1.99% | 6 | -1.051 | rejected |

### XRP/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | -1.48% | 6.62% | 12 | -1.224 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | -1.63% | 5.67% | 10 | -1.287 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | -0.28% | 4.27% | 7 | -0.64 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 3.55% | 0.57% | 2 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 1.70% | 2.25% | 10 | 0.755 | selected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | 0.05% | 1.12% | 4 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | -0.63% | 1.23% | 2 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -0.07% | 1.28% | 3 | -999.0 | rejected |

### LTC/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 1.76% | 9.49% | 31 | 0.185 | rejected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | -1.77% | 9.40% | 27 | -1.188 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 4.11% | 6.05% | 21 | 0.68 | selected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -0.05% | 3.06% | 9 | -0.61 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 1.19% | 4.49% | 32 | 0.265 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | -1.91% | 3.96% | 13 | -1.483 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | 1.20% | 1.64% | 8 | 0.36 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 1.03% | 0.95% | 4 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -0.43% | 3.37% | 13 | -1.129 | rejected |

### DOGE/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 2.64% | 4.71% | 8 | 0.337 | selected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 2.53% | 5.06% | 7 | 0.3 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 1.21% | 3.78% | 4 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -0.19% | 1.85% | 5 | -0.657 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | -2.82% | 3.58% | 10 | -1.786 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | 0.01% | 2.58% | 6 | 0.002 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | 0.39% | 0.31% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.36% | 0.29% | 1 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | -0.90% | 1.27% | 3 | -999.0 | rejected |

### MATIC/USD

| Strategy | Params | Test ret | Test DD | Trades | Score | Status |
|---|---|---:|---:|---:|---:|---|
| trend_breakout | `{"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0}` | 0.00% | -0.00% | 0 | -999.0 | selected |
| trend_breakout | `{"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| trend_breakout | `{"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| trend_pullback | `{"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
| mean_reversion | `{"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2}` | 0.00% | -0.00% | 0 | -999.0 | rejected |
