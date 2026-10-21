# BTC research & paper-trading system

A self-contained BTC/USDT **spot** research and paper-trading system: real Binance public
market data, six strategy families, a fee/slippage-aware backtester, walk-forward ML and
sentiment studies, a permanent trade journal, and a browser dashboard.

> **This system cannot place real orders.** There is no exchange credential handling, no
> withdrawal code and no live execution path. `trading.enable_live_orders` fails validation
> if set to `true`. Leverage is fixed at 1 and short selling is not implemented.

> **No profitability claim is made.** Every number in this repository is a simulation. The
> strategies tested here do **not** beat BTC buy-and-hold on absolute return; they trade a
> much smaller drawdown instead. See [reports/experiments.md](reports/experiments.md).

---

## 1. What is in the box

| Area | Implementation |
|---|---|
| Market data | Binance public REST (`/api/v3/klines`, `/api/v3/ticker/bookTicker`), UTC-normalised, validated, gap-checked, cached in SQLite, provenance recorded |
| Strategies | `ema_std` (EMA-65 ± 65-bar σ baseline), `trend_ema` (50/200 EMA), `breakout_donchian`, `meanrev_bb_rsi`, `regime_adaptive`, `ml_gated` |
| Backtester | Event-driven, next-bar-open fills, fees + spread + slippage on both sides, ATR stop, risk-sized orders, equity/drawdown tracking, benchmark comparison |
| Risk | Independent `RiskManager`: 0.5% risk/trade, 20% max exposure, 1 position, 2% daily loss halt, 10% drawdown halt, 3-loss cooldown, stale/abnormal data blocks |
| Paper trading | Live quotes, bar-close decisions, persisted broker/risk state, restart-safe, optional labelled historical replay seed |
| ML | Causal features, purged/embargoed walk-forward validation, gradient-boosted trees, promotion gate (OOF AUC ≥ 0.55) — currently **rejected** (0.533) |
| Sentiment | Alternative.me Fear & Greed history + optional RSS headlines; experimental, disabled for trading (no measured benefit) |
| Journal | Every closed trade with entry/exit rationale, regime, R-multiple, MFE/MAE, fees, anomalies; automatic (heuristic) weakness findings |
| Dashboard | FastAPI + vanilla JS + TradingView Lightweight Charts: candles, EMA/σ bands, markers, equity curve, journal, experiments, controls |
| Tests | 103 pytest tests, including look-ahead, restart, duplicate-order and risk-limit regression tests |

Documentation: [PRD.md](PRD.md) · [Architecture.md](Architecture.md) · [RESEARCH.md](RESEARCH.md) ·
[LICENSE_AUDIT.md](LICENSE_AUDIT.md) · [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) ·
[docs/RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md) · [docs/BACKTESTING.md](docs/BACKTESTING.md) ·
[docs/MODEL_VALIDATION.md](docs/MODEL_VALIDATION.md)

---

## 2. Requirements

* Python **3.11+** (developed and tested on CPython 3.14.6, Windows)
* No Docker, no database server, no API keys
* Outbound HTTPS to `api.binance.com` and (optionally) `api.alternative.me`

## 3. Install

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell):
./.venv/Scripts/python.exe -m pip install -r requirements.txt
# Linux / macOS:
# ./venv/bin/python -m pip install -r requirements.txt
```

Configuration lives in [config/config.json](config/config.json) and is validated on load.
`.env.example` documents the only optional override (`BTCBOT_CONFIG`, `BTCBOT_PORT`); no
secret is used anywhere.

## 4. First run

```bash
# create/validate the SQLite schema
./.venv/Scripts/python.exe -m btcbot.cli init

# download BTC/USDT 1d + 4h history (2017-08-17 -> now), validate, cache
./.venv/Scripts/python.exe -m btcbot.cli fetch

# one backtest of the baseline (--report also writes charts/CSV for that run)
./.venv/Scripts/python.exe -m btcbot.cli backtest --strategy ema_std --timeframe 1d --report

# all strategies + comparison table
./.venv/Scripts/python.exe -m btcbot.cli compare

# controlled experiments (band grid, stops, filters, OOS splits, risk budget)
./.venv/Scripts/python.exe -m btcbot.cli experiments

# ML and sentiment studies
./.venv/Scripts/python.exe -m btcbot.cli ml
./.venv/Scripts/python.exe -m btcbot.cli sentiment

# write research artifacts (charts, reports)
./.venv/Scripts/python.exe -m btcbot.cli report
```

Everything is also available through the dashboard.

## 5. Run the dashboard and paper trading

```bash
# dashboard only (no trading loop)
./.venv/Scripts/python.exe -m btcbot.cli serve --port 8601

# dashboard + paper loop from startup
./.venv/Scripts/python.exe -m btcbot.cli serve --port 8601 --paper

# headless paper trading in the foreground
./.venv/Scripts/python.exe -m btcbot.cli paper            # Ctrl+C to stop safely
./.venv/Scripts/python.exe -m btcbot.cli paper --once     # exactly one poll
```

Open <http://127.0.0.1:8601>. The **Controls** tab starts, pauses, resumes and stops the
session, resets the drawdown breaker, runs a backtest and exports the research bundle.

A session left `running` is **resumed automatically** when the API starts again
(`paper.resume_on_start`, default `true`). Broker cash, open position, risk counters and the
last processed bar are reloaded from SQLite, so a restart cannot duplicate a filled order.

### Replay seeding (clearly labelled)

`POST /api/paper/start {"replay_bars": 120}` (or the "Replay closed bars first" box) replays
the last N closed bars through the same broker to give the session a history, then continues
on live quotes. Replay fills are historical simulations. The dashboard shows a
**REPLAY-SEEDED** badge and the journal states that these records are not live performance.
A replay seed no longer closes an open position at the last bar — the position is carried
into live trading.

## 6. Tests

```bash
./.venv/Scripts/python.exe -m pytest -q      # 103 passed
```

The suite covers indicator maths (against a reference implementation), causal/no-look-ahead
signals, strategy signal timing, risk sizing and every circuit breaker, broker fees/slippage/
rejections, persistence across restarts, duplicate-order prevention, replay idempotency, API
consistency and the paper loop (paused, stopped, stop-loss, restart).

## 7. Configuration that matters

| Key | Default | Meaning |
|---|---|---|
| `costs.taker_fee_bps` | 10 | 0.10% per side (Binance spot taker) |
| `costs.spread_bps` | 2 | half-spread paid on every fill |
| `costs.slippage_bps` | 5 | extra adverse move modelled on every fill |
| `risk.risk_per_trade_pct` | 0.5 | equity risked between entry and stop |
| `risk.max_position_pct` | 20 | notional cap per position |
| `risk.daily_loss_limit_pct` | 2 | halts new entries for the UTC day |
| `risk.max_drawdown_pct` | 10 | halts new entries until a human resets it |
| `paper.resume_on_start` | true | resume an interrupted session on API start |
| `paper.strategy` | `ema_std` | strategy used by the paper loop |

Risk limits are read-only in the UI on purpose: changing them requires editing the config and
restarting, and the values used are stored in every run's metadata.

## 8. Where results live

```
reports/backtests/<run_id>/   SUMMARY.md, metrics.json, trades.csv, equity.csv,
                              chart.html, performance.html
reports/backtests/comparison.md   strategy comparison table
reports/experiments.md        controlled experiments, ML and sentiment verdicts
reports/models/               trained model artifacts
reports/research_bundle.json  full JSON export (runs, trades, sessions, models)
reports/screenshots/          dashboard captures
logs/                         application log
data/btcbot.sqlite3           candles, runs, trades, orders, fills, journal, sessions
```

## 9. Verified state (2026-10-21)

* `pytest` → **103 passed**.
* Data: 3,351 daily bars (2017-08-17 → 2026-10-19), 0 gaps; 20,093 4h bars, 16 exchange outages.
* 80 persisted backtest runs, 29 experiment records, 1 model (rejected).
* Paper session `paper_20261020_185842_f513cd` running against live Binance quotes with one
  open position (0.008 84 BTC, entry 69,376.38, stop 64,590.91) seeded by a labelled 120-bar
  replay; no live signal has fired since (daily strategy is in its "hold" state).

## 10. Limitations

* Long-only spot, one open position, one process, one user; no execution algorithm.
* Fill model is a quote + slippage approximation, not order-book simulation.
* Daily/weekly regime labels use rolling windows; regime attribution of trades is therefore
  approximate.
* Sentiment is limited to the Fear & Greed index (RSS headlines are stored as context only).
* ML was tested and **rejected**; it is not part of the running paper strategy.
* Nothing here demonstrates that paper results predict live fills, and nothing here is
  financial advice.
