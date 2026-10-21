# Architecture

Single-process Python application: FastAPI serves the dashboard and JSON API, a background
thread runs the paper loop, SQLite stores everything. No broker SDK, no exchange credentials,
no live execution path.

```
                         ┌──────────────────────────── web/ (vanilla JS + Lightweight Charts)
                         │                    Dashboard · Research · Journal · Learning
                         │                        Sentiment · Controls
                         ▼  HTTP (loopback)
┌──────────────────────────────────────────────────────────────────────────────┐
│ btcbot/api.py    FastAPI: /api/market /api/paper /api/trades /api/runs ...    │
│                  static file server (no-cache for un-hashed assets)           │
└──────────┬───────────────────────────────────────────────┬───────────────────┘
           │                                               │
┌──────────▼────────────┐                       ┌──────────▼──────────────────┐
│ PaperService (thread) │                       │ Backtest engine (sync)      │
│  poll every 60 s      │                       │  same broker + risk + data  │
└──────────┬────────────┘                       └──────────┬──────────────────┘
           │                                               │
┌──────────▼───────────────────────────────────────────────▼──────────────────┐
│ PaperTrader / run_backtest                                                  │
│   Strategy.prepare(df) → Strategy.signal(df, i, position) → Signal          │
│   RiskManager.size(...) / allow_entry(...)  → Order → PaperBroker.execute   │
│   TradeRecord ← PaperBroker.close_position(...)                             │
└──────────┬───────────────────────────┬───────────────────────┬──────────────┘
           │                           │                       │
┌──────────▼─────────┐      ┌──────────▼──────────┐   ┌────────▼─────────────┐
│ CandleIngestor     │      │ persist.py          │   │ journal.py           │
│ Binance public API │      │ orders/fills/trades │   │ findings, reviews    │
│ validation, cache  │      │ equity/signals/risk │   │                      │
└──────────┬─────────┘      └──────────┬──────────┘   └────────┬─────────────┘
           └───────────────────────────┴───────────────────────┘
                                  SQLite (data/btcbot.sqlite3)
```

## 1. Modules

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic models for every tunable; validation forbids leverage ≠ 1 and live orders |
| `db.py` | Connection (WAL, FK on), schema, migrations by `CREATE IF NOT EXISTS`, `system_state` KV, error log |
| `persist.py` | One place for the SQL that writes orders, fills, trades, equity, signals, risk events; run-scoped deletion |
| `data/exchange.py` | Retrying HTTP client (bounded retries, backoff, min request interval), klines, ticker, order-book top |
| `data/store.py` | Upsert/read candles, quality summary, dataset hash |
| `data/ingest.py` | Fill missing ranges, warm-up windows, recent closed bars |
| `indicators.py` | Causal EMA, SMA, population σ, ATR, RSI, Bollinger, Donchian, ADX |
| `regime.py` | Causal regime labels (`bull_trend`, `bear_trend`, `range_low_vol`, `range_high_vol`, `abnormal`) from EMA200 slope, ADX, realised vol |
| `strategies/base.py` | `Params`, `Signal`, `PositionView`, `Strategy` (prepare/signal/params_dict/version) |
| `strategies/*.py` | Six concrete strategies (see RESEARCH.md §4) |
| `execution/models.py` | `Order`, `Fill`, `Position`, `TradeRecord`; deterministic `unique_tag` for dedupe |
| `execution/paper.py` | Simulated broker: cash, fees, half-spread, slippage, rejection (min notional, qty step), stop handling |
| `risk.py` | Sizing from stop distance, exposure caps, daily loss, drawdown, loss-streak cooldown, stale/abnormal data blocks; serialisable state |
| `backtest/engine.py` | Event loop over bars; pending order model; stop checks; equity rows; forced end-of-window exit (optional) |
| `backtest/metrics.py` | Return, CAGR, drawdown, Sharpe/Sortino/Calmar, win rate, PF, expectancy, R, exposure, turnover, regime and monthly breakdowns, benchmarks |
| `paper.py` | `PaperTrader` (live poll, replay seed, stop monitoring, state persistence) and `PaperService` (thread) |
| `ml/features.py` | Causal features and horizon labels |
| `ml/train.py` | Purged, embargoed walk-forward CV, model registry, promotion gate |
| `sentiment/*` | Fear & Greed history, optional RSS collection, availability-correct features |
| `journal.py` | Journal queries, summary statistics, heuristic weakness findings |
| `reporting.py` | Per-run artifacts (`SUMMARY.md`, `metrics.json`, `trades.csv`, `equity.csv`, `chart.html`, `performance.html`), comparison table |
| `experiments.py` | Fixed experiment grid, OOS protocol, acceptance rules, Markdown log |
| `api.py` | HTTP surface + static hosting |
| `cli.py` | `init fetch status backtest compare experiments sentiment ml serve paper journal report export strategies` |

## 2. Data flow

### Historical
1. `fetch` asks `CandleIngestor` for each configured timeframe from `data.history_start`.
2. `ExchangeClient.klines` pages by candle step, retries with backoff, respects a 250 ms
   minimum interval.
3. Rows are validated (monotonic open time, `low ≤ open/close ≤ high`, positive volume,
   non-zero prices), deduplicated and upserted into `candles` with source and `ingested_at`.
4. A quality record is written to `data_quality_checks` (bars, expected, missing, duplicates,
   bad OHLC, zero volume, status). Gaps up to `max_gap_bars` are reported, not silently filled.

### Backtest
1. Load candles for the window, attach indicators once, compute regime labels.
2. For bar *i* from `warmup_bars` to end: check stop against the bar's high/low, execute any
   pending order at the **open of this bar**, then evaluate the strategy on the close of *i*
   and queue an order for *i+1*.
3. Risk manager sizes the order (stop distance → quantity) or rejects it with a reason.
4. Equity is marked at each close; metrics and benchmarks are computed at the end; everything
   is persisted atomically with the run row.

### Live paper
1. `PaperTrader.poll_once` fetches the ticker (live quote) and rebuilds a warm-up window of
   closed candles.
2. The protective stop is checked against the live quote on every poll (not just at bar close).
3. If a new closed bar appeared, the strategy decides; the risk manager gates; the broker fills
   at the live quote with costs; state is saved after every step.
4. Signals are recorded even when no order is placed (HOLD, blocked, paused), so the dashboard
   can explain idle periods.

## 3. Execution boundary

* Only `PaperBroker.execute` creates fills. It is the sole place where fees, spread and slippage
  are applied; it never talks to a network.
* There is no exchange order client, no signing code, no key loading. `TradingCfg` refuses
  `enable_live_orders=true`, so an accidental configuration cannot enable trading.
* Paper and backtest share the broker, so a strategy cannot behave differently in simulation.
* Order identity is `run_id:signal_bar_ts:side:strategy_version`; the database enforces
  uniqueness, and persistence never deletes a parent order that already has fills.

## 4. Persistence model

| Table | Contents |
|---|---|
| `candles` | OHLCV per symbol/timeframe/open time, source, ingest timestamp |
| `data_sources`, `data_quality_checks`, `collection_runs` | provenance and validation history |
| `backtest_runs` | params, cost model, risk model, dataset hash, metrics, benchmarks, notes |
| `paper_sessions` | id, mode (`live`), strategy/version, TF, equity, status, last bar, notes |
| `orders` / `fills` | order lifecycle, rejections, fees, slippage, spread |
| `trades` | round trips with rationale, regime, R, MFE/MAE, anomalies |
| `trade_reviews` | journal findings per trade |
| `equity_curve` | per-bar equity, cash, position, drawdown per run/session |
| `signals` | every decision with indicators, tag and meta |
| `risk_events` | breakers, blocks, stop exits |
| `experiments`, `models`, `model_predictions` | research history and model registry |
| `sentiment_observations` | timestamped Fear & Greed values and headlines |
| `system_state` | serialised broker/risk state per paper session (restart recovery) |
| `errors` | application errors with context |

`mode` (`backtest` / `paper`) is stored on every row; reporting never mixes them.

## 5. Restart and recovery

* Broker/risk/last-bar state is serialised to `system_state` after every processed bar.
* On API startup, a `running`/`paused` session is rehydrated and its loop restarted
  (`paper.resume_on_start`). A clean `stop` marks the session `stopped`, which is not resumed.
* If the process dies mid-bar, the next poll sees the same last processed bar and does nothing,
  so no duplicate order can be produced.
* Missing candles or ticker failures are logged, exposed in `/api/paper` and `/api/health`, and
  stop new entries rather than trading on stale data.

## 6. HTTP surface

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | version, mode, DB path, session/run counts, loop state |
| `GET /api/market?timeframe=&limit=` | candles + indicators, live quote, regime, latest signal, quality |
| `GET /api/paper` | session, equity, cash, position, risk, journal summary, loop state |
| `POST /api/paper/start|pause|resume|stop|tick` | operational control (paper only) |
| `POST /api/paper/reset-drawdown` | explicit human reset of the drawdown halt |
| `GET /api/signals`, `/api/trades`, `/api/trades/{id}`, `/api/journal`, `/api/learning` | decisions, trades, findings |
| `GET /api/runs`, `/api/runs/{id}`, `POST /api/backtest` | research results and on-demand runs |
| `GET /api/performance?run_id=&mode=` | equity curve + metrics |
| `GET /api/sentiment`, `POST /api/sentiment/refresh` | experimental context |
| `GET /api/config`, `/api/export` | effective configuration, JSON research bundle |

## 7. Frontend

Static files in `web/` (no build step): `index.html`, `styles.css`, `app.js`, and vendored
`lightweight-charts.standalone.production.js`. Charts are drawn only from API data; nothing is
fabricated. Markers are filtered to the loaded candle window (out-of-range markers would
otherwise be clamped to the edge bar). CSS grid uses `minmax(0, 1fr)` so a wide canvas cannot
stretch the page, and `[hidden]` is explicitly honoured for tab switching.

## 8. Decisions

| Decision | Reason |
|---|---|
| Own engine instead of Freqtrade | Freqtrade is GPL-3.0-or-later; vendoring it would force the whole project under GPL. Its 45-dependency stack and config surface also exceed what a single-pair research loop needs. |
| SQLite | Single user, single process, zero operations, easy inspection from any SQL client. |
| Next-bar-open fills | A signal computed from a close cannot be filled at that same close. |
| Risk manager outside strategies | A strategy (or model) must not be able to widen its own limits. |
| Separate `mode` per row | Backtest, replay and live-paper evidence must never be silently aggregated. |
| Vanilla JS + Lightweight Charts | No bundler, no node toolchain, ~35 KB JS, readable by a reviewer. |
| Copy no third-party code | Avoids licence obligations and unverifiable provenance; the engine is small enough to own. |
