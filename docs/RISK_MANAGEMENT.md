# Risk management

Risk lives in `btcbot/risk.py` and is independent of every strategy and model. A strategy
produces a *desire* (enter, with an optional stop). The risk manager decides whether an order
may exist and how large it may be. Nothing in the strategy layer can widen a limit.

## 1. Defaults (configurable, validated)

| Control | Default | Config key |
|---|---|---|
| Risk budget per trade | 0.5% of current equity | `risk.risk_per_trade_pct` (max 5) |
| Maximum position notional | 20% of equity | `risk.max_position_pct` |
| Simultaneous positions | 1 | `risk.max_positions` |
| Daily equity-loss breaker | 2% | `risk.daily_loss_limit_pct` |
| High-water-mark drawdown breaker | 10% | `risk.max_drawdown_pct` |
| Consecutive losses before cooldown | 3 | `risk.consecutive_losses_limit` |
| Cooldown length | 24 h | `risk.cooldown_hours` |
| Minimum stop distance | 1% | `risk.min_stop_distance_pct` |
| Maximum stop distance | 25% | `risk.max_stop_distance_pct` |
| Stale-bar guard | 2 bars | `risk.max_bars_stale` |
| Abnormal bar-return guard | 35% | `risk.max_abs_bar_return` |
| Entry gap buffer | 1% | `risk.entry_gap_buffer_pct` |
| Leverage | 1 (cannot be changed) | `risk.leverage` (validator rejects ≠ 1) |
| Real orders | disabled | `trading.enable_live_orders` (validator rejects `true`) |

## 2. Sizing

```
cost_rate   = fee + (spread/2 + slippage)          # fraction, per side
eff_entry   = entry * (1 + gap_buffer) * (1 + cost_rate)
eff_stop    = stop  * (1 - cost_rate)
risk/unit   = eff_entry - eff_stop

qty_risk     = equity * risk_pct / 100 / (risk/unit)
qty_exposure = equity * max_position_pct / 100 / eff_entry
qty_cash     = cash / (eff_entry * (1 + fee))

qty = floor_to_step( min(qty_risk, qty_exposure, qty_cash) )
```

Consequences by design:

* A wider stop **reduces** quantity; it never increases risk.
* Costs are inside the sizing input, so a tight stop cannot create a position whose round-trip
  costs exceed its risk budget.
* Quantity is floored to the exchange step (`0.00001 BTC`) and rejected if notional < $10.
* No martingale, no averaging down, no "recover the loss" resizing — the size formula has no
  memory of previous outcomes.

Every accepted order records which cap bound it (`risk_budget`, `max_exposure`,
`available_cash`) and every rejection records a machine-readable code:

`invalid`, `no_stop`, `stop_too_tight`, `stop_too_wide`, `leverage`, `max_positions`,
`pending_order`, `drawdown_halt`, `daily_loss_halt`, `loss_cooldown`, `reentry_cooldown`,
`stale_data`, `abnormal_price`, `no_cash`, plus broker codes `min_notional`, `qty_step`,
`insufficient_cash`, `no_position`.

## 3. Circuit breakers

**Daily loss (2%).** Evaluated on each closed bar against that UTC day's starting equity. When
the loss reaches the limit, new entries are blocked for the rest of the UTC day; the block is
recorded once per day in `risk_events`. Open positions are **not** liquidated by this breaker —
the strategy's stop still governs the position.

**Drawdown (10% from high-water mark).** Latches `drawdown_halted=True` and blocks new entries
indefinitely. Only an explicit human action clears it (`POST /api/paper/reset-drawdown`, or the
Controls button). The high-water mark is retained, so equity must genuinely recover; the breaker
does not "reset the clock".

**Consecutive losses (3).** A losing closed trade increments the counter; the third starts a
24 h cooldown and resets the counter. A winning trade resets the counter.

**Post-exit pause.** Every exit sets `no_entry_until = exit_ts + one bar`, so the bar that
closed a trade cannot immediately re-enter on the same signal.

## 4. Data and market-condition guards

| Condition | Behaviour |
|---|---|
| Latest closed bar older than 2 bars | New entries rejected (`stale_data`); exits/stops still processed |
| Single-bar return > 35% | New entries rejected (`abnormal_price`) |
| Ticker unavailable | Poll returns an error, the session keeps state, no order is placed |
| Candle fetch fails | Error logged to `errors`, no decision is taken on incomplete data |
| Missed bars (process asleep) | A warning is recorded; only the latest closed bar is evaluated |
| Order rejected by the broker (min notional, cash) | Order stored with status `rejected` and reason; no retry loop |
| Process restart | Broker/risk/last-bar state reloaded; same bar is not evaluated twice, so no duplicate order |

## 5. Stop-loss behaviour

* Every strategy must return a stop with an entry; a missing stop is rejected (`no_stop`).
* Backtest: the stop is checked against each bar's low/high before new signals. If a bar opens
  beyond the stop, the fill is at the open (gap), not at the stop price.
* Paper: the stop is checked against the live quote on **every poll**, not only at bar close.
* A stop exit is written as a trade with `exit_reason="stop-loss"` and a `risk_events` row.

## 6. Operational controls

| Action | Endpoint | Effect |
|---|---|---|
| Pause | `POST /api/paper/pause` | Decisions are still recorded; no orders are placed |
| Resume | `POST /api/paper/resume` | Loop continues from the persisted state |
| Stop | `POST /api/paper/stop` | Loop stops, session marked `stopped`, not resumed on restart |
| Reset drawdown | `POST /api/paper/reset-drawdown` | Clears the latch, keeps the high-water mark |
| Poll once | `POST /api/paper/tick` | One iteration, useful for debugging |

Real-money trading is not implemented anywhere in the codebase, so there is no emergency
"flatten" that would touch an exchange. Flattening a simulated position is done by stopping the
session and editing nothing else: the persisted position remains visible.

## 7. What is deliberately *not* automatic

* The drawdown breaker is never cleared by the program itself.
* A model cannot remove a stop or raise a limit; the ML strategy only emits enter/exit desires.
* No strategy may hold more than one position or exceed the exposure cap, regardless of its
  signal strength.
* Losses are never hidden: rejected orders, blocked signals and risk events are all persisted
  and shown in the dashboard.

## 8. Tests that pin this behaviour (`tests/test_risk.py`, `tests/test_paper_broker.py`)

| Behaviour | Test |
|---|---|
| Sizing respects the risk budget and stop distance | `test_position_size_respects_risk_budget_and_stop_distance` |
| Exposure cap binds for tight stops | `test_max_exposure_cap_binds_for_tight_stops` |
| Missing / too-tight / too-wide stops refused | `test_sizing_rejects_missing_tight_and_wide_stops` |
| Never exceeds available cash | `test_sizing_never_exceeds_available_cash` |
| Dust position refused | `test_position_too_small_is_rejected` |
| Daily loss breaker blocks entries | `test_daily_loss_circuit_breaker_blocks_new_entries` |
| Drawdown breaker latches until a manual reset | `test_drawdown_breaker_halts_until_manual_reset` |
| Three losses trigger a cooldown | `test_consecutive_loss_cooldown` |
| One-bar re-entry pause after every exit | `test_reentry_cooldown_after_every_exit` |
| Stale data and abnormal returns rejected | `test_stale_data_and_abnormal_price_are_rejected` |
| Open position / pending order block new entries | `test_existing_position_and_pending_order_block_entries` |
| Risk state round-trips through storage | `test_state_persistence_roundtrip` |
| Leverage ≠ 1 and unsafe config refused | `test_leverage_and_unsafe_config_refused` |
| Half-spread, slippage and fee applied per side | `test_buy_applies_half_spread_slippage_and_fee` |
| Quantity floored to the exchange step | `test_qty_is_floored_to_exchange_step` |
| Minimum notional / rounding / cash rejections | `test_min_notional_rejection`, `test_qty_rounding_to_zero_rejection`, `test_insufficient_cash_rejection` |
| Stop exit records the trade and the event | `test_live_stop_exit_records_trade` (paper loop) |
| Paused session records decisions without orders | `test_paused_session_records_signal_without_orders` |
| Restart does not duplicate an order | `test_live_poll_enters_and_persists_order`, `test_session_resumes_position_after_restart` |

All of the above pass (`pytest -q`, 103 tests).
