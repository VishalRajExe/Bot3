"""Event-driven backtest engine.

Execution convention (the core anti-look-ahead guarantee)
--------------------------------------------------------
For each bar ``i``:

1. Any order created from the close of bar ``i-1`` is executed at bar ``i``'s
   **open**.
2. Protective stops are checked against bar ``i``'s high/low *after* entries are
   processed.  A gap through the stop fills at the open, not at the stop price.
3. Equity is marked at bar ``i``'s close.
4. The strategy is evaluated on the close of bar ``i`` and may create an order
   for bar ``i+1``.

A strategy therefore never sees the bar it trades on, and no signal is filled at
the price that produced it.  The final open position is closed at the last close
and flagged as such.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data.store import dataset_hash
from ..data.timeframes import tf_seconds
from ..db import Database, utcnow_iso
from ..execution.models import Fill, Order, TradeRecord
from ..execution.paper import PaperBroker, RejectedOrder
from ..persist import (insert_equity, insert_fills, insert_orders, insert_risk_events,
                       insert_signals, insert_trades)
from ..risk import RiskManager
from ..strategies.base import PositionView, Signal, Strategy
from .metrics import buy_and_hold_benchmark, cash_benchmark, compute_metrics

log = logging.getLogger(__name__)

# Extra cushion used when converting an equity target into a quantity, so an
# adverse overnight gap cannot push exposure/risk materially above its limit.
GAP_BUFFER_PCT = 1.0


@dataclass
class RunResult:
    run_id: str
    strategy: str
    strategy_version: str
    timeframe: str
    params: dict
    initial_equity: float
    start_ts: int
    end_ts: int
    metrics: dict
    benchmark: dict
    trades: list[TradeRecord] = field(default_factory=list)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: list[dict] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    risk_events: list[dict] = field(default_factory=list)
    dataset_hash: str = ""
    bars: int = 0
    notes: str = ""
    persisted: bool = False
    mode: str = "backtest"
    broker_state: dict = field(default_factory=dict)
    risk_state: dict = field(default_factory=dict)
    last_bar_ts: int = 0


def _signal_record(*, run_id: str, mode: str, strategy: Strategy, bar_ts: int,
                   sig: Signal, price: float, extra: dict | None = None) -> dict:
    payload = {
        "run_id": run_id,
        "mode": mode,
        "strategy": strategy.name,
        "strategy_version": strategy.version,
        "bar_ts": int(bar_ts),
        "action": sig.action,
        "reason": sig.reason,
        "price": float(price),
        "indicators": sig.indicators,
        "tag": sig.tag,
    }
    if extra:
        payload.update(extra)
    return payload


def run_backtest(cfg: Config, db: Database | None, strategy: Strategy, df: pd.DataFrame, *,
                 timeframe: str | None = None, initial_equity: float | None = None,
                 run_id: str | None = None, persist: bool = True,
                 start_ts: int | None = None, end_ts: int | None = None,
                 notes: str = "", mode: str = "backtest",
                 close_at_end: bool = True) -> RunResult:
    tf = timeframe or strategy.timeframe
    tf_s = tf_seconds(tf)
    if df.empty:
        raise ValueError("cannot backtest on an empty candle frame")

    initial_equity = float(initial_equity if initial_equity is not None
                           else cfg.backtest.initial_equity)
    prepared = strategy.prepare(df)
    n = len(prepared)
    bar_ts_all = prepared["bar_ts"].to_numpy(dtype=np.int64)

    start_idx = max(strategy.warmup_bars, 0)
    if start_ts is not None:
        start_idx = max(start_idx, int(np.searchsorted(bar_ts_all, start_ts, side="left")))
    end_idx = n - 1
    if end_ts is not None:
        end_idx = min(end_idx, int(np.searchsorted(bar_ts_all, end_ts, side="left")) - 1)
    if end_idx - start_idx < 5:
        raise ValueError(
            f"evaluation window too short: {end_idx - start_idx + 1} bars "
            f"(warm-up needs {strategy.warmup_bars})"
        )

    run_id = run_id or f"bt_{time.strftime('%Y%m%d_%H%M%S')}_{strategy.name}_{uuid.uuid4().hex[:6]}"
    broker = PaperBroker(cfg.costs, symbol=cfg.exchange.pair, initial_equity=initial_equity,
                         run_id=run_id, mode=mode)
    risk = RiskManager(cfg.risk, cfg.costs, initial_equity, timeframe=tf)

    trades: list[TradeRecord] = []
    orders: list[Order] = []
    fills: list[Fill] = []
    signals: list[dict] = []
    risk_events: list[dict] = []
    equity_rows: list[tuple] = []
    pending: Order | None = None
    pending_meta: dict = {}
    peak = initial_equity
    bars_in_market = 0
    last_signal: Signal | None = None

    def record_orders(order: Order, fill: Fill | None = None, error: str | None = None) -> None:
        order.run_id = run_id
        order.mode = "backtest"
        orders.append(order)
        if fill is not None:
            fills.append(fill)
        if error:
            order.status = "rejected"
            order.reject_reason = error

    for i in range(start_idx, end_idx + 1):
        row = prepared.iloc[i]
        bar_ts = int(row["bar_ts"])
        ts_open = bar_ts
        ts_close = bar_ts + tf_s
        open_px = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        bar_return = float(close / open_px - 1.0) if open_px else None

        # 1. Execute the order queued by the previous bar's close.
        if pending is not None:
            try:
                fill = broker.execute(pending, open_px, ts_open)
                if pending.side == "buy" and broker.position is not None:
                    broker.position.entry_regime = pending_meta.get("regime")
                    broker.position.entry_indicators = pending_meta.get("indicators", {})
                    broker.position.risk_amount = float(pending_meta.get("risk_amount", 0.0))
                    broker.position.entry_order_id = pending.id
                record_orders(pending, fill)
                pending.status = "filled"
                if pending.side == "sell":
                    trade = broker.close_position(
                        fill, exit_reason=pending.reason, exit_signal="exit_long"
                    )
                    trades.append(trade)
                    risk.on_trade_closed(ts=ts_open, net_pnl=trade.net_pnl, timeframe=tf)
            except RejectedOrder as exc:
                record_orders(pending, None, str(exc))
                risk_events.append({
                    "ts": ts_open, "kind": "order_rejected", "action": "order_dropped",
                    "detail": f"{pending.side} rejected: {exc}", "value": {"reason": str(exc)},
                })
            pending = None
            pending_meta = {}

        # 2. Protective stop (after entries, before mark-to-market).
        pos = broker.position
        if pos is not None and pos.stop_price is not None and low <= pos.stop_price:
            reference = open_px if open_px <= pos.stop_price else pos.stop_price
            gapped = open_px <= pos.stop_price
            stop_order = Order(
                symbol=cfg.exchange.pair,
                side="sell",
                qty=pos.qty,
                reason=("stop-loss gap fill at open" if gapped
                        else f"stop-loss hit at {pos.stop_price:,.2f}"),
                strategy=strategy.name,
                strategy_version=strategy.version,
                signal_bar_ts=bar_ts,
                target_bar_ts=bar_ts,
                stop_price=None,
                tag="stop",
                run_id=run_id,
                mode=mode,
            )
            fill = broker.execute(stop_order, reference, ts_close)
            record_orders(stop_order, fill)
            stop_order.status = "filled"
            trade = broker.close_position(fill, exit_reason="stop-loss", exit_signal="stop")
            trades.append(trade)
            risk.on_trade_closed(ts=ts_close, net_pnl=trade.net_pnl, timeframe=tf)
            risk_events.append({
                "ts": ts_close, "kind": "stop_loss", "action": "position_closed",
                "detail": stop_order.reason,
                "value": {"stop": pos.stop_price, "fill": fill.price, "gapped": gapped},
            })

        # 3. Mark to market and update risk state.
        if broker.position is not None:
            broker.position.mark(high, low)
            bars_in_market += 1
        equity = broker.equity(close)
        peak = max(peak, equity)
        risk.on_equity(ts_close, equity)
        for ev in risk.drain_events():
            ev.update({"run_id": run_id, "mode": mode})
            risk_events.append(ev)
        equity_rows.append((
            run_id, ts_close, equity, broker.cash,
            broker.position.qty if broker.position else 0.0,
            close, (equity - peak) / peak * 100.0 if peak else 0.0,
        ))

        # 4. Strategy decision on the close of this bar.
        pos_view = None
        if broker.position is not None:
            p = broker.position
            pos_view = PositionView(qty=p.qty, entry_price=p.entry_price, entry_ts=p.entry_ts,
                                    stop_price=p.stop_price, entry_bar_ts=p.entry_bar_ts,
                                    entry_tag=p.entry_tag)
        sig = strategy.signal(prepared, i, pos_view)
        last_signal = sig
        extra = {}
        if pending is not None:
            extra["note"] = "order already pending"
        if sig.is_entry and broker.position is None:
            decision = risk.check_entry(
                ts=ts_close, equity=equity, cash=broker.cash, has_position=False,
                bar_ts=bar_ts, bar_return=bar_return, pending_order=pending is not None,
            )
            if not decision.allowed:
                extra["blocked"] = {"code": decision.code, "reason": decision.reason}
                risk_events.append({
                    "ts": ts_close, "run_id": run_id, "mode": mode,
                    "kind": f"entry_blocked:{decision.code}", "action": "no_trade",
                    "detail": decision.reason, "value": {"bar_ts": bar_ts},
                })
            else:
                sizing = risk.size_position(
                    equity=equity, cash=broker.cash,
                    entry_price=close,
                    stop_price=sig.stop_price,
                )
                if not sizing.allowed:
                    extra["blocked"] = {"code": sizing.capped_by, "reason": sizing.reason}
                    risk_events.append({
                        "ts": ts_close, "run_id": run_id, "mode": mode,
                        "kind": f"sizing_rejected:{sizing.capped_by}", "action": "no_trade",
                        "detail": sizing.reason, "value": {"bar_ts": bar_ts},
                    })
                else:
                    extra["sizing"] = {
                        "qty": sizing.qty, "notional": round(sizing.notional, 2),
                        "risk_amount": round(sizing.risk_amount, 2),
                        "stop_distance_pct": round(sizing.stop_distance_pct or 0, 3),
                        "capped_by": sizing.capped_by,
                    }
                    pending = Order(
                        symbol=cfg.exchange.pair,
                        side="buy",
                        qty=sizing.qty,
                        reason=sig.reason,
                        strategy=strategy.name,
                        strategy_version=strategy.version,
                        signal_bar_ts=bar_ts,
                        target_bar_ts=bar_ts + tf_s,
                        stop_price=sig.stop_price,
                        tag=sig.tag,
                        run_id=run_id,
                        mode=mode,
                    )
                    pending_meta = {
                        "regime": str(row.get("regime")) if "regime" in row else None,
                        "indicators": sig.indicators,
                        "risk_amount": sizing.risk_amount,
                    }
        elif sig.is_exit and broker.position is not None and pending is None:
            pending = Order(
                symbol=cfg.exchange.pair,
                side="sell",
                qty=broker.position.qty,
                reason=sig.reason,
                strategy=strategy.name,
                strategy_version=strategy.version,
                signal_bar_ts=bar_ts,
                target_bar_ts=bar_ts + tf_s,
                stop_price=None,
                tag=sig.tag,
                run_id=run_id,
                mode=mode,
            )
            pending_meta = {}

        signals.append(_signal_record(run_id=run_id, mode=mode, strategy=strategy,
                                      bar_ts=bar_ts, sig=sig, price=close, extra=extra))

    # 5. Close any remaining position at the final close (explicit, flagged).
    #    A paper replay passes close_at_end=False so the seeded session keeps
    #    the position the strategy still wants to hold, instead of inventing an
    #    exit that never occurred.
    final_row = prepared.iloc[end_idx]
    if broker.position is not None and close_at_end:
        pos = broker.position
        final_order = Order(
            symbol=cfg.exchange.pair, side="sell", qty=pos.qty,
            reason="end of evaluation window: position closed at final close",
            strategy=strategy.name, strategy_version=strategy.version,
            signal_bar_ts=int(final_row["bar_ts"]), target_bar_ts=None, tag="eod",
            run_id=run_id, mode=mode,
        )
        fill = broker.execute(final_order, float(final_row["close"]), int(final_row["bar_ts"]) + tf_s)
        record_orders(final_order, fill)
        final_order.status = "filled"
        trade = broker.close_position(fill, exit_reason="end_of_period", exit_signal="forced_exit")
        trade.anomalies.append("position forced closed at end of evaluation window")
        trades.append(trade)
        risk.on_trade_closed(ts=int(final_row["bar_ts"]) + tf_s, net_pnl=trade.net_pnl,
                             timeframe=tf)

    if pending is not None:
        pending.status = "cancelled"
        pending.reject_reason = "evaluation window ended before execution"
        record_orders(pending)

    equity_df = pd.DataFrame(equity_rows, columns=[
        "run_id", "ts", "equity", "cash", "position_qty", "price", "drawdown"
    ])
    metrics = compute_metrics(
        equity_df=equity_df, trades=trades, initial_equity=initial_equity,
        timeframe=tf, bars_in_market=bars_in_market, total_bars=len(equity_df),
    )
    window = prepared.iloc[start_idx:end_idx + 1]
    benchmark = {
        "buy_hold": buy_and_hold_benchmark(
            window, initial_equity,
            fee_bps=cfg.costs.taker_fee_bps,
            slippage_bps=cfg.costs.slippage_bps,
            spread_bps=cfg.costs.spread_bps,
        ),
        "cash": cash_benchmark(initial_equity),
    }
    bs = benchmark["buy_hold"].get("net_return_pct")
    if bs is not None and metrics.get("net_return_pct") is not None:
        metrics["excess_return_vs_buy_hold_pct"] = round(metrics["net_return_pct"] - bs, 4)

    result = RunResult(
        run_id=run_id,
        strategy=strategy.name,
        strategy_version=strategy.version,
        timeframe=tf,
        params=strategy.params_dict(),
        initial_equity=initial_equity,
        start_ts=int(equity_df["ts"].iloc[0]),
        end_ts=int(equity_df["ts"].iloc[-1]),
        metrics=metrics,
        benchmark=benchmark,
        trades=trades,
        equity=equity_df,
        signals=signals,
        orders=orders,
        fills=fills,
        risk_events=risk_events,
        dataset_hash=dataset_hash(df),
        bars=len(equity_df),
        notes=notes,
        mode=mode,
        broker_state=broker.to_dict(),
        risk_state=risk.to_dict(),
        last_bar_ts=int(final_row["bar_ts"]),
    )

    if persist and db is not None:
        persist_run(db, cfg, result, strategy)
        result.persisted = True
    log.info(
        "backtest %s %s v%s: %d trades, net %.2f%% vs buy&hold %.2f%%",
        run_id, strategy.name, strategy.version, len(trades),
        metrics.get("net_return_pct") or 0.0, bs if bs is not None else 0.0,
    )
    return result



def persist_run(db: Database, cfg: Config, result: RunResult, strategy: Strategy) -> None:
    """Write the full audit trail for a run."""
    db.execute(
        "INSERT OR REPLACE INTO backtest_runs (id, created_at, strategy, strategy_version,"
        " params_json, symbol, timeframe, start_ts, end_ts, initial_equity, costs_json,"
        " risk_json, dataset_hash, dataset_bars, code_version, metrics_json, benchmark_json,"
        " status, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            result.run_id, utcnow_iso(), result.strategy, result.strategy_version,
            json.dumps(result.params, default=str), cfg.exchange.pair, result.timeframe,
            result.start_ts, result.end_ts, result.initial_equity,
            json.dumps(cfg.costs.model_dump()), json.dumps(cfg.risk.model_dump()),
            result.dataset_hash, result.bars, strategy.version,
            json.dumps(result.metrics, default=str), json.dumps(result.benchmark, default=str),
            "completed", result.notes,
        ),
    )
    insert_equity(db, result.equity)
    insert_orders(db, result.orders)
    insert_fills(db, result.fills)
    insert_trades(db, result.trades)
    insert_signals(db, result.signals)
    insert_risk_events(db, result.risk_events)
