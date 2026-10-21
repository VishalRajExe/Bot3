"""Shared persistence helpers.

Both the backtester and the paper trader write the same shape of records, so the
SQL lives in one place.  ``mode`` distinguishes ``backtest``, ``paper_replay``
and ``paper`` rows; results are never mixed across modes in reporting.
"""

from __future__ import annotations

import json

from .db import Database, utcnow_iso
from .execution.models import Fill, Order, TradeRecord


def insert_orders(db: Database, orders: list[Order]) -> None:
    """Insert orders without ever deleting a row that fills may reference.

    ``INSERT OR REPLACE`` deletes the conflicting parent row first, which trips
    the ``fills.order_id`` foreign key whenever an order already has fills (for
    example when a paper replay is repeated, or an order is re-persisted after
    a restart).  ``unique_tag`` stays the dedupe key: conflicting tags are
    ignored, and the status of an existing row is updated in place instead.
    """
    if not orders:
        return
    db.executemany(
        "INSERT OR IGNORE INTO orders (id, run_id, mode, symbol, side, order_type, qty,"
        " requested_price, status, reject_reason, reason, strategy, strategy_version,"
        " signal_bar_ts, target_bar_ts, created_at, unique_tag)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(o.id, o.run_id, o.mode, o.symbol, o.side, o.order_type, o.qty, o.requested_price,
          o.status, o.reject_reason, o.reason, o.strategy, o.strategy_version,
          o.signal_bar_ts, o.target_bar_ts, utcnow_iso(), o.unique_tag) for o in orders],
    )
    db.executemany(
        "UPDATE orders SET qty=?, requested_price=?, status=?, reject_reason=?,"
        " target_bar_ts=? WHERE id=?",
        [(o.qty, o.requested_price, o.status, o.reject_reason, o.target_bar_ts, o.id)
         for o in orders],
    )


def clear_run_records(db: Database, run_id: str) -> None:
    """Drop every row owned by a run/session (fills first, to respect FKs)."""
    db.execute("DELETE FROM fills WHERE order_id IN"
               " (SELECT id FROM orders WHERE run_id=?)", (run_id,))
    for table in ("trades", "orders", "equity_curve", "signals", "risk_events"):
        column = "run_id"
        db.execute(f"DELETE FROM {table} WHERE {column}=?", (run_id,))


def insert_fills(db: Database, fills: list[Fill]) -> None:
    if not fills:
        return
    db.executemany(
        "INSERT OR REPLACE INTO fills (id, order_id, ts, price, qty, notional, fee, fee_bps,"
        " slippage_bps, spread_bps, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(f.id, f.order_id, f.ts, f.price, f.qty, f.notional, f.fee, f.fee_bps,
          f.slippage_bps, f.spread_bps, utcnow_iso()) for f in fills],
    )


def insert_trades(db: Database, trades: list[TradeRecord]) -> None:
    if not trades:
        return
    db.executemany(
        "INSERT OR REPLACE INTO trades (id, mode, run_id, symbol, side, strategy,"
        " strategy_version, model_version, entry_order_id, exit_order_id, entry_ts,"
        " entry_bar_ts, entry_price, entry_signal, entry_reason, exit_ts, exit_price,"
        " exit_signal, exit_reason, qty, initial_stop, intended_stop, stop_distance_pct,"
        " gross_pnl, fees, slippage_cost, net_pnl, return_pct, r_multiple, mfe_pct, mae_pct,"
        " holding_hours, regime, indicators_entry_json, sentiment_entry_json,"
        " risk_events_json, anomalies_json, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(t.id, t.mode, t.run_id, t.symbol, t.side, t.strategy, t.strategy_version,
          t.model_version, t.entry_order_id, t.exit_order_id, t.entry_ts, t.entry_bar_ts,
          t.entry_price, t.entry_signal, t.entry_reason, t.exit_ts, t.exit_price,
          t.exit_signal, t.exit_reason, t.qty, t.initial_stop, t.intended_stop,
          t.stop_distance_pct, t.gross_pnl, t.fees, t.slippage_cost, t.net_pnl,
          t.return_pct, t.r_multiple, t.mfe_pct, t.mae_pct, t.holding_hours, t.regime,
          json.dumps(t.indicators_entry, default=str),
          json.dumps(t.sentiment_entry, default=str),
          json.dumps(t.risk_events, default=str), json.dumps(t.anomalies, default=str),
          utcnow_iso()) for t in trades],
    )


def insert_equity(db: Database, equity_df, *, mode: str = "backtest") -> None:
    if equity_df is None or equity_df.empty:
        return
    db.executemany(
        "INSERT OR REPLACE INTO equity_curve (run_id, ts, equity, cash, position_qty, price,"
        " drawdown) VALUES (?,?,?,?,?,?,?)",
        [tuple(row) for row in equity_df.itertuples(index=False, name=None)],
    )


def insert_signals(db: Database, signals: list[dict]) -> None:
    if not signals:
        return
    rows = []
    for s in signals:
        meta = {k: v for k, v in s.items()
                if k in ("sizing", "blocked", "note", "mode")}
        rows.append((
            s.get("run_id"), s.get("mode", "backtest"), s["strategy"], s["strategy_version"],
            int(s["bar_ts"]), s["action"], s["reason"], s.get("price"),
            json.dumps({**s.get("indicators", {}), "_tag": s.get("tag"), "_meta": meta},
                       default=str),
            utcnow_iso(),
        ))
    db.executemany(
        "INSERT INTO signals (run_id, mode, strategy, strategy_version, bar_ts, action, reason,"
        " price, indicators_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def insert_risk_events(db: Database, events: list[dict]) -> None:
    if not events:
        return
    db.executemany(
        "INSERT INTO risk_events (ts, mode, run_id, kind, action, detail, value_json, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        [(int(e["ts"]), e.get("mode", "backtest"), e.get("run_id"), e["kind"], e["action"],
          e["detail"], json.dumps(e.get("value", {}), default=str), utcnow_iso())
         for e in events],
    )
