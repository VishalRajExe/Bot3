from __future__ import annotations

import pytest

from btcbot.backtest import run_backtest
from btcbot.config import Config
from btcbot.db import Database
from btcbot.execution.models import Order
from btcbot.journal import load_trades
from btcbot.persist import insert_orders, insert_trades
from btcbot.risk import RiskManager
from tests.helpers import DummyStrategy, make_candles


def test_run_artifacts_survive_reopen(tmp_path):
    path = tmp_path / "persist.sqlite3"
    cfg = Config()
    db = Database(path)
    df = make_candles(150, seed=3)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 12, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=True)
    counts = {
        "runs": db.scalar("SELECT COUNT(*) FROM backtest_runs"),
        "trades": db.scalar("SELECT COUNT(*) FROM trades WHERE run_id=?", (res.run_id,)),
        "orders": db.scalar("SELECT COUNT(*) FROM orders WHERE run_id=?", (res.run_id,)),
        "fills": db.scalar("SELECT COUNT(*) FROM fills"),
        "equity": db.scalar("SELECT COUNT(*) FROM equity_curve WHERE run_id=?", (res.run_id,)),
        "signals": db.scalar("SELECT COUNT(*) FROM signals WHERE run_id=?", (res.run_id,)),
    }
    db.close()

    db2 = Database(path)
    assert db2.scalar("SELECT COUNT(*) FROM backtest_runs") == 1
    assert db2.scalar("SELECT COUNT(*) FROM trades WHERE run_id=?", (res.run_id,)) == counts["trades"]
    assert db2.scalar("SELECT COUNT(*) FROM equity_curve WHERE run_id=?", (res.run_id,)) == counts["equity"]
    trades = load_trades(db2, run_id=res.run_id)
    assert len(trades) == len(res.trades)
    assert trades[0].net_pnl == pytest.approx(res.trades[0].net_pnl)
    db2.close()


def test_saving_the_same_run_twice_does_not_duplicate(tmp_path):
    db = Database(tmp_path / "dup.sqlite3")
    cfg = Config()
    df = make_candles(120, seed=4)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 12, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=True)
    before = db.scalar("SELECT COUNT(*) FROM trades WHERE run_id=?", (res.run_id,))
    from btcbot.backtest.engine import persist_run
    persist_run(db, cfg, res, strat)
    after = db.scalar("SELECT COUNT(*) FROM trades WHERE run_id=?", (res.run_id,))
    assert before == after == len(res.trades)
    db.close()


def test_duplicate_order_tag_is_rejected_by_unique_constraint(tmp_path):
    db = Database(tmp_path / "orders.sqlite3")
    order = Order(symbol="BTC/USDT", side="buy", qty=0.1, reason="x", strategy="dummy",
                  strategy_version="1", signal_bar_ts=1_600_000_000, run_id="run1", mode="paper")
    insert_orders(db, [order])
    order.status = "filled"
    insert_orders(db, [order])
    assert db.scalar("SELECT COUNT(*) FROM orders WHERE unique_tag=?", (order.unique_tag,)) == 1
    assert db.query_one("SELECT status FROM orders WHERE id=?", (order.id,))["status"] == "filled"
    db.close()


def test_risk_state_survives_restart(tmp_path):
    db = Database(tmp_path / "risk.sqlite3")
    cfg = Config()
    rm = RiskManager(cfg.risk, cfg.costs, 10_000)
    rm.on_equity(86_400 * 10, 9_000)
    rm.on_trade_closed(ts=86_400 * 10, net_pnl=-20)
    db.set_state("risk", rm.to_dict())

    rm2 = RiskManager(cfg.risk, cfg.costs, 10_000)
    rm2.load_dict(db.get_state("risk"))
    assert rm2.state.drawdown_halted is True
    assert rm2.state.consecutive_losses == 1
    db.close()


def test_system_state_roundtrip(tmp_path):
    db = Database(tmp_path / "state.sqlite3")
    db.set_state("k", {"a": 1, "b": [1, 2, 3]})
    assert db.get_state("k") == {"a": 1, "b": [1, 2, 3]}
    assert db.get_state("missing", "default") == "default"
    db.close()
