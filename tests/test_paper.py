from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from btcbot.config import Config
from btcbot.data.exchange import Ticker
from btcbot.data.store import CandleStore
from btcbot.db import Database
from btcbot.paper import PaperTrader
from tests.helpers import make_candles


class LastBarEntry:
    """Minimal strategy: enter on the latest bar, 2% stop, exit on demand."""

    name = "paper_test"
    version = "test"
    timeframe = "1d"
    warmup_bars = 2

    def __init__(self, exit: bool = False) -> None:
        self.exit = exit

    def params_dict(self) -> dict:
        return {"exit": self.exit}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def signal(self, df, i, position=None):
        from btcbot.strategies.base import Signal

        if position is None and not self.exit:
            close = float(df["close"].iloc[i])
            return Signal("enter_long", "paper test entry", close * 0.98, tag="test")
        if position is not None and self.exit:
            return Signal("exit_long", "paper test exit", None, tag="test")
        return Signal.hold("waiting")


class PeriodicStrategy:
    """Enters every 20 bars and exits 10 bars later; used for replay tests."""

    name = "periodic"
    version = "test"
    timeframe = "1d"
    warmup_bars = 5

    def params_dict(self) -> dict:
        return {}

    def prepare(self, df):
        return df.copy()

    def signal(self, df, i, position=None):
        from btcbot.strategies.base import Signal

        if position is None and i % 20 == 0:
            close = float(df["close"].iloc[i])
            return Signal("enter_long", "periodic entry", close * 0.90, tag="test")
        if position is not None and i % 20 == 10:
            return Signal("exit_long", "periodic exit", None, tag="test")
        return Signal.hold("periodic hold")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data.db_path = str(tmp_path / "paper.sqlite3")
    cfg.paper.poll_seconds = 1
    db = Database(cfg.db_path)
    day = 86_400
    end_ts = int(time.time()) // day * day - day  # yesterday's closed daily bar
    candles = make_candles(220, seed=41, end_ts=end_ts)
    CandleStore(db).upsert(candles, cfg.exchange.pair, "1d")
    trader = PaperTrader(cfg, db, mode="live")
    trader.strategy = LastBarEntry()
    trader.ingestor.recent_closed = lambda tf, bars: candles
    price = {"value": float(candles["close"].iloc[-1])}
    monkeypatch.setattr("btcbot.paper.get_ticker",
                        lambda cfg: Ticker(pair="BTC/USDT", price=price["value"],
                                           bid=price["value"] - 1, ask=price["value"] + 1,
                                           ts=int(pd.Timestamp.utcnow().timestamp()),
                                           source="test"))
    yield cfg, db, trader, candles, price
    db.close()


def test_live_poll_enters_and_persists_order(setup):
    cfg, db, trader, candles, price = setup
    res = trader.poll_once()
    assert res["ok"] and res["new_bar"]
    assert trader.broker.position is not None
    assert trader.broker.position.qty > 0
    assert db.scalar("SELECT COUNT(*) FROM orders WHERE mode='paper'") == 1
    fills = db.query("SELECT * FROM fills")
    assert len(fills) == 1 and fills[0]["fee"] > 0
    trades = db.query("SELECT * FROM trades")
    assert len(trades) == 0
    # Same bar polled again must not create another order.
    again = trader.poll_once()
    assert again["new_bar"] is False
    assert db.scalar("SELECT COUNT(*) FROM orders") == 1


def test_live_stop_exit_records_trade(setup):
    cfg, db, trader, candles, price = setup
    trader.poll_once()
    entry = trader.broker.position.entry_price
    price["value"] = entry * 0.90  # crash through the 2% stop
    trader.poll_once()
    assert trader.broker.position is None
    trades = db.query("SELECT * FROM trades")
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "stop-loss"
    assert trades[0]["net_pnl"] < 0
    assert db.scalar("SELECT COUNT(*) FROM risk_events WHERE kind='stop_loss'") >= 0


def test_session_resumes_position_after_restart(setup):
    cfg, db, trader, candles, price = setup
    trader.poll_once()
    qty = trader.broker.position.qty
    restarted = PaperTrader(cfg, db, session_id=trader.session_id, mode="live")
    assert restarted.broker.position is not None
    assert restarted.broker.position.qty == pytest.approx(qty)
    assert restarted.last_processed_bar == trader.last_processed_bar


def test_paused_session_records_signal_without_orders(setup):
    cfg, db, trader, candles, price = setup
    trader.set_status("paused")
    res = trader.poll_once()
    assert res["paused"] is True
    assert trader.broker.position is None
    assert db.scalar("SELECT COUNT(*) FROM orders") == 0
    signals = db.query("SELECT * FROM signals")
    assert len(signals) == 1 and "paused" in (signals[0]["indicators_json"] or "")


def test_replay_seeds_session_and_persists_equity(setup):
    cfg, db, trader, candles, price = setup
    trader.strategy = PeriodicStrategy()
    result = trader.run_replay(180)
    assert result["bars"] > 100
    assert db.scalar("SELECT COUNT(*) FROM paper_sessions WHERE id=?", (trader.session_id,)) == 1
    assert db.scalar("SELECT COUNT(*) FROM equity_curve WHERE run_id=?", (trader.session_id,)) > 100
    assert db.scalar("SELECT mode FROM paper_sessions WHERE id=?", (trader.session_id,)) == "live"
    assert trader.broker.n_trades >= 1
    # All replayed artifacts are tagged as paper, never as backtest.
    assert db.scalar("SELECT COUNT(*) FROM trades WHERE mode!='paper'") == 0


def test_replay_seed_keeps_open_position_instead_of_fake_exit(setup):
    """A replay seed must not invent an end-of-window exit for a held position."""
    cfg, db, trader, candles, price = setup
    trader.strategy = LastBarEntry()
    info = trader.run_replay(120)
    assert trader.broker.position is not None
    assert info["trades"] >= 0
    assert db.scalar("SELECT COUNT(*) FROM trades WHERE exit_reason='end_of_period'") == 0
    restarted = PaperTrader(cfg, db, session_id=trader.session_id, mode="live")
    assert restarted.broker.position is not None
    assert restarted.broker.position.qty == pytest.approx(trader.broker.position.qty)


def test_repeated_replay_replaces_rows_without_fk_violation(setup):
    """Regression: a second replay seed used to hit fills.order_id via OR REPLACE."""
    cfg, db, trader, candles, price = setup
    trader.strategy = PeriodicStrategy()
    trader.run_replay(180)
    first_trades = db.scalar("SELECT COUNT(*) FROM trades WHERE run_id=?", (trader.session_id,))
    first_orders = db.scalar("SELECT COUNT(*) FROM orders WHERE run_id=?", (trader.session_id,))
    assert first_trades >= 1

    trader.run_replay(180)  # must not raise sqlite3.IntegrityError

    assert db.scalar("SELECT COUNT(*) FROM trades WHERE run_id=?",
                     (trader.session_id,)) == first_trades
    assert db.scalar("SELECT COUNT(*) FROM orders WHERE run_id=?",
                     (trader.session_id,)) == first_orders
    orphan = db.scalar(
        "SELECT COUNT(*) FROM fills WHERE order_id NOT IN (SELECT id FROM orders)")
    assert orphan == 0


def test_stopped_session_does_not_trade(setup):
    cfg, db, trader, candles, price = setup
    trader.set_status("stopped")
    res = trader.poll_once()
    assert res["ok"] is False
    assert db.scalar("SELECT COUNT(*) FROM orders") == 0
