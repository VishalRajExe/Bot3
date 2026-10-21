from __future__ import annotations

import numpy as np
import pytest

from btcbot.backtest import run_backtest
from btcbot.backtest.metrics import buy_and_hold_benchmark, compute_metrics
from btcbot.config import Config
from btcbot.db import Database
from btcbot.execution.models import TradeRecord
from tests.helpers import DummyStrategy, make_candles


@pytest.fixture
def cfg():
    c = Config()
    c.data.db_path = ":memory:"
    return c


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "bt.sqlite3")
    yield d
    d.close()


def cost_rate(cfg: Config) -> float:
    return (cfg.costs.spread_bps / 2 + cfg.costs.slippage_bps) / 10_000.0


def test_entry_fills_at_next_bar_open_and_exit_at_following_open(cfg, db):
    df = make_candles(200, seed=42)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 12, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=True)
    assert len(res.trades) == 1
    trade = res.trades[0]
    expected_entry = df["open"].iloc[6] * (1 + cost_rate(cfg))
    expected_exit = df["open"].iloc[13] * (1 - cost_rate(cfg))
    assert trade.entry_price == pytest.approx(expected_entry, rel=2e-4)
    assert trade.exit_price == pytest.approx(expected_exit, rel=2e-4)
    assert trade.exit_reason == "dummy exit"
    assert trade.entry_bar_ts == int(df["bar_ts"].iloc[5])
    assert trade.risk_amount if hasattr(trade, "risk_amount") else True


def test_engine_does_not_use_signal_bar_price(cfg, db):
    df = make_candles(120, seed=5)
    # Make the signal bar close dramatically lower than the next open: a
    # look-ahead fill would show the low price, a correct fill shows the open.
    df.loc[5, "close"] = df.loc[6, "open"] * 0.8
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 20, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=False)
    entry = res.trades[0].entry_price
    assert entry > df["close"].iloc[5] * 1.1
    assert entry == pytest.approx(df["open"].iloc[6] * (1 + cost_rate(cfg)), rel=2e-4)


def test_stop_loss_fills_at_stop_and_marks_trade(cfg, db):
    df = make_candles(60, seed=8)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 40, "stop_pct": 0.02})
    prepared_stop = df["close"].iloc[5] * 0.98
    df.loc[7, "low"] = prepared_stop * 0.995
    res = run_backtest(cfg, db, strat, df, persist=False)
    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.exit_reason == "stop-loss"
    assert trade.exit_price <= prepared_stop * 1.0001
    assert any(e["kind"] == "stop_loss" for e in res.risk_events)


def test_gap_through_stop_fills_at_open_not_stop(cfg, db):
    df = make_candles(60, seed=9)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 40, "stop_pct": 0.02})
    stop = df["close"].iloc[5] * 0.98
    # Bar 7 opens far below the stop: it must fill near the open, not the stop.
    df.loc[7, "open"] = stop * 0.90
    df.loc[7, "high"] = max(df.loc[7, "high"], df.loc[7, "open"] * 1.001)
    df.loc[7, "low"] = df.loc[7, "open"] * 0.995
    df.loc[7, "close"] = df.loc[7, "open"]
    res = run_backtest(cfg, db, strat, df, persist=False)
    trade = res.trades[0]
    assert trade.exit_reason == "stop-loss"
    assert trade.exit_price < stop * 0.95


def test_single_order_per_signal_and_no_duplicates(cfg, db):
    df = make_candles(150, seed=12)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 12, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=False)
    tags = [o.unique_tag for o in res.orders]
    assert len(tags) == len(set(tags))
    assert len([o for o in res.orders if o.side == "buy"]) == 1
    assert len(res.fills) == 2


def test_open_position_is_forced_closed_at_end(cfg, db):
    df = make_candles(80, seed=15)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 999, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=False)
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "end_of_period"
    assert res.trades[0].anomalies


def test_backtest_is_reproducible(cfg, db):
    df = make_candles(400, seed=17)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 30, "stop_pct": 0.20})
    a = run_backtest(cfg, db, strat, df, persist=False)
    b = run_backtest(cfg, db, strat, df, persist=False)
    for key in ("net_return_pct", "max_drawdown_pct", "trades", "total_fees", "cagr_pct"):
        assert a.metrics[key] == b.metrics[key]
    assert [t.entry_price for t in a.trades] == [t.entry_price for t in b.trades]


def test_equity_curve_starts_at_initial_equity(cfg, db):
    df = make_candles(100, seed=19)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 10, "stop_pct": 0.20})
    res = run_backtest(cfg, db, strat, df, persist=False, initial_equity=5_000.0)
    assert res.equity["equity"].iloc[0] == pytest.approx(5_000.0)
    first_trade = res.trades[0]
    assert res.metrics["total_fees"] > 0
    assert first_trade.fees > 0


def test_buy_and_hold_benchmark_matches_manual_calculation(cfg):
    df = make_candles(300, seed=23)
    bench = buy_and_hold_benchmark(df, 10_000.0, fee_bps=10, slippage_bps=5, spread_bps=2)
    entry = df["open"].iloc[0] * (1 + (10 + 5 + 1) / 10_000)
    qty = 10_000 / (entry * 1.001)
    expected = qty * df["close"].iloc[-1] + (10_000 - qty * entry - qty * entry * 0.001)
    assert bench["final_equity"] == pytest.approx(expected, rel=1e-6)
    assert bench["max_drawdown_pct"] >= 0


def test_metrics_handle_no_trades():
    import pandas as pd

    eq = pd.DataFrame({"ts": [1_600_000_000 + i * 86_400 for i in range(100)],
                       "equity": [10_000.0] * 100})
    m = compute_metrics(equity_df=eq, trades=[], initial_equity=10_000.0, timeframe="1d",
                        bars_in_market=0, total_bars=100)
    assert m["net_return_pct"] == 0.0
    assert m["trades"] == 0
    assert m["win_rate_pct"] is None
    assert m["sharpe"] is None


def test_sizing_rejection_is_recorded(cfg, db):
    df = make_candles(80, seed=25)
    strat = DummyStrategy({"entry_bar": 5, "exit_bar": 30, "stop_pct": 0.5})  # 50% stop
    res = run_backtest(cfg, db, strat, df, persist=False)
    assert len(res.trades) == 0
    assert any("stop_too_wide" in e["kind"] for e in res.risk_events)
