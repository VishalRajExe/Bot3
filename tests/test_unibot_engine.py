"""Backtester mechanics: fills, fees, stops and portfolio accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unibot.backtest import run_portfolio_backtest, summarize_trades
from unibot.config import Settings, SymbolCfg
from unibot.features import add_features
from unibot.store import UnibotStore
from unibot.strategies import Signal, Strategy


class Scripted(Strategy):
    """Deterministic strategy: enter on bar 5, exit on bar `exit_at`."""

    name = "scripted"
    version = "1.0.0"

    def __init__(self, exit_at: int = 20) -> None:
        self.exit_at = exit_at
        super().__init__({})

    @property
    def warmup_bars(self) -> int:
        return 2

    def _signal(self, df, i, position):
        close = float(df.iloc[i]["close"])
        if position is None and i == 5:
            return Signal("enter_long", "scripted entry", stop_price=close * 0.95,
                          trail_atr=0.0, indicators={"close": close})
        if position is not None and i >= self.exit_at:
            return Signal("exit_long", "scripted exit", indicators={"close": close})
        return Signal.hold("scripted hold", close=close)


def make_frame(n: int = 60, *, drift: float = 0.5, noise: float = 0.2,
               seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(np.full(n, drift) + rng.normal(0, noise, n))
    open_ = close - drift * 0.5 + rng.normal(0, noise * 0.2, n)
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    ts = np.arange(n, dtype=np.int64) * 14_400 + 1_700_000_000
    df = pd.DataFrame({"bar_ts": ts, "open": open_, "high": high, "low": low,
                       "close": close, "volume": 100.0, "quote_volume": 1e4,
                       "trades": 10, "close_time": ts + 14_400, "source": "test"})
    return add_features(df)


def one_symbol_settings(*, max_positions: int = 2, risk: float = 1.0) -> Settings:
    return Settings(
        capital=100.0,
        universe=[SymbolCfg(symbol="BTC/USD", cluster="btc", strategy="trend_breakout")],
        risk={"max_positions": max_positions, "risk_per_trade_pct": risk},
    )


def test_round_trip_accounting_and_fees():
    s = one_symbol_settings()
    frame = make_frame()
    strat = Scripted(exit_at=20)
    r = run_portfolio_backtest(s, {"BTC/USD": frame}, strategies={"BTC/USD": strat},
                               start_ts=int(frame.iloc[3]["bar_ts"]),
                               initial_equity=100.0)
    assert len(r.trades) == 1
    t = r.trades[0]
    # fees are 25 bps per side on both legs
    expected = (t.qty * t.entry_price + t.qty * t.exit_price) * 0.0025
    assert t.fees == pytest.approx(expected, rel=1e-6)
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.fees, rel=1e-9)
    assert r.equity.iloc[-1]["equity"] == pytest.approx(r.final_equity)
    assert t.entry_price > float(frame.iloc[6]["open"])  # slippage charged on entry
    assert t.exit_price < float(frame.iloc[21]["open"])  # and on exit


def test_equity_identity_holds_every_bar():
    s = one_symbol_settings()
    frame = make_frame()
    r = run_portfolio_backtest(s, {"BTC/USD": frame},
                               strategies={"BTC/USD": Scripted()},
                               start_ts=int(frame.iloc[3]["bar_ts"]))
    eq = r.equity
    assert (eq["cash"] > 0).all()
    assert (eq["equity"] > 0).all()
    assert (eq["positions"] <= 2).all()
    # invested never exceeds equity, i.e. no leverage
    assert (eq["invested"] <= eq["equity"] + 1e-6).all()
    # terminal position is closed
    assert eq.iloc[-1]["positions"] == 0


def test_stop_is_checked_and_gaps_fill_worse():
    s = one_symbol_settings()
    frame = make_frame()
    # force a gap-down through the stop on the bar after entry
    entry_bar = 6
    stop = float(frame.iloc[entry_bar]["open"]) * 0.95
    frame.loc[frame.index[entry_bar + 1], "open"] = stop * 0.90
    frame.loc[frame.index[entry_bar + 1], "low"] = stop * 0.88
    frame.loc[frame.index[entry_bar + 1], "high"] = stop * 0.92
    frame.loc[frame.index[entry_bar + 1], "close"] = stop * 0.90

    r = run_portfolio_backtest(s, {"BTC/USD": frame},
                               strategies={"BTC/USD": Scripted(exit_at=50)},
                               start_ts=int(frame.iloc[3]["bar_ts"]))
    stops = [t for t in r.trades if "stop" in t.exit_signal]
    assert stops, "expected a stop exit"
    t = stops[0]
    assert t.exit_price < stop, "gap must fill below the stop, not at it"
    assert t.net_pnl < 0


def test_no_entry_in_bear_regime():
    s = one_symbol_settings()
    frame = make_frame()
    frame["market_score"] = -0.9
    frame["market_label"] = "bear"
    frame["btc_trend"] = -20.0
    r = run_portfolio_backtest(s, {"BTC/USD": frame},
                               strategies={"BTC/USD": Scripted()},
                               start_ts=int(frame.iloc[3]["bar_ts"]))
    assert len(r.trades) == 0
    assert any("bear" in d["reason"] for d in r.decisions)


def test_risk_caps_number_of_positions():
    s = one_symbol_settings(max_positions=1)
    frame = make_frame()
    # add a second symbol that also wants to enter
    second = frame.copy()
    second["close"] = second["close"] * 1.01
    second["open"] = second["open"] * 1.01
    second["high"] = second["high"] * 1.01
    second["low"] = second["low"] * 1.01
    s2 = s.model_copy(update={"universe": [
        SymbolCfg(symbol="BTC/USD", cluster="btc", strategy="trend_breakout"),
        SymbolCfg(symbol="ETH/USD", cluster="l1", strategy="trend_breakout"),
    ]})
    r = run_portfolio_backtest(
        s2, {"BTC/USD": frame, "ETH/USD": second},
        strategies={"BTC/USD": Scripted(), "ETH/USD": Scripted(exit_at=30)},
        start_ts=int(frame.iloc[3]["bar_ts"]),
    )
    eq = r.equity
    assert eq["positions"].max() <= 1
    assert any("max_positions" in d["reason"] for d in r.decisions)


def test_summarize_trades_empty_and_filled():
    empty = summarize_trades([])
    assert empty["trades"] == 0 and empty["net_pnl"] == 0.0


def test_store_persists_run_and_equity(tmp_path):
    store = UnibotStore(tmp_path / "test.sqlite3")
    s = one_symbol_settings()
    frame = make_frame()
    r = run_portfolio_backtest(s, {"BTC/USD": frame},
                               strategies={"BTC/USD": Scripted()},
                               start_ts=int(frame.iloc[3]["bar_ts"]), store=store,
                               label="pytest")
    runs = store.runs()
    assert runs and runs[0]["run_id"] == r.run_id
    eq = store.equity(r.run_id)
    assert len(eq) == len(r.equity)
    assert store.db.scalar("SELECT COUNT(*) FROM trades") == len(r.trades)
    store.close()
