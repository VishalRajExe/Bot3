from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcbot.strategies import STRATEGIES, build_strategy, list_strategies
from btcbot.strategies.base import PositionView
from btcbot.strategies.ema_std import EmaStd
from tests.helpers import make_candles


def test_registry_lists_all_strategies():
    names = {s["name"] for s in list_strategies()}
    assert {"ema_std", "trend_ema", "breakout_donchian", "meanrev_bb_rsi",
            "regime_adaptive"} <= names
    for name in names:
        strat = build_strategy(name)
        assert strat.version
        assert strat.description


def test_unknown_strategy_and_unknown_param_rejected():
    with pytest.raises(ValueError):
        build_strategy("does_not_exist")
    with pytest.raises(Exception):
        build_strategy("ema_std", {"not_a_param": 1})


def test_ema_std_state_machine_is_hysteretic():
    df = make_candles(900, seed=21, drift=0.002, vol=0.01)
    strat = build_strategy("ema_std", {"bull_z": 0.75, "bear_z": -0.75})
    prepared = strat.prepare(df)
    z = prepared["z65"]
    state = prepared["es_state"]
    bull_rows = z >= 0.75
    bear_rows = z <= -0.75
    assert (state[bull_rows] == 1).all()
    assert (state[bear_rows] == 0).all()
    # Between the thresholds the state must only ever repeat a previous state.
    for i in np.where(~(bull_rows | bear_rows))[0]:
        if i == 0:
            continue
        assert state.iloc[i] == state.iloc[i - 1]


def test_ema_std_enters_on_bullish_state_and_sets_stop():
    df = make_candles(1200, seed=4, drift=0.0, vol=0.03)
    strat = build_strategy("ema_std")
    prepared = strat.prepare(df)
    idx = int(np.where((prepared["es_state"] == 1) & (prepared.index > strat.warmup_bars))[0][0])
    sig = strat.signal(prepared, idx, None)
    assert sig.action == "enter_long"
    assert sig.stop_price is not None and sig.stop_price < prepared["close"].iloc[idx]
    assert "z=" in sig.reason
    # While holding, the strategy must exit once the state flips bearish.
    bear_idx = int(np.where((prepared["es_state"] == 0) &
                            (prepared.index > idx))[0][0])
    pos = PositionView(qty=1.0, entry_price=float(prepared["close"].iloc[idx]),
                       entry_ts=int(prepared["bar_ts"].iloc[idx]), stop_price=sig.stop_price,
                       entry_bar_ts=int(prepared["bar_ts"].iloc[idx]))
    assert strat.signal(prepared, bear_idx, pos).action == "exit_long"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_lookahead_signal_matches_truncated_history(name):
    """A signal must be identical whether or not future bars exist."""
    strat = build_strategy(name)
    df = make_candles(700, seed=9, drift=0.0015, vol=0.018)
    full = strat.prepare(df)
    for frac in (0.55, 0.8, 0.97):
        i = int(len(df) * frac)
        trunc = strat.prepare(df.iloc[: i + 1])
        pos = None
        if name == "regime_adaptive":
            pos = PositionView(qty=0.01, entry_price=float(df["close"].iloc[i - 5]),
                               entry_ts=int(df["bar_ts"].iloc[i - 5]),
                               stop_price=float(df["close"].iloc[i - 5]) * 0.9,
                               entry_bar_ts=int(df["bar_ts"].iloc[i - 5]),
                               entry_tag="breakout")
        a = strat.signal(full, i, pos)
        b = strat.signal(trunc, i, pos)
        assert a.action == b.action, f"{name} action changed with future data at {frac}"
        assert a.reason == b.reason, f"{name} reason changed with future data at {frac}"
        if a.stop_price is None:
            assert b.stop_price is None
        else:
            assert b.stop_price == pytest.approx(a.stop_price, rel=1e-12)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategies_are_flat_during_warmup(name):
    strat = build_strategy(name)
    df = make_candles(400, seed=2)
    prepared = strat.prepare(df)
    for i in range(min(strat.warmup_bars, 50)):
        assert strat.signal(prepared, i, None).action == "hold"


@pytest.mark.parametrize("name", [n for n in sorted(STRATEGIES) if n != "meanrev_bb_rsi"])
def test_entry_signals_always_carry_a_stop(name):
    strat = build_strategy(name)
    df = make_candles(1200, seed=13, drift=0.0012, vol=0.02)
    prepared = strat.prepare(df)
    entries = 0
    for i in range(strat.warmup_bars, len(prepared)):
        sig = strat.signal(prepared, i, None)
        if sig.action == "enter_long":
            entries += 1
            assert sig.stop_price is not None and sig.stop_price > 0
            assert sig.stop_price < float(prepared["close"].iloc[i])
            assert sig.reason
    assert entries > 0, f"{name} produced no entries on a trending synthetic series"


def test_regime_adaptive_does_not_trade_high_volatility_regime():
    df = make_candles(800, seed=31, drift=0.0, vol=0.05)
    strat = build_strategy("regime_adaptive")
    prepared = strat.prepare(df)
    high_vol = prepared["regime"] == "high_vol"
    for i in np.where(high_vol)[0]:
        if i < strat.warmup_bars:
            continue
        assert strat.signal(prepared, i, None).action == "hold"


def test_ema_std_trend_filter_blocks_entries_below_ema():
    df = make_candles(900, seed=8, drift=-0.0005, vol=0.02)
    strat = EmaStd({"ema_trend_filter": 200})
    prepared = strat.prepare(df)
    for i in range(strat.warmup_bars, len(prepared)):
        below = prepared["close"].iloc[i] < prepared["ema_trend"].iloc[i]
        if below and prepared["es_state"].iloc[i] == 1:
            assert strat.signal(prepared, i, None).action == "hold"


def test_mean_reversion_requires_an_oversold_dip_inside_an_uptrend():
    # Choppy series with occasional sharp sell-offs and a positive drift.
    df = make_candles(1600, seed=77, drift=0.0015, vol=0.045)
    strat = build_strategy("meanrev_bb_rsi")
    prepared = strat.prepare(df)
    entries = []
    for i in range(strat.warmup_bars, len(prepared)):
        sig = strat.signal(prepared, i, None)
        if sig.action == "enter_long":
            entries.append(i)
            assert sig.stop_price is not None and sig.stop_price < prepared["close"].iloc[i]
            assert prepared["close"].iloc[i] < prepared["bb_lower"].iloc[i]
            assert prepared["rsi"].iloc[i] < strat.params.rsi_entry
            assert prepared["close"].iloc[i] > prepared["ema_trend"].iloc[i]
    assert entries, "expected at least one mean-reversion entry on a choppy series"
