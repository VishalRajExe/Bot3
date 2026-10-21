from __future__ import annotations

import pytest

from btcbot.config import CostsCfg, RiskCfg
from btcbot.risk import RiskManager

DAY = 86_400


def manager(equity: float = 10_000.0, timeframe: str = "1d", **overrides) -> RiskManager:
    cfg = RiskCfg(**overrides)
    rm = RiskManager(cfg, CostsCfg(), equity, timeframe=timeframe)
    rm.on_equity(DAY * 100, equity)
    return rm


def test_position_size_respects_risk_budget_and_stop_distance():
    rm = manager()
    res = rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=92.0)
    assert res.allowed
    risk_at_stop = (res.qty * 100.0) - (res.qty * 92.0)
    # 0.5% of 10k = 50; gap buffer makes the planned risk slightly smaller
    assert risk_at_stop <= 50.0
    assert risk_at_stop >= 40.0
    assert res.notional <= 0.20 * 10_000 + 1e-6


def test_max_exposure_cap_binds_for_tight_stops():
    rm = manager()
    res = rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=98.9)
    assert res.allowed
    assert res.capped_by == "max_exposure"
    assert res.notional <= 0.20 * 10_000 * 1.02


def test_sizing_rejects_missing_tight_and_wide_stops():
    rm = manager()
    assert not rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=None).allowed
    tight = rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=99.9)
    assert not tight.allowed and tight.capped_by == "stop_too_tight"
    assert rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=98.0).allowed
    wide = rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=60.0)
    assert not wide.allowed and wide.capped_by == "stop_too_wide"
    invalid = rm.size_position(equity=10_000, cash=10_000, entry_price=100.0, stop_price=110.0)
    assert not invalid.allowed


def test_sizing_never_exceeds_available_cash():
    rm = manager()
    res = rm.size_position(equity=10_000, cash=300.0, entry_price=100.0, stop_price=92.0)
    assert res.allowed
    assert res.notional < 300.0
    assert res.capped_by == "available_cash"


def test_position_too_small_is_rejected():
    rm = manager(equity=100.0)
    res = rm.size_position(equity=100.0, cash=100.0, entry_price=80_000.0, stop_price=70_000.0)
    assert not res.allowed


def test_daily_loss_circuit_breaker_blocks_new_entries():
    rm = manager()
    rm.on_equity(DAY * 100 + 3_600, 9_700)  # -3% on the day
    decision = rm.check_entry(ts=DAY * 100 + 7_200, equity=9_700, cash=9_700,
                              has_position=False, bar_ts=DAY * 100, bar_return=0.0)
    assert not decision.allowed
    assert decision.code == "daily_loss_halt"
    events = rm.drain_events()
    assert any(e["kind"] == "daily_loss_limit" for e in events)
    # Next UTC day resets the daily breaker.
    rm.on_equity(DAY * 101 + 3_600, 9_700)
    assert rm.check_entry(ts=DAY * 101 + 3_600, equity=9_700, cash=9_700, has_position=False,
                          bar_ts=DAY * 101, bar_return=0.0).allowed


def test_drawdown_breaker_halts_until_manual_reset():
    rm = manager()
    rm.on_equity(DAY * 101, 8_900)  # 11% below the 10k HWM
    assert rm.state.drawdown_halted
    assert rm.check_entry(ts=DAY * 101 + 100, equity=8_900, cash=8_900, has_position=False,
                          bar_ts=DAY * 101, bar_return=0.0).code == "drawdown_halt"
    rm.reset_drawdown_halt(DAY * 102)
    assert rm.check_entry(ts=DAY * 102 + 100, equity=8_900, cash=8_900, has_position=False,
                          bar_ts=DAY * 102, bar_return=0.0).allowed


def test_consecutive_loss_cooldown():
    rm = manager()
    ts = DAY * 103
    for k in range(3):
        rm.on_trade_closed(ts=ts + k * 3_600, net_pnl=-5.0)
    assert rm.state.cooldown_until == ts + 2 * 3_600 + int(24 * 3_600)
    decision = rm.check_entry(ts=ts + 3 * 3_600, equity=10_000, cash=10_000,
                              has_position=False, bar_ts=DAY * 103, bar_return=0.0)
    assert not decision.allowed
    assert decision.code == "loss_cooldown"
    # A later win clears the streak.
    rm2 = manager()
    rm2.on_trade_closed(ts=ts, net_pnl=-5.0)
    rm2.on_trade_closed(ts=ts + 100, net_pnl=15.0)
    assert rm2.state.consecutive_losses == 0


def test_reentry_cooldown_after_every_exit():
    rm = manager()
    rm.on_trade_closed(ts=DAY * 104, net_pnl=10.0)
    decision = rm.check_entry(ts=DAY * 104, equity=10_000, cash=10_000, has_position=False,
                              bar_ts=DAY * 104, bar_return=0.0)
    assert not decision.allowed and decision.code == "reentry_cooldown"
    assert rm.check_entry(ts=DAY * 104 + DAY, equity=10_000, cash=10_000, has_position=False,
                          bar_ts=DAY * 105, bar_return=0.0).allowed


def test_stale_data_and_abnormal_price_are_rejected():
    now = DAY * 200
    rm = manager()
    rm.on_equity(now, 10_000)
    stale = rm.check_entry(ts=now, equity=10_000, cash=10_000, has_position=False,
                           bar_ts=now - 5 * DAY, bar_return=0.0)
    assert not stale.allowed and stale.code == "stale_data"
    abnormal = rm.check_entry(ts=now, equity=10_000, cash=10_000, has_position=False,
                              bar_ts=now - DAY, bar_return=0.60)
    assert not abnormal.allowed and abnormal.code == "abnormal_price"


def test_existing_position_and_pending_order_block_entries():
    rm = manager()
    assert rm.check_entry(ts=DAY * 105, equity=10_000, cash=10_000, has_position=True,
                          bar_ts=DAY * 104, bar_return=0.0).code == "max_positions"
    assert rm.check_entry(ts=DAY * 105, equity=10_000, cash=10_000, has_position=False,
                          bar_ts=DAY * 104, bar_return=0.0, pending_order=True).code == "pending_order"


def test_state_persistence_roundtrip():
    rm = manager()
    rm.on_equity(DAY * 150, 9_000)
    rm.on_trade_closed(ts=DAY * 150, net_pnl=-10)
    dump = rm.to_dict()
    rm2 = manager()
    rm2.load_dict(dump)
    assert rm2.state.hwm == rm.state.hwm
    assert rm2.state.drawdown_halted == rm.state.drawdown_halted
    assert rm2.status(9_000)["drawdown_pct"] == rm.status(9_000)["drawdown_pct"]


def test_leverage_and_unsafe_config_refused():
    with pytest.raises(Exception):
        RiskCfg(leverage=5)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        RiskCfg(risk_per_trade_pct=10.0)
    with pytest.raises(Exception):
        RiskCfg(daily_loss_limit_pct=-1.0)
