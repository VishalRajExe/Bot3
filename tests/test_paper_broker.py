from __future__ import annotations

import pytest

from btcbot.config import CostsCfg
from btcbot.execution.models import Order
from btcbot.execution.paper import PaperBroker, RejectedOrder, floor_to_step


def broker(cash: float = 10_000.0, **cost_overrides) -> PaperBroker:
    costs = CostsCfg(**cost_overrides)
    return PaperBroker(costs, symbol="BTC/USDT", initial_equity=cash, run_id="test", mode="paper")


def buy_order(qty: float, stop: float | None = None) -> Order:
    return Order(symbol="BTC/USDT", side="buy", qty=qty, reason="test", strategy="dummy",
                 strategy_version="1", signal_bar_ts=1_600_000_000, stop_price=stop,
                 run_id="test", mode="paper")


def sell_order(qty: float) -> Order:
    return Order(symbol="BTC/USDT", side="sell", qty=qty, reason="test exit", strategy="dummy",
                 strategy_version="1", signal_bar_ts=1_600_086_400, run_id="test", mode="paper")


def test_buy_applies_half_spread_slippage_and_fee():
    b = broker(taker_fee_bps=10.0, spread_bps=2.0, slippage_bps=5.0)
    fill = b.execute(buy_order(1.0, stop=90.0), 100.0, 1_600_000_000)
    expected_price = 100.0 * (1 + (2 / 2 + 5) / 10_000)
    assert fill.price == pytest.approx(expected_price, rel=1e-6)
    assert fill.fee == pytest.approx(fill.notional * 0.001, rel=1e-9)
    assert b.cash == pytest.approx(10_000 - fill.notional - fill.fee)
    assert b.position is not None and b.position.qty == 1.0
    assert b.position.stop_price == 90.0


def test_qty_is_floored_to_exchange_step():
    b = broker(qty_step=0.001)
    fill = b.execute(buy_order(1.0123456), 100.0, 1_600_000_000)
    assert fill.qty == pytest.approx(1.012)
    assert b.position.qty == pytest.approx(1.012)


def test_min_notional_rejection():
    b = broker(min_notional=10.0)
    with pytest.raises(RejectedOrder, match="minimum"):
        b.execute(buy_order(0.01), 100.0, 1_600_000_000)


def test_qty_rounding_to_zero_rejection():
    b = broker(qty_step=0.00001)
    with pytest.raises(RejectedOrder, match="rounds to zero"):
        b.execute(buy_order(0.000001), 100.0, 1_600_000_000)


def test_insufficient_cash_rejection():
    b = broker(cash=100.0)
    with pytest.raises(RejectedOrder, match="insufficient"):
        b.execute(buy_order(1.0), 500.0, 1_600_000_000)


def test_second_position_rejected():
    b = broker()
    b.execute(buy_order(0.1), 100.0, 1_600_000_000)
    with pytest.raises(RejectedOrder, match="position already open"):
        b.execute(buy_order(0.1), 100.0, 1_600_000_000)


def test_sell_without_position_rejected():
    b = broker()
    with pytest.raises(RejectedOrder, match="no open position"):
        b.execute(sell_order(1.0), 100.0, 1_600_000_000)


def test_round_trip_cash_and_pnl_are_consistent():
    b = broker()
    entry = b.execute(buy_order(0.5, stop=80.0), 100.0, 1_600_000_000)
    exit_fill = b.execute(sell_order(0.5), 110.0, 1_600_086_400)
    trade = b.close_position(exit_fill, "signal exit")
    assert trade.gross_pnl == pytest.approx((exit_fill.price - entry.price) * 0.5)
    assert trade.fees == pytest.approx(entry.fee + exit_fill.fee)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.fees)
    assert b.cash == pytest.approx(10_000 + trade.net_pnl)
    assert b.position is None
    assert trade.return_pct == pytest.approx(trade.net_pnl / (entry.price * 0.5) * 100)


def test_sell_always_closes_the_whole_position():
    b = broker()
    b.execute(buy_order(0.2), 100.0, 1_600_000_000)
    fill = b.execute(sell_order(0.2), 105.0, 1_600_086_400)
    trade = b.close_position(fill, "test")
    assert fill.qty == pytest.approx(0.2)
    assert trade.qty == pytest.approx(0.2)


def test_mfe_mae_tracking():
    b = broker()
    b.execute(buy_order(0.1), 100.0, 1_600_000_000)
    entry = b.position.entry_price
    b.position.mark(high=120.0, low=90.0)
    assert b.position.mfe_pct == pytest.approx((120.0 - entry) / entry * 100)
    assert b.position.mae_pct == pytest.approx((90.0 - entry) / entry * 100)


def test_fractional_step_helper():
    assert floor_to_step(1.99999, 0.01) == pytest.approx(1.99)
    assert floor_to_step(0.000019, 0.00001) == pytest.approx(0.00001)
    assert floor_to_step(3.0, 1.0) == pytest.approx(3.0)


def test_state_roundtrip_restores_position():
    b = broker()
    b.execute(buy_order(0.1, stop=80.0), 100.0, 1_600_000_000)
    state = b.to_dict()
    b2 = broker()
    b2.load_dict(state)
    assert b2.position is not None
    assert b2.position.qty == b.position.qty
    assert b2.cash == pytest.approx(b.cash)
