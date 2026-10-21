"""Paper-trading broker.

Simulates spot market orders with realistic frictions:

* taker fee on both sides (default 10 bps/side, Binance spot standard tier),
* half the quoted spread paid per side,
* adverse slippage,
* tick/step rounding and minimum notional enforcement,
* insufficient-cash and no-position rejections,
* conservative stop handling (a gap through the stop fills at the open, not at
  the stop price).

There is **no live execution path**.  This module cannot place a real order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict

from ..config import CostsCfg
from ..data.timeframes import human_duration
from .models import Fill, Order, Position, TradeRecord, new_id

log = logging.getLogger(__name__)


class RejectedOrder(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def floor_to_step(value: float, step: float) -> float:
    """Floor a quantity to an exchange step without binary-float truncation.

    ``1.0 / 0.00001`` evaluates to ``99999.99999999999``; a naive floor would
    turn 1 BTC into 0.99999 BTC.  The relative epsilon absorbs that error while
    staying far below one step.
    """
    if step <= 0:
        return value
    units = value / step
    units = math.floor(units + 1e-9 * max(1.0, abs(units)))
    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(units * step, decimals + 1)


class PaperBroker:
    def __init__(self, costs: CostsCfg, *, symbol: str, initial_equity: float,
                 run_id: str | None = None, mode: str = "paper") -> None:
        if initial_equity <= 0:
            raise ValueError("initial equity must be positive")
        self.costs = costs
        self.symbol = symbol
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.position: Position | None = None
        self.run_id = run_id
        self.mode = mode
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage_cost = 0.0
        self.n_trades = 0
        self.last_price: float | None = None
        self.trades: list[TradeRecord] = []

    # ------------------------------------------------------------- valuation
    def equity(self, price: float | None = None) -> float:
        mark = price if price is not None else self.last_price
        if self.position is None or mark is None:
            return self.cash
        return self.cash + self.position.qty * mark

    def exposure_pct(self, price: float | None = None) -> float:
        if self.position is None:
            return 0.0
        mark = price if price is not None else (self.last_price or self.position.entry_price)
        eq = self.equity(mark)
        return (self.position.qty * mark) / eq * 100.0 if eq > 0 else 0.0

    # ------------------------------------------------------------ cost model
    def _half_spread(self) -> float:
        return self.costs.spread_bps / 2.0 / 10_000.0

    def _slippage(self) -> float:
        return self.costs.slippage_bps / 10_000.0

    def _execution_price(self, market_price: float, side: str) -> float:
        adverse = self._half_spread() + self._slippage()
        raw = market_price * (1 + adverse) if side == "buy" else market_price * (1 - adverse)
        tick = self.costs.price_tick
        if tick > 0:
            raw = round(raw / tick) * tick
            decimals = max(0, int(round(-math.log10(tick))))
            raw = round(raw, decimals)
        return raw

    def _quantize_qty(self, qty: float) -> float:
        return floor_to_step(qty, self.costs.qty_step)

    # --------------------------------------------------------------- orders
    def execute(self, order: Order, market_price: float, ts: int) -> Fill:
        """Execute a market order at ``market_price`` or raise RejectedOrder."""
        if not (market_price > 0):
            raise RejectedOrder("invalid market price")
        price = self._execution_price(market_price, order.side)
        qty = self._quantize_qty(order.qty)
        if qty <= 0:
            raise RejectedOrder(f"quantity {order.qty:.10f} rounds to zero at step {self.costs.qty_step}")
        notional = qty * price
        if notional < self.costs.min_notional:
            raise RejectedOrder(
                f"notional {notional:.2f} below minimum {self.costs.min_notional:.2f}"
            )
        fee = notional * self.costs.taker_fee_bps / 10_000.0
        slippage_bps = abs(price - market_price) / market_price * 10_000.0

        if order.side == "buy":
            if self.position is not None:
                raise RejectedOrder("position already open (max 1 position)")
            if self.cash < notional + fee:
                raise RejectedOrder(
                    f"insufficient paper cash: need {notional + fee:.2f}, have {self.cash:.2f}"
                )
            self.cash -= notional + fee
            self.position = Position(
                qty=qty,
                entry_price=price,
                entry_ts=ts,
                entry_bar_ts=order.signal_bar_ts,
                stop_price=order.stop_price,
                initial_stop=order.stop_price,
                entry_reason=order.reason,
                entry_signal=f"enter_long@{order.signal_bar_ts}",
                entry_order_id=order.id,
                entry_fees=fee,
                entry_slippage_cost=abs(price - market_price) * qty,
                strategy=order.strategy,
                strategy_version=order.strategy_version,
                entry_tag=order.tag,
            )
        elif order.side == "sell":
            if self.position is None:
                raise RejectedOrder("no open position to sell")
            if qty > self.position.qty + 1e-12:
                raise RejectedOrder(
                    f"cannot sell {qty:.8f}: position is {self.position.qty:.8f}"
                )
            qty = self.position.qty  # sell the whole spot position
            notional = qty * price
            fee = notional * self.costs.taker_fee_bps / 10_000.0
            self.cash += notional - fee
        else:
            raise RejectedOrder(f"unsupported side {order.side}")

        self.total_fees += fee
        self.total_slippage_cost += abs(price - market_price) * qty
        self.last_price = market_price
        return Fill(
            order_id=order.id,
            ts=ts,
            price=price,
            qty=qty,
            notional=notional,
            fee=fee,
            fee_bps=self.costs.taker_fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=self.costs.spread_bps,
            reference_price=market_price,
        )

    def close_position(self, fill: Fill, exit_reason: str, exit_signal: str = "exit_long",
                       extra_fees: float = 0.0) -> TradeRecord:
        pos = self.position
        if pos is None:
            raise RuntimeError("no open position to close")
        gross = (fill.price - pos.entry_price) * pos.qty
        fees = pos.entry_fees + fill.fee + extra_fees
        net = gross - fees
        denom = pos.entry_price * pos.qty
        record = TradeRecord(
            id=pos.trade_id,
            mode=self.mode,
            run_id=self.run_id,
            symbol=self.symbol,
            side="long",
            strategy=pos.strategy,
            strategy_version=pos.strategy_version,
            entry_ts=pos.entry_ts,
            entry_bar_ts=pos.entry_bar_ts,
            entry_price=pos.entry_price,
            entry_signal=pos.entry_signal,
            entry_reason=pos.entry_reason,
            exit_ts=fill.ts,
            exit_price=fill.price,
            exit_signal=exit_signal,
            exit_reason=exit_reason,
            qty=pos.qty,
            initial_stop=pos.initial_stop,
            intended_stop=pos.stop_price,
            stop_distance_pct=(
                (pos.entry_price - pos.initial_stop) / pos.entry_price * 100.0
                if pos.initial_stop else None
            ),
            gross_pnl=gross,
            fees=fees,
            slippage_cost=pos.entry_slippage_cost + abs(
                fill.price - (fill.reference_price or fill.price)
            ) * pos.qty,
            net_pnl=net,
            return_pct=net / denom * 100.0 if denom else 0.0,
            r_multiple=(net / pos.risk_amount) if pos.risk_amount > 0 else None,
            mfe_pct=pos.mfe_pct,
            mae_pct=pos.mae_pct,
            holding_hours=(fill.ts - pos.entry_ts) / 3600.0,
            regime=pos.entry_regime,
            indicators_entry=pos.entry_indicators,
            anomalies=list(pos.anomalies),
            entry_order_id=pos.entry_order_id,
            exit_order_id=fill.order_id,
        )
        self.realized_pnl += net
        self.n_trades += 1
        self.trades.append(record)
        self.position = None
        return record

    def trade_summary(self) -> str:
        if not self.trades:
            return "no closed trades"
        last = self.trades[-1]
        return f"{last.describe()} after {human_duration(last.holding_hours)}"

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "position": asdict(self.position) if self.position else None,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
            "total_slippage_cost": self.total_slippage_cost,
            "n_trades": self.n_trades,
            "last_price": self.last_price,
            "run_id": self.run_id,
            "mode": self.mode,
        }

    def load_dict(self, data: dict) -> None:
        self.cash = float(data.get("cash", self.initial_equity))
        self.realized_pnl = float(data.get("realized_pnl", 0.0))
        self.total_fees = float(data.get("total_fees", 0.0))
        self.total_slippage_cost = float(data.get("total_slippage_cost", 0.0))
        self.n_trades = int(data.get("n_trades", 0))
        self.last_price = data.get("last_price")
        pos = data.get("position")
        self.position = Position(**pos) if pos else None
