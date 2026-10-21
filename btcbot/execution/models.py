"""Order, fill and position models shared by backtest and paper trading."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..data.timeframes import human_duration


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


@dataclass
class Order:
    symbol: str
    side: str  # 'buy' | 'sell'
    qty: float
    reason: str
    strategy: str
    strategy_version: str
    signal_bar_ts: int
    target_bar_ts: int | None = None
    order_type: str = "market"
    requested_price: float | None = None
    stop_price: float | None = None
    tag: str = "default"
    id: str = field(default_factory=lambda: new_id("ord"))
    status: str = "pending"
    reject_reason: str | None = None
    run_id: str | None = None
    mode: str = "backtest"

    @property
    def unique_tag(self) -> str:
        return f"{self.run_id}:{self.signal_bar_ts}:{self.side}:{self.strategy_version}"


@dataclass
class Fill:
    order_id: str
    ts: int
    price: float
    qty: float
    notional: float
    fee: float
    fee_bps: float
    slippage_bps: float
    spread_bps: float
    reference_price: float | None = None
    id: str = field(default_factory=lambda: new_id("fil"))


@dataclass
class Position:
    qty: float
    entry_price: float
    entry_ts: int
    entry_bar_ts: int
    stop_price: float | None
    initial_stop: float | None
    entry_reason: str
    entry_signal: str
    entry_order_id: str
    entry_fees: float
    entry_slippage_cost: float = 0.0
    strategy: str = ""
    strategy_version: str = ""
    entry_tag: str = "default"
    trade_id: str = field(default_factory=lambda: new_id("trd"))
    risk_amount: float = 0.0
    max_price: float = 0.0
    min_price: float = float("inf")
    entry_indicators: dict = field(default_factory=dict)
    entry_regime: str | None = None
    anomalies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_price = max(self.max_price, self.entry_price)
        self.min_price = min(self.min_price, self.entry_price)

    @property
    def notional_at_entry(self) -> float:
        return self.qty * self.entry_price

    def mark(self, high: float, low: float) -> None:
        self.max_price = max(self.max_price, high)
        self.min_price = min(self.min_price, low)

    @property
    def mfe_pct(self) -> float:
        return (self.max_price - self.entry_price) / self.entry_price * 100.0

    @property
    def mae_pct(self) -> float:
        return (self.min_price - self.entry_price) / self.entry_price * 100.0


@dataclass
class TradeRecord:
    """A completed round trip.  This is the permanent journal entry."""

    id: str
    mode: str
    run_id: str | None
    symbol: str
    side: str
    strategy: str
    strategy_version: str
    entry_ts: int
    entry_bar_ts: int
    entry_price: float
    entry_signal: str
    entry_reason: str
    exit_ts: int
    exit_price: float
    exit_signal: str
    exit_reason: str
    qty: float
    initial_stop: float | None
    intended_stop: float | None
    stop_distance_pct: float | None
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    return_pct: float
    r_multiple: float | None
    mfe_pct: float
    mae_pct: float
    holding_hours: float
    regime: str | None = None
    model_version: str | None = None
    indicators_entry: dict = field(default_factory=dict)
    sentiment_entry: dict = field(default_factory=dict)
    risk_events: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    entry_order_id: str | None = None
    exit_order_id: str | None = None

    def describe(self) -> str:
        return (
            f"{self.strategy} v{self.strategy_version} {self.mode}: "
            f"{self.qty:.6f} BTC {self.entry_price:,.2f} -> {self.exit_price:,.2f} "
            f"({self.return_pct:+.2f}%, net {self.net_pnl:+.2f}, {human_duration(self.holding_hours)})"
        )
