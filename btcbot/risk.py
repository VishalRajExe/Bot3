"""Centralised risk management.

Every proposed entry passes through :class:`RiskManager`.  Strategies and ML
models have no ability to bypass, widen or disable these limits.  Exits are
never blocked: risk controls may stop new risk, never trap an open position.

Implemented controls
--------------------
* risk-based position sizing (0.5% of equity at the stop by default)
* maximum position notional (20% of equity)
* maximum one simultaneous position, spot only, no leverage
* daily equity-loss circuit breaker (2% from UTC day start)
* high-water-mark drawdown breaker (10%) -- requires an explicit operator reset
* consecutive-loss cooldown (3 losses -> 24h)
* stale-data guard and abnormal single-bar return guard
* minimum/maximum stop distance sanity checks
* short re-entry cooldown after any exit
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import CostsCfg, RiskCfg
from .data.timeframes import tf_seconds


@dataclass
class RiskDecision:
    allowed: bool
    code: str
    reason: str


@dataclass
class SizingResult:
    allowed: bool
    qty: float
    notional: float
    risk_amount: float
    stop_distance_pct: float | None
    capped_by: str
    reason: str


@dataclass
class RiskState:
    hwm: float
    day_key: int | None = None
    day_start_equity: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: int = 0
    drawdown_halted: bool = False
    drawdown_halted_at: int | None = None
    daily_halt_day: int | None = None
    no_entry_until: int = 0
    last_event_keys: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: RiskCfg, costs: CostsCfg, initial_equity: float,
                 *, timeframe: str = "1d") -> None:
        self.cfg = cfg
        self.costs = costs
        self.initial_equity = float(initial_equity)
        self.timeframe = timeframe
        self.state = RiskState(hwm=float(initial_equity), day_start_equity=float(initial_equity))
        self.events: list[dict] = []
        self.last_rejection: RiskDecision | None = None

    # ------------------------------------------------------------- events
    def _emit(self, ts: int, kind: str, action: str, detail: str, value: dict | None = None,
              once_key: str | None = None) -> None:
        if once_key:
            if once_key in self.state.last_event_keys:
                return
            self.state.last_event_keys.append(once_key)
            self.state.last_event_keys = self.state.last_event_keys[-50:]
        self.events.append({"ts": int(ts), "kind": kind, "action": action,
                            "detail": detail, "value": value or {}})

    def drain_events(self) -> list[dict]:
        out = self.events
        self.events = []
        return out

    # ------------------------------------------------------------- equity
    def on_equity(self, ts: int, equity: float, *, idle_bars: int = 0) -> None:
        """Update HWM, day anchor and circuit breakers.  Call once per closed bar."""
        st = self.state
        day = int(ts) // 86_400
        if st.day_key is None:
            st.day_key = day
            st.day_start_equity = equity
        elif day != st.day_key:
            st.day_key = day
            st.day_start_equity = equity
            st.daily_halt_day = None

        if equity > st.hwm:
            st.hwm = equity

        loss_pct = (st.day_start_equity - equity) / st.day_start_equity * 100.0
        if loss_pct >= self.cfg.daily_loss_limit_pct and st.daily_halt_day != day:
            st.daily_halt_day = day
            self._emit(
                ts, "daily_loss_limit", "block_new_entries",
                f"daily equity loss {loss_pct:.2f}% >= limit {self.cfg.daily_loss_limit_pct:.2f}%",
                {"loss_pct": round(loss_pct, 4), "day": day},
                once_key=f"daily_loss:{day}",
            )

        dd_pct = (st.hwm - equity) / st.hwm * 100.0
        if dd_pct >= self.cfg.max_drawdown_pct and not st.drawdown_halted:
            st.drawdown_halted = True
            st.drawdown_halted_at = int(ts)
            self._emit(
                ts, "max_drawdown", "halt_new_entries",
                f"drawdown {dd_pct:.2f}% from high-water mark exceeded "
                f"{self.cfg.max_drawdown_pct:.2f}%; operator reset required",
                {"drawdown_pct": round(dd_pct, 4), "hwm": round(st.hwm, 2), "equity": round(equity, 2)},
                once_key=f"drawdown:{st.hwm:.2f}",
            )

    # ------------------------------------------------------- entry gating
    def check_entry(self, *, ts: int, equity: float, cash: float, has_position: bool,
                    bar_ts: int, bar_return: float | None, pending_order: bool = False) -> RiskDecision:
        st = self.state
        cfg = self.cfg

        def reject(code: str, reason: str) -> RiskDecision:
            d = RiskDecision(False, code, reason)
            self.last_rejection = d
            return d

        if cfg.leverage != 1:
            return reject("leverage", "leverage must be 1")
        if has_position:
            return reject("max_positions", "a position is already open")
        if pending_order:
            return reject("pending_order", "an order is already pending")
        if st.drawdown_halted:
            return reject(
                "drawdown_halt",
                f"max drawdown breaker active ({cfg.max_drawdown_pct:.1f}% from HWM); "
                "operator reset required",
            )
        if st.daily_halt_day is not None and st.daily_halt_day == int(ts) // 86_400:
            return reject("daily_loss_halt", "daily loss circuit breaker active for this UTC day")
        if ts < st.cooldown_until:
            hours = (st.cooldown_until - ts) / 3600.0
            return reject(
                "loss_cooldown",
                f"{st.consecutive_losses} consecutive losses: {hours:.1f}h cooldown remaining",
            )
        if ts < st.no_entry_until:
            return reject("reentry_cooldown", "post-exit re-entry cooldown active")

        bar_age = int(ts) - (int(bar_ts) + tf_seconds(self.timeframe))
        if bar_age > cfg.max_bars_stale * tf_seconds(self.timeframe):
            return reject("stale_data", f"latest closed bar is {bar_age / 3600:.1f}h old")

        if bar_return is not None and abs(bar_return) > cfg.max_abs_bar_return:
            return reject(
                "abnormal_price",
                f"abnormal single-bar return {bar_return * 100:+.2f}% exceeds "
                f"{cfg.max_abs_bar_return * 100:.0f}% guard",
            )
        if cash <= 0:
            return reject("no_cash", "no simulated cash available")
        return RiskDecision(True, "ok", "entry permitted")

    # ---------------------------------------------------------- position size
    def size_position(self, *, equity: float, cash: float, entry_price: float,
                      stop_price: float | None) -> SizingResult:
        cfg = self.cfg
        if entry_price <= 0:
            return SizingResult(False, 0.0, 0.0, 0.0, None, "invalid",
                                "invalid entry price")
        if stop_price is None or stop_price >= entry_price:
            return SizingResult(
                False, 0.0, 0.0, 0.0, None, "no_stop",
                "strategy produced no valid stop-loss; position cannot be sized safely",
            )

        stop_distance_pct = (entry_price - stop_price) / entry_price * 100.0
        if stop_distance_pct < cfg.min_stop_distance_pct:
            return SizingResult(False, 0.0, 0.0, 0.0, stop_distance_pct, "stop_too_tight",
                                f"stop distance {stop_distance_pct:.2f}% below minimum "
                                f"{cfg.min_stop_distance_pct:.2f}%")
        if stop_distance_pct > cfg.max_stop_distance_pct:
            return SizingResult(False, 0.0, 0.0, 0.0, stop_distance_pct, "stop_too_wide",
                                f"stop distance {stop_distance_pct:.2f}% above maximum "
                                f"{cfg.max_stop_distance_pct:.2f}%")

        fee_rate = self.costs.taker_fee_bps / 10_000.0
        cost_rate = fee_rate + (self.costs.spread_bps / 2.0 + self.costs.slippage_bps) / 10_000.0
        gap_buffer = 1.0 + cfg.entry_gap_buffer_pct / 100.0
        eff_entry = entry_price * gap_buffer * (1.0 + cost_rate)
        eff_stop = stop_price * (1.0 - cost_rate)
        risk_per_unit = eff_entry - eff_stop
        if risk_per_unit <= 0:
            return SizingResult(False, 0.0, 0.0, 0.0, stop_distance_pct, "invalid",
                                "stop must be below entry after costs")

        risk_amount = equity * cfg.risk_per_trade_pct / 100.0
        qty_risk = risk_amount / risk_per_unit
        qty_exposure = equity * cfg.max_position_pct / 100.0 / eff_entry
        qty_cash = max(cash, 0.0) / (eff_entry * (1.0 + fee_rate))

        qty = min(qty_risk, qty_exposure, qty_cash)
        capped_by = min(
            [("risk_budget", qty_risk), ("max_exposure", qty_exposure), ("available_cash", qty_cash)],
            key=lambda kv: kv[1],
        )[0]
        step = self.costs.qty_step
        qty = math.floor(qty / step + 1e-12) * step if step > 0 else qty
        qty = round(qty, 10)
        notional = qty * eff_entry
        if qty <= 0 or notional < self.costs.min_notional:
            return SizingResult(
                False, 0.0, notional, risk_amount, stop_distance_pct, capped_by,
                f"position too small: qty {qty:.8f}, notional {notional:.2f} below minimum "
                f"{self.costs.min_notional:.2f}",
            )
        return SizingResult(
            True, qty, notional, risk_amount, stop_distance_pct, capped_by,
            f"sized {qty:.8f} BTC (${notional:,.2f}) at {stop_distance_pct:.2f}% stop, "
            f"risk ${risk_amount:,.2f}, capped by {capped_by}",
        )

    # ------------------------------------------------------------- outcomes
    def on_trade_closed(self, *, ts: int, net_pnl: float, timeframe: str | None = None) -> None:
        st = self.state
        tf = timeframe or self.timeframe
        # Any exit imposes a one-bar re-entry pause; the trade's own exit bar
        # cannot immediately re-lever into the same signal.
        st.no_entry_until = int(ts) + tf_seconds(tf)

        if net_pnl < 0:
            st.consecutive_losses += 1
            if st.consecutive_losses >= self.cfg.consecutive_losses_limit:
                st.cooldown_until = int(ts) + int(self.cfg.cooldown_hours * 3600)
                self._emit(
                    ts, "loss_cooldown", "block_new_entries",
                    f"{st.consecutive_losses} consecutive losing trades: "
                    f"{self.cfg.cooldown_hours:.0f}h cooldown",
                    {"consecutive_losses": st.consecutive_losses,
                     "cooldown_until": st.cooldown_until},
                    once_key=f"cooldown:{st.cooldown_until}",
                )
                st.consecutive_losses = 0
        else:
            st.consecutive_losses = 0

    def reset_drawdown_halt(self, ts: int | None = None) -> None:
        """Explicit operator reset of the high-water-mark breaker."""
        if self.state.drawdown_halted:
            self.state.drawdown_halted = False
            self.state.drawdown_halted_at = None
            self.state.hwm = self.state.hwm  # keep HWM; equity must climb back naturally
            if ts is not None:
                self._emit(ts, "manual_reset", "allow_new_entries",
                           "operator reset the drawdown circuit breaker", once_key=None)

    # ---------------------------------------------------------------- status
    def status(self, equity: float | None = None) -> dict:
        st = self.state
        eq = st.hwm if equity is None else equity
        dd = (st.hwm - eq) / st.hwm * 100.0 if st.hwm else 0.0
        day_loss = (
            (st.day_start_equity - eq) / st.day_start_equity * 100.0
            if st.day_start_equity else 0.0
        )
        return {
            "high_water_mark": round(st.hwm, 2),
            "drawdown_pct": round(dd, 3),
            "daily_loss_pct": round(day_loss, 3),
            "consecutive_losses": st.consecutive_losses,
            "cooldown_until": st.cooldown_until,
            "no_entry_until": st.no_entry_until,
            "drawdown_halted": st.drawdown_halted,
            "daily_halt_active": st.daily_halt_day is not None
            and st.day_key is not None and st.daily_halt_day == st.day_key,
            "limits": {
                "risk_per_trade_pct": self.cfg.risk_per_trade_pct,
                "max_position_pct": self.cfg.max_position_pct,
                "max_positions": self.cfg.max_positions,
                "daily_loss_limit_pct": self.cfg.daily_loss_limit_pct,
                "max_drawdown_pct": self.cfg.max_drawdown_pct,
                "consecutive_losses_limit": self.cfg.consecutive_losses_limit,
                "cooldown_hours": self.cfg.cooldown_hours,
                "leverage": self.cfg.leverage,
            },
        }

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "hwm": self.state.hwm,
            "day_key": self.state.day_key,
            "day_start_equity": self.state.day_start_equity,
            "consecutive_losses": self.state.consecutive_losses,
            "cooldown_until": self.state.cooldown_until,
            "drawdown_halted": self.state.drawdown_halted,
            "drawdown_halted_at": self.state.drawdown_halted_at,
            "daily_halt_day": self.state.daily_halt_day,
            "no_entry_until": self.state.no_entry_until,
            "last_event_keys": self.state.last_event_keys,
        }

    def load_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
