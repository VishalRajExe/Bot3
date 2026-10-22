"""Portfolio risk for a $100 spot account.

Design rules:

* Risk is defined by the stop, never by conviction: notional = risk$ / stop%.
* Position notional is additionally capped by equity percentage and by the cash
  that is not reserved, so the account can never be fully deployed.
* Portfolio heat (sum of open risk at stops) is capped, so two simultaneous
  losers cost a known, small amount.
* Daily loss, peak drawdown and consecutive-loss breakers are latched in state
  and require an operator reset for the drawdown latch.
* Every rejection returns a machine-readable reason so the dashboard and
  Telegram can show *why* a trade was not taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import CostsCfg, RiskCfg
from .regime import RegimeState


@dataclass
class OpenRisk:
    symbol: str
    cluster: str
    notional: float
    risk_usd: float
    stop_pct: float


@dataclass
class Decision:
    approved: bool
    reason: str
    size_usd: float = 0.0
    risk_usd: float = 0.0
    stop_pct: float = 0.0
    checks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "size_usd": round(self.size_usd, 4),
            "risk_usd": round(self.risk_usd, 4),
            "stop_pct": round(self.stop_pct, 4),
            "checks": self.checks,
        }


class RiskManager:
    def __init__(self, cfg: RiskCfg, costs: CostsCfg) -> None:
        self.cfg = cfg
        self.costs = costs

    # ------------------------------------------------------------------ state
    @staticmethod
    def day_start_equity(equity: float, stored: dict | None, day: int) -> float:
        stored = stored or {}
        if stored.get("day") != day or not stored.get("equity"):
            return float(equity)
        return float(stored["equity"])

    def breaker_state(self, *, equity: float, hwm: float, day_start: float) -> dict:
        dd = (hwm - equity) / hwm * 100.0 if hwm > 0 else 0.0
        day_loss = (day_start - equity) / day_start * 100.0 if day_start > 0 else 0.0
        return {
            "drawdown_pct": round(dd, 3),
            "daily_loss_pct": round(day_loss, 3),
            "drawdown_halted": dd >= self.cfg.max_drawdown_pct,
            "daily_halted": day_loss >= self.cfg.daily_loss_limit_pct,
        }

    # ------------------------------------------------------------------ sizing
    def position_size(self, *, equity: float, cash: float, price: float, stop: float,
                      regime: RegimeState, symbol_multiplier: float = 1.0,
                      prob: float | None = None) -> Decision:
        cfg = self.cfg
        checks: dict = {}
        if price <= 0 or stop <= 0 or stop >= price:
            return Decision(False, "invalid stop: must be below entry", checks=checks)
        stop_pct = (price - stop) / price * 100.0
        checks["stop_pct"] = round(stop_pct, 3)

        if stop_pct < 0.5:
            return Decision(False, f"stop {stop_pct:.2f}% too tight for {self.costs.round_trip_pct:.2f}% round-trip costs",
                            stop_pct=stop_pct, checks=checks)
        if stop_pct > 20.0:
            return Decision(False, f"stop {stop_pct:.2f}% wider than 20%", stop_pct=stop_pct,
                            checks=checks)

        edge_bps = 2.0 * stop_pct * 100.0 - self.costs.round_trip_bps
        checks["expected_edge_bps"] = round(edge_bps, 1)
        if edge_bps < cfg.min_expected_edge_bps:
            return Decision(
                False,
                f"2R edge {edge_bps:.0f}bps below required {cfg.min_expected_edge_bps:.0f}bps "
                f"after {self.costs.round_trip_bps:.0f}bps costs",
                stop_pct=stop_pct, checks=checks,
            )

        reg_mult = cfg.regime_multipliers.get(regime.label, 0.5)
        prob_mult = 1.0
        if prob is not None:
            prob_mult = max(0.5, min(1.0, 0.5 + (prob - 0.5) * 2.0))
        risk_usd = equity * (cfg.risk_per_trade_pct / 100.0) * reg_mult * symbol_multiplier * prob_mult
        checks.update({"regime_mult": reg_mult, "symbol_mult": round(symbol_multiplier, 3),
                       "prob_mult": round(prob_mult, 3), "risk_usd": round(risk_usd, 4)})

        notional_by_risk = risk_usd / (stop_pct / 100.0)
        notional_by_equity = equity * cfg.max_position_pct / 100.0
        spendable = max(0.0, cash - equity * cfg.cash_reserve_pct / 100.0)
        notional = min(notional_by_risk, notional_by_equity, spendable)
        checks.update({"risk_based_usd": round(notional_by_risk, 2),
                       "equity_cap_usd": round(notional_by_equity, 2),
                       "spendable_usd": round(spendable, 2)})

        if notional < self.costs.min_notional:
            return Decision(
                False,
                f"sized notional ${notional:.2f} below the ${self.costs.min_notional:.2f} minimum",
                stop_pct=stop_pct, checks=checks,
            )
        fees = notional * self.costs.taker_fee_bps / 10_000.0
        if notional + fees > cash:
            return Decision(False, "not enough cash for notional plus fee", stop_pct=stop_pct,
                            checks=checks)
        checks["fee_usd"] = round(fees, 4)
        return Decision(True, "approved", size_usd=round(notional, 4),
                        risk_usd=round(notional * stop_pct / 100.0, 4),
                        stop_pct=stop_pct, checks=checks)

    # -------------------------------------------------------------- portfolio
    def can_enter(self, *, symbol: str, cluster: str, equity: float, cash: float,
                  open_positions: list[dict], open_risk: list[OpenRisk],
                  regime: RegimeState, breaker: dict, cooldown_until: int,
                  now: int, bar_return_pct: float) -> Decision:
        cfg = self.cfg
        if breaker.get("drawdown_halted"):
            return Decision(False, "drawdown circuit breaker latched (operator reset required)")
        if breaker.get("daily_halted"):
            return Decision(False, "daily loss limit hit; no new entries until UTC midnight")
        if cooldown_until and now < cooldown_until:
            mins = int((cooldown_until - now) / 60)
            return Decision(False, f"symbol cooldown active for {mins} more minutes")
        if len(open_positions) >= cfg.max_positions:
            return Decision(False, f"max_positions {cfg.max_positions} reached")
        if any(p["symbol"] == symbol for p in open_positions):
            return Decision(False, "position already open in this symbol")
        same_cluster = [p for p in open_positions if p.get("cluster") == cluster]
        if len(same_cluster) >= cfg.max_per_cluster:
            return Decision(False, f"cluster '{cluster}' already at cap {cfg.max_per_cluster}")
        heat = sum(r.risk_usd for r in open_risk)
        max_heat = equity * cfg.portfolio_heat_pct / 100.0
        if heat >= max_heat:
            return Decision(False, f"portfolio heat ${heat:.2f} at cap ${max_heat:.2f}")
        if abs(bar_return_pct) > cfg.max_bar_return_pct:
            return Decision(False, f"bar return {bar_return_pct:.1f}% looks like bad data")
        if regime.label == "bear":
            return Decision(False, f"market regime is bear (score {regime.score:.2f}); "
                                   f"long-only system stays in cash")
        return Decision(True, "portfolio checks passed")

    def remaining_heat(self, equity: float, open_risk: list[OpenRisk]) -> float:
        return max(0.0, equity * self.cfg.portfolio_heat_pct / 100.0
                   - sum(r.risk_usd for r in open_risk))
