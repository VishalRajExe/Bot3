"""Strategy registry.

Adding a strategy means implementing :class:`Strategy` and registering it here;
nothing else in the engine changes.
"""

from __future__ import annotations

from .base import Params, PositionView, Signal, Strategy
from .breakout import DonchianBreakout
from .ema_std import EmaStd
from .mean_reversion import BollingerRsiMeanReversion
from .regime_adaptive import RegimeAdaptive
from .trend import EmaCrossTrend

STRATEGIES: dict[str, type[Strategy]] = {
    EmaStd.name: EmaStd,
    EmaCrossTrend.name: EmaCrossTrend,
    DonchianBreakout.name: DonchianBreakout,
    BollingerRsiMeanReversion.name: BollingerRsiMeanReversion,
    RegimeAdaptive.name: RegimeAdaptive,
}


def build_strategy(name: str, params: dict | None = None, timeframe: str | None = None) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown strategy '{name}'; available: {sorted(STRATEGIES)}") from exc
    return cls(params=params, timeframe=timeframe)


def list_strategies() -> list[dict]:
    out = []
    for name, cls in STRATEGIES.items():
        inst = cls()
        out.append({
            "name": name,
            "version": inst.version,
            "timeframe": inst.default_timeframe,
            "description": inst.description,
            "params": inst.params_dict(),
            "warmup_bars": inst.warmup_bars,
        })
    return out


__all__ = ["Strategy", "Signal", "PositionView", "Params", "STRATEGIES",
           "build_strategy", "list_strategies"]
