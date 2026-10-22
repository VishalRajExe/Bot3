"""Three long-only spot strategy families, each with its own parameter set.

The bot does not run one universal rule on every coin.  Each symbol is assigned
a family (trend breakout, trend pullback or mean reversion) and its own
parameters in ``config/unibot.json``; the assignment is also the thing the
research script tests and reports, so a switch is an evidence-backed change.

Every signal is long-only, exposes the stop it wants, and explains itself with
the indicator values that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from .features import DONCHIAN_WINDOWS

Action = Literal["enter_long", "exit_long", "hold"]


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BreakoutParams(_Params):
    donchian: int = 55
    trend_ema: Literal[50, 100, 200] = 100
    atr_window: int = 14
    stop_atr: float = 3.0
    trail_atr: float = 4.0
    min_atr_pct: float = 0.8
    max_atr_pct: float = 18.0
    min_vol_ratio: float = 1.0

    @field_validator("donchian")
    @classmethod
    def _channel(cls, v: int) -> int:
        if v not in DONCHIAN_WINDOWS:
            raise ValueError(f"donchian must be one of {DONCHIAN_WINDOWS}")
        return v

    @field_validator("stop_atr", "trail_atr")
    @classmethod
    def _positive_atr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ATR multiples must be positive")
        return v


class PullbackParams(_Params):
    ema_fast: Literal[20, 50] = 50
    ema_slow: Literal[100, 200] = 200
    rsi_window: int = 14
    rsi_reclaim: float = 42.0
    rsi_take_profit: float = 72.0
    stop_atr: float = 2.5
    trail_atr: float = 3.2
    min_atr_pct: float = 1.0
    max_atr_pct: float = 20.0

    @field_validator("stop_atr", "trail_atr")
    @classmethod
    def _positive_atr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ATR multiples must be positive")
        return v


class MeanRevParams(_Params):
    bb_window: int = 20
    bb_std: float = 2.0
    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    adx_max: float = 22.0
    stop_atr: float = 2.4
    trail_atr: float = 3.0
    min_atr_pct: float = 1.0
    max_atr_pct: float = 20.0

    @field_validator("bb_std")
    @classmethod
    def _positive_std(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("bb_std must be positive")
        return v

    @field_validator("rsi_entry")
    @classmethod
    def _entry_band(cls, v: float) -> float:
        if not 0 < v < 50:
            raise ValueError("rsi_entry must be between 0 and 50")
        return v


class SmcParams(_Params):
    bias_ema: Literal[20, 50, 100, 200] = 50
    fvg_lookback_bars: int = 40
    min_fvg_atr_ratio: float = 0.2
    rr_ratio: float = 2.0
    stop_buffer_atr: float = 0.2
    min_atr_pct: float = 0.5
    max_atr_pct: float = 25.0

    @field_validator("rr_ratio")
    @classmethod
    def _positive_rr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("rr_ratio must be positive")
        return v


@dataclass(frozen=True)
class PositionView:
    qty: float
    entry_price: float
    entry_ts: int
    stop_price: float | None
    entry_bar_ts: int
    trail_atr: float = 0.0


@dataclass
class Signal:
    action: Action
    reason: str
    stop_price: float | None = None
    trail_atr: float = 0.0
    indicators: dict = field(default_factory=dict)
    tag: str = "default"

    @classmethod
    def hold(cls, reason: str, **indicators) -> "Signal":
        return cls("hold", reason, None, 0.0, indicators)


class Strategy:
    name = "base"
    version = "0.0.0"
    PARAMS: type[_Params] = _Params

    def __init__(self, params: dict | None = None) -> None:
        self.params = self.PARAMS.model_validate(params or {})

    @property
    def warmup_bars(self) -> int:
        raise NotImplementedError

    def signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        if i < self.warmup_bars or i >= len(df):
            return Signal.hold("warm-up")
        return self._signal(df, i, position)

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        raise NotImplementedError

    def params_dict(self) -> dict:
        return self.params.model_dump()

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version,
                "params": self.params_dict(), "warmup_bars": self.warmup_bars}


class TrendBreakout(Strategy):
    """Buy new N-bar highs inside an established uptrend, ride with a trail."""

    name = "trend_breakout"
    version = "1.0.0"
    PARAMS = BreakoutParams

    @property
    def warmup_bars(self) -> int:
        return max(int(self.params.trend_ema), int(self.params.donchian)) + 20

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        p = self.params
        row = df.iloc[i]
        close = float(row["close"])
        atr_abs = float(row["atr14"])
        atr_pct = float(row["atr_pct"])
        hi = float(row[f"don_hi_{p.donchian}"])
        trend = float(row[f"ema{p.trend_ema}"])
        vol_ratio = float(row["vol_ratio"])
        ind = {"close": close, "atr_pct": round(atr_pct, 3), "trend_ema": round(trend, 6),
               "prior_high": round(hi, 6), "vol_ratio": round(vol_ratio, 2),
               "rsi14": round(float(row["rsi14"]), 2)}

        if not (p.min_atr_pct <= atr_pct <= p.max_atr_pct):
            return Signal.hold(f"volatility {atr_pct:.2f}% outside [{p.min_atr_pct}, {p.max_atr_pct}]", **ind)

        if position is None:
            if close > hi and close > trend and vol_ratio >= p.min_vol_ratio:
                stop = close - p.stop_atr * atr_abs
                return Signal(
                    "enter_long",
                    f"broke {p.donchian}-bar high {hi:.6g} above EMA{p.trend_ema} "
                    f"with {vol_ratio:.2f}x volume",
                    stop_price=stop,
                    trail_atr=p.trail_atr,
                    indicators={**ind, "stop": round(stop, 6)},
                    tag="breakout",
                )
            why = ("below prior high" if close <= hi else "below trend EMA")
            return Signal.hold(f"no breakout: {why}", **ind)

        # --- exit rules (the engine also enforces the trailing stop)
        if close < trend:
            return Signal("exit_long", f"close {close:.6g} lost EMA{p.trend_ema} {trend:.6g}",
                          indicators=ind)
        if close < float(row["don_lo_20"]):
            return Signal("exit_long", f"close broke 20-bar low {float(row['don_lo_20']):.6g}",
                          indicators=ind)
        return Signal.hold("uptrend intact", **ind)


class TrendPullback(Strategy):
    """Buy a momentum reset (RSI reclaim) while the medium trend is up."""

    name = "trend_pullback"
    version = "1.0.0"
    PARAMS = PullbackParams

    @property
    def warmup_bars(self) -> int:
        return int(self.params.ema_slow) + 20

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        p = self.params
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = float(row["close"])
        atr_abs = float(row["atr14"])
        atr_pct = float(row["atr_pct"])
        rsi_now = float(row["rsi14"])
        rsi_prev = float(prev["rsi14"])
        fast = float(row[f"ema{p.ema_fast}"])
        slow = float(row[f"ema{p.ema_slow}"])
        ind = {"close": close, "rsi14": round(rsi_now, 2), "atr_pct": round(atr_pct, 3),
               "ema_fast": round(fast, 6), "ema_slow": round(slow, 6)}

        if not (p.min_atr_pct <= atr_pct <= p.max_atr_pct):
            return Signal.hold(f"volatility {atr_pct:.2f}% outside range", **ind)

        if position is None:
            reclaim = rsi_prev < p.rsi_reclaim <= rsi_now
            if reclaim and close > slow and fast > slow:
                stop = close - p.stop_atr * atr_abs
                return Signal(
                    "enter_long",
                    f"RSI reclaimed {p.rsi_reclaim:.0f} ({rsi_prev:.1f}->{rsi_now:.1f}) "
                    f"in an EMA{p.ema_fast}/EMA{p.ema_slow} uptrend",
                    stop_price=stop,
                    trail_atr=p.trail_atr,
                    indicators={**ind, "stop": round(stop, 6)},
                    tag="pullback",
                )
            why = "no RSI reclaim" if not reclaim else "trend not aligned"
            return Signal.hold(f"no entry: {why}", **ind)

        if rsi_now >= p.rsi_take_profit:
            return Signal("exit_long", f"RSI {rsi_now:.1f} >= {p.rsi_take_profit:.0f} (extended)",
                          indicators=ind)
        if close < slow:
            return Signal("exit_long", f"close lost EMA{p.ema_slow} {slow:.6g}", indicators=ind)
        if fast < slow:
            return Signal("exit_long", "fast EMA crossed below slow EMA", indicators=ind)
        return Signal.hold("trend intact, holding", **ind)


class MeanReversion(Strategy):
    """Fade a stretched move inside a non-trending market, exit at the mean."""

    name = "mean_reversion"
    version = "1.0.0"
    PARAMS = MeanRevParams

    @property
    def warmup_bars(self) -> int:
        return max(int(self.params.bb_window), 40) + 40

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        p = self.params
        row = df.iloc[i]
        close = float(row["close"])
        atr_abs = float(row["atr14"])
        atr_pct = float(row["atr_pct"])
        rsi_now = float(row["rsi14"])
        adx_now = float(row["adx14"])
        lower = float(row["bb_lower"])
        mid = float(row["bb_mid"])
        ind = {"close": close, "rsi14": round(rsi_now, 2), "adx14": round(adx_now, 2),
               "atr_pct": round(atr_pct, 3), "bb_lower": round(lower, 6),
               "bb_mid": round(mid, 6)}

        if not (p.min_atr_pct <= atr_pct <= p.max_atr_pct):
            return Signal.hold(f"volatility {atr_pct:.2f}% outside range", **ind)

        if position is None:
            if close < lower and rsi_now < p.rsi_entry and adx_now < p.adx_max:
                stop = close - p.stop_atr * atr_abs
                return Signal(
                    "enter_long",
                    f"close {close:.6g} below lower band {lower:.6g}, RSI {rsi_now:.1f}, "
                    f"ADX {adx_now:.1f} (range market)",
                    stop_price=stop,
                    trail_atr=p.trail_atr,
                    indicators={**ind, "stop": round(stop, 6)},
                    tag="mean_reversion",
                )
            why = "not below band" if close >= lower else (
                "RSI not stretched" if rsi_now >= p.rsi_entry else "trending (ADX high)")
            return Signal.hold(f"no entry: {why}", **ind)

        if close >= mid:
            return Signal("exit_long", f"reverted to mean {mid:.6g}", indicators=ind)
        if rsi_now > p.rsi_exit:
            return Signal("exit_long", f"RSI {rsi_now:.1f} above exit {p.rsi_exit:.0f}",
                          indicators=ind)
        return Signal.hold("waiting for mean reversion", **ind)


class SmcFvg(Strategy):
    """Institutional Order Flow: 1H Bias + 1H FVG/OB Zone + Confirmation Retest."""

    name = "smc_fvg"
    version = "1.0.0"
    PARAMS = SmcParams

    @property
    def warmup_bars(self) -> int:
        return int(self.params.bias_ema) + self.params.fvg_lookback_bars + 10

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        p = self.params
        row = df.iloc[i]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr_abs = float(row.get("atr14", close * 0.02))
        atr_pct = float(row.get("atr_pct", 2.0))
        bias_val = float(row[f"ema{p.bias_ema}"])

        ind = {"close": close, "bias_ema": round(bias_val, 4), "atr_pct": round(atr_pct, 3)}

        if not (p.min_atr_pct <= atr_pct <= p.max_atr_pct):
            return Signal.hold(f"volatility {atr_pct:.2f}% outside range", **ind)

        # --- Exit Rules
        if position is not None:
            entry = position.entry_price
            stop = position.stop_price or (entry - 2.0 * atr_abs)
            risk = max(1e-6, entry - stop)
            tp = entry + p.rr_ratio * risk
            ind["take_profit"] = round(tp, 4)
            ind["stop_price"] = round(stop, 4)

            # Fixed 1:2 Take Profit
            if high >= tp:
                return Signal("exit_long", f"Take profit 1:{p.rr_ratio:.0f} reached at {tp:.4g}",
                              indicators=ind)
            # Stop loss at opposite boundary of 1H zone
            if low <= stop:
                return Signal("exit_long", f"Stop loss hit at zone boundary {stop:.4g}",
                              indicators=ind)
            # Trend breakdown
            if close < bias_val * 0.95:
                return Signal("exit_long", f"Close broke below 1H bias EMA{p.bias_ema}", indicators=ind)
            return Signal.hold(f"holding SMC (target: {tp:.4g}, stop: {stop:.4g})", **ind)

        # --- Entry Rules
        btc_trend = float(row.get("btc_trend", 0.0))
        rsi = float(row.get("rsi14", 50.0))
        if btc_trend < 0 or rsi < 42:
            return Signal.hold(f"SMC setup suppressed: btc_trend={btc_trend:.2f}, rsi={rsi:.1f}", **ind)

        # 1. 1H Bullish Bias
        if close <= bias_val:
            return Signal.hold(f"bearish bias: close {close:.4g} <= EMA{p.bias_ema} {bias_val:.4g}", **ind)

        # 2. Look for valid unmitigated 1H FVG + OB zone
        start_idx = max(2, i - p.fvg_lookback_bars)
        recent_zone = None

        for k in range(i - 1, start_idx, -1):
            bar_k = df.iloc[k]
            bar_k2 = df.iloc[k - 2]
            if float(bar_k["low"]) > float(bar_k2["high"]):
                gap = float(bar_k["low"]) - float(bar_k2["high"])
                if gap >= p.min_fvg_atr_ratio * atr_abs:
                    ob_low = min(float(bar_k2["low"]), float(df.iloc[k - 1]["low"]))
                    fvg_top = float(bar_k["low"])
                    zone_bottom = ob_low - p.stop_buffer_atr * atr_abs
                    sub_lows = df.iloc[k:i]["low"].values
                    if len(sub_lows) == 0 or np.min(sub_lows) >= zone_bottom:
                        recent_zone = {"top": fvg_top, "bottom": zone_bottom, "k": k}
                        break

        if not recent_zone:
            return Signal.hold("no valid unmitigated 1H FVG + OB zone in lookback", **ind)

        # 3. Check if 1H zone was tapped
        zone_tapped = low <= recent_zone["top"] and high >= recent_zone["bottom"]
        if not zone_tapped:
            return Signal.hold(f"1H FVG+OB zone active ({recent_zone['bottom']:.4g}-{recent_zone['top']:.4g}) but not tapped", **ind)

        # 4. Confirmation: Reaction / displacement off the zone
        prev = df.iloc[i - 1]
        reaction = close > float(prev["high"]) or close > float(row["open"])
        if not reaction:
            return Signal.hold("zone tapped, waiting for confirmation displacement", **ind)

        # 5. Trigger Buy with stop at opposite boundary of 1H zone
        stop_price = max(recent_zone["bottom"], close - 5.0 * atr_abs)
        if stop_price >= close * 0.998:
            stop_price = close - 1.5 * atr_abs

        tp_target = close + p.rr_ratio * (close - stop_price)
        ind["zone_bottom"] = round(recent_zone["bottom"], 4)
        ind["zone_top"] = round(recent_zone["top"], 4)
        ind["stop_loss"] = round(stop_price, 4)
        ind["take_profit"] = round(tp_target, 4)

        return Signal(
            "enter_long",
            f"1H Bullish Bias + 1H FVG/OB Zone ({recent_zone['bottom']:.4g}-{recent_zone['top']:.4g}) tapped & confirmed (1:{p.rr_ratio:.0f} R:R)",
            stop_price=stop_price,
            trail_atr=0.0,
            indicators=ind,
            tag="smc_fvg",
        )


REGISTRY: dict[str, type[Strategy]] = {
    "trend_breakout": TrendBreakout,
    "trend_pullback": TrendPullback,
    "mean_reversion": MeanReversion,
    "smc_fvg": SmcFvg,
}


def build(symbol_cfg) -> Strategy:
    """Instantiate the strategy assigned to a universe symbol."""
    cls = REGISTRY[symbol_cfg.strategy]
    return cls(symbol_cfg.params)


def describe_all() -> list[dict]:
    return [cls({}).describe() for cls in REGISTRY.values()]
