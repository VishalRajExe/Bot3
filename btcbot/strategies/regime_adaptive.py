"""Strategy E -- regime-aware switching between breakout and mean reversion.

The strategy only trades regimes in which its component rules have a plausible
edge, and stands aside in high-volatility or downtrend regimes.  Regime labels
are computed causally in ``btcbot.regime``.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, bollinger, donchian, ema, rsi, sma
from .base import Params, PositionView, Signal, Strategy


class RegimeAdaptiveParams(Params):
    adx_min: float = 20.0
    entry_window: int = 20
    exit_window: int = 10
    bb_window: int = 20
    bb_std: float = 2.0
    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    atr_window: int = 14
    stop_atr_mult: float = 2.5
    volume_filter: bool = True
    allow_breakout: bool = True
    allow_meanrev: bool = True


class RegimeAdaptive(Strategy):
    name = "regime_adaptive"
    version = "1.0.0"
    default_timeframe = "4h"
    description = (
        "Regime switch: Donchian breakouts in trending bull regimes, Bollinger/RSI mean "
        "reversion in low-volatility ranges, no entries in bear or high-volatility regimes."
    )
    PARAMS = RegimeAdaptiveParams

    @property
    def warmup_bars(self) -> int:
        return max(self.params.entry_window, self.params.bb_window, self.params.rsi_window,
                   200, self.params.atr_window) + 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        entry_ch = donchian(df["high"], df["low"], p.entry_window)
        exit_ch = donchian(df["high"], df["low"], p.exit_window)
        df["dc_upper"] = entry_ch["upper"]
        df["dc_lower"] = exit_ch["lower"]
        bb = bollinger(close, p.bb_window, p.bb_std)
        df["bb_mid"] = bb["mid"]
        df["bb_lower"] = bb["lower"]
        df["rsi"] = rsi(close, p.rsi_window)
        df["atr"] = atr(df["high"], df["low"], close, p.atr_window)
        df["vol_sma"] = sma(df["volume"], 20)
        df["ema200"] = ema(close, 200)
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        close = float(row["close"])
        regime = str(row.get("regime", "unknown"))
        upper = self._val(row, "dc_upper")
        lower = self._val(row, "dc_lower")
        mid = self._val(row, "bb_mid")
        bb_lower = self._val(row, "bb_lower")
        rsi_val = self._val(row, "rsi")
        atr_val = self._val(row, "atr")
        ind = self._snapshot(row, ["close", "dc_upper", "dc_lower", "bb_mid", "bb_lower", "rsi",
                                   "atr", "regime_adx", "regime_vol_ratio"])
        ind["regime"] = regime

        if position is None:
            if regime == "bull_trend" and self.params.allow_breakout and upper is not None:
                vol_ok = (not self.params.volume_filter) or bool(row.get("vol_ok", True)) or (
                    self._val(row, "vol_sma") is not None and float(row["volume"]) > float(row["vol_sma"])
                )
                if close > upper and vol_ok:
                    stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
                    return Signal(
                        "enter_long",
                        f"bull regime breakout: close {close:.2f} > prior {self.params.entry_window}-bar high {upper:.2f}",
                        stop, ind, tag="breakout",
                    )
            if regime == "range_low_vol" and self.params.allow_meanrev and bb_lower is not None and rsi_val is not None:
                trend_ok = self._val(row, "ema200") is None or close > float(row["ema200"])
                if close < bb_lower and rsi_val < self.params.rsi_entry and trend_ok:
                    stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
                    return Signal(
                        "enter_long",
                        f"range mean reversion: close {close:.2f} < lower band {bb_lower:.2f}, RSI {rsi_val:.1f}",
                        stop, ind, tag="meanrev",
                    )
            return Signal.hold(f"no setup in regime '{regime}'", **ind)

        tag = position.entry_tag or "default"
        if tag == "breakout" and lower is not None and close < lower:
            return Signal("exit_long",
                          f"breakout exit: close {close:.2f} < prior {self.params.exit_window}-bar low {lower:.2f}",
                          None, ind, tag="breakout")
        if tag == "meanrev" and ((mid is not None and close >= mid) or
                                 (rsi_val is not None and rsi_val >= self.params.rsi_exit)):
            return Signal("exit_long",
                          f"mean-reversion exit: close {close:.2f}, mid {mid}, RSI {rsi_val}",
                          None, ind, tag="meanrev")
        if regime == "high_vol":
            return Signal("exit_long", "high-volatility regime: reduce exposure", None, ind, tag="regime")
        return Signal.hold(f"holding {tag} position in regime '{regime}'", **ind)
