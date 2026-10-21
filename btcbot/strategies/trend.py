"""Strategy A -- trend following with an EMA cross and ATR stop."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, ema
from .base import Params, PositionView, Signal, Strategy


class TrendParams(Params):
    fast: int = 50
    slow: int = 200
    atr_window: int = 14
    stop_atr_mult: float = 3.0
    entry_mode: str = "cross"  # "cross" (golden cross only) or "state"


class EmaCrossTrend(Strategy):
    name = "trend_ema"
    version = "1.0.0"
    default_timeframe = "1d"
    description = (
        "Daily long-only EMA(50/200) trend system: enter on a golden cross while price is "
        "above the slow EMA, exit on a death cross, ATR-based initial stop."
    )
    PARAMS = TrendParams

    @property
    def warmup_bars(self) -> int:
        return max(self.params.slow, self.params.atr_window) + 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        df["ema_fast"] = ema(close, p.fast)
        df["ema_slow"] = ema(close, p.slow)
        df["atr"] = atr(df["high"], df["low"], close, p.atr_window)
        df["cross_up"] = (df["ema_fast"] > df["ema_slow"]) & (
            df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)
        )
        df["cross_down"] = (df["ema_fast"] < df["ema_slow"]) & (
            df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)
        )
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        close = float(row["close"])
        fast = self._val(row, "ema_fast")
        slow = self._val(row, "ema_slow")
        atr_val = self._val(row, "atr")
        if fast is None or slow is None:
            return Signal.hold("EMAs unavailable")
        ind = self._snapshot(row, ["close", "ema_fast", "ema_slow", "atr", "cross_up", "cross_down"])

        if position is None:
            armed = bool(row["cross_up"]) if self.params.entry_mode == "cross" else fast > slow
            if armed and close > slow:
                stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
                return Signal(
                    "enter_long",
                    f"EMA{self.params.fast} above EMA{self.params.slow}"
                    + (" (fresh cross)" if self.params.entry_mode == "cross" else ""),
                    stop,
                    ind,
                    tag="trend",
                )
            return Signal.hold("no bullish cross / price below slow EMA", **ind)

        if bool(row["cross_down"]) or fast < slow:
            return Signal(
                "exit_long",
                f"EMA{self.params.fast} below EMA{self.params.slow}",
                None,
                ind,
                tag="trend",
            )
        return Signal.hold("trend intact", **ind)
