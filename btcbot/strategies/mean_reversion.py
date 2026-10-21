"""Strategy C -- mean reversion inside an established uptrend."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, bollinger, ema, rsi
from .base import Params, PositionView, Signal, Strategy


class MeanRevParams(Params):
    bb_window: int = 20
    bb_std: float = 2.0
    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 55.0
    trend_ema: int = 200
    require_uptrend: bool = True
    atr_window: int = 14
    stop_atr_mult: float = 2.0


class BollingerRsiMeanReversion(Strategy):
    name = "meanrev_bb_rsi"
    version = "1.0.0"
    default_timeframe = "4h"
    description = (
        "Buy a close below the lower Bollinger band with RSI oversold while price is above a "
        "long EMA; exit at the middle band or when RSI recovers; ATR stop. Deliberately avoids "
        "buying dips in a downtrend."
    )
    PARAMS = MeanRevParams

    @property
    def warmup_bars(self) -> int:
        return max(self.params.bb_window, self.params.rsi_window, self.params.trend_ema,
                   self.params.atr_window) + 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        bb = bollinger(close, p.bb_window, p.bb_std)
        df["bb_mid"] = bb["mid"]
        df["bb_upper"] = bb["upper"]
        df["bb_lower"] = bb["lower"]
        df["rsi"] = rsi(close, p.rsi_window)
        df["atr"] = atr(df["high"], df["low"], close, p.atr_window)
        if p.require_uptrend:
            df["ema_trend"] = ema(close, p.trend_ema)
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        close = float(row["close"])
        lower = self._val(row, "bb_lower")
        mid = self._val(row, "bb_mid")
        rsi_val = self._val(row, "rsi")
        atr_val = self._val(row, "atr")
        if lower is None or mid is None or rsi_val is None:
            return Signal.hold("indicators warming up")
        ind = self._snapshot(row, ["close", "bb_mid", "bb_upper", "bb_lower", "rsi", "atr", "ema_trend"])

        if position is None:
            trend_ok = True
            if self.params.require_uptrend:
                trend_val = self._val(row, "ema_trend")
                trend_ok = trend_val is not None and close > trend_val
            if not trend_ok:
                return Signal.hold("oversold but price below long EMA: no counter-trend buys", **ind)
            if close < lower and rsi_val < self.params.rsi_entry:
                stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
                return Signal(
                    "enter_long",
                    f"mean reversion: close {close:.2f} < lower band {lower:.2f}, RSI {rsi_val:.1f} "
                    f"< {self.params.rsi_entry:.0f}, price above EMA{self.params.trend_ema}",
                    stop,
                    ind,
                    tag="meanrev",
                )
            return Signal.hold(
                f"no setup (close {close:.2f} vs lower {lower:.2f}, RSI {rsi_val:.1f})", **ind
            )

        if close >= mid or rsi_val >= self.params.rsi_exit:
            return Signal(
                "exit_long",
                f"mean-reversion exit: close {close:.2f} vs mid {mid:.2f}, RSI {rsi_val:.1f}",
                None,
                ind,
                tag="meanrev",
            )
        return Signal.hold("waiting for reversion to the mean", **ind)
