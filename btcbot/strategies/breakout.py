"""Strategy B -- Donchian breakout with volume confirmation."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, donchian, ema, sma
from .base import Params, PositionView, Signal, Strategy


class BreakoutParams(Params):
    entry_window: int = 20
    exit_window: int = 10
    atr_window: int = 14
    stop_atr_mult: float = 2.5
    volume_filter: bool = True
    volume_window: int = 20
    volume_mult: float = 1.0
    trend_filter: int = 0  # e.g. 200 to require close > EMA200


class DonchianBreakout(Strategy):
    name = "breakout_donchian"
    version = "1.0.0"
    default_timeframe = "4h"
    description = (
        "Long-only Donchian breakout: buy a close above the prior N-bar high with above-average "
        "volume, exit a close below the prior M-bar low, ATR stop."
    )
    PARAMS = BreakoutParams

    @property
    def warmup_bars(self) -> int:
        return max(
            self.params.entry_window,
            self.params.exit_window,
            self.params.volume_window,
            self.params.atr_window,
            self.params.trend_filter,
        ) + 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        entry_ch = donchian(df["high"], df["low"], p.entry_window)
        exit_ch = donchian(df["high"], df["low"], p.exit_window)
        df["dc_upper"] = entry_ch["upper"]
        df["dc_lower"] = exit_ch["lower"]
        df["atr"] = atr(df["high"], df["low"], close, p.atr_window)
        df["vol_sma"] = sma(df["volume"], p.volume_window)
        if p.volume_filter:
            df["vol_ok"] = df["volume"] > p.volume_mult * df["vol_sma"]
        else:
            df["vol_ok"] = True
        if p.trend_filter:
            df["ema_trend"] = ema(close, p.trend_filter)
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        close = float(row["close"])
        upper = self._val(row, "dc_upper")
        lower = self._val(row, "dc_lower")
        atr_val = self._val(row, "atr")
        if upper is None or lower is None:
            return Signal.hold("channel unavailable")
        ind = self._snapshot(row, ["close", "dc_upper", "dc_lower", "atr", "vol_sma", "vol_ok"])

        if position is None:
            broke = close > upper
            vol_ok = bool(row["vol_ok"]) if "vol_ok" in row else True
            trend_ok = True
            if self.params.trend_filter:
                trend_val = self._val(row, "ema_trend")
                trend_ok = trend_val is not None and close > trend_val
            if broke and vol_ok and trend_ok:
                stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
                return Signal(
                    "enter_long",
                    f"close {close:.2f} above prior {self.params.entry_window}-bar high {upper:.2f}"
                    + (f", volume {row['volume']:.2f} > {self.params.volume_mult}x SMA" if self.params.volume_filter else "")
                    + (f", trend filter EMA{self.params.trend_filter} passed" if self.params.trend_filter else ""),
                    stop,
                    ind,
                    tag="breakout",
                )
            if broke and not vol_ok:
                return Signal.hold(f"breakout above {upper:.2f} rejected: volume below filter", **ind)
            return Signal.hold(f"close {close:.2f} inside channel (<= {upper:.2f})", **ind)

        if close < lower:
            return Signal(
                "exit_long",
                f"close {close:.2f} below prior {self.params.exit_window}-bar low {lower:.2f}",
                None,
                ind,
                tag="breakout",
            )
        return Signal.hold("channel exit not triggered", **ind)
