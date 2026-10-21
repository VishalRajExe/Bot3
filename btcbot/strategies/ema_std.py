"""Strategy D -- EMA-65 / 65-bar standard-deviation band system (baseline).

Provenance
----------
The brief refers to an existing Bitcoin daily strategy built on a 65-period EMA
and a 65-period standard deviation.  No strategy document existed in this
workspace, and web research could only find an *unverified* Reddit post
describing a z-score state machine (no reproducible rules or band values).  The
rules below are therefore an explicit **reconstruction**, not the original:

1. ``z = (close - EMA65) / StdDev65`` (population std, computed on closes).
2. The system is in a *bullish state* when ``z >= bull_z`` (default +0.75).
3. It is in a *bearish state* when ``z <= bear_z`` (default -0.75).
4. Between the thresholds the previous state persists (hysteresis).
5. Flat-to-long when bullish, long-to-flat when bearish.  Long-only spot.
6. A protective stop at ``entry - stop_atr_mult * ATR14`` is added because this
   project requires every strategy to define a hard stop.  Set
   ``use_stop=false`` to reproduce the pure state rule for comparison.

The band multiplier is a configuration parameter precisely because the original
value is unknown; the research report shows a sensitivity grid instead of
claiming one optimal setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr, ema, rolling_std
from .base import Params, PositionView, Signal, Strategy


class EmaStdParams(Params):
    window: int = 65
    bull_z: float = 0.75
    bear_z: float = -0.75
    atr_window: int = 14
    stop_atr_mult: float = 3.0
    use_stop: bool = True
    ema_trend_filter: int = 0  # 0 disables; e.g. 200 requires close > EMA200
    std_ddof: int = 0


class EmaStd(Strategy):
    name = "ema_std"
    version = "1.0.0"
    default_timeframe = "1d"
    description = (
        "Daily EMA-65 standard-deviation band state machine (reconstruction): long above "
        "the upper z-band, flat below the lower z-band, hold state in between."
    )
    PARAMS = EmaStdParams

    @property
    def warmup_bars(self) -> int:
        return max(
            self.params.window,
            self.params.atr_window,
            self.params.ema_trend_filter,
        ) + 1

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        df["ema65"] = ema(close, p.window)
        df["std65"] = rolling_std(close, p.window, ddof=p.std_ddof)
        df["z65"] = (close - df["ema65"]) / df["std65"].replace(0.0, np.nan)
        df["atr"] = atr(df["high"], df["low"], close, p.atr_window)
        df["band_1u"] = df["ema65"] + df["std65"]
        df["band_1l"] = df["ema65"] - df["std65"]
        df["band_2u"] = df["ema65"] + 2.0 * df["std65"]
        df["band_2l"] = df["ema65"] - 2.0 * df["std65"]
        if p.ema_trend_filter:
            df["ema_trend"] = ema(close, p.ema_trend_filter)

        z = df["z65"].to_numpy(dtype=float)
        state = np.zeros(len(df), dtype=np.int8)
        cur = 0
        for i in range(len(df)):
            zi = z[i]
            if zi == zi:  # not NaN
                if zi >= p.bull_z:
                    cur = 1
                elif zi <= p.bear_z:
                    cur = 0
            state[i] = cur
        df["es_state"] = state
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        z = self._val(row, "z65")
        if z is None:
            return Signal.hold("z-score unavailable")
        close = float(row["close"])
        state = int(row["es_state"])
        atr_val = self._val(row, "atr")
        ind = self._snapshot(row, ["close", "ema65", "std65", "z65", "atr", "es_state", "ema_trend"])

        trend_ok = True
        trend_note = ""
        if self.params.ema_trend_filter:
            trend_val = self._val(row, "ema_trend")
            trend_ok = trend_val is not None and close > trend_val
            if not trend_ok:
                trend_note = f", blocked by EMA{self.params.ema_trend_filter} trend filter"

        if position is None:
            if state == 1 and trend_ok:
                stop = None
                if self.params.use_stop and atr_val is not None:
                    stop = close - self.params.stop_atr_mult * atr_val
                return Signal(
                    "enter_long",
                    f"bullish state: z={z:.2f} >= {self.params.bull_z:.2f}{trend_note}",
                    stop,
                    ind,
                    tag="ema_std",
                )
            if state == 1 and not trend_ok:
                return Signal.hold(f"bullish z but trend filter blocks entry (z={z:.2f})", **ind)
            return Signal.hold(f"flat: z={z:.2f} below bull threshold", **ind)

        if state == 0:
            return Signal(
                "exit_long",
                f"bearish state: z={z:.2f} <= {self.params.bear_z:.2f}",
                None,
                ind,
                tag="ema_std",
            )
        return Signal.hold(f"holding: z={z:.2f} above bear threshold", **ind)
