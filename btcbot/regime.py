"""Market-regime classification.

Regime labels are descriptive, causal, and used for (a) journaling every trade
and (b) an optional strategy filter.  They are not a prediction.

Definitions (all computed from closed bars only):

* ``trend``      : EMA(fast) vs EMA(slow) separation, normalised by ATR.
* ``strength``   : ADX(14).
* ``vol_state``  : realised 20-bar volatility vs its own trailing 2-year median.
* Regime label   : one of ``bull_trend``, ``bear_trend``, ``range_low_vol``,
                   ``range_high_vol``, ``high_vol``.

``high_vol`` overrides the others: in extreme volatility the system prefers to
stand aside rather than classify a direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, atr, ema, realized_vol

REGIMES = ("bull_trend", "bear_trend", "range_low_vol", "range_high_vol", "high_vol", "unknown")


def add_regime_columns(df: pd.DataFrame, *, fast: int = 50, slow: int = 200,
                       adx_window: int = 14, vol_window: int = 20,
                       vol_lookback: int = 730) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    atr14 = atr(out["high"], out["low"], close, adx_window)
    out["regime_ema_fast"] = ema_fast
    out["regime_ema_slow"] = ema_slow
    out["regime_adx"] = adx(out["high"], out["low"], close, adx_window)
    out["regime_vol"] = realized_vol(close, vol_window)
    # Trailing median only: no future information.
    vol_median = out["regime_vol"].rolling(vol_lookback, min_periods=max(60, vol_lookback // 6)).median()
    out["regime_vol_median"] = vol_median
    out["regime_vol_ratio"] = out["regime_vol"] / vol_median.replace(0.0, np.nan)
    separation = (ema_fast - ema_slow) / atr14.replace(0.0, np.nan)
    out["regime_separation"] = separation

    bull = (ema_fast > ema_slow) & (separation > 0.25) & (out["regime_adx"] >= 20)
    bear = (ema_fast < ema_slow) & (separation < -0.25) & (out["regime_adx"] >= 20)
    high_vol = out["regime_vol_ratio"] > 2.2

    label = pd.Series("range_low_vol", index=out.index, dtype=object)
    label = label.where(out["regime_vol_ratio"] <= 1.15, "range_high_vol")
    label = label.where(~bull, "bull_trend")
    label = label.where(~bear, "bear_trend")
    label = label.where(~high_vol, "high_vol")
    label = label.where(out["regime_adx"].notna() & out["regime_vol_ratio"].notna(), "unknown")
    out["regime"] = label
    return out


def regime_distribution(df: pd.DataFrame) -> dict[str, int]:
    if "regime" not in df:
        df = add_regime_columns(df)
    return {k: int(v) for k, v in df["regime"].value_counts().items()}
