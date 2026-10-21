"""Technical indicators.

All functions are pure, vectorised and *causal*: the value at bar ``i`` depends
only on bars ``<= i``.  Strategies that must exclude the current bar (a breakout
of a prior range, for example) shift the series explicitly.

Conventions
-----------
* ``ema`` uses ``adjust=False`` (recursive EMA, the standard trading definition).
* ``rolling_std`` uses population standard deviation (``ddof=0``), matching the
  common Bollinger/band definition.
* ``atr`` and ``rsi`` use Wilder smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=ddof)


def zscore(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    """Standardised distance from the rolling mean."""
    mean = sma(series, window)
    std = rolling_std(series, window, ddof=ddof)
    return (series - mean) / std.replace(0.0, np.nan)


def ema_zscore(close: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    """(close - EMA_window) / StdDev_window -- the EMA-65 band family."""
    mean = ema(close, window)
    std = rolling_std(close, window, ddof=ddof)
    return (close - mean) / std.replace(0.0, np.nan)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # 100 when there were no losses at all in the window.
    out = out.where(avg_loss != 0.0, 100.0)
    return out.where(avg_gain.notna())


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = rolling_std(close, window)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std}
    )


def donchian(high: pd.Series, low: pd.Series, window: int) -> pd.DataFrame:
    """Prior-bar channel: bar *i* contains the range of bars i-window..i-1.

    Excluding the current bar is what makes a breakout test causal.
    """
    upper = high.rolling(window=window, min_periods=window).max().shift(1)
    lower = low.rolling(window=window, min_periods=window).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean() / atr_s.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean() / atr_s.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def realized_vol(close: pd.Series, window: int = 20, periods_per_year: int = 365) -> pd.Series:
    """Annualised realised volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(periods_per_year)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Normalised linear-regression slope over a rolling window.

    Expressed as fractional change per bar so it is comparable across price
    levels.
    """
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def _slope(values: np.ndarray) -> float:
        return float(np.dot(x_centered, values - values.mean()) / denom)

    raw = series.rolling(window=window, min_periods=window).apply(_slope, raw=True)
    return raw / series.replace(0.0, np.nan)


def crossed_above(series: pd.Series, level: float | pd.Series) -> pd.Series:
    prev = series.shift(1)
    if isinstance(level, pd.Series):
        return (series > level) & (prev <= level.shift(1))
    return (series > level) & (prev <= level)


def crossed_below(series: pd.Series, level: float | pd.Series) -> pd.Series:
    prev = series.shift(1)
    if isinstance(level, pd.Series):
        return (series < level) & (prev >= level.shift(1))
    return (series < level) & (prev >= level)
