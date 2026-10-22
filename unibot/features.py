"""Causal feature engineering shared by strategies, the regime model and ML.

Every column at bar ``i`` uses only bars ``<= i``.  The few features that must
exclude the current bar (prior-high channels) are shifted explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcbot.data.timeframes import tf_seconds
from btcbot.indicators import adx, atr, bollinger, ema, realized_vol, rolling_slope, rsi

DONCHIAN_WINDOWS = (20, 35, 40, 45, 50, 55)

FEATURE_COLUMNS = [
    "rsi14",
    "adx14",
    "atr_pct",
    "bb_pos",
    "dist_ema50_pct",
    "dist_ema200_pct",
    "ema_slope_pct",
    "vol_ratio",
    "range_pos_20",
    "donchian_pos",
    "ret_6",
    "ret_24",
    "ret_72",
    "realized_vol",
    "market_score",
    "breadth",
    "btc_trend",
]


def resample_ohlcv(df: pd.DataFrame, from_tf: str, to_tf: str) -> pd.DataFrame:
    """Aggregate OHLCV bars onto a coarser UTC-aligned grid."""
    if df.empty:
        return df.copy()
    step = tf_seconds(to_tf)
    src = tf_seconds(from_tf)
    if step % src != 0:
        raise ValueError(f"cannot resample {from_tf} -> {to_tf}")
    out = df.copy()
    out["bucket"] = (out["bar_ts"] // step) * step
    grouped = out.groupby("bucket", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trades=("trades", "sum") if "trades" in out.columns else ("volume", "size"),
        n_bars=("close", "size"),
    )
    grouped = grouped.reset_index().rename(columns={"bucket": "bar_ts"})
    # Only keep buckets that are fully populated, so no partial bar can ever
    # masquerade as a closed bar.
    grouped = grouped[grouped["n_bars"] == step // src].drop(columns=["n_bars"])
    grouped["close_time"] = grouped["bar_ts"] + step
    grouped["dt"] = pd.to_datetime(grouped["bar_ts"], unit="s", utc=True)
    grouped["source"] = "resampled"
    return grouped.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the indicator set used by strategies and models."""
    if df.empty:
        return df.copy()
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema100"] = ema(close, 100)
    out["ema200"] = ema(close, 200)
    out["rsi14"] = rsi(close, 14)
    out["atr14"] = atr(high, low, close, 14)
    out["atr_pct"] = out["atr14"] / close * 100.0
    bb = bollinger(close, 20, 2.0)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bb["mid"], bb["upper"], bb["lower"]
    width = (bb["upper"] - bb["lower"]).replace(0.0, np.nan)
    out["bb_pos"] = (close - bb["lower"]) / width
    out["adx14"] = adx(high, low, close, 14)
    out["dist_ema50_pct"] = (close / out["ema50"] - 1.0) * 100.0
    out["dist_ema200_pct"] = (close / out["ema200"] - 1.0) * 100.0
    out["ema_slope_pct"] = rolling_slope(close, 24) * 100.0
    out["realized_vol"] = realized_vol(close, 20)

    vol_ma = out["volume"].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = (out["volume"] / vol_ma.replace(0.0, np.nan)).clip(0, 20)

    lo20 = low.rolling(20, min_periods=20).min()
    hi20 = high.rolling(20, min_periods=20).max()
    out["range_pos_20"] = (close - lo20) / (hi20 - lo20).replace(0.0, np.nan)

    # Prior-bar channels (exclude the current bar: causal breakout test).
    for w in DONCHIAN_WINDOWS:
        out[f"don_hi_{w}"] = high.rolling(w, min_periods=w).max().shift(1)
        out[f"don_lo_{w}"] = low.rolling(w, min_periods=w).min().shift(1)
    rng = (out["don_hi_55"] - out["don_lo_55"]).replace(0.0, np.nan)
    out["donchian_pos"] = (close - out["don_lo_55"]) / rng

    for h in (1, 3, 6, 12, 24, 72):
        out[f"ret_{h}"] = close.pct_change(h) * 100.0
    out["ret_1"] = out["ret_1"].fillna(0.0)

    # Simple candle anatomy, useful to the model.
    out["body_pct"] = (close - out["open"]) / out["open"] * 100.0
    out["upper_wick_pct"] = (high - np.maximum(close, out["open"])) / close * 100.0
    out["lower_wick_pct"] = (np.minimum(close, out["open"]) - low) / close * 100.0
    return out


def attach_market_context(bars: pd.DataFrame, market: pd.DataFrame,
                          timeframe: str = "4h") -> pd.DataFrame:
    """Join the causal market regime known at each bar's close."""
    if market.empty or bars.empty:
        out = bars.copy()
        for col in ("market_score", "breadth", "btc_trend"):
            out[col] = 0.0
        out["market_label"] = "chop"
        return out
    left = bars.copy()
    left["_known"] = left["bar_ts"] + tf_seconds(timeframe)
    right = market[["known_ts", "score", "breadth", "btc_trend", "label"]].sort_values("known_ts")
    merged = pd.merge_asof(
        left.sort_values("_known"),
        right,
        left_on="_known",
        right_on="known_ts",
        direction="backward",
    )
    merged = merged.rename(columns={"score": "market_score"})
    merged["market_score"] = merged["market_score"].fillna(0.0)
    merged["breadth"] = merged["breadth"].fillna(0.5)
    merged["btc_trend"] = merged["btc_trend"].fillna(0.0)
    merged["market_label"] = merged["label"].fillna("chop") if "label" in merged else "chop"
    return merged.drop(columns=["_known", "known_ts", "label"], errors="ignore").reset_index(drop=True)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Model-ready features, forward-filled only *backwards* from known values."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[cols].astype(float).copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.ffill()


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    upper_atr: float,
    lower_atr: float,
    fee_pct: float,
) -> pd.DataFrame:
    """Label each bar by which barrier price reaches first within ``horizon``.

    Entry reference is the next bar's open (the earliest executable price), the
    barriers are scaled by the ATR known at the signal bar, and ties inside one
    bar resolve to the stop first (pessimistic).  Bars whose trade cannot clear
    round-trip costs are labelled 0.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index, columns=["label", "ret", "bars_held", "outcome"])
    if n < 2:
        return out
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr_abs = df["atr14"].to_numpy(float) if "atr14" in df else np.zeros(n)

    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    held = np.full(n, np.nan)
    outcome = np.empty(n, dtype=object)
    for i in range(n - 1):
        entry = o[i + 1]
        a = atr_abs[i]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(a) or a <= 0:
            continue
        up = entry + upper_atr * a
        dn = entry - lower_atr * a
        end = min(n - 1, i + horizon)
        result = None
        exit_px = c[end]
        exit_idx = end
        for j in range(i + 1, end + 1):
            if l[j] <= dn:  # stop first when both are touched
                result, exit_px, exit_idx = "loss", dn, j
                break
            if h[j] >= up:
                result, exit_px, exit_idx = "win", up, j
                break
        if result is None:
            result = "timeout"
        r = (exit_px / entry - 1.0) * 100.0 - fee_pct
        labels[i] = 1.0 if r > 0 else 0.0
        rets[i] = r
        held[i] = exit_idx - i
        outcome[i] = result
    out["label"] = labels
    out["ret"] = rets
    out["bars_held"] = held
    out["outcome"] = outcome
    return out


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if "dt" not in df.columns:
        df = df.copy()
        df["dt"] = pd.to_datetime(df["bar_ts"], unit="s", utc=True)
    return df
