"""Feature construction.

Every feature at bar ``t`` uses only information available at the close of bar
``t``.  The label is the *forward* return and is only ever used for training,
never as a feature.  ``build_dataset`` returns features and labels aligned to the
signal bar so walk-forward splitting can purge the overlap between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import adx, atr, ema, realized_vol, rsi

FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "ret_30",
    "vol_10", "vol_30", "atr_pct", "rsi_14", "adx_14",
    "z65", "ema_dist_50", "ema_dist_200", "vol_z", "range_pos",
    "regime_trend", "regime_high_vol",
]

# Optional sentiment features, appended only when a caller has attached them
# with ``btcbot.sentiment.features.attach_fear_greed`` (point-in-time safe).
FG_FEATURES = ["fg_norm", "fg_change_7", "fg_extreme_fear", "fg_extreme_greed"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame indexed like ``df`` with causal features only."""
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    log_ret = np.log(close / close.shift(1))
    for h in (1, 3, 7, 14, 30):
        out[f"ret_{h}"] = np.log(close / close.shift(h))
    out["vol_10"] = log_ret.rolling(10, min_periods=10).std(ddof=0)
    out["vol_30"] = log_ret.rolling(30, min_periods=30).std(ddof=0)
    atr14 = atr(df["high"], df["low"], close, 14)
    out["atr_pct"] = atr14 / close
    out["rsi_14"] = rsi(close, 14)
    out["adx_14"] = adx(df["high"], df["low"], close, 14)

    ema65 = ema(close, 65)
    std65 = close.rolling(65, min_periods=65).std(ddof=0)
    out["z65"] = (close - ema65) / std65.replace(0.0, np.nan)
    out["ema_dist_50"] = close / ema(close, 50) - 1.0
    out["ema_dist_200"] = close / ema(close, 200) - 1.0

    vol_mean = df["volume"].rolling(30, min_periods=30).mean()
    vol_std = df["volume"].rolling(30, min_periods=30).std(ddof=0)
    out["vol_z"] = (df["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    hi20 = df["high"].rolling(20, min_periods=20).max()
    lo20 = df["low"].rolling(20, min_periods=20).min()
    out["range_pos"] = (close - lo20) / (hi20 - lo20).replace(0.0, np.nan)
    out["regime_trend"] = ((out["adx_14"] >= 20) & (out["ema_dist_200"].abs() > 0.02)).astype(float)
    rv = realized_vol(close, 20)
    out["regime_high_vol"] = (rv > rv.rolling(365, min_periods=60).median()).astype(float)
    for name in FG_FEATURES:
        if name in df.columns:
            out[name] = df[name].astype(float)
    return out


def build_dataset(df: pd.DataFrame, *, horizon: int, cost_bps: float) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features, binary label and raw forward return, aligned by bar.

    Label is 1 when the forward return over ``horizon`` bars exceeds the
    round-trip cost assumption, i.e. "a long trade would have cleared costs".
    """
    feats = build_features(df)
    close = df["close"].astype(float)
    fwd_ret = close.shift(-horizon) / close - 1.0
    label = (fwd_ret > cost_bps / 10_000.0).astype("float")
    fwd_ret = fwd_ret.where(fwd_ret.notna())
    label = label.where(fwd_ret.notna())
    valid = feats.notna().all(axis=1) & label.notna()
    feats = feats.loc[valid].copy()
    feats["bar_ts"] = df.loc[valid, "bar_ts"].astype("int64")
    return feats, label.loc[valid].astype(int), fwd_ret.loc[valid]
