"""Point-in-time sentiment feature joins.

``merge_asof`` on ``available_ts`` guarantees that a bar only sees sentiment that
was already published.  Fear & Greed is shifted to the end of the day it
describes, so a daily bar cannot use that day's own value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..db import Database
from .fng import load_fng


def attach_fear_greed(df: pd.DataFrame, db: Database) -> pd.DataFrame:
    """Add causal Fear & Greed features to a candle frame."""
    fng = load_fng(db)
    out = df.copy()
    if fng.empty:
        out["fg_value"] = np.nan
        out["fg_norm"] = np.nan
        out["fg_change_7"] = np.nan
        out["fg_extreme_fear"] = 0.0
        out["fg_extreme_greed"] = 0.0
        return out

    fng = fng.sort_values("available_ts")
    fng["fg_value"] = fng["value"].astype(float)
    fng["fg_norm"] = (fng["fg_value"] - 50.0) / 50.0
    fng["fg_change_7"] = fng["fg_value"].diff(7) / 50.0
    fng["fg_extreme_fear"] = (fng["fg_value"] <= 25).astype(float)
    fng["fg_extreme_greed"] = (fng["fg_value"] >= 75).astype(float)

    left = out[["bar_ts"]].copy()
    left["bar_ts"] = left["bar_ts"].astype("int64")
    merged = pd.merge_asof(
        left.sort_values("bar_ts"),
        fng[["available_ts", "fg_value", "fg_norm", "fg_change_7",
             "fg_extreme_fear", "fg_extreme_greed"]].sort_values("available_ts"),
        left_on="bar_ts", right_on="available_ts", direction="backward",
    )
    for col in ("fg_value", "fg_norm", "fg_change_7", "fg_extreme_fear", "fg_extreme_greed"):
        out[col] = merged[col].to_numpy()
    return out


def sentiment_features_available(df: pd.DataFrame) -> bool:
    return "fg_value" in df.columns and df["fg_value"].notna().sum() > 100
