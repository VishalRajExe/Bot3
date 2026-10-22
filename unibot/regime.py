"""Market regime: is the whole crypto market risk-on, choppy or risk-off?

The regime is computed from *daily* data only and is joined to intraday bars
using the most recent daily close that had already happened.  That makes the
filter causal: a 4h bar at 08:00 can only know yesterday's completed daily bar.

Inputs are deliberately few and interpretable:

* BTC distance from its 100-day EMA (trend),
* breadth: share of the universe trading above its 200-day EMA,
* BTC volatility percentile over the past year (risk).

The resulting ``score`` in [-1, 1] scales position risk; it never picks trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcbot.indicators import ema, realized_vol


@dataclass(frozen=True)
class RegimeState:
    label: str
    score: float
    btc_trend: float
    breadth: float
    vol_percentile: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "btc_trend": round(self.btc_trend, 3),
            "breadth": round(self.breadth, 3),
            "vol_percentile": round(self.vol_percentile, 3),
        }


def _label(score: float, btc_trend: float) -> str:
    if score <= -0.25 or btc_trend < -8.0:
        return "bear"
    if score >= 0.25 and btc_trend > 0:
        return "bull"
    return "chop"


def build_market_series(daily: dict[str, pd.DataFrame], *, btc: str = "BTC/USD") -> pd.DataFrame:
    """Daily regime series with a ``known_ts`` column (the bar's close time)."""
    if btc not in daily or daily[btc].empty:
        return pd.DataFrame(columns=["known_ts", "score", "breadth", "btc_trend",
                                     "vol_pct", "label"])

    frames: dict[str, pd.DataFrame] = {}
    for sym, df in daily.items():
        if df is None or df.empty or len(df) < 210:
            continue
        f = df[["bar_ts", "close"]].copy()
        f["ema100"] = ema(f["close"], 100)
        f["ema200"] = ema(f["close"], 200)
        frames[sym] = f.set_index("bar_ts")

    anchor = frames[btc]
    trend = (anchor["close"] / anchor["ema100"] - 1.0) * 100.0
    vol = realized_vol(anchor["close"], 20, periods_per_year=365)
    # Percentile of today's volatility inside the trailing year (365 obs).
    vol_pct = vol.rolling(365, min_periods=60).apply(
        lambda w: float(np.mean(w <= w[-1])), raw=True
    )

    above = []
    for sym, f in frames.items():
        if sym == btc:
            continue
        above.append((f["close"] > f["ema200"]).astype(float))
    if above:
        breadth = pd.concat(above, axis=1).mean(axis=1, skipna=True)
    else:
        breadth = pd.Series(1.0, index=anchor.index)

    idx = anchor.index
    breadth = breadth.reindex(idx).ffill().fillna(0.5)
    trend_component = np.tanh(trend / 10.0)
    breadth_component = (breadth - 0.5) * 2.0
    vol_component = (1.0 - 2.0 * vol_pct.fillna(0.5)).clip(-1, 1)
    score = (0.5 * trend_component + 0.3 * breadth_component + 0.2 * vol_component).clip(-1, 1)

    out = pd.DataFrame({
        "known_ts": idx + 86_400,
        "score": score.to_numpy(),
        "breadth": breadth.to_numpy(),
        "btc_trend": trend.to_numpy(),
        "vol_pct": vol_pct.to_numpy(),
    })
    out["label"] = [
        _label(float(s), float(t) if np.isfinite(t) else 0.0)
        for s, t in zip(out["score"], out["btc_trend"])
    ]
    return out.dropna(subset=["score"]).reset_index(drop=True)


def regime_at(market: pd.DataFrame, ts: int) -> RegimeState:
    """Regime known at epoch ``ts`` (callers pass a bar close time)."""
    if market.empty:
        return RegimeState("chop", 0.0, 0.0, 0.5, 0.5)
    idx = int(np.searchsorted(market["known_ts"].to_numpy(np.int64), int(ts), side="right") - 1)
    if idx < 0:
        return RegimeState("chop", 0.0, 0.0, 0.5, 0.5)
    row = market.iloc[idx]
    return RegimeState(
        label=str(row["label"]),
        score=float(row["score"]),
        btc_trend=float(row["btc_trend"]),
        breadth=float(row["breadth"]),
        vol_percentile=float(row["vol_pct"]) if np.isfinite(row["vol_pct"]) else 0.5,
    )
