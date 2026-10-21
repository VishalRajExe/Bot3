"""Strategy F -- ML-gated long-only strategy.

The strategy does **not** train anything.  It consumes out-of-fold (or, in live
mode, final-model) probabilities produced by :mod:`btcbot.ml.train` and combines
them with a long-term trend filter and a hard ATR stop.  If predictions are
missing for a bar it emits no signal, so a broken model cannot create trades.

Parameters are fixed before evaluation; the model cannot widen risk limits.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, ema
from .base import Params, PositionView, Signal, Strategy


class MLParams(Params):
    entry_prob: float = 0.55
    exit_prob: float = 0.45
    trend_ema: int = 200
    atr_window: int = 14
    stop_atr_mult: float = 2.5
    require_trend: bool = True


class MLGated(Strategy):
    name = "ml_gated"
    version = "1.0.0"
    default_timeframe = "1d"
    description = (
        "Long when the walk-forward probability of a cost-clearing forward return exceeds a "
        "threshold and price is above its 200-day EMA; exit when the probability decays or "
        "trend breaks. Predictions are out-of-fold during backtests."
    )
    PARAMS = MLParams

    def __init__(self, params: dict | None = None, timeframe: str | None = None,
                 predictions: pd.DataFrame | None = None, model_version: str | None = None) -> None:
        super().__init__(params, timeframe)
        self.predictions = predictions
        self.model_version = model_version

    @property
    def warmup_bars(self) -> int:
        return max(self.params.trend_ema, self.params.atr_window) + 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        df["atr"] = atr(df["high"], df["low"], close, p.atr_window)
        if p.require_trend:
            df["ema_trend"] = ema(close, p.trend_ema)
        if self.predictions is not None and not self.predictions.empty:
            preds = self.predictions[["bar_ts", "prob_up"]].copy()
            preds["bar_ts"] = preds["bar_ts"].astype("int64")
            df = df.merge(preds, on="bar_ts", how="left")
        else:
            df["prob_up"] = float("nan")
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        prob = self._val(row, "prob_up")
        close = float(row["close"])
        atr_val = self._val(row, "atr")
        ind = self._snapshot(row, ["close", "prob_up", "atr", "ema_trend"])
        if prob is None:
            return Signal.hold("no model prediction for this bar", **ind)

        trend_ok = True
        if self.params.require_trend:
            ema_val = self._val(row, "ema_trend")
            trend_ok = ema_val is not None and close > ema_val

        if position is None:
            if prob >= self.params.entry_prob and trend_ok:
                stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
                return Signal(
                    "enter_long",
                    f"model P(up over horizon)={prob:.3f} >= {self.params.entry_prob:.2f} "
                    f"with price above EMA{self.params.trend_ema}",
                    stop, ind, tag="ml",
                )
            if prob >= self.params.entry_prob and not trend_ok:
                return Signal.hold(f"model bullish ({prob:.3f}) but trend filter blocks entry", **ind)
            return Signal.hold(f"model probability {prob:.3f} below entry threshold", **ind)

        if prob <= self.params.exit_prob or not trend_ok:
            why = (f"model probability decayed to {prob:.3f}" if prob <= self.params.exit_prob
                   else "price fell below trend EMA")
            return Signal("exit_long", f"ML exit: {why}", None, ind, tag="ml")
        return Signal.hold(f"holding: model probability {prob:.3f}", **ind)
