"""Deterministic helpers for the test-suite (no network, no randomness leaks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcbot.data.timeframes import tf_seconds
from btcbot.indicators import atr
from btcbot.strategies.base import Params, PositionView, Signal, Strategy

BASE_TS = 1_600_000_000 - (1_600_000_000 % 86_400)


def make_candles(n: int = 600, seed: int = 7, price: float = 10_000.0,
                 drift: float = 0.0008, vol: float = 0.02,
                 timeframe: str = "1d", start: int | None = None,
                 end_ts: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = tf_seconds(timeframe)
    rets = rng.normal(drift, vol, n)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.001, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.lognormal(2.0, 0.4, n)
    base = BASE_TS if start is None else int(start)
    if end_ts is not None:
        base = int(end_ts) - (n - 1) * step
    ts = base + np.arange(n, dtype=np.int64) * step
    df = pd.DataFrame({
        "bar_ts": ts,
        "open": open_,
        "high": high,
        "low": np.maximum(low, 1e-6),
        "close": close,
        "volume": volume,
        "quote_volume": volume * close,
        "trades": np.full(n, 100),
        "close_time": ts + step,
        "source": "synthetic",
    })
    df["dt"] = pd.to_datetime(df["bar_ts"], unit="s", utc=True)
    return df


class DummyParams(Params):
    entry_bar: int = 5
    exit_bar: int = 10
    stop_pct: float = 0.5  # very wide, so stops are not hit by accident
    use_stop: bool = True


class DummyStrategy(Strategy):
    """Signals at fixed bar indices; used to test engine mechanics."""

    name = "dummy"
    version = "test"
    default_timeframe = "1d"
    PARAMS = DummyParams

    @property
    def warmup_bars(self) -> int:
        return 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df["high"], df["low"], df["close"], 14)
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        close = float(df["close"].iloc[i])
        if position is None and i == self.params.entry_bar:
            stop = close * (1 - self.params.stop_pct) if self.params.use_stop else None
            return Signal("enter_long", "dummy entry", stop, {"close": close}, tag="dummy")
        if position is not None and i == self.params.exit_bar:
            return Signal("exit_long", "dummy exit", None, {"close": close}, tag="dummy")
        return Signal.hold("dummy hold")
