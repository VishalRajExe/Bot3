"""Strategy interface.

Design rules that every strategy must obey:

* ``prepare()`` may only compute indicators from past/current values.  It never
  shifts information backwards and never uses ``.shift(-n)``.
* ``signal(df, i, position)`` sees the prepared frame and bar index ``i``.  It
  must only read rows ``<= i``.  The backtest engine calls it *after* bar ``i``
  has closed and executes any order at bar ``i+1``'s open.
* Every signal carries a human-readable reason and the indicator values that
  produced it, so the journal and dashboard can explain decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from ..regime import add_regime_columns

Action = Literal["enter_long", "exit_long", "hold"]


@dataclass(frozen=True)
class PositionView:
    """What a strategy is allowed to know about the open position."""

    qty: float
    entry_price: float
    entry_ts: int
    stop_price: float | None
    entry_bar_ts: int
    entry_tag: str | None = None


@dataclass
class Signal:
    action: Action
    reason: str
    stop_price: float | None = None
    indicators: dict = field(default_factory=dict)
    tag: str = "default"

    @classmethod
    def hold(cls, reason: str = "no signal", **indicators) -> "Signal":
        return cls("hold", reason, None, indicators)

    @property
    def is_entry(self) -> bool:
        return self.action == "enter_long"

    @property
    def is_exit(self) -> bool:
        return self.action == "exit_long"


class Params(BaseModel):
    """Base class for strategy parameters; subclasses forbid unknown keys."""

    model_config = {"extra": "forbid"}


class Strategy(ABC):
    name: str = "base"
    version: str = "0.0.0"
    default_timeframe: str = "1d"
    description: str = ""
    PARAMS: type[Params] = Params

    def __init__(self, params: dict | None = None, timeframe: str | None = None) -> None:
        self.params = self.PARAMS.model_validate(params or {})
        self.timeframe = timeframe or self.default_timeframe
        self._prepared_index: pd.Index | None = None

    # ------------------------------------------------------------- lifecycle
    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Bars required before a signal can be trusted."""

    @abstractmethod
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with indicator columns; must remain causal."""

    @abstractmethod
    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        """Signal for bar ``i`` using only rows ``<= i``."""

    # ---------------------------------------------------------------- public
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._add_indicators(df.copy())
        if "regime" not in out.columns:
            out = add_regime_columns(out)
        self._prepared_index = out.index
        return out

    def signal(self, df: pd.DataFrame, i: int, position: PositionView | None = None) -> Signal:
        if i < 0 or i >= len(df):
            raise IndexError(f"bar index {i} out of range")
        if i < self.warmup_bars:
            return Signal.hold("warm-up period")
        sig = self._signal(df, i, position)
        if sig.stop_price is not None and not (sig.stop_price > 0):
            return Signal.hold("invalid stop price")
        return sig

    def params_dict(self) -> dict:
        return self.params.model_dump()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "timeframe": self.timeframe,
            "description": self.description,
            "params": self.params_dict(),
            "warmup_bars": self.warmup_bars,
        }

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _val(row: pd.Series, key: str) -> float | None:
        if key not in row:
            return None
        v = row[key]
        if v is None or pd.isna(v):
            return None
        return float(v)

    def _snapshot(self, row: pd.Series, keys: list[str]) -> dict:
        return {k: self._val(row, k) for k in keys if k in row.index}
