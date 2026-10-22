"""Configuration for the unified multi-asset bot.

One JSON file drives the whole system: the tradable universe, each symbol's
strategy and parameters, Alpaca cost assumptions, portfolio risk limits and the
model/learning schedule.  Environment variables supply credentials only and are
never written into the config or the database.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CostsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taker_fee_bps: float = 25.0
    maker_fee_bps: float = 15.0
    spread_bps: float = 4.0
    slippage_bps: float = 5.0
    min_notional: float = 1.0
    qty_step: float = 1e-9
    price_tick: float = 1e-9

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.taker_fee_bps + self.spread_bps / 2.0 + self.slippage_bps)

    @property
    def round_trip_pct(self) -> float:
        return self.round_trip_bps / 100.0


class RiskCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_per_trade_pct: float = 1.0
    max_positions: int = 2
    max_per_cluster: int = 1
    max_position_pct: float = 35.0
    cash_reserve_pct: float = 25.0
    portfolio_heat_pct: float = 2.0
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 12.0
    consecutive_losses_limit: int = 3
    cooldown_hours: float = 12.0
    loss_cooldown_bars: int = 3
    regime_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"bull": 1.0, "chop": 0.6, "bear": 0.25}
    )
    min_expected_edge_bps: float = 150.0
    max_bar_return_pct: float = 25.0

    @model_validator(mode="after")
    def _bounded(self) -> "RiskCfg":
        if not 0 < self.risk_per_trade_pct <= 2.0:
            raise ValueError("risk_per_trade_pct must be in (0, 2] for a small account")
        if not 0 < self.max_position_pct <= 50:
            raise ValueError("max_position_pct must be in (0, 50]")
        if self.max_positions < 1 or self.max_positions > 5:
            raise ValueError("max_positions must be between 1 and 5")
        if self.cash_reserve_pct < 0 or self.cash_reserve_pct >= 100:
            raise ValueError("cash_reserve_pct must be in [0, 100)")
        return self


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    algorithm: Literal["gradient_boosting", "random_forest"] = "gradient_boosting"
    horizon_bars: int = 18
    u_barrier_atr: float = 2.0
    l_barrier_atr: float = 1.0
    min_samples: int = 250
    min_test_samples: int = 60
    min_auc: float = 0.55
    min_oos_improvement_pct: float = 0.0
    train_months: int = 18
    test_months: int = 4
    step_months: int = 4
    purge_bars: int = 24
    retrain_every_trades: int = 25


class AlpacaCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_base_url: str = "https://paper-api.alpaca.markets"
    data_base_url: str = "https://data.alpaca.markets"
    data_feed: str = "us"
    use_stop_orders: bool = True
    stop_limit_buffer_pct: float = 1.5
    reconcile_seconds: int = 300


class TelegramCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    notify_holds: bool = True
    hold_digest_minutes: int = 240
    rate_limit_per_minute: int = 18


class PaperCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poll_seconds: int = 60
    decision_timeframe: str = "4h"
    trade_on_bar_close: bool = True
    max_order_slippage_bps: float = 40.0


class ApiCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8610


class PathsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db: str = "data/unibot.sqlite3"
    models: str = "data/models"
    reports: str = "reports/unibot"
    dashboard: str = "web/unibot"


class SymbolCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    cluster: str
    strategy: Literal["trend_breakout", "trend_pullback", "mean_reversion"]
    trade_enabled: bool = True
    role: str = "tradable"
    execution_symbol: str | None = None
    note: str = ""
    params: dict = Field(default_factory=dict)

    @property
    def exec_symbol(self) -> str:
        return self.execution_symbol or self.symbol


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capital: float = 100.0
    base_timeframe: str = "1h"
    timeframe: str = "4h"
    history_start: str = "2021-01-01"
    hourly_start: str = "2024-01-01"   # Alpaca caps 1Hour pages at ~170 bars
    warmup_bars: int = 420
    universe: list[SymbolCfg]
    costs: CostsCfg = Field(default_factory=CostsCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    models: ModelCfg = Field(default_factory=ModelCfg)
    alpaca: AlpacaCfg = Field(default_factory=AlpacaCfg)
    telegram: TelegramCfg = Field(default_factory=TelegramCfg)
    paper: PaperCfg = Field(default_factory=PaperCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)

    @model_validator(mode="after")
    def _sane(self) -> "Settings":
        names = [s.symbol for s in self.universe]
        if len(set(names)) != len(names):
            raise ValueError("duplicate symbols in universe")
        if self.timeframe not in {"1h", "2h", "4h", "6h", "12h", "1d"}:
            raise ValueError("unsupported decision timeframe")
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if self.warmup_bars < 200:
            raise ValueError("warmup_bars below 200 leaves long indicators undefined")
        return self

    # ------------------------------------------------------------ accessors
    @property
    def symbols(self) -> list[str]:
        return [s.symbol for s in self.universe]

    @property
    def tradable(self) -> list[SymbolCfg]:
        return [s for s in self.universe if s.trade_enabled]

    def symbol(self, name: str) -> SymbolCfg:
        for s in self.universe:
            if s.symbol == name:
                return s
        raise KeyError(name)

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.universe:
            out.setdefault(s.cluster, []).append(s.symbol)
        return out

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path = "config/unibot.json") -> "Settings":
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


@dataclass(frozen=True)
class Credentials:
    alpaca_key: str
    alpaca_secret: str
    telegram_token: str
    telegram_chat_id: str
    alpaca_base_url: str

    @property
    def alpaca_configured(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """Minimal .env reader (does not export to os.environ)."""
    p = Path(path)
    values: dict[str, str] = {}
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip("\"'")
    return values


def load_credentials(env_path: str | Path = ".env") -> Credentials:
    env = {**load_env(env_path), **{k: v for k, v in os.environ.items() if v}}
    base = env.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    if "paper" not in base:
        raise ValueError("refusing non-paper Alpaca endpoint; this system is paper only")
    key = env.get("ALPACA_API_KEY") or env.get("ALPACA_PAPER_API_KEY", "")
    secret = env.get("ALPACA_SECRET_KEY") or env.get("ALPACA_PAPER_API_SECRET", "")
    return Credentials(
        alpaca_key=key,
        alpaca_secret=secret,
        telegram_token=env.get("TELEGRAM_BOT_TOKEN") or env.get("TELEGRAM_TOKEN", ""),
        telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
        alpaca_base_url=base,
    )
