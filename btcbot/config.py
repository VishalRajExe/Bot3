"""Configuration loading and validation.

Every knob that can change trading behaviour lives here so experiments are
reproducible from a single JSON file.  Validation is deliberately strict: the
paper-trading system refuses to start with an unsafe configuration (negative
fees, leverage enabled, or any attempt to enable real order submission).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


class ExchangeCfg(BaseModel):
    """Public market-data venue.  No credentials are ever needed for this."""

    name: Literal["binance", "kraken", "coinbase"] = "binance"
    base: str = "BTC"
    quote: str = "USDT"
    api_base: str | None = None  # override for mirrors/tests
    request_timeout_s: float = 15.0
    max_retries: int = 4
    min_request_interval_s: float = 0.25

    @property
    def pair(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def exchange_symbol(self) -> str:
        return f"{self.base}{self.quote}"


class DataCfg(BaseModel):
    timeframes: list[str] = Field(default_factory=lambda: ["1d", "4h"])
    history_start: str = "2017-08-17"
    db_path: str = "data/btcbot.sqlite3"
    max_gap_bars: int = 2  # tolerated gaps before data is marked unusable
    candle_cache_dir: str = "data/candles"

    @field_validator("timeframes")
    @classmethod
    def _known_timeframes(cls, v: list[str]) -> list[str]:
        allowed = {"1h", "2h", "4h", "6h", "12h", "1d"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"unsupported timeframes: {sorted(bad)}")
        if not v:
            raise ValueError("at least one timeframe is required")
        return v


class CostsCfg(BaseModel):
    """Transaction-cost model applied to *both* backtest and paper trading."""

    taker_fee_bps: float = 10.0  # Binance spot standard tier, per side
    maker_fee_bps: float = 10.0
    spread_bps: float = 2.0  # full quoted spread; half is paid per side
    slippage_bps: float = 5.0  # adverse slippage on market orders
    min_notional: float = 10.0  # Binance BTCUSDT MIN_NOTIONAL
    qty_step: float = 0.00001  # LOT_SIZE step
    price_tick: float = 0.01  # PRICE_FILTER tick

    @field_validator("taker_fee_bps", "maker_fee_bps", "spread_bps", "slippage_bps")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cost assumptions cannot be negative")
        return v


class RiskCfg(BaseModel):
    """Hard risk limits.  Applied by a manager the strategies cannot bypass."""

    risk_per_trade_pct: float = 0.5  # of current equity
    max_position_pct: float = 20.0  # of current equity, position notional
    max_positions: int = 1
    daily_loss_limit_pct: float = 2.0  # intraday equity drop from day-start
    max_drawdown_pct: float = 10.0  # from high-water mark, stops new entries
    consecutive_losses_limit: int = 3
    cooldown_hours: float = 24.0
    leverage: Literal[1] = 1  # leverage is not implemented; must stay 1
    max_bars_stale: int = 2  # reject signals on stale data
    max_abs_bar_return: float = 0.35  # abnormal-price circuit breaker
    min_stop_distance_pct: float = 1.0
    max_stop_distance_pct: float = 25.0
    # Extra cushion so an adverse gap between signal and next-open fill cannot
    # push exposure/risk materially past their limits.
    entry_gap_buffer_pct: float = 1.0

    @field_validator("risk_per_trade_pct", "max_position_pct", "daily_loss_limit_pct", "max_drawdown_pct")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("risk percentages must be > 0")
        return v

    @field_validator("risk_per_trade_pct")
    @classmethod
    def _sane_risk(cls, v: float) -> float:
        if v > 5:
            raise ValueError("risk_per_trade_pct above 5% is not permitted for this system")
        return v

    @model_validator(mode="after")
    def _no_leverage(self) -> "RiskCfg":
        if self.leverage != 1:
            raise ValueError("leverage is not supported")
        return self


class BacktestCfg(BaseModel):
    initial_equity: float = 10_000.0
    start: str | None = None  # ISO date; defaults to data start after warm-up
    end: str | None = None
    include_benchmarks: bool = True


class PaperCfg(BaseModel):
    initial_equity: float = 10_000.0
    strategy: str = "ema_std"
    strategy_params: dict = Field(default_factory=dict)
    timeframe: Literal["1h", "2h", "4h", "6h", "12h", "1d"] = "1d"
    poll_seconds: int = 60
    replay_bars: int = 0  # >0 seeds the paper session by replaying closed bars
    warmup_bars: int = 120
    resume_on_start: bool = True  # resume the newest live session when the API starts


class ApiCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8601


class TradingCfg(BaseModel):
    """Safety gates.  Real-money trading is not implemented at all."""

    mode: Literal["paper"] = "paper"
    enable_live_orders: bool = False

    @model_validator(mode="after")
    def _live_disabled(self) -> "TradingCfg":
        if self.enable_live_orders:
            raise ValueError(
                "enable_live_orders must remain false: this project intentionally "
                "implements paper trading only and has no live execution path."
            )
        return self


class Config(BaseModel):
    exchange: ExchangeCfg = Field(default_factory=ExchangeCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    paper: PaperCfg = Field(default_factory=PaperCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    trading: TradingCfg = Field(default_factory=TradingCfg)
    log_level: str = "INFO"

    def resolve(self, p: str | Path) -> Path:
        """Resolve a config-relative path against the project root."""
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def db_path(self) -> Path:
        return self.resolve(self.data.db_path)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from JSON, creating the default file on first run."""
    env_path = os.environ.get("BTCBOT_CONFIG")
    cfg_path = Path(path or env_path or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        cfg = Config()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2), encoding="utf-8")
        return cfg
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return Config.model_validate(raw)
