"""Read-only market-data clients.

Only public endpoints are used.  There is no authentication, no API-key
handling and no order-submission code anywhere in this module.

Primary source: Binance spot REST (``/api/v3/klines``), which provides full
BTCUSDT history and is the venue the paper-trading fees/limits are modelled on.
A Coinbase public ticker is used only as a fallback reference price so a
Binance outage does not blind the dashboard to the market.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from ..config import ExchangeCfg
from .timeframes import tf_seconds

log = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com"
COINBASE_BASE = "https://api.exchange.coinbase.com"
KRAKEN_BASE = "https://api.kraken.com"

USER_AGENT = "btcbot-research/0.1 (paper trading; contact: local operator)"


class MarketDataError(RuntimeError):
    """Raised when market data could not be obtained or is unusable."""


@dataclass
class Ticker:
    pair: str
    price: float
    bid: float | None
    ask: float | None
    ts: int
    source: str


class _Http:
    """Small retrying HTTP helper with a polite minimum request interval."""

    def __init__(self, cfg: ExchangeCfg) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._last_call = 0.0

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            throttle = self.cfg.min_request_interval_s - (time.monotonic() - self._last_call)
            if throttle > 0:
                time.sleep(throttle)
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.request_timeout_s)
                self._last_call = time.monotonic()
            except requests.RequestException as exc:
                last_err = exc
                self._backoff(attempt, f"network error: {exc}")
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (418, 429):
                retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                log.warning("rate limited by %s (HTTP %s), sleeping %.1fs",
                            url.split("/")[2], resp.status_code, retry_after)
                time.sleep(min(retry_after, 60.0))
                last_err = MarketDataError(f"HTTP {resp.status_code} from {url}")
                continue
            if 500 <= resp.status_code < 600:
                last_err = MarketDataError(f"HTTP {resp.status_code} from {url}")
                self._backoff(attempt, str(last_err))
                continue
            # 4xx other than rate limits: do not retry, it will not improve.
            raise MarketDataError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
        raise MarketDataError(f"request to {url} failed after retries: {last_err}")

    @staticmethod
    def _backoff(attempt: int, msg: str) -> None:
        delay = min(2**attempt * 0.5, 8.0)
        log.warning("%s (retry %d in %.1fs)", msg, attempt + 1, delay)
        time.sleep(delay)


class BinancePublic:
    """Binance spot public market data (read-only)."""

    name = "binance"

    def __init__(self, cfg: ExchangeCfg) -> None:
        self.cfg = cfg
        self.base = (cfg.api_base or BINANCE_BASE).rstrip("/")
        self.http = _Http(cfg)

    # ------------------------------------------------------------ klines
    def fetch_klines(self, timeframe: str, start_s: int, end_s: int,
                     symbol: str | None = None) -> list[list]:
        """Fetch raw klines in ``[start_s, end_s)``, paginating 1000 bars at a time."""
        sym = symbol or self.cfg.exchange_symbol
        step = tf_seconds(timeframe)
        out: list[list] = []
        cursor = int(start_s)
        end_s = int(end_s)
        max_pages = 10_000
        for _ in range(max_pages):
            if cursor >= end_s:
                break
            params = {
                "symbol": sym,
                "interval": timeframe,
                "startTime": cursor * 1000,
                "endTime": min(end_s, cursor + 1000 * step) * 1000 - 1,
                "limit": 1000,
            }
            batch = self.http.get(f"{self.base}/api/v3/klines", params)
            if not isinstance(batch, list):
                raise MarketDataError(f"unexpected klines payload: {str(batch)[:200]}")
            if not batch:
                break
            out.extend(batch)
            last_open = int(batch[-1][0]) // 1000
            if last_open < cursor:  # defensive: never loop forever
                break
            cursor = last_open + step
        return out

    def fetch_recent(self, timeframe: str, limit: int = 500,
                     symbol: str | None = None) -> list[list]:
        sym = symbol or self.cfg.exchange_symbol
        return self.http.get(
            f"{self.base}/api/v3/klines",
            {"symbol": sym, "interval": timeframe, "limit": int(limit)},
        )

    def server_time(self) -> int:
        data = self.http.get(f"{self.base}/api/v3/time")
        return int(data["serverTime"]) // 1000

    def ticker(self) -> Ticker:
        book = self.http.get(
            f"{self.base}/api/v3/ticker/bookTicker", {"symbol": self.cfg.exchange_symbol}
        )
        bid = float(book["bidPrice"])
        ask = float(book["askPrice"])
        return Ticker(
            pair=self.cfg.pair,
            price=(bid + ask) / 2.0,
            bid=bid,
            ask=ask,
            ts=int(time.time()),
            source="binance",
        )


class CoinbasePublic:
    """Fallback read-only ticker (USD reference)."""

    name = "coinbase"

    def __init__(self, timeout: float = 10.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.timeout = timeout

    def ticker(self, pair: str = "BTC-USD") -> Ticker:
        resp = self.session.get(f"{COINBASE_BASE}/products/{pair}/ticker", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return Ticker(
            pair=pair.replace("-", "/"),
            price=float(data["price"]),
            bid=float(data["bid"]),
            ask=float(data["ask"]),
            ts=int(time.time()),
            source="coinbase",
        )


def make_client(cfg: ExchangeCfg) -> BinancePublic | CoinbasePublic:
    if cfg.name == "binance":
        return BinancePublic(cfg)
    if cfg.name == "coinbase":
        return CoinbasePublic()
    raise ValueError(f"unsupported exchange: {cfg.name}")


def get_ticker(cfg: ExchangeCfg) -> Ticker:
    """Best-effort ticker with a fallback venue, used by the dashboard and loop."""
    errors: list[str] = []
    if cfg.name == "binance":
        try:
            return BinancePublic(cfg).ticker()
        except Exception as exc:  # noqa: BLE001 - fallback path
            errors.append(f"binance: {exc}")
        try:
            return CoinbasePublic().ticker("BTC-USD")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"coinbase: {exc}")
    else:
        try:
            return CoinbasePublic().ticker("BTC-USD")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"coinbase: {exc}")
    raise MarketDataError("; ".join(errors))


__all__ = [
    "MarketDataError",
    "Ticker",
    "BinancePublic",
    "CoinbasePublic",
    "make_client",
    "get_ticker",
    "BINANCE_BASE",
    "COINBASE_BASE",
    "KRAKEN_BASE",
]
