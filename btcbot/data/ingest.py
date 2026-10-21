"""Market-data ingestion.

Two rules that protect research integrity:

1. **Only closed candles are stored or returned.**  A candle is stored once
   ``open_time + timeframe <= now``.  Partial candles would leak future
   information into signals.
2. **Every ingestion path records provenance and a quality report**, so a
   backtest can be traced to the exact bars it used.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import Config
from ..db import Database, utcnow_iso
from .exchange import BinancePublic, MarketDataError
from .store import CandleStore, QualityReport, dataset_hash, parse_klines, validate_frame
from .timeframes import add_bars, parse_iso_date, tf_seconds, to_datetime

log = logging.getLogger(__name__)


class CandleIngestor:
    def __init__(self, cfg: Config, db: Database, client: BinancePublic | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.store = CandleStore(db)
        self.client = client or BinancePublic(cfg.exchange)
        self.symbol = cfg.exchange.pair

    # ------------------------------------------------------------------ sync
    def sync(self, timeframe: str, start: int | None = None, end: int | None = None,
             *, force_full: bool = False) -> QualityReport:
        """Download any missing history for one timeframe."""
        tf = tf_seconds(timeframe)
        now = int(time.time())
        end = min(end if end is not None else now, now)

        if start is not None:
            cursor = start
        elif force_full:
            cursor = parse_iso_date(self.cfg.data.history_start)
        else:
            last = self.store.last_open_time(self.symbol, timeframe)
            cursor = add_bars(last, timeframe, 1) if last is not None else parse_iso_date(
                self.cfg.data.history_start
            )

        if cursor >= end:
            df = self.store.load(self.symbol, timeframe)
            report = validate_frame(df, timeframe, symbol=self.symbol,
                                    max_gap_bars=self.cfg.data.max_gap_bars, now_s=now)
            return report

        started = utcnow_iso()
        log.info("syncing %s %s from %s", self.symbol, timeframe, to_datetime(cursor).date())
        raw = self.client.fetch_klines(timeframe, cursor, end)
        if not raw:
            raise MarketDataError(f"no klines returned for {timeframe} from {to_datetime(cursor)}")

        df = parse_klines(raw, timeframe, self.client.name)
        # Drop any still-forming candle.
        df = df[df["bar_ts"] + tf <= now].reset_index(drop=True)
        if df.empty:
            raise MarketDataError("all fetched candles are still forming")

        self.store.upsert(df, self.symbol, timeframe)
        self.store.record_source(
            f"{self.client.name}:{self.symbol}:{timeframe}",
            f"{self.client.base if hasattr(self.client, 'base') else 'public'} klines",
            notes="public read-only OHLCV",
        )
        full = self.store.load(self.symbol, timeframe)
        report = validate_frame(full, timeframe, symbol=self.symbol,
                                max_gap_bars=self.cfg.data.max_gap_bars, now_s=now)
        self.store.record_quality(report)
        self.store.record_collection(
            f"klines:{timeframe}", started, len(df), report.status,
            f"range {to_datetime(df['bar_ts'].iloc[0])} .. {to_datetime(df['bar_ts'].iloc[-1])}",
        )
        self.db.set_state(f"sync:{timeframe}", {"at": utcnow_iso(), "bars": report.bars,
                                                "status": report.status,
                                                "hash": dataset_hash(full)})
        log.info("synced %d new %s bars (total %d, quality=%s, missing=%d)",
                 len(df), timeframe, report.bars, report.status, report.missing_bars)
        return report

    def sync_all(self, force_full: bool = False) -> dict[str, QualityReport]:
        out: dict[str, QualityReport] = {}
        for tf in self.cfg.data.timeframes:
            try:
                out[tf] = self.sync(tf, force_full=force_full)
            except MarketDataError as exc:
                log.error("sync failed for %s: %s", tf, exc)
                self.db.log_error(f"candle sync failed: {exc}", logger=__name__,
                                  context={"timeframe": tf})
                raise
        return out

    # --------------------------------------------------------------- reading
    def recent_closed(self, timeframe: str, bars: int = 300) -> pd.DataFrame:
        """Fetch the most recent closed bars from the venue (live path)."""
        now = int(time.time())
        tf = tf_seconds(timeframe)
        raw = self.client.fetch_recent(timeframe, limit=min(max(bars, 10), 1000))
        df = parse_klines(raw, timeframe, self.client.name)
        df = df[df["bar_ts"] + tf <= now].reset_index(drop=True)
        if df.empty:
            raise MarketDataError(f"no closed {timeframe} candles returned")
        return df.tail(bars).reset_index(drop=True)

    def load_history(self, timeframe: str, start: int | None = None,
                     end: int | None = None) -> pd.DataFrame:
        return self.store.load(self.symbol, timeframe, start, end)

    def ensure_warmup(self, timeframe: str, warmup_bars: int) -> pd.DataFrame:
        """Return history, syncing on demand so a fresh clone can run."""
        df = self.load_history(timeframe)
        if len(df) < warmup_bars + 5:
            log.info("insufficient cached history (%d bars), syncing", len(df))
            self.sync(timeframe, start=parse_iso_date(self.cfg.data.history_start))
            df = self.load_history(timeframe)
        return df
