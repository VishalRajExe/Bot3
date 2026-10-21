"""Candle parsing, validation, caching and provenance.

The store keeps raw OHLCV in SQLite so research is reproducible offline and a
paper session does not re-download history on every restart.  Every load runs
structural validation before returning data to a strategy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..db import Database, utcnow_iso
from .timeframes import expected_bar_times, tf_seconds, to_datetime

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class CandleFormatError(ValueError):
    """Raised when candle data violates structural invariants."""


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    bars: int
    expected_bars: int
    missing_bars: int
    duplicate_bars: int
    bad_ohlc: int
    zero_volume_bars: int
    first_ts: int | None
    last_ts: int | None
    status: str
    details: dict = field(default_factory=dict)

    @property
    def missing_fraction(self) -> float:
        return self.missing_bars / self.expected_bars if self.expected_bars else 0.0

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars": self.bars,
            "expected_bars": self.expected_bars,
            "missing_bars": self.missing_bars,
            "missing_fraction": round(self.missing_fraction, 6),
            "duplicate_bars": self.duplicate_bars,
            "bad_ohlc": self.bad_ohlc,
            "zero_volume_bars": self.zero_volume_bars,
            "first_ts": self.first_ts,
            "last_ts": to_datetime(self.last_ts).isoformat() if self.last_ts else None,
            "status": self.status,
            "details": self.details,
        }


def parse_klines(raw: list[list], timeframe: str, source: str) -> pd.DataFrame:
    """Convert Binance kline arrays into a validated, de-duplicated frame."""
    if not raw:
        return empty_frame()
    rows = []
    for k in raw:
        rows.append(
            {
                "bar_ts": int(k[0]) // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]) if len(k) > 7 else np.nan,
                "trades": int(k[8]) if len(k) > 8 else np.nan,
                "close_time": int(k[6]) // 1000 if len(k) > 6 else int(k[0]) // 1000 + tf_seconds(timeframe),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("bar_ts").drop_duplicates(subset=["bar_ts"], keep="last")
    df = df.reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["bar_ts"], unit="s", utc=True)
    df["source"] = source
    return df


def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=["bar_ts", *OHLCV_COLUMNS, "quote_volume", "trades", "close_time", "dt", "source"])
    df["bar_ts"] = df["bar_ts"].astype("int64")
    return df


def validate_frame(df: pd.DataFrame, timeframe: str, *, symbol: str = "BTC/USDT",
                   max_gap_bars: int = 2, now_s: int | None = None) -> QualityReport:
    """Structural validation.  Raises only for unusable formats."""
    if df.empty:
        return QualityReport(symbol, timeframe, 0, 0, 0, 0, 0, 0, None, None, "fail",
                             {"reason": "no candles"})

    required = {"bar_ts", *OHLCV_COLUMNS}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise CandleFormatError(f"missing columns: {sorted(missing_cols)}")
    if df["bar_ts"].duplicated().any():
        raise CandleFormatError("duplicate bar timestamps")

    ts = df["bar_ts"].to_numpy(dtype=np.int64)
    step = tf_seconds(timeframe)
    if np.any(np.diff(ts) <= 0):
        raise CandleFormatError("bars are not strictly increasing in time")
    off_grid = int(np.sum(ts % step != 0))
    if off_grid:
        raise CandleFormatError(f"{off_grid} bars are not aligned to the {timeframe} UTC grid")

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    positive = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    coherent = (h >= np.maximum(o, c) - 1e-9) & (l <= np.minimum(o, c) + 1e-9) & (h >= l)
    bad_ohlc = int(np.sum(~(finite & positive & coherent)))

    expected = expected_bar_times(int(ts[0]), int(ts[-1]) + step, timeframe)
    missing = max(len(expected) - len(ts), 0)
    zero_vol = int(np.sum(v == 0))

    stale = False
    if now_s is not None and len(ts):
        last_close = int(ts[-1]) + step
        stale = (now_s - last_close) > max_gap_bars * step

    status = "ok"
    missing_frac = missing / len(expected) if expected else 0.0
    if bad_ohlc > 0 or missing_frac > 0.02 or stale:
        status = "fail"
    elif missing > 0 or zero_vol > 0 or off_grid:
        status = "warn"

    return QualityReport(
        symbol=symbol,
        timeframe=timeframe,
        bars=len(df),
        expected_bars=len(expected),
        missing_bars=missing,
        duplicate_bars=0,
        bad_ohlc=bad_ohlc,
        zero_volume_bars=zero_vol,
        first_ts=int(ts[0]),
        last_ts=int(ts[-1]),
        status=status,
        details={
            "stale": stale,
            "missing_fraction": round(missing_frac, 6),
            "first": to_datetime(int(ts[0])).isoformat(),
            "last": to_datetime(int(ts[-1])).isoformat(),
            "source": str(df["source"].iloc[0]) if "source" in df else "unknown",
        },
    )


def dataset_hash(df: pd.DataFrame) -> str:
    """Stable fingerprint of the exact bars used in a run."""
    if df.empty:
        return "empty"
    h = hashlib.sha256()
    ts = df["bar_ts"].to_numpy(dtype=np.int64)
    h.update(ts.tobytes())
    for col in OHLCV_COLUMNS:
        h.update(np.round(df[col].to_numpy(dtype=float), 8).tobytes())
    return h.hexdigest()[:32]


class CandleStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -------------------------------------------------------------- writing
    def upsert(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        if df.empty:
            return 0
        now = utcnow_iso()
        rows = [
            (
                symbol,
                timeframe,
                int(r.bar_ts),
                int(r.close_time),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
                None if pd.isna(r.quote_volume) else float(r.quote_volume),
                None if pd.isna(r.trades) else int(r.trades),
                str(r.source),
                now,
            )
            for r in df.itertuples(index=False)
        ]
        self.db.executemany(
            "INSERT INTO candles (symbol, timeframe, open_time, close_time, open, high, low, close,"
            " volume, quote_volume, trades, source, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(symbol, timeframe, open_time) DO UPDATE SET"
            " close=excluded.close, high=excluded.high, low=excluded.low, volume=excluded.volume,"
            " quote_volume=excluded.quote_volume, trades=excluded.trades,"
            " source=excluded.source, ingested_at=excluded.ingested_at",
            rows,
        )
        return len(rows)

    def record_source(self, name: str, endpoint: str, notes: str = "") -> None:
        now = utcnow_iso()
        self.db.execute(
            "INSERT INTO data_sources (name, endpoint, first_seen, last_success, notes)"
            " VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET last_success=excluded.last_success,"
            " endpoint=excluded.endpoint, notes=excluded.notes",
            (name, endpoint, now, now, notes),
        )

    def record_quality(self, report: QualityReport) -> None:
        self.db.execute(
            "INSERT INTO data_quality_checks (symbol, timeframe, checked_at, start_ts, end_ts, bars,"
            " expected_bars, missing_bars, duplicate_bars, bad_ohlc, zero_volume_bars, status,"
            " details_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                report.symbol, report.timeframe, utcnow_iso(), report.first_ts, report.last_ts,
                report.bars, report.expected_bars, report.missing_bars, report.duplicate_bars,
                report.bad_ohlc, report.zero_volume_bars, report.status,
                json.dumps(report.details),
            ),
        )

    def record_collection(self, source: str, started_at: str, rows_added: int, status: str,
                          detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO collection_runs (source, started_at, finished_at, rows_added, status, detail)"
            " VALUES (?,?,?,?,?,?)",
            (source, started_at, utcnow_iso(), rows_added, status, detail[:2000]),
        )

    # -------------------------------------------------------------- reading
    def load(self, symbol: str, timeframe: str, start: int | None = None,
             end: int | None = None) -> pd.DataFrame:
        clauses = ["symbol = ?", "timeframe = ?"]
        params: list = [symbol, timeframe]
        if start is not None:
            clauses.append("open_time >= ?")
            params.append(int(start))
        if end is not None:
            clauses.append("open_time < ?")
            params.append(int(end))
        sql = (
            "SELECT open_time AS bar_ts, open, high, low, close, volume, quote_volume, trades,"
            " close_time, source FROM candles WHERE "
            + " AND ".join(clauses)
            + " ORDER BY open_time"
        )
        rows = self.db.query(sql, params)
        if not rows:
            return empty_frame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["bar_ts"] = df["bar_ts"].astype("int64")
        df["dt"] = pd.to_datetime(df["bar_ts"], unit="s", utc=True)
        return df

    def last_open_time(self, symbol: str, timeframe: str) -> int | None:
        val = self.db.scalar(
            "SELECT MAX(open_time) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        )
        return None if val is None else int(val)

    def count(self, symbol: str, timeframe: str) -> int:
        return int(self.db.scalar(
            "SELECT COUNT(*) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ) or 0)

    def summary(self, symbol: str, timeframe: str) -> dict:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n, MIN(open_time) AS first, MAX(open_time) AS last,"
            " MIN(ingested_at) AS first_ingest, MAX(ingested_at) AS last_ingest"
            " FROM candles WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        )
        if row is None or row["n"] == 0:
            return {"symbol": symbol, "timeframe": timeframe, "bars": 0}
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": row["n"],
            "first": to_datetime(row["first"]).isoformat(),
            "last": to_datetime(row["last"]).isoformat(),
            "last_ingested_at": row["last_ingest"],
            "age_seconds": int(datetime.now(timezone.utc).timestamp() - row["last"]),
        }
