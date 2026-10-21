"""SQLite persistence.

A single database holds research and paper-trading state:

* ``candles``                  cached OHLCV (raw research input)
* ``backtest_runs``            one row per reproducible backtest
* ``paper_sessions``           paper-trading sessions (live and replay)
* ``orders`` / ``fills``       simulated order lifecycle
* ``trades``                   completed round trips == the permanent journal
* ``equity_curve``             per-bar equity for every run
* ``signals``                  every generated decision, including no-ops
* ``risk_events``              every risk veto / circuit breaker
* ``experiments``              recorded strategy change and its verdict
* ``models``                   ML model registry with promotion records
* ``sentiment_observations``   timestamped sentiment features
* ``system_state``             restart-safe key/value state
* ``errors``                   unexpected errors with context

WAL mode keeps the API readable while the paper loop writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,   -- UTC epoch seconds, bar open
    close_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL,
    trades INTEGER,
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, open_time)
);
CREATE INDEX IF NOT EXISTS idx_candles_sym_tf ON candles (symbol, timeframe, open_time);

CREATE TABLE IF NOT EXISTS data_sources (
    name TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_success TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS data_quality_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    start_ts INTEGER,
    end_ts INTEGER,
    bars INTEGER,
    expected_bars INTEGER,
    missing_bars INTEGER,
    duplicate_bars INTEGER,
    bad_ohlc INTEGER,
    zero_volume_bars INTEGER,
    status TEXT NOT NULL,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    initial_equity REAL NOT NULL,
    costs_json TEXT NOT NULL,
    risk_json TEXT NOT NULL,
    dataset_hash TEXT,
    dataset_bars INTEGER,
    code_version TEXT,
    metrics_json TEXT,
    benchmark_json TEXT,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS paper_sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    stopped_at TEXT,
    mode TEXT NOT NULL,             -- 'live' or 'replay'
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    initial_equity REAL NOT NULL,
    status TEXT NOT NULL,           -- 'running', 'stopped', 'paused'
    last_processed_bar INTEGER,
    config_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    run_id TEXT,                    -- backtest run id or paper session id
    mode TEXT NOT NULL,             -- 'backtest' | 'paper'
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    qty REAL NOT NULL,
    requested_price REAL,
    status TEXT NOT NULL,           -- 'pending' | 'filled' | 'rejected' | 'cancelled'
    reject_reason TEXT,
    reason TEXT,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    signal_bar_ts INTEGER NOT NULL, -- bar whose close produced the signal
    target_bar_ts INTEGER,          -- bar at whose open execution is attempted
    created_at TEXT NOT NULL,
    unique_tag TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders (run_id, mode);

CREATE TABLE IF NOT EXISTS fills (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    ts INTEGER NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    notional REAL NOT NULL,
    fee REAL NOT NULL,
    fee_bps REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    spread_bps REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    run_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_version TEXT,
    entry_order_id TEXT,
    exit_order_id TEXT,
    entry_ts INTEGER NOT NULL,
    entry_bar_ts INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_signal TEXT,
    entry_reason TEXT,
    exit_ts INTEGER,
    exit_price REAL,
    exit_signal TEXT,
    exit_reason TEXT,
    qty REAL NOT NULL,
    initial_stop REAL,
    intended_stop REAL,
    stop_distance_pct REAL,
    gross_pnl REAL,
    fees REAL,
    slippage_cost REAL,
    net_pnl REAL,
    return_pct REAL,
    r_multiple REAL,
    mfe_pct REAL,
    mae_pct REAL,
    holding_hours REAL,
    regime TEXT,
    indicators_entry_json TEXT,
    sentiment_entry_json TEXT,
    risk_events_json TEXT,
    anomalies_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades (mode, exit_ts);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades (run_id);

CREATE TABLE IF NOT EXISTS trade_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL REFERENCES trades(id),
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    finding TEXT NOT NULL,
    evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    run_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    position_qty REAL NOT NULL,
    price REAL NOT NULL,
    drawdown REAL NOT NULL,
    PRIMARY KEY (run_id, ts)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    mode TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    bar_ts INTEGER NOT NULL,
    action TEXT NOT NULL,             -- 'buy' | 'sell' | 'hold' | 'exit'
    reason TEXT NOT NULL,
    price REAL,
    indicators_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_run ON signals (run_id, bar_ts);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    mode TEXT NOT NULL,
    run_id TEXT,
    kind TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    value_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    baseline_run_id TEXT,
    candidate_run_id TEXT,
    verdict TEXT NOT NULL,            -- 'accepted' | 'rejected' | 'inconclusive'
    notes TEXT,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    feature_names_json TEXT NOT NULL,
    label_definition TEXT NOT NULL,
    train_start INTEGER,
    train_end INTEGER,
    test_start INTEGER,
    test_end INTEGER,
    metrics_json TEXT,
    artifact_path TEXT,
    status TEXT NOT NULL,             -- 'candidate' | 'validated' | 'rejected' | 'active'
    promoted_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    ts INTEGER NOT NULL,
    bar_ts INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    prob_up REAL,
    expected_return REAL,
    used_in_trading INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (model_version, bar_ts, horizon)
);

CREATE TABLE IF NOT EXISTS sentiment_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ts INTEGER NOT NULL,             -- time the observation describes
    available_ts INTEGER NOT NULL,   -- first time the system could know it
    kind TEXT NOT NULL,              -- 'fear_greed' | 'news_lexicon' | ...
    value REAL,
    label TEXT,
    payload_json TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE (source, ts, kind)
);

CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    rows_added INTEGER,
    status TEXT NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    logger TEXT,
    level TEXT,
    message TEXT NOT NULL,
    context_json TEXT
);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-safe SQLite wrapper.

    One connection guarded by a lock keeps the paper loop and the read-only API
    consistent without a connection pool.  Write volume is tiny (a few rows per
    bar), so this is more than adequate.
    """

    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30.0, isolation_level="DEFERRED"
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if create:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()

    # ------------------------------------------------------------------ core
    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock, self.conn:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock, self.conn:
            self.conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        return None if row is None else row[0]

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------- key/value
    def set_state(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO system_state (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), utcnow_iso()),
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM system_state WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    # ------------------------------------------------------------ error log
    def log_error(self, message: str, *, logger: str | None = None, level: str = "ERROR",
                  context: dict | None = None) -> None:
        self.execute(
            "INSERT INTO errors (ts, logger, level, message, context_json) VALUES (?,?,?,?,?)",
            (utcnow_iso(), logger, level, message, json.dumps(context or {})),
        )

    # ------------------------------------------------------------ risk events
    def record_risk_event(self, *, ts: int, mode: str, run_id: str | None, kind: str,
                          action: str, detail: str, value: dict | None = None) -> None:
        self.execute(
            "INSERT INTO risk_events (ts, mode, run_id, kind, action, detail, value_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (int(ts), mode, run_id, kind, action, detail, json.dumps(value or {}), utcnow_iso()),
        )

    # --------------------------------------------------------------- signals
    def record_signal(self, *, run_id: str | None, mode: str, strategy: str,
                      strategy_version: str, bar_ts: int, action: str, reason: str,
                      price: float | None, indicators: dict | None = None) -> None:
        self.execute(
            "INSERT INTO signals (run_id, mode, strategy, strategy_version, bar_ts, action, "
            "reason, price, indicators_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, mode, strategy, strategy_version, int(bar_ts), action, reason, price,
             json.dumps(indicators or {}, default=str), utcnow_iso()),
        )
