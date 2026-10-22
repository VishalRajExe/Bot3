"""Durable state for the unified bot.

The base schema (candles, orders, fills, trades, equity, signals, models, ...)
is shared with the BTC research engine so both systems read one database.  This
module adds the multi-asset tables: per-symbol decisions, live positions, model
promotions, lessons and per-symbol statistics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btcbot.db import Database, utcnow_iso

UNIBOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS uni_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    bar_ts INTEGER NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    price REAL,
    prob REAL,
    size_usd REAL,
    stop_price REAL,
    regime TEXT,
    blocked_by TEXT,
    features_json TEXT,
    notified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uni_dec_ts ON uni_decisions (ts DESC);
CREATE INDEX IF NOT EXISTS idx_uni_dec_sym ON uni_decisions (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS uni_positions (
    symbol TEXT PRIMARY KEY,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_ts INTEGER NOT NULL,
    stop_price REAL,
    initial_stop REAL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_version TEXT,
    prob REAL,
    notional REAL NOT NULL,
    risk_usd REAL NOT NULL,
    stop_order_id TEXT,
    opened_by TEXT NOT NULL,          -- 'live' | 'imported'
    meta_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uni_symbol_stats (
    symbol TEXT PRIMARY KEY,
    n_trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    net_pnl REAL NOT NULL DEFAULT 0.0,
    gross_profit REAL NOT NULL DEFAULT 0.0,
    gross_loss REAL NOT NULL DEFAULT 0.0,
    consec_losses INTEGER NOT NULL DEFAULT 0,
    risk_multiplier REAL NOT NULL DEFAULT 1.0,
    cooldown_until INTEGER NOT NULL DEFAULT 0,
    last_trade_ts INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uni_models (
    version TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    train_start INTEGER,
    train_end INTEGER,
    test_start INTEGER,
    test_end INTEGER,
    n_train INTEGER,
    n_test INTEGER,
    auc REAL,
    brier REAL,
    base_rate REAL,
    threshold REAL,
    oos_trades INTEGER,
    oos_net_pnl REAL,
    oos_return_pct REAL,
    oos_improvement_pct REAL,
    status TEXT NOT NULL,             -- active | candidate | rejected | insufficient
    artifact_path TEXT,
    metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_uni_models_sym ON uni_models (symbol, trained_at DESC);

CREATE TABLE IF NOT EXISTS uni_model_predictions (
    model_version TEXT NOT NULL,
    ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    prob REAL NOT NULL,
    used_in_trading INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model_version, symbol, ts)
);

CREATE TABLE IF NOT EXISTS uni_lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    title TEXT NOT NULL,
    finding TEXT NOT NULL,
    evidence_json TEXT,
    sample_size INTEGER,
    status TEXT NOT NULL DEFAULT 'open',   -- open | confirmed | retired
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uni_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    entry_ts INTEGER NOT NULL,
    features_json TEXT NOT NULL,
    prob REAL,
    outcome INTEGER NOT NULL,
    return_pct REAL NOT NULL,
    net_pnl REAL NOT NULL,
    regime TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uni_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,               -- portfolio_backtest | symbol_backtest | live
    label TEXT NOT NULL,
    start_ts INTEGER,
    end_ts INTEGER,
    initial_equity REAL NOT NULL,
    final_equity REAL,
    metrics_json TEXT,
    params_json TEXT,
    symbols_json TEXT,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS uni_equity (
    run_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    invested REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    drawdown REAL NOT NULL,
    positions INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, ts)
);

CREATE TABLE IF NOT EXISTS uni_regime (
    ts INTEGER NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    btc_trend REAL,
    breadth REAL,
    vol_percentile REAL,
    payload_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (ts)
);

CREATE TABLE IF NOT EXISTS uni_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    text TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS uni_recon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS uni_portfolio (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class UnibotStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db = Database(db_path)
        self.db.conn.executescript(UNIBOT_SCHEMA)
        self.db.conn.commit()

    # ------------------------------------------------------------- decisions
    def record_decision(
        self,
        *,
        ts: int,
        bar_ts: int,
        mode: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        strategy_version: str,
        action: str,
        reason: str,
        price: float | None = None,
        prob: float | None = None,
        size_usd: float | None = None,
        stop_price: float | None = None,
        regime: str | None = None,
        blocked_by: str | None = None,
        features: dict | None = None,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO uni_decisions (ts, bar_ts, mode, symbol, timeframe, strategy,"
            " strategy_version, action, reason, price, prob, size_usd, stop_price, regime,"
            " blocked_by, features_json, notified, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
            (int(ts), int(bar_ts), mode, symbol, timeframe, strategy, strategy_version,
             action, reason, price, prob, size_usd, stop_price, regime, blocked_by,
             json.dumps(features or {}, default=str), utcnow_iso()),
        )
        return int(cur.lastrowid or 0)

    def mark_notified(self, decision_id: int) -> None:
        self.db.execute("UPDATE uni_decisions SET notified=1 WHERE id=?", (decision_id,))

    def decisions(self, limit: int = 100, symbol: str | None = None,
                  since: int | None = None) -> list[dict]:
        sql = "SELECT * FROM uni_decisions"
        clauses, params = [], []
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol)
        if since:
            clauses.append("ts>=?")
            params.append(int(since))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.db.query(sql, params)]

    def latest_decision_per_symbol(self) -> dict[str, dict]:
        rows = self.db.query(
            "SELECT d.* FROM uni_decisions d JOIN ("
            " SELECT symbol, MAX(ts) AS mts FROM uni_decisions GROUP BY symbol"
            ") m ON d.symbol=m.symbol AND d.ts=m.mts"
        )
        out: dict[str, dict] = {}
        for r in rows:
            out[r["symbol"]] = dict(r)
        return out

    # -------------------------------------------------------------- positions
    def save_position(
        self,
        *,
        symbol: str,
        qty: float,
        entry_price: float,
        entry_ts: int,
        stop_price: float | None,
        initial_stop: float | None,
        strategy: str,
        strategy_version: str,
        model_version: str | None,
        prob: float | None,
        notional: float,
        risk_usd: float,
        stop_order_id: str | None = None,
        opened_by: str = "live",
        meta: dict | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO uni_positions (symbol, qty, entry_price, entry_ts, stop_price,"
            " initial_stop, strategy, strategy_version, model_version, prob, notional,"
            " risk_usd, stop_order_id, opened_by, meta_json, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty,"
            " entry_price=excluded.entry_price, entry_ts=excluded.entry_ts,"
            " stop_price=excluded.stop_price, initial_stop=excluded.initial_stop,"
            " strategy=excluded.strategy, strategy_version=excluded.strategy_version,"
            " model_version=excluded.model_version, prob=excluded.prob,"
            " notional=excluded.notional, risk_usd=excluded.risk_usd,"
            " stop_order_id=excluded.stop_order_id, opened_by=excluded.opened_by,"
            " meta_json=excluded.meta_json, updated_at=excluded.updated_at",
            (symbol, qty, entry_price, int(entry_ts), stop_price, initial_stop, strategy,
             strategy_version, model_version, prob, notional, risk_usd, stop_order_id,
             opened_by, json.dumps(meta or {}, default=str), utcnow_iso()),
        )

    def update_stop(self, symbol: str, stop_price: float, stop_order_id: str | None = None) -> None:
        if stop_order_id is None:
            self.db.execute(
                "UPDATE uni_positions SET stop_price=?, updated_at=? WHERE symbol=?",
                (stop_price, utcnow_iso(), symbol),
            )
        else:
            self.db.execute(
                "UPDATE uni_positions SET stop_price=?, stop_order_id=?, updated_at=? WHERE symbol=?",
                (stop_price, stop_order_id, utcnow_iso(), symbol),
            )

    def delete_position(self, symbol: str) -> None:
        self.db.execute("DELETE FROM uni_positions WHERE symbol=?", (symbol,))

    def positions(self) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT * FROM uni_positions ORDER BY symbol")]

    def position(self, symbol: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM uni_positions WHERE symbol=?", (symbol,))
        return dict(row) if row else None

    # ----------------------------------------------------------------- stats
    def symbol_stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.db.query("SELECT * FROM uni_symbol_stats"):
            d = dict(r)
            out[d["symbol"]] = d
        return out

    def stats_row(self, symbol: str) -> dict:
        row = self.db.query_one("SELECT * FROM uni_symbol_stats WHERE symbol=?", (symbol,))
        if row:
            return dict(row)
        return {
            "symbol": symbol, "n_trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "consec_losses": 0,
            "risk_multiplier": 1.0, "cooldown_until": 0, "last_trade_ts": None,
        }

    def apply_trade_outcome(
        self,
        *,
        symbol: str,
        net_pnl: float,
        return_pct: float,
        ts: int,
        trade_id: str,
        strategy: str,
        features: dict | None = None,
        prob: float | None = None,
        regime: str | None = None,
        cooldown_bars_seconds: int = 0,
        consecutive_limit: int = 3,
        cooldown_hours: float = 12.0,
        min_samples_for_adaptation: int = 20,
    ) -> dict:
        """Update per-symbol statistics, memory and bounded risk multiplier."""
        st = self.stats_row(symbol)
        wins = int(st["wins"]) + int(net_pnl > 0)
        losses = int(st["losses"]) + int(net_pnl <= 0)
        n = int(st["n_trades"]) + 1
        gross_profit = float(st["gross_profit"]) + max(net_pnl, 0.0)
        gross_loss = float(st["gross_loss"]) + max(-net_pnl, 0.0)
        consec = 0 if net_pnl > 0 else int(st["consec_losses"]) + 1
        cooldown_until = int(st["cooldown_until"] or 0)

        if cooldown_bars_seconds:
            cooldown_until = max(cooldown_until, int(ts) + int(cooldown_bars_seconds))
        if consec >= consecutive_limit:
            cooldown_until = max(cooldown_until, int(ts) + int(cooldown_hours * 3600))
            consec = 0

        # Bounded, evidence-gated adaptation.  Nothing changes on a sample
        # smaller than min_samples_for_adaptation; the range is [0.5, 1.25].
        multiplier = float(st["risk_multiplier"] or 1.0)
        if n >= min_samples_for_adaptation:
            win_rate = wins / n
            # Laplace-smoothed edge over a 40% baseline.
            smoothed = (wins + 8) / (n + 20)
            target = max(0.5, min(1.25, 0.5 + (smoothed - 0.40) * 3.0))
            multiplier = round(0.7 * multiplier + 0.3 * target, 4)

        self.db.execute(
            "INSERT INTO uni_symbol_stats (symbol, n_trades, wins, losses, net_pnl,"
            " gross_profit, gross_loss, consec_losses, risk_multiplier, cooldown_until,"
            " last_trade_ts, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(symbol) DO UPDATE SET n_trades=excluded.n_trades,"
            " wins=excluded.wins, losses=excluded.losses, net_pnl=excluded.net_pnl,"
            " gross_profit=excluded.gross_profit, gross_loss=excluded.gross_loss,"
            " consec_losses=excluded.consec_losses, risk_multiplier=excluded.risk_multiplier,"
            " cooldown_until=excluded.cooldown_until, last_trade_ts=excluded.last_trade_ts,"
            " updated_at=excluded.updated_at",
            (symbol, n, wins, losses, float(st["net_pnl"]) + net_pnl, gross_profit,
             gross_loss, consec, multiplier, cooldown_until, int(ts), utcnow_iso()),
        )
        self.remember_trade(
            trade_id=trade_id, symbol=symbol, strategy=strategy, entry_ts=int(ts),
            features=features or {}, prob=prob, net_pnl=net_pnl, return_pct=return_pct,
            regime=regime,
        )
        return {"n_trades": n, "wins": wins, "losses": losses, "consec_losses": consec,
                "risk_multiplier": multiplier, "cooldown_until": cooldown_until}

    def remember_trade(self, *, trade_id: str, symbol: str, strategy: str, entry_ts: int,
                       features: dict, prob: float | None, net_pnl: float, return_pct: float,
                       regime: str | None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO uni_memory (trade_id, symbol, strategy, entry_ts,"
            " features_json, prob, outcome, return_pct, net_pnl, regime, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trade_id, symbol, strategy, int(entry_ts), json.dumps(features, default=str),
             prob, int(net_pnl > 0), return_pct, net_pnl, regime, utcnow_iso()),
        )

    def memory(self, symbol: str | None = None, limit: int = 5000) -> list[dict]:
        if symbol:
            rows = self.db.query(
                "SELECT * FROM uni_memory WHERE symbol=? ORDER BY entry_ts DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM uni_memory ORDER BY entry_ts DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- models
    def save_model(self, row: dict) -> None:
        keys = [
            "version", "symbol", "strategy", "timeframe", "algorithm", "trained_at",
            "train_start", "train_end", "test_start", "test_end", "n_train", "n_test",
            "auc", "brier", "base_rate", "threshold", "oos_trades", "oos_net_pnl",
            "oos_return_pct", "oos_improvement_pct", "status", "artifact_path", "metrics_json",
        ]
        vals = [row.get(k) for k in keys]
        self.db.execute(
            f"INSERT OR REPLACE INTO uni_models ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
            vals,
        )

    def models(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            rows = self.db.query(
                "SELECT * FROM uni_models WHERE symbol=? ORDER BY trained_at DESC", (symbol,)
            )
        else:
            rows = self.db.query("SELECT * FROM uni_models ORDER BY trained_at DESC")
        return [dict(r) for r in rows]

    def active_model(self, symbol: str) -> dict | None:
        row = self.db.query_one(
            "SELECT * FROM uni_models WHERE symbol=? AND status='active'"
            " ORDER BY trained_at DESC LIMIT 1",
            (symbol,),
        )
        return dict(row) if row else None

    # --------------------------------------------------------------- lessons
    def add_lesson(self, *, kind: str, symbol: str | None, title: str, finding: str,
                   evidence: dict | None = None, sample_size: int | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO uni_lessons (ts, kind, symbol, title, finding, evidence_json,"
            " sample_size, status, created_at) VALUES (?,?,?,?,?,?,?,'open',?)",
            (_now_ts(), kind, symbol, title, finding, json.dumps(evidence or {}, default=str),
             sample_size, utcnow_iso()),
        )
        return int(cur.lastrowid or 0)

    def lessons(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM uni_lessons ORDER BY ts DESC LIMIT ?", (limit,)
        )]

    # ------------------------------------------------------------------ runs
    def save_run(self, *, run_id: str, kind: str, label: str, start_ts: int | None,
                 end_ts: int | None, initial_equity: float, final_equity: float | None,
                 metrics: dict, params: dict, symbols: list[str], status: str = "complete",
                 notes: str = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO uni_runs (run_id, created_at, kind, label, start_ts,"
            " end_ts, initial_equity, final_equity, metrics_json, params_json, symbols_json,"
            " status, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, utcnow_iso(), kind, label, start_ts, end_ts, initial_equity,
             final_equity, json.dumps(metrics, default=str), json.dumps(params, default=str),
             json.dumps(symbols), status, notes),
        )

    def runs(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM uni_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )]

    def save_equity(self, run_id: str, rows: list[tuple]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO uni_equity (run_id, ts, equity, cash, invested,"
            " gross_exposure, drawdown, positions) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )

    def equity(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM uni_equity WHERE run_id=? ORDER BY ts", (run_id,)
        )]

    # --------------------------------------------------------------- regime
    def save_regime(self, *, ts: int, label: str, score: float, btc_trend: float | None,
                    breadth: float | None, vol_percentile: float | None,
                    payload: dict | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO uni_regime (ts, label, score, btc_trend, breadth,"
            " vol_percentile, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (int(ts), label, score, btc_trend, breadth, vol_percentile,
             json.dumps(payload or {}, default=str), utcnow_iso()),
        )

    def latest_regime(self) -> dict | None:
        row = self.db.query_one("SELECT * FROM uni_regime ORDER BY ts DESC LIMIT 1")
        return dict(row) if row else None

    # ----------------------------------------------------------------- trades
    def save_trade(self, trade) -> None:
        """Persist a closed trade into the shared journal (idempotent by id)."""
        import json as _json
        self.db.execute(
            "INSERT OR REPLACE INTO trades (id, mode, run_id, symbol, side, strategy,"
            " strategy_version, model_version, entry_order_id, exit_order_id, entry_ts,"
            " entry_bar_ts, entry_price, entry_signal, entry_reason, exit_ts, exit_price,"
            " exit_signal, exit_reason, qty, initial_stop, intended_stop, stop_distance_pct,"
            " gross_pnl, fees, slippage_cost, net_pnl, return_pct, r_multiple, mfe_pct,"
            " mae_pct, holding_hours, regime, indicators_entry_json, sentiment_entry_json,"
            " risk_events_json, anomalies_json, created_at) VALUES"
            " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade.id, trade.mode, trade.run_id, trade.symbol, trade.side, trade.strategy,
             trade.strategy_version, trade.model_version, trade.entry_order_id,
             trade.exit_order_id, trade.entry_ts, trade.entry_bar_ts, trade.entry_price,
             trade.entry_signal, trade.entry_reason, trade.exit_ts, trade.exit_price,
             trade.exit_signal, trade.exit_reason, trade.qty, trade.initial_stop,
             trade.intended_stop, trade.stop_distance_pct, trade.gross_pnl, trade.fees,
             trade.slippage_cost, trade.net_pnl, trade.return_pct, trade.r_multiple,
             trade.mfe_pct, trade.mae_pct, trade.holding_hours, trade.regime,
             _json.dumps(trade.indicators_entry, default=str), "{}", "[]", "[]",
             utcnow_iso()),
        )

    # --------------------------------------------------------------- alerts
    def log_alert(self, *, kind: str, text: str, symbol: str | None = None,
                  delivered: bool = False, error: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO uni_alerts (ts, kind, symbol, text, delivered, error)"
            " VALUES (?,?,?,?,?,?)",
            (utcnow_iso(), kind, symbol, text[:4000], int(delivered), error),
        )

    def alerts(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM uni_alerts ORDER BY id DESC LIMIT ?", (limit,)
        )]

    def log_recon(self, *, status: str, detail: str, payload: dict | None = None) -> None:
        self.db.execute(
            "INSERT INTO uni_recon (ts, status, detail, payload_json) VALUES (?,?,?,?)",
            (utcnow_iso(), status, detail, json.dumps(payload or {}, default=str)),
        )

    def latest_recon(self) -> dict | None:
        row = self.db.query_one("SELECT * FROM uni_recon ORDER BY id DESC LIMIT 1")
        return dict(row) if row else None

    # ------------------------------------------------------------ portfolio kv
    def save_portfolio(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO uni_portfolio (key, value, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), utcnow_iso()),
        )

    def get_portfolio(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value FROM uni_portfolio WHERE key=?", (key,))
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def close(self) -> None:
        self.db.close()
