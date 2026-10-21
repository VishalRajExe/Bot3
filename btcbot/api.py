"""FastAPI application + dashboard backend.

Read-only endpoints expose cached research data and stored results.  Control
endpoints (paper session management, backtests) are intentionally explicit and
idempotent where possible.  Nothing here can place a real exchange order.

Binding defaults to 127.0.0.1 and the app is meant for single-operator local
use; there is no authentication layer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .backtest.engine import run_backtest
from .backtest.metrics import compute_metrics
from .config import PROJECT_ROOT, Config, load_config
from .data.exchange import get_ticker
from .data.ingest import CandleIngestor
from .data.store import CandleStore
from .data.timeframes import tf_seconds, to_datetime
from .db import Database
from .indicators import atr, ema, rolling_std
from .journal import (journal_summary, load_trades, review_trades, row_to_trade,
                       trade_to_dict)
from .paper import PaperService, PaperTrader
from .regime import add_regime_columns
from .risk import RiskManager
from .sentiment.fng import latest_value as latest_fng
from .sentiment.news import latest_headlines
from .strategies import STRATEGIES, build_strategy, list_strategies

log = logging.getLogger(__name__)

WEB_DIR = PROJECT_ROOT / "web"


class BacktestRequest(BaseModel):
    strategy: str
    timeframe: str | None = None
    params: dict = Field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    initial_equity: float | None = None
    notes: str = ""


class PaperStartRequest(BaseModel):
    replay_bars: int = 0
    strategy: str | None = None
    timeframe: str | None = None


class _TickerCache:
    def __init__(self, cfg: Config, ttl: float = 10.0) -> None:
        self.cfg = cfg
        self.ttl = ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._ticker = None
        self._error: str | None = None
        self._price_history: list[tuple[int, float]] = []

    def get(self):
        with self._lock:
            now = time.monotonic()
            if self._ticker is not None and now - self._at < self.ttl:
                return self._ticker, self._error
            try:
                t = get_ticker(self.cfg.exchange)
                self._ticker = t
                self._error = None
                self._price_history.append((t.ts, t.price))
                self._price_history = self._price_history[-500:]
            except Exception as exc:  # noqa: BLE001 - report staleness, never invent a price
                self._error = str(exc)
                log.warning("ticker unavailable: %s", exc)
            self._at = now
            return self._ticker, self._error

    @property
    def age(self) -> float | None:
        if self._ticker is None:
            return None
        return time.time() - self._ticker.ts


class AppState:
    def __init__(self, cfg: Config, db: Database) -> None:
        self.cfg = cfg
        self.db = db
        self.ingestor = CandleIngestor(cfg, db)
        self.store = CandleStore(db)
        self.tickers = _TickerCache(cfg)
        self.service: PaperService | None = None
        self.trader: PaperTrader | None = None
        self.lock = threading.Lock()


def _session_summary(state: AppState, session_id: str | None = None) -> dict:
    db = state.db
    if session_id:
        row = db.query_one("SELECT * FROM paper_sessions WHERE id=?", (session_id,))
    else:
        row = db.query_one("SELECT * FROM paper_sessions ORDER BY started_at DESC LIMIT 1")
    if row is None:
        return {"session": None, "message": "no paper session has been created yet"}

    sid = row["id"]
    stored = db.get_state(f"paper_session:{sid}") or {}
    broker = stored.get("broker", {})
    ticker, err = state.tickers.get()
    price = ticker.price if ticker else stored.get("last_price")
    cash = float(broker.get("cash", row["initial_equity"]))
    pos = broker.get("position")
    qty = float(pos["qty"]) if pos else 0.0
    equity = cash + qty * price if price else cash
    trades = load_trades(db, mode="paper", run_id=sid)
    risk = RiskManager(state.cfg.risk, state.cfg.costs,
                       float(row["initial_equity"]), timeframe=row["timeframe"])
    risk.load_dict(stored.get("risk", {}))
    return {
        "session": {
            "id": sid, "mode": row["mode"], "status": row["status"],
            "started_at": row["started_at"], "stopped_at": row["stopped_at"],
            "strategy": row["strategy"], "strategy_version": row["strategy_version"],
            "timeframe": row["timeframe"], "initial_equity": row["initial_equity"],
            "last_processed_bar": to_datetime(row["last_processed_bar"]).isoformat()
            if row["last_processed_bar"] else None,
            "notes": row["notes"],
            "replay_seeded": bool(row["notes"] and "replay" in str(row["notes"]).lower()),
        },
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "position_qty": qty,
        "position": ({
            **pos,
            "entry_time": to_datetime(int(pos["entry_ts"])).isoformat(),
            "unrealized_pnl": round((price - pos["entry_price"]) * qty, 2) if price else None,
        } if pos else None),
        "mark_price": price,
        "mark_source": ticker.source if ticker else None,
        "price_age_seconds": state.tickers.age,
        "price_error": err,
        "realized_pnl": round(float(broker.get("realized_pnl", 0.0)), 2),
        "total_fees": round(float(broker.get("total_fees", 0.0)), 2),
        "trades": len(trades),
        "risk": risk.status(equity),
        "data_fresh": (to_datetime(int(row["last_processed_bar"])).isoformat()
                       if row["last_processed_bar"] else None),
        "journal": journal_summary(trades),
    }


def _attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema65"] = ema(out["close"], 65)
    out["std65"] = rolling_std(out["close"], 65)
    out["band_1u"] = out["ema65"] + out["std65"]
    out["band_1l"] = out["ema65"] - out["std65"]
    out["band_2u"] = out["ema65"] + 2 * out["std65"]
    out["band_2l"] = out["ema65"] - 2 * out["std65"]
    out["atr"] = atr(out["high"], out["low"], out["close"], 14)
    out["ema200"] = ema(out["close"], 200)
    return add_regime_columns(out)


def _series(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {"time": [int(x) for x in df["bar_ts"]]}
    for c in cols:
        if c in df:
            vals = df[c].tolist()
            out[c] = [None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                      else (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                      for v in vals]
    return out


def create_app(cfg: Config | None = None, db: Database | None = None,
               *, web_dir: Path | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or Database(cfg.db_path)
    state = AppState(cfg, db)

    app = FastAPI(title="BTC research & paper trading", version=__version__)
    app.state.ctx = state

    @app.on_event("startup")
    def _resume_paper_session() -> None:
        """Continue an interrupted paper session instead of silently forgetting it.

        The broker, risk state and last processed bar are reloaded from the
        database, so a restart cannot duplicate an already-filled order.
        """
        if not cfg.paper.resume_on_start:
            return
        row = db.query_one(
            "SELECT * FROM paper_sessions WHERE status IN ('running','paused')"
            " ORDER BY started_at DESC LIMIT 1")
        if row is None:
            return
        state.trader = PaperTrader(cfg, db, strategy=row["strategy"],
                                   timeframe=row["timeframe"],
                                   session_id=row["id"], mode=row["mode"])
        state.service = PaperService(state.trader)
        state.service.start()
        log.info("resumed paper session %s (%s)", row["id"], row["status"])

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health() -> dict:
        sessions = db.query_one("SELECT COUNT(*) AS n FROM paper_sessions")
        runs = db.query_one("SELECT COUNT(*) AS n FROM backtest_runs")
        return {
            "status": "ok",
            "version": __version__,
            "time": datetime.now(timezone.utc).isoformat(),
            "trading_mode": cfg.trading.mode,
            "live_orders_enabled": cfg.trading.enable_live_orders,
            "leverage": cfg.risk.leverage,
            "symbol": cfg.exchange.pair,
            "database": str(cfg.db_path),
            "paper_sessions": sessions["n"] if sessions else 0,
            "backtest_runs": runs["n"] if runs else 0,
            "paper_loop_running": bool(state.service and state.service.running),
        }

    # ------------------------------------------------------------------ market
    @app.get("/api/market")
    def market(timeframe: str = Query("1d"), limit: int = Query(400, ge=50, le=3000)) -> dict:
        if timeframe not in cfg.data.timeframes:
            raise HTTPException(400, f"timeframe {timeframe} not in configured data timeframes")
        df = state.store.load(cfg.exchange.pair, timeframe)
        summary = state.store.summary(cfg.exchange.pair, timeframe)
        if df.empty:
            raise HTTPException(503, "no cached candles; run `python -m btcbot.cli fetch`")
        full = _attach_indicators(df)
        view = full.tail(limit)
        ticker, err = state.tickers.get()
        latest = state.db.query_one(
            "SELECT action, reason, bar_ts, price, strategy, strategy_version, created_at"
            " FROM signals WHERE mode='paper' ORDER BY id DESC LIMIT 1"
        )
        quality = state.db.query_one(
            "SELECT status, missing_bars, bad_ohlc, checked_at FROM data_quality_checks"
            " WHERE timeframe=? ORDER BY id DESC LIMIT 1", (timeframe,)
        )
        if quality is None:
            from .data.store import validate_frame
            rep = validate_frame(df, timeframe, symbol=cfg.exchange.pair,
                                 max_gap_bars=cfg.data.max_gap_bars)
            quality = {"status": rep.status, "missing_bars": rep.missing_bars,
                       "bad_ohlc": rep.bad_ohlc, "checked_at": "computed on request"}
        last_bar_ts = int(df["bar_ts"].iloc[-1])
        bar_age = int(datetime.now(timezone.utc).timestamp()) - (last_bar_ts + tf_seconds(timeframe))
        return {
            "symbol": cfg.exchange.pair,
            "timeframe": timeframe,
            "candles": _series(view, ["open", "high", "low", "close", "volume", "ema65",
                                      "band_1u", "band_1l", "band_2u", "band_2l", "ema200", "atr"]),
            "steps": len(view),
            "spot": ({
                "price": ticker.price, "bid": ticker.bid, "ask": ticker.ask,
                "source": ticker.source, "ts": ticker.ts,
                "age_seconds": state.tickers.age,
                "is_live_quote": True,
            } if ticker else {"price": None, "error": err}),
            "last_closed_bar": to_datetime(last_bar_ts).isoformat(),
            "last_bar_age_seconds": bar_age,
            "stale": bar_age > cfg.risk.max_bars_stale * tf_seconds(timeframe),
            "data_summary": summary,
            "quality": dict(quality) if quality else None,
            "latest_signal": dict(latest) if latest else None,
            "regime": None if pd.isna(full["regime"].iloc[-1]) else str(full["regime"].iloc[-1]),
        }

    # ------------------------------------------------------------- strategies
    @app.get("/api/strategies")
    def strategies() -> dict:
        return {"strategies": list_strategies()}

    # -------------------------------------------------------------- backtests
    @app.get("/api/runs")
    def runs(limit: int = Query(50, ge=1, le=500)) -> dict:
        rows = db.query(
            "SELECT id, created_at, strategy, strategy_version, timeframe, start_ts, end_ts,"
            " metrics_json, benchmark_json, params_json, status FROM backtest_runs"
            " ORDER BY created_at DESC LIMIT ?", (limit,),
        )
        out = []
        for r in rows:
            m = json.loads(r["metrics_json"] or "{}")
            b = json.loads(r["benchmark_json"] or "{}")
            out.append({
                "id": r["id"], "created_at": r["created_at"], "strategy": r["strategy"],
                "strategy_version": r["strategy_version"], "timeframe": r["timeframe"],
                "start": to_datetime(r["start_ts"]).isoformat(),
                "end": to_datetime(r["end_ts"]).isoformat(),
                "status": r["status"],
                "net_return_pct": m.get("net_return_pct"),
                "cagr_pct": m.get("cagr_pct"),
                "max_drawdown_pct": m.get("max_drawdown_pct"),
                "sharpe": m.get("sharpe"),
                "trades": m.get("trades"),
                "win_rate_pct": m.get("win_rate_pct"),
                "profit_factor": m.get("profit_factor"),
                "buy_hold_return_pct": b.get("buy_hold", {}).get("net_return_pct"),
                "params": json.loads(r["params_json"] or "{}"),
            })
        return {"runs": out}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        row = db.query_one("SELECT * FROM backtest_runs WHERE id=?", (run_id,))
        if row is None:
            raise HTTPException(404, "run not found")
        equity = db.query(
            "SELECT ts, equity, cash, position_qty, price, drawdown FROM equity_curve"
            " WHERE run_id=? ORDER BY ts", (run_id,)
        )
        trades = load_trades(db, run_id=run_id)
        markers = [{
            "entry_ts": t.entry_ts, "entry_price": t.entry_price,
            "exit_ts": t.exit_ts, "exit_price": t.exit_price,
            "net_pnl": t.net_pnl, "return_pct": t.return_pct, "id": t.id,
            "stop": t.intended_stop, "regime": t.regime, "exit_reason": t.exit_reason,
        } for t in trades]
        return {
            "run": {k: row[k] for k in row.keys()},
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "benchmark": json.loads(row["benchmark_json"] or "{}"),
            "params": json.loads(row["params_json"] or "{}"),
            "equity": [dict(r) for r in equity],
            "trades": [trade_to_dict(t) for t in trades],
            "markers": markers,
        }

    @app.post("/api/backtest")
    def backtest(req: BacktestRequest) -> dict:
        if req.strategy not in STRATEGIES:
            raise HTTPException(400, f"unknown strategy {req.strategy}")
        tf = req.timeframe or STRATEGIES[req.strategy]().default_timeframe
        df = state.store.load(cfg.exchange.pair, tf)
        if df.empty:
            raise HTTPException(503, "no candles for timeframe; fetch data first")
        strategy = build_strategy(req.strategy, req.params, tf)
        from .data.timeframes import parse_iso_date

        result = run_backtest(
            cfg, db, strategy, df, timeframe=tf,
            initial_equity=req.initial_equity, persist=True,
            start_ts=parse_iso_date(req.start) if req.start else None,
            end_ts=parse_iso_date(req.end) if req.end else None,
            notes=req.notes or "requested from dashboard",
        )
        return {"run_id": result.run_id, "metrics": result.metrics,
                "benchmark": result.benchmark, "trades": len(result.trades)}

    # ------------------------------------------------------------ paper trade
    @app.get("/api/paper")
    def paper_status(session_id: str | None = None) -> dict:
        summary = _session_summary(state, session_id)
        loop_running = bool(state.service and state.service.running)
        summary["loop_running"] = loop_running
        if state.trader is not None and (
            session_id is None or state.trader.session_id == session_id
        ):
            live = state.trader.status()
            summary["live"] = {
                "last_error": live.get("last_error"),
                "last_signal": live.get("last_signal"),
                "last_price": live.get("last_price"),
                "price_source": live.get("price_source"),
            }
        return summary

    @app.get("/api/paper/sessions")
    def paper_sessions() -> dict:
        rows = db.query(
            "SELECT id, started_at, stopped_at, mode, strategy, timeframe, status,"
            " initial_equity, last_processed_bar FROM paper_sessions ORDER BY started_at DESC"
        )
        return {"sessions": [dict(r) for r in rows]}

    @app.post("/api/paper/start")
    def paper_start(req: PaperStartRequest) -> dict:
        with state.lock:
            if state.trader is None:
                state.trader = PaperTrader(cfg, db, strategy=req.strategy, timeframe=req.timeframe,
                                           mode="live")
                state.service = PaperService(state.trader)
            trader = state.trader
            if trader._session_status() == "stopped":
                raise HTTPException(409, "this session is stopped; start a new server session")
            if trader._session_status() == "paused":
                trader.set_status("running")
            replay = None
            if req.replay_bars and req.replay_bars > 0:
                replay = trader.run_replay(req.replay_bars,
                                           notes=f"seeded by replaying {req.replay_bars} closed bars")
            if not state.service.running:
                state.service.start()
            return {"session_id": trader.session_id, "mode": trader.mode,
                    "loop_running": state.service.running, "replay": replay,
                    "notice": ("Paper trading only. Live executions are simulated at the current "
                               "public quote; no exchange order is ever sent.")}

    @app.post("/api/paper/pause")
    def paper_pause() -> dict:
        if state.trader is None:
            raise HTTPException(404, "no active paper session")
        state.trader.set_status("paused")
        return {"status": "paused", "session_id": state.trader.session_id}

    @app.post("/api/paper/resume")
    def paper_resume() -> dict:
        if state.trader is None:
            raise HTTPException(404, "no active paper session")
        state.trader.set_status("running")
        return {"status": "running", "session_id": state.trader.session_id}

    @app.post("/api/paper/stop")
    def paper_stop() -> dict:
        if state.trader is None:
            raise HTTPException(404, "no active paper session")
        if state.service is not None:
            state.service.stop()
        state.trader.set_status("stopped")
        return {"status": "stopped", "session_id": state.trader.session_id}

    @app.post("/api/paper/reset-drawdown")
    def paper_reset_drawdown() -> dict:
        if state.trader is None:
            raise HTTPException(404, "no active paper session")
        state.trader.reset_drawdown_halt()
        return {"status": "ok", "risk": state.trader.status()["risk"]}

    @app.post("/api/paper/tick")
    def paper_tick() -> dict:
        if state.trader is None:
            raise HTTPException(404, "no active paper session")
        return state.trader.poll_once()

    # ---------------------------------------------------------------- journal
    @app.get("/api/signals")
    def signals(mode: str | None = None, limit: int = Query(50, ge=1, le=500)) -> dict:
        rows = db.query(
            "SELECT id, run_id, mode, strategy, strategy_version, bar_ts, action, reason, price,"
            " indicators_json, created_at FROM signals WHERE (? IS NULL OR mode = ?)"
            " ORDER BY id DESC LIMIT ?", (mode, mode, limit)
        )
        return {"signals": [{**dict(r),
                             "indicators": json.loads(r["indicators_json"] or "{}")}
                            for r in rows]}

    @app.get("/api/trades")
    def trades(mode: str | None = None, limit: int = Query(200, ge=1, le=2000)) -> dict:
        rows = load_trades(db, mode=mode, limit=limit)
        return {"trades": [trade_to_dict(t) for t in rows], "summary": journal_summary(rows)}

    @app.get("/api/trades/{trade_id}")
    def trade_detail(trade_id: str) -> dict:
        row = db.query_one("SELECT * FROM trades WHERE id=?", (trade_id,))
        if row is None:
            raise HTTPException(404, "trade not found")
        detail = trade_to_dict(row_to_trade(row))
        fills = db.query(
            "SELECT f.*, o.side, o.reason FROM fills f JOIN orders o ON o.id = f.order_id"
            " WHERE o.id IN (?, ?) ORDER BY f.ts",
            (row["entry_order_id"], row["exit_order_id"]),
        )
        risk = db.query("SELECT * FROM risk_events WHERE run_id=? ORDER BY ts LIMIT 50",
                        (row["run_id"],))
        return {"trade": detail, "fills": [dict(f) for f in fills],
                "risk_events": [dict(r) for r in risk]}

    @app.get("/api/journal")
    def journal(mode: str = "backtest", limit: int = Query(200, ge=1, le=2000),
                review: bool = False) -> dict:
        rows = load_trades(db, mode=mode, limit=limit)
        findings = review_trades(db, rows, persist=False) if review else []
        risk_events = db.query(
            "SELECT * FROM risk_events WHERE mode=? ORDER BY ts DESC LIMIT 50", (mode,)
        )
        reviews = db.query(
            "SELECT * FROM trade_reviews ORDER BY id DESC LIMIT 50"
        )
        return {
            "mode": mode,
            "trades": [trade_to_dict(t) for t in rows],
            "summary": journal_summary(rows),
            "findings": findings,
            "risk_events": [dict(r) for r in risk_events],
            "stored_reviews": [dict(r) for r in reviews],
        }

    @app.get("/api/learning")
    def learning() -> dict:
        experiments = db.query("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 100")
        models = db.query("SELECT * FROM models ORDER BY created_at DESC LIMIT 50")
        errors = db.query("SELECT * FROM errors ORDER BY id DESC LIMIT 50")
        reviews = db.query("SELECT * FROM trade_reviews ORDER BY id DESC LIMIT 50")
        return {
            "experiments": [{**dict(e), "metrics": json.loads(e["metrics_json"] or "{}")}
                            for e in experiments],
            "models": [{**dict(m),
                        "metrics": json.loads(m["metrics_json"] or "{}"),
                        "features": json.loads(m["feature_names_json"] or "[]")}
                       for m in models],
            "errors": [dict(e) for e in errors],
            "reviews": [dict(r) for r in reviews],
            "promotion_policy": {
                "min_oof_auc": 0.55,
                "min_predictions": 400,
                "requires_monotonic_calibration": True,
                "requires_beating_baseline_after_costs": True,
                "note": "Models are never auto-promoted; the default paper strategy does not use ML.",
            },
        }

    # ------------------------------------------------------------ performance
    @app.get("/api/performance")
    def performance(mode: str = "paper", run_id: str | None = None) -> dict:
        rows = db.query("SELECT ts, equity, cash, position_qty, price, drawdown FROM equity_curve"
                        " WHERE run_id=? ORDER BY ts", (run_id,)) if run_id else []
        if not rows:
            if run_id:
                raise HTTPException(404, "no equity data for this run")
            return {"metrics": None, "message": "select a run_id or start a paper session"}
        eq = pd.DataFrame([dict(r) for r in rows])
        trades = load_trades(db, run_id=run_id)
        bars_in_market = int((eq["position_qty"] > 0).sum())
        metrics = compute_metrics(equity_df=eq, trades=trades,
                                  initial_equity=cfg.paper.initial_equity if mode == "paper"
                                  else cfg.backtest.initial_equity,
                                  timeframe=cfg.paper.timeframe, bars_in_market=bars_in_market,
                                  total_bars=len(eq))
        # An open position has already paid entry costs that no closed trade
        # reports; surface them instead of showing a misleading zero.
        stored = db.get_state(f"paper_session:{run_id}") if mode == "paper" else None
        pos = (stored or {}).get("broker", {}).get("position")
        if pos:
            metrics["open_position_fees"] = round(float(pos.get("entry_fees") or 0.0), 2)
            metrics["open_position_slippage"] = round(
                float(pos.get("entry_slippage_cost") or 0.0), 2)
        return {"metrics": metrics, "equity": [dict(r) for r in rows]}

    @app.get("/api/export")
    def export_bundle() -> dict:
        runs = [dict(r) for r in db.query("SELECT * FROM backtest_runs ORDER BY created_at")]
        for r in runs:
            r["metrics"] = json.loads(r.pop("metrics_json") or "{}")
            r["benchmark"] = json.loads(r.pop("benchmark_json") or "{}")
            r["params"] = json.loads(r.pop("params_json") or "{}")
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system": {"version": __version__, "symbol": cfg.exchange.pair,
                       "mode": cfg.trading.mode, "live_orders": cfg.trading.enable_live_orders},
            "runs": runs,
            "experiments": [dict(r) for r in db.query("SELECT * FROM experiments")],
            "models": [dict(r) for r in db.query("SELECT * FROM models")],
            "paper_sessions": [dict(r) for r in db.query("SELECT * FROM paper_sessions")],
            "trades": [trade_to_dict(t) for t in load_trades(db, limit=5000)],
        }

    # --------------------------------------------------------------- sentiment
    @app.get("/api/sentiment")
    def sentiment() -> dict:
        fng = latest_fng(db)
        stories = latest_headlines(db, 12)
        rows = db.query(
            "SELECT kind, COUNT(*) AS n, MAX(ts) AS last FROM sentiment_observations GROUP BY kind"
        )
        return {
            "fear_greed": fng,
            "headlines": stories,
            "coverage": {r["kind"]: {"count": r["n"],
                                     "last": to_datetime(r["last"]).isoformat()}
                         for r in rows},
            "note": ("Sentiment is experimental context. It is disabled for trading by default "
                     "and has not been shown to improve out-of-sample results."),
        }

    @app.post("/api/sentiment/refresh")
    def sentiment_refresh() -> dict:
        from .sentiment.fng import store_fng
        from .sentiment.news import collect

        out: dict = {}
        try:
            out["fear_greed_rows"] = store_fng(db)
        except Exception as exc:  # noqa: BLE001
            out["fear_greed_error"] = str(exc)
        out.update(collect(db))
        return out

    # ------------------------------------------------------------------ config
    @app.get("/api/config")
    def config_view() -> dict:
        return {
            "symbol": cfg.exchange.pair,
            "exchange": cfg.exchange.name,
            "trading": cfg.trading.model_dump(),
            "costs": cfg.costs.model_dump(),
            "risk": cfg.risk.model_dump(),
            "paper": cfg.paper.model_dump(),
            "timeframes": cfg.data.timeframes,
            "disclaimer": ("Paper trading only. Results are simulations, do not include "
                           "real liquidity/outages, and are not financial advice."),
        }

    # ------------------------------------------------------------------- static
    mount_dir = web_dir or WEB_DIR
    if mount_dir.exists():
        @app.middleware("http")
        async def _revalidate_static(request: Request, call_next):  # noqa: ANN001
            """Let the browser revalidate the un-hashed JS/CSS instead of caching it."""
            response = await call_next(request)
            path = request.url.path
            if path == "/" or path.endswith((".js", ".css", ".html")):
                response.headers["Cache-Control"] = "no-cache"
            return response

        app.mount("/", StaticFiles(directory=str(mount_dir), html=True), name="web")
    else:  # pragma: no cover
        @app.get("/")
        def no_web() -> JSONResponse:
            return JSONResponse({"error": "web directory missing"}, status_code=500)

    return app


app = create_app()
