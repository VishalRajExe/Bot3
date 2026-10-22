"""FastAPI service for the unified bot dashboard."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from btcbot.db import Database
from btcbot.data.timeframes import tf_seconds

from .config import Settings, load_credentials
from .datafeed import build_market, coverage, prepare_all, prepare_bars
from .live import LiveEngine
from .learning import learning_state
from .market import AlpacaClient, data_quality
from .models import model_report
from .regime import regime_at
from .store import UnibotStore


def _json_load(raw) -> dict:
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}


def create_app(settings: Settings | None = None, *, db_path: str | None = None,
               web_dir: str | None = None) -> FastAPI:
    settings = settings or Settings.load("config/unibot.json")
    store = UnibotStore(db_path or settings.paths.db)
    app = FastAPI(title="Unified crypto bot", version="1.0.0",
                  docs_url="/api/docs", openapi_url="/api/openapi.json")

    def engine() -> LiveEngine:
        return LiveEngine(settings, store)

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health() -> dict:
        last = store.latest_recon()
        snap = store.get_portfolio("equity_snapshot", {}) or {}
        age = int(time.time()) - int(snap.get("ts") or 0)
        return {
            "status": "ok" if snap else "starting",
            "mode": "paper",
            "live_orders_enabled": False,
            "broker": "alpaca-paper",
            "service_time": datetime.now(timezone.utc).isoformat(),
            "last_cycle_seconds_ago": age,
            "loop_alive": age < 600,
            "reconcile": (last or {}).get("status"),
            "symbols": len(settings.universe),
            "capital": settings.capital,
            "timeframe": settings.timeframe,
        }

    # ---------------------------------------------------------------- overview
    @app.get("/api/overview")
    def overview() -> dict:  # noqa: C901
        snap = store.get_portfolio("equity_snapshot", {}) or {}
        breakers = store.get_portfolio("breakers", {}) or {}
        positions = store.positions()
        stats = store.symbol_stats()
        closed = store.db.query(
            "SELECT COUNT(*) n, COALESCE(SUM(net_pnl),0) pnl, COALESCE(SUM(fees),0) fees"
            " FROM trades WHERE mode='live'"
        )
        row = dict(closed[0]) if closed else {"n": 0, "pnl": 0, "fees": 0}
        regime = store.latest_regime() or {}
        wins = store.db.scalar(
            "SELECT COUNT(*) FROM trades WHERE mode='live' AND net_pnl > 0") or 0
        recon = store.latest_recon() or {}
        marks = {}
        live_prices = {}
        try:
            creds = load_credentials(".env")
            client = AlpacaClient(creds.alpaca_key, creds.alpaca_secret,
                                  paper_base=creds.alpaca_base_url,
                                  data_base=settings.alpaca.data_base_url)
            acct = client.account()
            live_prices = client.latest_prices(settings.symbols)
            marks = {s: d for s, d in live_prices.items() if s in [p["symbol"] for p in positions]}
        except Exception:
            acct = {}
        unrealized = 0.0
        pos_view = []
        for p in positions:
            px = float((marks.get(p["symbol"]) or {}).get("price") or p["entry_price"])
            value = px * float(p["qty"])
            pnl = (px - float(p["entry_price"])) * float(p["qty"])
            unrealized += pnl
            meta = _json_load(p.get("meta_json"))
            pos_view.append({
                "symbol": p["symbol"], "qty": p["qty"], "entry_price": p["entry_price"],
                "price": round(px, 8), "notional": round(value, 4),
                "unrealized": round(pnl, 4),
                "return_pct": round((px / float(p["entry_price"]) - 1) * 100, 3),
                "stop": p["stop_price"], "strategy": p["strategy"],
                "prob": p["prob"], "risk_usd": p["risk_usd"],
                "trail_atr": meta.get("trail_atr"),
            })
        equity = float(snap.get("equity") or acct.get("equity") or settings.capital)
        cash = float(snap.get("cash") or acct.get("cash") or settings.capital)
        return {
            "equity": round(equity, 4),
            "cash": round(cash, 4),
            "invested": round(sum(p["notional"] for p in pos_view), 4),
            "unrealized_pnl": round(unrealized, 4),
            "realized_pnl": round(float(row["pnl"]), 4),
            "fees_paid": round(float(row["fees"]), 4),
            "trades": int(row["n"]),
            "win_rate_pct": round(wins / row["n"] * 100, 2) if row["n"] else None,
            "positions": pos_view,
            "regime": {
                "label": regime.get("label"), "score": regime.get("score"),
                "breadth": regime.get("breadth"), "btc_trend": regime.get("btc_trend"),
                "vol_percentile": regime.get("vol_percentile"),
                "known_at": datetime.fromtimestamp(regime["ts"], tz=timezone.utc).isoformat()
                if regime.get("ts") else None,
            },
            "breakers": breakers,
            "reconcile": {"status": recon.get("status"), "detail": recon.get("detail"),
                          "at": recon.get("ts")},
            "last_cycle": snap.get("ts"),
            "strategy_stats": [{"symbol": k, **v} for k, v in stats.items()],
            "account_status": acct.get("status"),
            "paper": True,
            "n_trades": int(row.get("n") or 0),
            "total_fees": round(float(row.get("fees") or 0), 4),
            "realized_pnl": round(float(row.get("pnl") or 0), 4),
            "win_rate_pct": round(wins / row["n"] * 100, 2) if row.get("n") else None,
            "expectancy": round(float(row.get("pnl") or 0) / row["n"], 4) if row.get("n") else None,
            "max_drawdown_pct": round(float(breakers.get("max_dd") or 0), 3),
            "capital": settings.capital,
            "risk_config": settings.risk.model_dump(),
            "latest_prices": {s: d.get("price") for s, d in live_prices.items() if d.get("price")},
            "regime_ts": regime.get("ts"),
            # Equity history for sparkline (last 90 snapshots)
            "equity_history": [
                {"ts": r["ts"], "equity": round(r["equity"], 4)}
                for r in store.db.query(
                    "SELECT ts, equity FROM uni_equity WHERE run_id='live'"
                    " ORDER BY ts DESC LIMIT 90"
                )
            ][::-1],
        }

    # ----------------------------------------------------------------- symbols
    @app.get("/api/symbols")
    def symbols() -> dict:
        latest = store.latest_decision_per_symbol()
        stats = store.symbol_stats()
        models = {m["symbol"]: m for m in store.models()}
        active = {s: store.active_model(s) for s in settings.symbols}
        market = build_market(store, settings)
        out = []
        for cfg in settings.universe:
            df = prepare_bars(store, settings, cfg.symbol, market)
            last = df.iloc[-1] if not df.empty else None
            prev = df.iloc[-7] if len(df) > 7 else (df.iloc[0] if not df.empty else None)
            dec = latest.get(cfg.symbol, {})
            regime = regime_at(market, int(last["bar_ts"]) + tf_seconds(settings.timeframe)) \
                if last is not None else None
            model = active.get(cfg.symbol)
            out.append({
                "symbol": cfg.symbol,
                "cluster": cfg.cluster,
                "strategy": cfg.strategy,
                "trade_enabled": cfg.trade_enabled,
                "note": cfg.note,
                "price": round(float(last["close"]), 8) if last is not None else None,
                "change_24h_pct": round((float(last["close"]) / float(prev["close"]) - 1) * 100, 3)
                if last is not None and prev is not None and float(prev["close"]) else None,
                "atr_pct": round(float(last["atr_pct"]), 3) if last is not None else None,
                "rsi14": round(float(last["rsi14"]), 2) if last is not None else None,
                "adx14": round(float(last["adx14"]), 2) if last is not None else None,
                "trend": ("up" if last is not None and float(last["close"]) > float(last["ema200"])
                          else "down") if last is not None else None,
                "decision": dec.get("action"),
                "reason": dec.get("reason"),
                "decided_at": dec.get("ts"),
                "prob": dec.get("prob"),
                "model": ({**model, "status": model.get("status")} if model else
                          ({"status": models[cfg.symbol]["status"]} if cfg.symbol in models
                           else {"status": "untrained"})),
                "stats": stats.get(cfg.symbol, {}),
                "regime": regime.as_dict() if regime else None,
            })
        return {"symbols": out, "timeframe": settings.timeframe,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    # ----------------------------------------------------------------- candles
    @app.get("/api/candles/{symbol:path}")
    def candles(
        symbol: str,
        limit: int = Query(400, ge=1, le=5000),
        timeframe: str = Query("4h"),
    ) -> dict:
        try:
            settings.symbol(symbol)
        except KeyError:
            raise HTTPException(404, f"unknown symbol {symbol}")
        market = build_market(store, settings)
        df = prepare_bars(store, settings, symbol, market, timeframe=timeframe)
        if df.empty:
            return {"symbol": symbol, "candles": [], "signals": [], "last_ts": None,
                    "snapshot": {}}
        df = df.tail(limit)
        # resolve optional columns safely
        def _safe(row, col):
            v = getattr(row, col, None)
            if v is None:
                return None
            try:
                return None if np.isnan(float(v)) else round(float(v), 8)
            except (TypeError, ValueError):
                return None

        rows = []
        for r in df.itertuples(index=False):
            o = round(float(r.open), 8)
            h = round(float(r.high), 8)
            l = round(float(r.low), 8)
            c = round(float(r.close), 8)
            v = round(float(r.volume), 4)
            rows.append({
                "ts": int(r.bar_ts),
                "open":   o,
                "high":   h,
                "low":    l,
                "close":  c,
                "volume": v,
                "o":      o,
                "h":      h,
                "l":      l,
                "c":      c,
                "v":      v,
                "ema50":  _safe(r, "ema50"),
                "ema200": _safe(r, "ema200"),
                "rsi14":  _safe(r, "rsi14"),
                "atr_pct":_safe(r, "atr_pct"),
                "bb_upper":_safe(r, "bb_upper"),
                "bb_lower":_safe(r, "bb_lower"),
            })
        signals = [
            {"ts": r["ts"], "action": r["action"], "price": r["price"], "reason": r["reason"]}
            for r in store.db.query(
                "SELECT ts, action, price, reason FROM uni_decisions"
                " WHERE symbol=? AND action IN ('enter_long','exit_long','buy_filled','exit_filled','buy_signal')"
                " ORDER BY ts DESC LIMIT 100",
                (symbol,),
            )
        ]
        last = df.iloc[-1]
        snapshot = {
            "price":   round(float(last["close"]), 8),
            "rsi14":   _safe(last, "rsi14"),
            "atr_pct": _safe(last, "atr_pct"),
            "ema50":   _safe(last, "ema50"),
            "ema200":  _safe(last, "ema200"),
            "n_bars":  len(df),
            "trend":   "up" if _safe(last,"ema50") and _safe(last,"ema200") and
                       _safe(last,"ema50") > _safe(last,"ema200") else "down",
        }
        return {
            "symbol": symbol, "candles": rows, "signals": signals,
            "last_ts": int(last["bar_ts"]), "snapshot": snapshot,
            "quality": data_quality(df, settings.timeframe),
        }

    # ------------------------------------------------------------------ trades
    @app.get("/api/trades")
    def trades(limit: int = Query(200, ge=1, le=2000), mode: str = "live") -> dict:
        rows = store.db.query(
            "SELECT * FROM trades WHERE mode=? ORDER BY entry_ts DESC LIMIT ?", (mode, limit)
        )
        out = []
        for r in rows:
            d = dict(r)
            d["indicators_entry"] = _json_load(d.pop("indicators_entry_json", "{}"))
            out.append(d)
        agg = store.db.query(
            "SELECT COUNT(*) n, COALESCE(SUM(net_pnl),0) pnl, COALESCE(SUM(fees),0) fees,"
            " COALESCE(AVG(return_pct),0) avg_ret FROM trades WHERE mode=?", (mode,)
        )
        return {"trades": out, "summary": dict(agg[0]) if agg else {}, "mode": mode}

    # ------------------------------------------------------------------ equity
    @app.get("/api/equity")
    def equity(run_id: str | None = None, limit: int = 2000) -> dict:
        rid = run_id or "live"
        rows = store.db.query(
            "SELECT ts, equity, cash, invested, drawdown, positions FROM uni_equity"
            " WHERE run_id=? ORDER BY ts DESC LIMIT ?", (rid, limit)
        )
        pts = [{"t": r["ts"], "equity": round(r["equity"], 4), "cash": round(r["cash"], 4),
                "invested": round(r["invested"], 4), "drawdown": round(r["drawdown"], 3),
                "positions": r["positions"]} for r in reversed(rows)]
        return {"run_id": rid, "points": pts}

    # ------------------------------------------------------------------ models
    @app.get("/api/learning")
    def learning() -> dict:
        return learning_state(store, settings)

    @app.get("/api/decisions")
    def decisions(limit: int = Query(120, ge=1, le=1000), symbol: str | None = None) -> dict:
        rows = store.decisions(limit=limit, symbol=symbol)
        for r in rows:
            r["features"] = _json_load(r.pop("features_json", "{}"))
        return {"decisions": rows}

    # --------------------------------------------------------------- backtests / runs (alias)
    @app.get("/api/backtests")
    @app.get("/api/runs")
    def backtests(limit: int = 60) -> dict:
        rows = store.runs(limit)
        for r in rows:
            r["metrics"] = _json_load(r.pop("metrics_json", "{}"))
            r["symbols"] = _json_load(r.pop("symbols_json", "[]"))
        return {"runs": rows}

    @app.get("/api/backtests/{run_id}")
    @app.get("/api/runs/{run_id}")
    def backtest_detail(run_id: str) -> dict:
        row = store.db.query_one("SELECT * FROM uni_runs WHERE run_id=?", (run_id,))
        if row is None:
            raise HTTPException(404, "unknown run")
        d = dict(row)
        d["metrics"] = _json_load(d.pop("metrics_json", "{}"))
        d["params"] = _json_load(d.pop("params_json", "{}"))
        d["symbols"] = _json_load(d.pop("symbols_json", "[]"))
        trades = [dict(r) for r in store.db.query(
            "SELECT symbol, strategy, entry_ts, exit_ts, entry_price, exit_price, qty,"
            " net_pnl, return_pct, exit_reason, holding_hours, mfe_pct, mae_pct"
            " FROM trades WHERE run_id=? ORDER BY entry_ts", (run_id,))]
        equity_curve = [
            {"ts": r["ts"], "equity": round(r["equity"],4), "drawdown": round(r["drawdown"],3)}
            for r in store.db.query(
                "SELECT ts, equity, drawdown FROM uni_equity WHERE run_id=? ORDER BY ts",
                (run_id,))
        ]
        metrics = _json_load(d.get("metrics_json") or "{}") if "metrics_json" in d else d.get("metrics", {})
        return {
            "run_id": run_id, "start_ts": d.get("start_ts"), "end_ts": d.get("end_ts"),
            "metrics": metrics, "trades": trades, "equity_curve": equity_curve,
        }

    # ------------------------------------------- backtest trigger (POST)
    class BacktestRequest(BaseModel):
        start_date: Optional[str] = None
        end_date:   Optional[str] = None
        initial_equity: float = 100.0

    @app.post("/api/backtest")
    def trigger_backtest(req: BacktestRequest) -> dict:
        from .backtest import run_portfolio_backtest
        from .datafeed import prepare_all
        from btcbot.data.timeframes import to_epoch
        import datetime as _dt
        run_id = f"ui_{_dt.datetime.now(_dt.timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        start_ts = to_epoch(req.start_date) if req.start_date else None
        end_ts   = to_epoch(req.end_date)   if req.end_date   else None
        frames = prepare_all(store, settings)
        result = run_portfolio_backtest(
            settings=settings, frames=frames, store=store,
            run_id=run_id, initial_equity=req.initial_equity,
            start_ts=start_ts, end_ts=end_ts,
            label=f"UI backtest {req.start_date or 'all'} → {req.end_date or 'now'}",
        )
        m = result.metrics
        return {
            "run_id": run_id,
            "n_trades": m.get("n_trades", 0),
            "net_return_pct": m.get("net_return_pct"),
            "max_drawdown_pct": m.get("max_drawdown_pct"),
        }

    # ------------------------------------------------------------ risk / data
    @app.get("/api/risk")
    def risk_info() -> dict:
        risk_data = settings.risk.model_dump()
        return {
            "limits": risk_data,
            "risk": risk_data,
            "costs": settings.costs.model_dump(),
            "breakers": store.get_portfolio("breakers", {}),
            "reconcile": store.latest_recon() or {},
            "alerts": store.alerts(50),
        }

    @app.get("/api/logs")
    def system_logs(limit: int = 150) -> dict:
        log_file = Path("logs/unibot.log")
        lines = []
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                    lines = [l.strip() for l in all_lines[-limit:]]
            except Exception:
                pass
        return {"logs": lines, "count": len(lines)}

    @app.get("/api/notifications")
    def notifications_feed(limit: int = 60) -> dict:
        alerts = store.alerts(limit)
        return {"notifications": alerts, "count": len(alerts)}

    @app.get("/api/coverage")
    @app.get("/api/data")
    def data_coverage() -> dict:
        return {"coverage": coverage(store, settings)}

    @app.get("/api/models")
    def models_info() -> dict:
        rpt = model_report(store, settings)
        ls  = learning_state(store, settings)
        lessons = []
        if store.db.table_exists("uni_lessons"):
            try:
                lessons = [
                    {"symbol": r["symbol"], "lesson": r["lesson"], "ts": r["created_at"]}
                    for r in store.db.query(
                        "SELECT symbol, lesson, created_at FROM uni_lessons ORDER BY id DESC LIMIT 40"
                    )
                ]
            except Exception:
                pass
        strat_map = {cfg.symbol: cfg.strategy for cfg in settings.universe}
        cohorts = []
        if store.db.table_exists("uni_symbol_stats"):
            try:
                for r in store.db.query(
                    "SELECT symbol, n_trades, wins, losses, net_pnl,"
                    " risk_multiplier, CAST(wins*100.0/MAX(n_trades,1) AS REAL) win_rate_pct,"
                    " CAST(gross_profit/MAX(gross_loss,0.0001) AS REAL) profit_factor"
                    " FROM uni_symbol_stats ORDER BY n_trades DESC"
                ):
                    d = dict(r)
                    d["strategy"] = strat_map.get(d["symbol"], "trend_breakout")
                    cohorts.append(d)
            except Exception:
                pass
        exps = []
        if store.db.table_exists("experiments"):
            try:
                exps = [
                    dict(r) for r in store.db.query(
                        "SELECT symbol, kind, status, hypothesis, base_return_pct, variant_return_pct, created_at"
                        " FROM experiments ORDER BY id DESC LIMIT 30"
                    )
                ]
            except Exception:
                pass
        return {
            "models": rpt, "learning_state": ls,
            "lessons": lessons, "cohorts": cohorts, "experiments": exps,
        }

    @app.get("/api/regime/history")
    def regime_history(limit: int = 400) -> dict:
        rows = store.db.query(
            "SELECT ts, label, score, btc_trend, breadth, vol_percentile FROM uni_regime"
            " ORDER BY ts DESC LIMIT ?", (limit,)
        )
        return {"points": [dict(r) for r in reversed(rows)]}

    # --------------------------------------------------------------- actions
    @app.post("/api/notify/test")
    def notify_test() -> dict:
        creds = load_credentials(".env")
        eng = engine()
        eng.notifier.send("🧪 Test message from the UniBot dashboard.", kind="test")
        return {"configured": creds.telegram_configured,
                "chat_id_present": bool(creds.telegram_chat_id)}

    @app.post("/api/cycle")
    def run_cycle() -> JSONResponse:
        """Manual trading cycle – only callable from the local dashboard."""
        eng = engine()
        try:
            result = eng.run_cycle()
        except Exception as exc:
            raise HTTPException(500, f"cycle failed: {exc}")
        return JSONResponse({"equity": result["equity"], "decisions": result["decisions"],
                             "regime": result["regime"]})

    @app.post("/api/pause")
    def pause_entries() -> dict:
        store.save_portfolio("entries_paused", {"paused": True, "ts": int(time.time())})
        return {"status": "paused"}

    @app.post("/api/resume")
    def resume_entries() -> dict:
        store.save_portfolio("entries_paused", {"paused": False, "ts": int(time.time())})
        return {"status": "resumed"}

    @app.post("/api/reset_drawdown")
    def reset_drawdown() -> dict:
        stored = store.get_portfolio("breakers", {}) or {}
        stored["drawdown_halted"] = False
        store.save_portfolio("breakers", stored)
        return {"message": "Drawdown circuit breaker reset. New entries allowed again.",
                "ts": int(time.time())}

    @app.get("/api/export")
    def export_data() -> JSONResponse:
        """Export all research data as a single JSON snapshot."""
        trades = [dict(r) for r in store.db.query(
            "SELECT * FROM trades ORDER BY entry_ts DESC LIMIT 2000")]
        runs   = [dict(r) for r in store.db.query(
            "SELECT run_id, label, initial_equity, final_equity, start_ts, end_ts"
            " FROM uni_runs ORDER BY created_at DESC LIMIT 30")]
        return JSONResponse({
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "capital": settings.capital,
            "universe": settings.symbols,
            "trades": trades,
            "runs": runs,
        })

    # ---------------------------------------------------------------- static
    directory = Path(web_dir or settings.paths.dashboard)
    if directory.exists():
        app.mount("/", StaticFiles(directory=str(directory), html=True), name="dashboard")
    else:
        @app.get("/")
        def root() -> dict:
            return {"service": "unified crypto bot",
                    "note": f"dashboard directory missing: {directory}"}

    return app


def serve(settings: Settings | None = None) -> None:
    import uvicorn
    s = settings or Settings.load("config/unibot.json")
    uvicorn.run(create_app(s), host=s.api.host, port=s.api.port, log_level="info")
