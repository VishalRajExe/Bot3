"""Paper trading.

Two clearly separated modes, both stored with an explicit ``mode`` column so
results can never be confused with each other or with a backtest:

``live``
    Polls the venue's public API.  A decision is made only on a **closed**
    candle and is executed against the first observed quote after that close
    (mid price, plus the same fee/spread/slippage model as the backtest).
    Between candles, protective stops are monitored against the live quote.

``replay``
    Seeds a session by running the identical execution logic over the most
    recent N *closed historical* bars (next-bar-open fills).  This is a
    simulation of history, not evidence of live performance, and the dashboard
    labels it as such.

No real order can be sent: the only executor is :class:`PaperBroker`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from .backtest.engine import run_backtest
from .config import Config
from .data.exchange import MarketDataError, Ticker, get_ticker
from .data.ingest import CandleIngestor
from .data.timeframes import tf_seconds, to_datetime
from .db import Database, utcnow_iso
from .execution.models import Order
from .execution.paper import PaperBroker, RejectedOrder
from .persist import (clear_run_records, insert_equity, insert_fills, insert_orders,
                      insert_risk_events, insert_signals, insert_trades)
from .risk import RiskManager
from .strategies import build_strategy
from .strategies.base import PositionView, Strategy

log = logging.getLogger(__name__)


class PaperTrader:
    def __init__(self, cfg: Config, db: Database, *, strategy: str | None = None,
                 timeframe: str | None = None, session_id: str | None = None,
                 mode: str = "live") -> None:
        self.cfg = cfg
        self.db = db
        self.ingestor = CandleIngestor(cfg, db)
        self.strategy: Strategy = build_strategy(
            strategy or cfg.paper.strategy,
            cfg.paper.strategy_params or None,
            timeframe or cfg.paper.timeframe,
        )
        self.timeframe = self.strategy.timeframe
        self.tf_s = tf_seconds(self.timeframe)
        self.mode = mode
        self.session_id = session_id or f"paper_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.broker = PaperBroker(cfg.costs, symbol=cfg.exchange.pair,
                                  initial_equity=cfg.paper.initial_equity,
                                  run_id=self.session_id, mode="paper")
        self.risk = RiskManager(cfg.risk, cfg.costs, cfg.paper.initial_equity,
                                timeframe=self.timeframe)
        self.last_processed_bar: int | None = None
        self.last_price: float | None = None
        self.last_ticker: Ticker | None = None
        self.last_error: str | None = None
        self.last_signal: dict | None = None
        self._lock = threading.RLock()
        self._ensure_session()

    # -------------------------------------------------------------- session
    def _ensure_session(self) -> None:
        row = self.db.query_one("SELECT * FROM paper_sessions WHERE id=?", (self.session_id,))
        if row is None:
            self.db.execute(
                "INSERT INTO paper_sessions (id, started_at, mode, strategy, strategy_version,"
                " timeframe, initial_equity, status, last_processed_bar, config_json, notes)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.session_id, utcnow_iso(), self.mode, self.strategy.name,
                 self.strategy.version, self.timeframe, self.cfg.paper.initial_equity,
                 "running", None, json.dumps({"costs": self.cfg.costs.model_dump(),
                                              "risk": self.cfg.risk.model_dump()}), None),
            )
            log.info("started paper session %s (%s, %s)", self.session_id, self.mode, self.timeframe)
            return

        self.mode = row["mode"]
        state = self.db.get_state(f"paper_session:{self.session_id}")
        if state:
            self.broker.load_dict(state.get("broker", {}))
            self.risk.load_dict(state.get("risk", {}))
            self.last_processed_bar = state.get("last_processed_bar")
            self.last_price = state.get("last_price")
            log.info("resumed paper session %s at bar %s", self.session_id,
                     to_datetime(self.last_processed_bar) if self.last_processed_bar else None)
        else:
            self.last_processed_bar = row["last_processed_bar"]

    def _save_state(self) -> None:
        self.db.set_state(f"paper_session:{self.session_id}", {
            "broker": self.broker.to_dict(),
            "risk": self.risk.to_dict(),
            "last_processed_bar": self.last_processed_bar,
            "last_price": self.last_price,
            "mode": self.mode,
            "updated_at": utcnow_iso(),
        })
        self.db.execute(
            "UPDATE paper_sessions SET last_processed_bar=?, status=? WHERE id=?",
            (self.last_processed_bar, self._session_status(), self.session_id),
        )

    def _session_status(self) -> str:
        row = self.db.query_one("SELECT status FROM paper_sessions WHERE id=?", (self.session_id,))
        return row["status"] if row else "running"

    # ---------------------------------------------------------------- replay
    def run_replay(self, bars: int, *, persist: bool = True, notes: str = "") -> dict:
        """Seed the session from the most recent closed historical bars."""
        if bars < 20:
            raise ValueError("replay requires at least 20 bars")
        df = self.ingestor.ensure_warmup(self.timeframe, self.cfg.paper.warmup_bars)
        if df.empty:
            raise MarketDataError("no cached candles for replay; run `fetch` first")
        bars = min(bars, len(df) - 1)
        start_ts = int(df["bar_ts"].iloc[-bars])
        log.info("replaying %d %s bars from %s for session %s", bars, self.timeframe,
                 to_datetime(start_ts).date(), self.session_id)
        result = run_backtest(
            self.cfg, self.db, self.strategy, df, timeframe=self.timeframe,
            initial_equity=self.cfg.paper.initial_equity, run_id=self.session_id,
            persist=False, start_ts=start_ts, notes=notes or "paper replay", mode="paper",
            close_at_end=False,
        )
        self.broker.load_dict(result.broker_state)
        self.risk.load_dict(result.risk_state)
        self.last_processed_bar = result.last_bar_ts
        self.last_price = float(df["close"].iloc[-1])
        if persist:
            # A replay re-baselines the session: replace its previous rows so a
            # repeated seed cannot double-count trades or collide with fills.
            clear_run_records(self.db, self.session_id)
            insert_trades(self.db, result.trades)
            insert_orders(self.db, result.orders)
            insert_fills(self.db, result.fills)
            insert_equity(self.db, result.equity)
            insert_signals(self.db, result.signals)
            insert_risk_events(self.db, result.risk_events)
            self.db.execute(
                "UPDATE paper_sessions SET status='running', last_processed_bar=?, notes=?"
                " WHERE id=?",
                (self.last_processed_bar, notes or f"replay of last {bars} bars", self.session_id),
            )
            self._save_state()
        return {
            "session_id": self.session_id,
            "bars": result.bars,
            "trades": len(result.trades),
            "net_return_pct": result.metrics.get("net_return_pct"),
            "last_bar": to_datetime(result.last_bar_ts).isoformat(),
            "mode": "replay",
        }

    # ------------------------------------------------------------------ live
    def poll_once(self) -> dict:
        """One live iteration.  Safe to call repeatedly."""
        with self._lock:
            started = time.time()
            status_row = self.db.query_one("SELECT status FROM paper_sessions WHERE id=?",
                                           (self.session_id,))
            if status_row is None or status_row["status"] in ("stopped",):
                return {"ok": False, "reason": "session stopped"}

            try:
                ticker = get_ticker(self.cfg.exchange)
                self.last_ticker = ticker
                self.last_price = ticker.price
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - report, do not trade blind
                self.last_error = f"market data unavailable: {exc}"
                log.error(self.last_error)
                self.db.log_error(self.last_error, logger=__name__,
                                  context={"session": self.session_id})
                return {"ok": False, "reason": self.last_error}

            try:
                df = self.ingestor.recent_closed(self.timeframe, bars=max(self.cfg.paper.warmup_bars + 20, 200))
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"candle fetch failed: {exc}"
                log.error(self.last_error)
                self.db.log_error(self.last_error, logger=__name__,
                                  context={"session": self.session_id})
                return {"ok": False, "reason": self.last_error}

            prepared = self.strategy.prepare(df)
            i = len(prepared) - 1
            bar_ts = int(prepared["bar_ts"].iloc[i])
            now = int(time.time())

            # Monitor the protective stop against the live quote on every poll.
            self._check_live_stop(ticker, now)

            equity = self.broker.equity(ticker.price)
            self.risk.on_equity(now, equity)
            self._drain_risk_events(now)

            new_bar = self.last_processed_bar is None or bar_ts > int(self.last_processed_bar)
            if not new_bar:
                self._check_freshness(now, bar_ts)
                return {"ok": True, "new_bar": False, "bar": to_datetime(bar_ts).isoformat(),
                        "equity": round(equity, 2), "price": ticker.price}

            missed = 0
            if self.last_processed_bar is not None:
                missed = max(int((bar_ts - self.last_processed_bar) / self.tf_s) - 1, 0)
            if missed:
                self.db.log_error(
                    f"paper session missed {missed} closed bar(s); only the latest is evaluated",
                    logger=__name__, level="WARNING",
                    context={"session": self.session_id, "bar": bar_ts},
                )

            pos_view = None
            if self.broker.position is not None:
                p = self.broker.position
                pos_view = PositionView(qty=p.qty, entry_price=p.entry_price, entry_ts=p.entry_ts,
                                        stop_price=p.stop_price, entry_bar_ts=p.entry_bar_ts,
                                        entry_tag=p.entry_tag)
            sig = self.strategy.signal(prepared, i, pos_view)
            row = prepared.iloc[i]
            bar_return = float(row["close"] / row["open"] - 1.0) if row["open"] else None
            record = {
                "run_id": self.session_id, "mode": "paper", "strategy": self.strategy.name,
                "strategy_version": self.strategy.version, "bar_ts": bar_ts,
                "action": sig.action, "reason": sig.reason, "price": float(row["close"]),
                "indicators": sig.indicators, "tag": sig.tag,
            }
            session_status = status_row["status"]
            if session_status == "paused":
                record["note"] = "session paused: decisions recorded, no orders placed"
                insert_signals(self.db, [record])
                self.last_signal = record
                self.last_processed_bar = bar_ts
                self._save_state()
                return {"ok": True, "new_bar": True, "paused": True, "signal": sig.action,
                        "reason": sig.reason}

            if sig.is_entry and self.broker.position is None:
                decision = self.risk.check_entry(
                    ts=now, equity=equity, cash=self.broker.cash, has_position=False,
                    bar_ts=bar_ts, bar_return=bar_return, pending_order=False,
                )
                if decision.allowed:
                    sizing = self.risk.size_position(
                        equity=equity, cash=self.broker.cash,
                        entry_price=float(row["close"]), stop_price=sig.stop_price,
                    )
                    if sizing.allowed:
                        order = Order(
                            symbol=self.cfg.exchange.pair, side="buy", qty=sizing.qty,
                            reason=sig.reason, strategy=self.strategy.name,
                            strategy_version=self.strategy.version, signal_bar_ts=bar_ts,
                            target_bar_ts=None, stop_price=sig.stop_price, tag=sig.tag,
                            run_id=self.session_id, mode="paper",
                        )
                        self._execute(order, ticker.price, now, "enter", record, sizing)
                    else:
                        record["blocked"] = {"code": sizing.capped_by, "reason": sizing.reason}
                else:
                    record["blocked"] = {"code": decision.code, "reason": decision.reason}
                    self._record_risk(now, f"entry_blocked:{decision.code}", "no_trade",
                                      decision.reason)
            elif sig.is_exit and self.broker.position is not None:
                order = Order(
                    symbol=self.cfg.exchange.pair, side="sell",
                    qty=self.broker.position.qty, reason=sig.reason,
                    strategy=self.strategy.name, strategy_version=self.strategy.version,
                    signal_bar_ts=bar_ts, target_bar_ts=None, stop_price=None, tag=sig.tag,
                    run_id=self.session_id, mode="paper",
                )
                self._execute(order, ticker.price, now, "exit", record, None)

            insert_signals(self.db, [record])
            self.last_signal = record
            self.last_processed_bar = bar_ts
            self._snapshot_equity(now, ticker.price)
            self._save_state()
            log.info("paper %s bar %s: %s (%s)", self.session_id, to_datetime(bar_ts).date(),
                     sig.action, sig.reason)
            return {
                "ok": True, "new_bar": True, "bar": to_datetime(bar_ts).isoformat(),
                "signal": sig.action, "reason": sig.reason, "equity": round(equity, 2),
                "elapsed_s": round(time.time() - started, 3),
            }

    # ------------------------------------------------------------- internals
    def _check_live_stop(self, ticker: Ticker, now: int) -> None:
        pos = self.broker.position
        if pos is None or pos.stop_price is None:
            return
        if ticker.price <= pos.stop_price:
            order = Order(
                symbol=self.cfg.exchange.pair, side="sell", qty=pos.qty,
                reason=f"live stop-loss hit: quote {ticker.price:,.2f} <= stop {pos.stop_price:,.2f}",
                strategy=self.strategy.name, strategy_version=self.strategy.version,
                signal_bar_ts=self.last_processed_bar or now, target_bar_ts=None,
                stop_price=None, tag="stop", run_id=self.session_id, mode="paper",
            )
            self._execute(order, ticker.price, now, "stop", None, None)

    def _execute(self, order: Order, reference_price: float, ts: int, kind: str,
                 record: dict | None, sizing) -> None:
        try:
            fill = self.broker.execute(order, reference_price, ts)
        except RejectedOrder as exc:
            order.status = "rejected"
            order.reject_reason = str(exc)
            insert_orders(self.db, [order])
            if record is not None:
                record["blocked"] = {"code": "order_rejected", "reason": str(exc)}
            self._record_risk(ts, "order_rejected", "order_dropped", str(exc))
            return
        order.status = "filled"
        insert_orders(self.db, [order])
        insert_fills(self.db, [fill])
        if order.side == "buy":
            pos = self.broker.position
            if pos is not None:
                pos.entry_regime = record.get("indicators", {}).get("regime") if record else None
                pos.entry_indicators = record.get("indicators", {}) if record else {}
                if sizing is not None:
                    pos.risk_amount = sizing.risk_amount
                pos.entry_order_id = order.id
        else:
            trade = self.broker.close_position(
                fill, exit_reason=order.reason if kind != "stop" else "stop-loss",
                exit_signal="exit_long" if kind == "exit" else "stop",
            )
            trade.model_version = None
            insert_trades(self.db, [trade])
            self.risk.on_trade_closed(ts=ts, net_pnl=trade.net_pnl, timeframe=self.timeframe)
            self._drain_risk_events(ts)
        log.info("paper fill %s %s %.6f @ %.2f (fee %.4f)", self.session_id, order.side,
                 fill.qty, fill.price, fill.fee)

    def _record_risk(self, ts: int, kind: str, action: str, detail: str) -> None:
        insert_risk_events(self.db, [{"ts": ts, "mode": "paper", "run_id": self.session_id,
                                      "kind": kind, "action": action, "detail": detail,
                                      "value": {}}])

    def _drain_risk_events(self, ts: int) -> None:
        events = self.risk.drain_events()
        if events:
            insert_risk_events(self.db, [{**e, "mode": "paper", "run_id": self.session_id}
                                         for e in events])

    def _snapshot_equity(self, ts: int, price: float) -> None:
        equity = self.broker.equity(price)
        peak = max(self.risk.state.hwm, equity)
        df = pd.DataFrame([[self.session_id, ts, equity, self.broker.cash,
                            self.broker.position.qty if self.broker.position else 0.0,
                            price, (equity - peak) / peak * 100.0 if peak else 0.0]],
                          columns=["run_id", "ts", "equity", "cash", "position_qty", "price", "drawdown"])
        insert_equity(self.db, df)

    def _check_freshness(self, now: int, bar_ts: int) -> None:
        age = now - (bar_ts + self.tf_s)
        if age > self.cfg.risk.max_bars_stale * self.tf_s:
            msg = f"stale market data: last closed bar is {age / 3600:.1f}h old"
            if self.last_error != msg:
                log.warning(msg)
            self.last_error = msg

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        row = self.db.query_one("SELECT * FROM paper_sessions WHERE id=?", (self.session_id,))
        price = self.last_price or (self.broker.last_price or 0.0)
        equity = self.broker.equity(price)
        pos = self.broker.position
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "status": row["status"] if row else "unknown",
            "started_at": row["started_at"] if row else None,
            "strategy": self.strategy.name,
            "strategy_version": self.strategy.version,
            "timeframe": self.timeframe,
            "last_processed_bar": to_datetime(self.last_processed_bar).isoformat()
            if self.last_processed_bar else None,
            "last_price": price,
            "price_source": self.last_ticker.source if self.last_ticker else None,
            "equity": round(equity, 2),
            "cash": round(self.broker.cash, 2),
            "starting_equity": self.cfg.paper.initial_equity,
            "realized_pnl": round(self.broker.realized_pnl, 2),
            "unrealized_pnl": round((price - pos.entry_price) * pos.qty, 2) if pos else 0.0,
            "position": {
                "qty": pos.qty, "entry_price": pos.entry_price,
                "entry_ts": to_datetime(pos.entry_ts).isoformat(),
                "stop": pos.stop_price,
                "mfe_pct": round(pos.mfe_pct, 3), "mae_pct": round(pos.mae_pct, 3),
                "entry_reason": pos.entry_reason,
            } if pos else None,
            "risk": self.risk.status(equity),
            "last_error": self.last_error,
            "last_signal": self.last_signal,
            "trades": self.broker.n_trades,
        }

    def set_status(self, status: str) -> None:
        if status not in ("running", "paused", "stopped"):
            raise ValueError("invalid session status")
        self.db.execute("UPDATE paper_sessions SET status=?, stopped_at=? WHERE id=?",
                        (status, utcnow_iso() if status == "stopped" else None, self.session_id))
        self._save_state()
        log.info("paper session %s -> %s", self.session_id, status)

    def reset_drawdown_halt(self) -> None:
        self.risk.reset_drawdown_halt(int(time.time()))
        self._drain_risk_events(int(time.time()))
        self._save_state()


class PaperService:
    """Background loop wrapper used by the API/CLI."""

    def __init__(self, trader: PaperTrader, poll_seconds: int | None = None) -> None:
        self.trader = trader
        self.poll_seconds = poll_seconds or trader.cfg.paper.poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.iterations = 0
        self.last_result: dict | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick(self) -> dict:
        try:
            self.last_result = self.trader.poll_once()
        except Exception as exc:  # noqa: BLE001 - loop must not die
            log.exception("paper loop iteration failed")
            self.trader.db.log_error(str(exc), logger=__name__,
                                     context={"session": self.trader.session_id})
            self.last_result = {"ok": False, "reason": str(exc)}
        self.iterations += 1
        return self.last_result or {}

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()

        def loop() -> None:
            log.info("paper service started (poll %ss)", self.poll_seconds)
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(self.poll_seconds)
            log.info("paper service stopped")

        self._thread = threading.Thread(target=loop, name="paper-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
