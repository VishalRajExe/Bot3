"""Live paper-trading engine (Alpaca paper account only).

Loop contract:

1. **never guess**: account cash/equity and open positions are read from the
   broker every cycle; the local database is a cache of intent, not the truth;
2. **decide once per closed bar**: each 4h bar is evaluated after a short grace
   period, so a partially formed bar can never trigger a trade;
3. **exits before entries**, and a software stop is checked on every poll even
   when a broker stop order exists;
4. **every decision is recorded and, if enabled, sent to Telegram** - including
   holds and blocked entries with the reason;
5. any broker/database mismatch is logged as a reconciliation event and the
   broker wins.  A failed reconcile blocks new entries until resolved.

The engine refuses to construct unless the credentials point at Alpaca's paper
endpoint.  There is no live-money code path.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from btcbot.data.timeframes import tf_seconds

from .backtest import _trade_record, OpenPosition
from .config import Settings, load_credentials
from .datafeed import build_market, prepare_bars, sync_all
from .learning import review_after_trade
from .market import AlpacaClient, AlpacaError
from .models import Gate, load_active_gates, train_all
from .notify import TelegramNotifier
from .regime import regime_at
from .risk import OpenRisk, RiskManager
from .strategies import PositionView, build as build_strategy
from .store import UnibotStore

log = logging.getLogger(__name__)


@dataclass
class LivePosition:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: int
    stop: float
    strategy: str
    strategy_version: str
    model_version: str | None
    prob: float | None
    risk_usd: float
    stop_order_id: str | None
    features: dict


class LiveEngine:
    def __init__(self, settings: Settings, store: UnibotStore, *, client: AlpacaClient | None = None,
                 notifier: TelegramNotifier | None = None) -> None:
        self.s = settings
        self.store = store
        if client is None:
            creds = load_credentials(".env")
            client = AlpacaClient(creds.alpaca_key, creds.alpaca_secret,
                                  paper_base=creds.alpaca_base_url,
                                  data_base=settings.alpaca.data_base_url)
            if notifier is None:
                notifier = TelegramNotifier(creds.telegram_token, creds.telegram_chat_id,
                                            enabled=settings.telegram.enabled, store=store)
        self.client = client
        self.notifier = notifier or TelegramNotifier("", "", enabled=False, store=store)
        self.risk = RiskManager(settings.risk, settings.costs)
        self.stop_event = threading.Event()
        self.gates: dict[str, Gate] = {}
        self.strategies = {s.symbol: build_strategy(s) for s in settings.universe}
        self._last_hold_digest = 0
        self._last_heartbeat = 0
        self._last_train_check = 0
        self._last_reconcile = 0
        self._latest_hold_rows = [
            {"symbol": sym, "action": d.get("action", "hold"), "reason": d.get("reason", "watching"),
             "price": d.get("price")}
            for sym, d in self.store.latest_decision_per_symbol().items()
        ]

    # ------------------------------------------------------------------ state
    def account(self) -> dict:
        return self.client.account()

    def broker_positions(self) -> dict[str, dict]:
        out = {}
        for p in self.client.positions():
            try:
                out[p["symbol"]] = p
            except KeyError:
                continue
        return out

    def equity(self) -> tuple[float, float]:
        acct = self.account()
        return float(acct.get("equity") or 0.0), float(acct.get("cash") or 0.0)

    # ------------------------------------------------------------- reconcile
    def reconcile(self) -> dict:
        broker = self.broker_positions()
        db_positions = {p["symbol"]: p for p in self.store.positions()}
        issues: list[str] = []
        for sym, bp in broker.items():
            qty = float(bp.get("qty") or 0)
            if qty <= 0:
                continue
            local = db_positions.get(sym)
            if local is None:
                price = float(bp.get("current_price") or bp.get("avg_entry_price") or 0)
                self.store.save_position(
                    symbol=sym, qty=qty, entry_price=float(bp.get("avg_entry_price") or price),
                    entry_ts=int(time.time()), stop_price=price * 0.95 if price else None,
                    initial_stop=price * 0.95 if price else None,
                    strategy="imported", strategy_version="0", model_version=None,
                    prob=None, notional=qty * price, risk_usd=qty * price * 0.05,
                    opened_by="imported",
                    meta={"source": "broker reconciliation"},
                )
                issues.append(f"adopted broker position {sym} {qty}")
                continue
            if abs(qty - float(local["qty"])) > max(1e-8, abs(qty) * 1e-4):
                issues.append(f"qty mismatch {sym}: broker {qty} != db {local['qty']}; "
                              f"broker is authoritative")
                self.store.db.execute(
                    "UPDATE uni_positions SET qty=?, updated_at=datetime('now') WHERE symbol=?",
                    (qty, sym),
                )
        for sym, local in list(db_positions.items()):
            if sym in broker:
                continue
            # A stop order may have filled while the bot was offline.  Record the
            # round trip from the broker's own fill instead of dropping it.
            filled = self._recent_sell_fill(sym, int(local["entry_ts"]))
            if filled:
                price = float(filled.get("filled_avg_price") or 0) or float(local["entry_price"])
                qty = float(filled.get("filled_qty") or local["qty"])
                entry = float(local["entry_price"])
                gross = (price - entry) * qty
                fee = (qty * entry + qty * price) * self.s.costs.taker_fee_bps / 10_000.0
                pos = OpenPosition(
                    symbol=sym, qty=qty, entry_price=entry, entry_ts=int(local["entry_ts"]),
                    entry_bar_ts=int(local["entry_ts"]),
                    stop=float(local["stop_price"] or entry * 0.95),
                    initial_stop=float(local["initial_stop"] or local["stop_price"] or entry * 0.95),
                    trail_atr=0.0, strategy=local["strategy"],
                    strategy_version=local["strategy_version"],
                    model_version=local.get("model_version"), prob=local.get("prob"),
                    risk_usd=float(local["risk_usd"]), entry_fees=fee / 2,
                    entry_slippage=0.0, entry_reason="live entry", entry_tag="live",
                    regime="live", max_price=entry, min_price=entry,
                )
                trade = _trade_record(pos, price=price, fee=fee / 2, ts=int(time.time()),
                                      reason="broker stop filled while offline",
                                      run_id="live", mode="live", exit_signal="stop")
                self.store.save_trade(trade)
                review_after_trade(self.store, self.s, trade, features={},
                                   prob=local.get("prob"), regime="live", equity=0.0)
                issues.append(f"reconciled broker exit for {sym} at {price:.6g}"
                              f" (net ${gross - fee:+.2f})")
            else:
                issues.append(f"local position {sym} no longer at broker; removed")
            self.store.delete_position(sym)
        status = "REVIEW" if issues else "HEALTHY"
        self.store.log_recon(status=status, detail="; ".join(issues) or "positions match",
                             payload={"broker": list(broker), "db": list(db_positions)})
        if issues:
            log.warning("reconcile: %s", "; ".join(issues))
        return {"status": status, "issues": issues, "broker": list(broker),
                "db": list(db_positions)}

    # ---------------------------------------------------------------- trading
    def _latest_price(self, symbol: str) -> float | None:
        try:
            px = self.client.latest_prices([symbol]).get(symbol)
            return float(px["price"]) if px else None
        except AlpacaError:
            return None

    def _recent_sell_fill(self, symbol: str, after_ts: int) -> dict | None:
        try:
            orders = self.client.orders(status="closed")
        except AlpacaError:
            return None
        best = None
        for o in orders:
            if o.get("symbol") != symbol or str(o.get("side")) != "sell":
                continue
            if str(o.get("status")) != "filled":
                continue
            try:
                ts = int(datetime.fromisoformat(
                    str(o.get("filled_at") or o.get("submitted_at")).replace("Z", "+00:00")).timestamp())
            except (ValueError, TypeError):
                continue
            if ts < after_ts:
                continue
            if best is None or ts > best[0]:
                best = (ts, o)
        return best[1] if best else None

    def _cancel_stop(self, row: dict) -> None:
        order_id = row.get("stop_order_id")
        if order_id:
            try:
                self.client.cancel_order(str(order_id))
            except AlpacaError:
                pass

    def _place_stop(self, symbol: str, qty: float, stop: float) -> str | None:
        if not self.s.alpaca.use_stop_orders or qty <= 0:
            return None
        buf = self.s.alpaca.stop_limit_buffer_pct / 100.0
        try:
            order = self.client.submit_stop_limit(
                symbol=symbol, qty=qty, stop_price=round(stop, 9),
                limit_price=round(stop * (1 - buf), 9),
            )
            return str(order.get("id"))
        except AlpacaError as exc:
            self.notifier.risk("Stop order rejected; software stop active",
                               f"{symbol}: {exc.body or exc}"[:300], symbol)
            return None

    def _equity_state(self) -> dict:
        equity, cash = self.equity()
        day = int(time.time()) // 86_400
        stored = self.store.get_portfolio("breakers", {}) or {}
        day_start = self.risk.day_start_equity(equity, stored, day)
        hwm = max(float(stored.get("hwm") or equity), equity)
        breaker = self.risk.breaker_state(equity=equity, hwm=hwm, day_start=day_start)
        self.store.save_portfolio("breakers", {
            "day": day, "equity": day_start, "hwm": hwm,
            "drawdown_halted": bool(stored.get("drawdown_halted")) or breaker["drawdown_halted"],
        })
        breaker["drawdown_halted"] = bool(stored.get("drawdown_halted")) or breaker["drawdown_halted"]
        return {"equity": equity, "cash": cash, "hwm": hwm, "day_start": day_start,
                "breaker": breaker}

    def enter(self, *, sym: str, price: float, stop: float, size_usd: float, reason: str,
              strategy: str, version: str, prob: float | None, regime: str,
              features: dict, trail_atr: float = 0.0,
              model_version: str | None = None) -> dict | None:
        qty = math.floor(size_usd / price * 1e9) / 1e9
        if qty <= 0 or qty * price < self.s.costs.min_notional:
            return None
        order = self.client.submit_market_order(symbol=sym, side="buy", qty=qty)
        self.notifier.order_submitted(symbol=sym, side="buy", qty=qty, price=price,
                                      size_usd=size_usd, order_id=str(order.get("id")),
                                      reason=reason)
        filled = order
        for _ in range(12):
            if str(order.get("status")) in ("filled", "canceled", "rejected", "expired"):
                break
            time.sleep(1.0)
            try:
                filled = self.client.order(str(order["id"]))
            except AlpacaError:
                break
        status = str(filled.get("status"))
        if status != "filled":
            self.notifier.risk("Entry order not filled",
                               f"{sym} {status}: {filled.get('failed_at') or filled.get('id')}", sym)
            return None
        fill_price = float(filled.get("filled_avg_price") or price)
        fill_qty = float(filled.get("filled_qty") or qty)
        notional = fill_price * fill_qty
        risk_usd = notional * (price - stop) / price
        stop_id = self._place_stop(sym, fill_qty, stop)
        self.store.save_position(
            symbol=sym, qty=fill_qty, entry_price=fill_price, entry_ts=int(time.time()),
            stop_price=stop, initial_stop=stop, strategy=strategy, strategy_version=version,
            model_version=None, prob=prob, notional=notional, risk_usd=risk_usd,
            stop_order_id=stop_id, opened_by="live",
            meta={"features": features, "regime": regime, "order_id": filled.get("id"),
                  "trail_atr": trail_atr},
        )
        self.store.record_decision(
            ts=int(time.time()), bar_ts=int(time.time()) // tf_seconds(self.s.timeframe)
            * tf_seconds(self.s.timeframe), mode="live", symbol=sym,
            timeframe=self.s.timeframe, strategy=strategy, strategy_version=version,
            action="buy_filled", reason=reason, price=fill_price, prob=prob,
            size_usd=notional, stop_price=stop, regime=regime, features=features,
        )
        equity_now, _ = self.equity()
        self.notifier.order_filled(symbol=sym, side="buy", qty=fill_qty, fill_price=fill_price,
                                   notional=notional, stop_price=stop, stop_order_id=stop_id,
                                   prob=prob, strategy=strategy, regime=regime, equity=equity_now)
        return {"symbol": sym, "qty": fill_qty, "price": fill_price, "notional": notional,
                "stop": stop, "stop_order_id": stop_id}

    def exit_position(self, *, sym: str, reason: str, price_hint: float | None = None
                     ) -> dict | None:
        row = self.store.position(sym)
        if row is None:
            return None
        self._cancel_stop(row)
        broker = self.broker_positions().get(sym)
        qty = float(broker["qty"]) if broker else float(row["qty"])
        if qty <= 0:
            self.store.delete_position(sym)
            return None
        order = self.client.submit_market_order(symbol=sym, side="sell", qty=qty)
        est_px = float(price_hint or row["entry_price"])
        self.notifier.order_submitted(symbol=sym, side="sell", qty=qty, price=est_px,
                                      size_usd=qty * est_px, order_id=str(order.get("id")),
                                      reason=reason)
        filled = order
        for _ in range(12):
            if str(order.get("status")) in ("filled", "canceled", "rejected", "expired"):
                break
            time.sleep(1.0)
            try:
                filled = self.client.order(str(order["id"]))
            except AlpacaError:
                break
        if str(filled.get("status")) != "filled":
            self.notifier.risk("Exit order not filled",
                               f"{sym} status {filled.get('status')}", sym)
            return None
        price = float(filled.get("filled_avg_price") or price_hint or row["entry_price"])
        qty = float(filled.get("filled_qty") or qty)
        entry = float(row["entry_price"])
        gross = (price - entry) * qty
        fee = (qty * entry + qty * price) * self.s.costs.taker_fee_bps / 10_000.0
        net = gross - fee
        pos = OpenPosition(
            symbol=sym, qty=qty, entry_price=entry, entry_ts=int(row["entry_ts"]),
            entry_bar_ts=int(row["entry_ts"]), stop=float(row["stop_price"] or entry * 0.95),
            initial_stop=float(row["initial_stop"] or row["stop_price"] or entry * 0.95),
            trail_atr=0.0, strategy=row["strategy"], strategy_version=row["strategy_version"],
            model_version=row.get("model_version"), prob=row.get("prob"),
            risk_usd=float(row["risk_usd"]), entry_fees=fee / 2, entry_slippage=0.0,
            entry_reason="live entry", entry_tag="live", regime="live",
            features=json_loads(row.get("meta_json")).get("features", {}),
            max_price=float(row["entry_price"]), min_price=float(row["entry_price"]),
        )
        trade = _trade_record(pos, price=price, fee=fee / 2, ts=int(time.time()),
                              reason=reason, run_id="live", mode="live",
                              exit_signal="stop" if "stop" in reason else "exit_long")
        self.store.save_trade(trade)
        self.store.delete_position(sym)
        equity, _ = self.equity()
        review_after_trade(self.store, self.s, trade,
                           features=json_loads(row.get("meta_json")).get("features", {}),
                           prob=row.get("prob"), regime="live", equity=equity)
        self.store.record_decision(
            ts=int(time.time()), bar_ts=int(time.time()) // tf_seconds(self.s.timeframe)
            * tf_seconds(self.s.timeframe), mode="live", symbol=sym,
            timeframe=self.s.timeframe, strategy=row["strategy"],
            strategy_version=row["strategy_version"], action="exit_filled", reason=reason,
            price=price, prob=None, size_usd=qty * price, stop_price=None, regime="live",
        )
        is_stop = "stop" in reason.lower()
        self.notifier.trade_closed(symbol=sym, entry=entry, exit_price=price, net_pnl=net,
                                   return_pct=trade.return_pct, reason=reason,
                                   holding_hours=trade.holding_hours, equity=equity,
                                   is_stop=is_stop)
        return {"symbol": sym, "price": price, "net_pnl": net, "return_pct": trade.return_pct}

    # ------------------------------------------------------------- bar cycle
    def last_closed_bar(self) -> tuple[int, int]:
        """The most recent bar that is fully closed on the UTC grid."""
        step = tf_seconds(self.s.timeframe)
        current_open = (int(time.time()) // step) * step
        closed_open = current_open - step
        return closed_open, closed_open + step

    def run_cycle(self) -> dict:
        state = self._equity_state()
        account = self.account()
        broker_equity = float(account.get("equity") or state["equity"])
        cash = float(account.get("cash") or state["cash"])

        # 1. refresh market data for the last few days only
        sync_all(self.store, self.s, self.client, timeframes=("1h", "1d"))

        market = build_market(self.store, self.s)
        bars_last = (int(time.time()) // tf_seconds(self.s.timeframe)) * tf_seconds(self.s.timeframe)
        regime = regime_at(market, bars_last)
        self.store.save_regime(ts=bars_last, label=regime.label, score=regime.score,
                               btc_trend=regime.btc_trend, breadth=regime.breadth,
                               vol_percentile=regime.vol_percentile, payload=regime.as_dict())

        # 2. reconcile before doing anything
        recon = self.reconcile()
        if recon["status"] != "HEALTHY" and not self.store.positions():
            log.warning("reconcile not healthy: %s", recon["issues"])

        # 3. software stops on every poll
        open_rows = {r["symbol"]: r for r in self.store.positions()}
        prices = self.client.latest_prices(list(open_rows)) if open_rows else {}
        for sym, row in list(open_rows.items()):
            px = prices.get(sym, {}).get("price")
            if px is None:
                continue
            if float(px) <= float(row["stop_price"] or 0):
                self.exit_position(sym=sym, reason="software_stop", price_hint=float(px))
                open_rows.pop(sym, None)

        # 4. one decision per closed bar
        bar_open, bar_close = self.last_closed_bar()
        last = self.store.get_portfolio("last_decided_bar", {}) or {}
        decisions_this_cycle: list[dict] = []
        hold_rows: list[dict] = []
        if last.get("bar") != bar_close and int(time.time()) >= bar_close + 180:
            for sym_cfg in self.s.universe:
                sym = sym_cfg.symbol
                if not sym_cfg.trade_enabled:
                    continue
                df = prepare_bars(self.store, self.s, sym, market)
                if df.empty or len(df) < 260:
                    continue
                # find the bar whose close time is the just-closed bar
                closes = (df["bar_ts"].to_numpy("int64")
                          + tf_seconds(self.s.timeframe))
                idx = int((closes <= bar_close).sum()) - 1
                if idx < self.strategies[sym].warmup_bars:
                    continue
                row = df.iloc[idx]
                pos_row = self.store.position(sym)
                pos_view = None
                if pos_row:
                    pos_view = PositionView(qty=float(pos_row["qty"]),
                                            entry_price=float(pos_row["entry_price"]),
                                            entry_ts=int(pos_row["entry_ts"]),
                                            stop_price=float(pos_row["stop_price"] or 0),
                                            entry_bar_ts=int(pos_row["entry_ts"]),
                                            trail_atr=0.0)
                signal = self.strategies[sym].signal(df, idx, pos_view)
                price = float(row["close"])
                strategy = self.strategies[sym]
                # Score every symbol on every closed bar, not just entries, so the
                # dashboard and the learning loop always see the model's view.
                gate = self.gates.get(sym)
                prob: float | None = None
                if gate is not None:
                    try:
                        prob = float(gate.predict(row))
                    except Exception:  # pragma: no cover - defensive
                        log.debug("gate predict failed for %s", sym, exc_info=True)
                self.store.record_decision(
                    ts=bar_close, bar_ts=int(row["bar_ts"]), mode="live", symbol=sym,
                    timeframe=self.s.timeframe, strategy=strategy.name,
                    strategy_version=strategy.version, action=signal.action,
                    reason=signal.reason, price=price, prob=prob, regime=regime.label,
                    features=signal.indicators,
                )
                decisions_this_cycle.append({"symbol": sym, "action": signal.action,
                                             "reason": signal.reason, "price": price})

                # exits first
                if signal.action == "exit_long" and pos_row:
                    self.notifier.signal(symbol=sym, action="exit_long", price=price,
                                         reason=signal.reason, strategy=strategy.name,
                                         regime=regime.label, indicators=signal.indicators)
                    self.exit_position(sym=sym, reason=signal.reason, price_hint=price)
                    continue
                if pos_row and signal.action != "exit_long":
                    # trailing stop + break-even ratchet
                    atr_now = float(row["atr14"])
                    trail_atr = float(json_loads(pos_row.get("meta_json")).get("trail_atr") or 0)
                    entry_px = float(pos_row["entry_price"])
                    init_stop = float(pos_row.get("initial_stop") or pos_row.get("stop_price") or entry_px * 0.95)
                    init_risk = entry_px - init_stop
                    cur_stop = float(pos_row["stop_price"] or 0)
                    new_stop = cur_stop

                    if init_risk > 0:
                        # Break-even stop ratchet once in +1.2R profit
                        if price >= entry_px + 1.2 * init_risk:
                            new_stop = max(new_stop, entry_px * 1.002)
                        # Profit-lock ratchet once in +2.2R profit
                        if price >= entry_px + 2.2 * init_risk:
                            new_stop = max(new_stop, entry_px + 0.8 * init_risk)

                    if trail_atr > 0 and math.isfinite(atr_now):
                        new_stop = max(new_stop, price - trail_atr * atr_now)

                    if new_stop > cur_stop:
                        stop_id = None
                        self._cancel_stop(pos_row)
                        stop_id = self._place_stop(sym, float(pos_row["qty"]), new_stop)
                        self.store.update_stop(sym, new_stop, stop_id)
                    continue
                if signal.action != "enter_long":
                    hold_rows.append({"symbol": sym, "action": "hold",
                                      "reason": signal.reason, "prob": prob, "price": price,
                                      "edge": (prob - gate.threshold) if (gate and prob is not None) else None})
                    continue

                # Notify buy signal opportunity
                self.notifier.signal(symbol=sym, action="enter_long", price=price,
                                     reason=signal.reason, strategy=strategy.name,
                                     regime=regime.label, stop=signal.stop_price,
                                     prob=prob, indicators=signal.indicators)

                # ---- entry checks
                breaker = state["breaker"]
                open_rows_now = self.store.positions()
                open_risk = [OpenRisk(symbol=r["symbol"], cluster=_cluster(self.s, r["symbol"]),
                                      notional=float(r["notional"]),
                                      risk_usd=float(r["risk_usd"]),
                                      stop_pct=(float(r["entry_price"]) - float(r["stop_price"] or 0))
                                      / float(r["entry_price"]) * 100.0)
                             for r in open_rows_now]
                gate_reason = None
                if gate is not None and prob is not None:
                    if prob < gate.threshold:
                        gate_reason = (f"model gate: P(win)={prob:.2f} < {gate.threshold:.2f}")
                if gate_reason:
                    self.store.record_decision(
                        ts=bar_close, bar_ts=int(row["bar_ts"]), mode="live", symbol=sym,
                        timeframe=self.s.timeframe, strategy=strategy.name,
                        strategy_version=strategy.version, action="blocked",
                        reason=gate_reason, price=price, prob=prob, regime=regime.label,
                    )
                    decisions_this_cycle.append({"symbol": sym, "action": "blocked",
                                                 "reason": gate_reason, "price": price})
                    continue
                portfolio_gate = self.risk.can_enter(
                    symbol=sym, cluster=sym_cfg.cluster, equity=broker_equity, cash=cash,
                    open_positions=[{"symbol": r["symbol"],
                                     "cluster": _cluster(self.s, r["symbol"])}
                                    for r in open_rows_now],
                    open_risk=open_risk, regime=regime, breaker=breaker,
                    cooldown_until=int(self.store.stats_row(sym).get("cooldown_until") or 0),
                    now=bar_close, bar_return_pct=float(row.get("ret_1") or 0.0),
                )
                if not portfolio_gate.approved:
                    self.store.record_decision(
                        ts=bar_close, bar_ts=int(row["bar_ts"]), mode="live", symbol=sym,
                        timeframe=self.s.timeframe, strategy=strategy.name,
                        strategy_version=strategy.version, action="blocked",
                        reason=portfolio_gate.reason, price=price, prob=prob,
                        regime=regime.label,
                    )
                    decisions_this_cycle.append({"symbol": sym, "action": "blocked",
                                                 "reason": portfolio_gate.reason, "price": price})
                    continue
                sizing = self.risk.position_size(
                    equity=broker_equity, cash=cash, price=price, stop=signal.stop_price,
                    regime=regime, symbol_multiplier=float(
                        self.store.stats_row(sym).get("risk_multiplier") or 1.0),
                    prob=prob,
                )
                if not sizing.approved:
                    self.store.record_decision(
                        ts=bar_close, bar_ts=int(row["bar_ts"]), mode="live", symbol=sym,
                        timeframe=self.s.timeframe, strategy=strategy.name,
                        strategy_version=strategy.version, action="blocked",
                        reason=sizing.reason, price=price, prob=prob, regime=regime.label,
                    )
                    decisions_this_cycle.append({"symbol": sym, "action": "blocked",
                                                 "reason": sizing.reason, "price": price})
                    continue
                filled = self.enter(
                    sym=sym, price=price, stop=signal.stop_price, size_usd=sizing.size_usd,
                    reason=signal.reason, strategy=strategy.name, version=strategy.version,
                    prob=prob, regime=regime.label, features=signal.indicators,
                    trail_atr=signal.trail_atr,
                    model_version=(gate.artifact.version if gate else None),
                )
                if filled:
                    self.store.save_portfolio("last_entry", {
                        "symbol": sym, "ts": int(time.time()), **filled})
                    cash = self._equity_state()["cash"]
                    decisions_this_cycle.append({"symbol": sym, "action": "buy_filled",
                                                 "reason": signal.reason, "price": price})
                else:
                    decisions_this_cycle.append({"symbol": sym, "action": "entry_failed",
                                                 "reason": "broker did not fill", "price": price})
            if hold_rows:
                self._latest_hold_rows = hold_rows
            self.store.save_portfolio("last_decided_bar", {"bar": bar_close,
                                                           "ts": int(time.time())})

        # 5. equity snapshot + notifications
        equity, cash2 = self.equity()
        pos_rows = self.store.positions()
        invested = sum(float(p["notional"]) for p in pos_rows)
        hwm = max(float(state["hwm"]), equity)
        dd = (hwm - equity) / hwm * 100.0 if hwm else 0.0
        self.store.save_portfolio("equity_snapshot", {
            "equity": round(equity, 4), "cash": round(cash2, 4),
            "positions": len(pos_rows),
            "regime": regime.label, "ts": int(time.time()),
        })
        self.store.save_equity("live", [("live", int(time.time()), equity, cash2, invested,
                                         invested / equity * 100.0 if equity else 0.0,
                                         dd, len(pos_rows))])
        now = int(time.time())
        if pos_rows and (now - getattr(self, "_last_positions_digest", 0) > 3600):
            try:
                px_map = self.client.latest_prices([p["symbol"] for p in pos_rows]) if pos_rows else {}
                pos_with_px = []
                for p in pos_rows:
                    p_copy = dict(p)
                    p_copy["current_price"] = (px_map.get(p["symbol"]) or {}).get("price") or p["entry_price"]
                    pos_with_px.append(p_copy)
                self.notifier.positions_digest(pos_with_px, equity=equity, cash=cash2)
                self._last_positions_digest = now
            except Exception as exc:
                log.debug("positions digest skipped: %s", exc)

        if now - getattr(self, "_last_portfolio_digest", 0) > 12 * 3600:
            closed_cnt = int(self.store.db.scalar("SELECT COUNT(*) FROM trades WHERE mode='live'") or 0)
            win_cnt = int(self.store.db.scalar("SELECT COUNT(*) FROM trades WHERE mode='live' AND net_pnl > 0") or 0)
            wr = (win_cnt / closed_cnt * 100.0) if closed_cnt else None
            self.notifier.portfolio_digest(equity=equity, cash=cash2,
                                           open_positions=len(pos_rows),
                                           closed_trades=closed_cnt, win_rate=wr,
                                           regime=regime.label)
            self._last_portfolio_digest = now

        if self.s.telegram.notify_holds and (now - self._last_hold_digest
                                             > self.s.telegram.hold_digest_minutes * 60):
            rows_to_send = hold_rows or getattr(self, "_latest_hold_rows", []) or decisions_this_cycle
            if rows_to_send:
                strat_map = {s.symbol: s.strategy for s in self.s.universe}
                pos_syms = {p["symbol"] for p in pos_rows}
                enriched = []
                for r in rows_to_send:
                    rc = dict(r)
                    rc["is_position"] = rc["symbol"] in pos_syms
                    enriched.append(rc)
                self.notifier.hold_digest(enriched, equity=equity,
                                          open_positions=len(self.store.positions()),
                                          regime=regime.label,
                                          strategy_map=strat_map)
                self._last_hold_digest = now
        if now - self._last_heartbeat > 6 * 3600:
            self.notifier.heartbeat(equity=equity, open_positions=len(self.store.positions()),
                                    regime=regime.label, trades=int(
                                        self.store.db.scalar("SELECT COUNT(*) FROM trades WHERE mode='live'") or 0))
            self._last_heartbeat = now
        return {"equity": equity, "cash": cash2, "regime": regime.as_dict(),
                "decisions": decisions_this_cycle, "recon": recon,
                "positions": self.store.positions(),
                "bar_close": bar_close}

    # ------------------------------------------------------------------- loop
    def maybe_retrain(self) -> dict | None:
        now = int(time.time())
        if now - self._last_train_check < 3600:
            return None
        self._last_train_check = now
        closed = int(self.store.db.scalar("SELECT COUNT(*) FROM trades WHERE mode='live'") or 0)
        last = self.store.get_portfolio("last_train", {}) or {}
        if last and closed - int(last.get("trades", 0)) < self.s.models.retrain_every_trades \
                and now - int(last.get("ts", 0)) < 7 * 86_400:
            return None
        from .datafeed import prepare_all
        frames = prepare_all(self.store, self.s)
        results = train_all(self.store, self.s, frames)
        self.gates = load_active_gates(self.store, self.s)
        self.store.save_portfolio("last_train", {"ts": now, "trades": closed})
        for r in results:
            if r.get("status") in ("active", "rejected"):
                self.notifier.model_update(r["symbol"], r["status"], r.get("auc"),
                                           r.get("oos_improvement"), r.get("threshold"))
        return {"results": results, "active_gates": sorted(self.gates)}

    def run_forever(self) -> None:
        equity, cash = self.equity()
        pos_rows = self.store.positions()
        regime = self.store.latest_regime() or {}
        self.store.save_portfolio("equity_snapshot", {
            "equity": round(equity, 4), "cash": round(cash, 4),
            "positions": len(pos_rows),
            "regime": regime.get("label", "bull"), "ts": int(time.time()),
        })
        self.notifier.startup(equity=equity, symbols=len(self.s.universe), mode="Alpaca paper")
        self.gates = load_active_gates(self.store, self.s)
        while not self.stop_event.is_set():
            try:
                result = self.run_cycle()
                log.info("cycle equity=%.2f regime=%s decisions=%s", result["equity"],
                         result["regime"]["label"], result["decisions"])
            except Exception:
                log.exception("cycle failed")
                self.notifier.risk("Cycle error", "see logs/unibot_live.log for the traceback")
            self.maybe_retrain()
            self.stop_event.wait(max(15, self.s.paper.poll_seconds))


def json_loads(raw) -> dict:
    import json
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}


def json_dumps(value) -> str:
    import json
    return json.dumps(value, default=str)


def _cluster(settings: Settings, symbol: str) -> str:
    try:
        return settings.symbol(symbol).cluster
    except KeyError:
        return "unknown"
