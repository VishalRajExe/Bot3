"""Event-driven, multi-asset portfolio backtester.

Modelling choices (all deliberately conservative):

* signals are evaluated on a **closed** bar and executed at the **next** bar's
  open, so no bar's high/low can be used before it exists;
* stops are checked against the bar's low and fill at ``min(open, stop)``, so a
  gap through the stop costs more than the stop, never less;
* fees are charged on both sides at the Alpaca crypto taker rate plus half the
  spread plus slippage on every fill;
* one shared cash balance and one risk manager govern all symbols, so the
  backtest measures the portfolio the account would actually have run;
* positions are long-only spot: no leverage, no shorting, no averaging down.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcbot.backtest.metrics import compute_metrics, max_drawdown, sharpe_sortino
from btcbot.data.timeframes import tf_seconds
from btcbot.execution.models import TradeRecord

from .config import Settings
from .regime import RegimeState
from .risk import OpenRisk, RiskManager
from .strategies import PositionView, Strategy, build as build_strategy


@dataclass
class OpenPosition:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: int
    entry_bar_ts: int
    stop: float
    initial_stop: float
    trail_atr: float
    strategy: str
    strategy_version: str
    model_version: str | None
    prob: float | None
    risk_usd: float
    entry_fees: float
    entry_slippage: float
    entry_reason: str
    entry_tag: str
    regime: str
    features: dict = field(default_factory=dict)
    max_price: float = 0.0
    min_price: float = float("inf")

    def mark(self, high: float, low: float) -> None:
        self.max_price = max(self.max_price, high)
        self.min_price = min(self.min_price, low)

    @property
    def mfe_pct(self) -> float:
        return (self.max_price - self.entry_price) / self.entry_price * 100.0

    @property
    def mae_pct(self) -> float:
        return (self.min_price - self.entry_price) / self.entry_price * 100.0


@dataclass
class PendingOrder:
    side: str
    symbol: str
    size_usd: float
    stop: float
    trail_atr: float
    reason: str
    tag: str
    prob: float | None
    strategy: str
    strategy_version: str
    signal_bar_ts: int
    features: dict
    regime: str


@dataclass
class PortfolioResult:
    run_id: str
    equity: pd.DataFrame
    trades: list[TradeRecord]
    metrics: dict
    decisions: list[dict]
    risk_events: list[dict]
    per_symbol: dict
    per_strategy: dict
    benchmarks: dict
    final_equity: float


class _Ledger:
    """Cash + positions with Alpaca-style fees and slippage."""

    def __init__(self, settings: Settings, initial_equity: float) -> None:
        self.s = settings
        self.cash = float(initial_equity)
        self.initial = float(initial_equity)
        self.positions: dict[str, OpenPosition] = {}
        self.fees = 0.0
        self.slippage = 0.0

    @property
    def slip_rate(self) -> float:
        return (self.s.costs.spread_bps / 2.0 + self.s.costs.slippage_bps) / 10_000.0

    def equity(self, marks: dict[str, float]) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            value += pos.qty * marks.get(sym, pos.entry_price)
        return value

    def buy(self, *, symbol: str, size_usd: float, open_price: float, ts: int,
            bar_ts: int, stop: float, trail_atr: float, strategy: str, version: str,
            model_version: str | None, prob: float | None, risk_usd: float,
            reason: str, tag: str, regime: str, features: dict) -> OpenPosition:
        price = open_price * (1.0 + self.slip_rate)
        qty = size_usd / price
        notional = qty * price
        fee = notional * self.s.costs.taker_fee_bps / 10_000.0
        if notional + fee > self.cash + 1e-9:
            raise ValueError(f"insufficient cash: need {notional + fee:.4f}, have {self.cash:.4f}")
        self.cash -= notional + fee
        self.fees += fee
        slip_cost = qty * (price - open_price)
        self.slippage += slip_cost
        pos = OpenPosition(
            symbol=symbol, qty=qty, entry_price=price, entry_ts=ts + tf_seconds(self.s.timeframe),
            entry_bar_ts=bar_ts, stop=stop, initial_stop=stop, trail_atr=trail_atr,
            strategy=strategy, strategy_version=version, model_version=model_version,
            prob=prob, risk_usd=risk_usd, entry_fees=fee, entry_slippage=slip_cost,
            entry_reason=reason, entry_tag=tag, regime=regime, features=features,
            max_price=price, min_price=price,
        )
        self.positions[symbol] = pos
        return pos

    def sell(self, *, symbol: str, price_ref: float, ts: int
             ) -> tuple[OpenPosition, float, float]:
        pos = self.positions.pop(symbol)
        price = price_ref * (1.0 - self.slip_rate)
        notional = pos.qty * price
        fee = notional * self.s.costs.taker_fee_bps / 10_000.0
        self.cash += notional - fee
        self.fees += fee
        self.slippage += pos.qty * (price_ref - price)
        return pos, price, fee


def _trade_record(pos: OpenPosition, *, price: float, fee: float, ts: int, reason: str,
                  run_id: str, mode: str, exit_signal: str) -> TradeRecord:
    gross = (price - pos.entry_price) * pos.qty
    fees = pos.entry_fees + fee
    net = gross - fees
    denom = pos.entry_price * pos.qty
    return TradeRecord(
        id=f"trd_{uuid.uuid4().hex[:20]}",
        mode=mode,
        run_id=run_id,
        symbol=pos.symbol,
        side="long",
        strategy=pos.strategy,
        strategy_version=pos.strategy_version,
        entry_ts=pos.entry_ts,
        entry_bar_ts=pos.entry_bar_ts,
        entry_price=pos.entry_price,
        entry_signal=f"enter_long@{pos.entry_bar_ts}",
        entry_reason=pos.entry_reason,
        exit_ts=ts,
        exit_price=price,
        exit_signal=exit_signal,
        exit_reason=reason,
        qty=pos.qty,
        initial_stop=pos.initial_stop,
        intended_stop=pos.stop,
        stop_distance_pct=(pos.entry_price - pos.initial_stop) / pos.entry_price * 100.0,
        gross_pnl=gross,
        fees=fees,
        slippage_cost=pos.entry_slippage,
        net_pnl=net,
        return_pct=net / denom * 100.0 if denom else 0.0,
        r_multiple=net / pos.risk_usd if pos.risk_usd > 0 else None,
        mfe_pct=pos.mfe_pct,
        mae_pct=pos.mae_pct,
        holding_hours=(ts - pos.entry_ts) / 3600.0,
        regime=pos.regime,
        model_version=pos.model_version,
        indicators_entry={**pos.features, "model_prob": pos.prob, "entry_tag": pos.entry_tag},
    )


def summarize_trades(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate_pct": None, "net_pnl": 0.0, "avg_return_pct": None,
                "profit_factor": None, "expectancy": None, "avg_r": None,
                "avg_hold_hours": None}
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gp = sum(t.net_pnl for t in wins)
    gl = abs(sum(t.net_pnl for t in losses))
    r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
    return {
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 2),
        "net_pnl": round(sum(t.net_pnl for t in trades), 4),
        "avg_return_pct": round(float(np.mean([t.return_pct for t in trades])), 4),
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "expectancy": round(float(np.mean([t.net_pnl for t in trades])), 4),
        "avg_r": round(float(np.mean(r_multiples)), 3) if r_multiples else None,
        "avg_hold_hours": round(float(np.mean([t.holding_hours for t in trades])), 1),
    }


def _cluster(settings: Settings, symbol: str) -> str:
    try:
        return settings.symbol(symbol).cluster
    except KeyError:
        return "unknown"


def _exit_mix(trades: list[TradeRecord]) -> dict:
    mix: dict[str, int] = {}
    for t in trades:
        key = "stop" if "stop" in t.exit_signal else (
            "strategy" if t.exit_signal == "exit_long" else t.exit_signal)
        mix[key] = mix.get(key, 0) + 1
    return mix


def run_portfolio_backtest(
    settings: Settings,
    frames: dict[str, pd.DataFrame],
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
    initial_equity: float | None = None,
    run_id: str | None = None,
    mode: str = "backtest",
    store=None,
    label: str = "portfolio",
    gates: dict | None = None,
    strategies: dict[str, Strategy] | None = None,
) -> PortfolioResult:
    tf = settings.timeframe
    step = tf_seconds(tf)
    equity0 = float(initial_equity if initial_equity is not None else settings.capital)
    run_id = run_id or f"uni_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    strategies = strategies or {s.symbol: build_strategy(s) for s in settings.universe}
    tradable = {s.symbol for s in settings.tradable}
    frames = {sym: df.reset_index(drop=True) for sym, df in frames.items() if not df.empty}
    if not frames:
        raise ValueError("no bars to backtest")

    window_start = int(start_ts) if start_ts else None
    window_end = int(end_ts) if end_ts else None

    ledger = _Ledger(settings, equity0)
    risk = RiskManager(settings.risk, settings.costs)
    positions = ledger.positions
    pending: dict[str, PendingOrder] = {}
    trades: list[TradeRecord] = []
    decisions: list[dict] = []
    risk_events: list[dict] = []
    equity_rows: list[tuple] = []

    hwm = equity0
    day: int | None = None
    day_start = equity0
    halted_dd = False
    cooldown_until: dict[str, int] = {s.symbol: 0 for s in settings.universe}
    last_close: dict[str, float] = {}

    index_by_ts: dict[str, dict[int, int]] = {
        sym: {int(t): i for i, t in enumerate(df["bar_ts"].to_numpy("int64"))}
        for sym, df in frames.items()
    }
    timeline = sorted({int(t) for df in frames.values()
                       for t in df["bar_ts"].to_numpy("int64")})
    if window_start is not None:
        timeline = [t for t in timeline if t + step >= window_start]
    if window_end is not None:
        timeline = [t for t in timeline if t + step <= window_end]
    if not timeline:
        raise ValueError("evaluation window contains no bars")

    def marks_at(ts: int) -> dict[str, float]:
        marks = dict(last_close)
        for sym, idx in index_by_ts.items():
            j = idx.get(ts)
            if j is not None:
                marks[sym] = float(frames[sym]["close"].iloc[j])
        return marks

    def decide(ts: int, symbol: str, action: str, reason: str, price: float | None,
               strategy: str, kind: str, **extra) -> None:
        decisions.append({"ts": ts, "symbol": symbol, "action": action, "reason": reason,
                          "price": price, "strategy": strategy, "kind": kind, **extra})

    for t in timeline:
        marks = marks_at(t)
        equity_now = ledger.equity(marks)
        daytime = t // 86_400
        if day is None or daytime != day:
            day = daytime
            day_start = equity_now
        hwm = max(hwm, equity_now)
        breaker = risk.breaker_state(equity=equity_now, hwm=hwm, day_start=day_start)
        if breaker["drawdown_halted"] and not halted_dd:
            halted_dd = True
            risk_events.append({"ts": t + step, "kind": "drawdown_halt",
                                "action": "block_new_entries",
                                "detail": f"drawdown {breaker['drawdown_pct']}% >= "
                                          f"{settings.risk.max_drawdown_pct}%",
                                "value": breaker})
        if halted_dd:
            breaker["drawdown_halted"] = True
        if breaker["daily_halted"]:
            risk_events.append({"ts": t + step, "kind": "daily_halt",
                                "action": "block_new_entries",
                                "detail": f"daily loss {breaker['daily_loss_pct']}%",
                                "value": breaker})

        # ------------------------------------------------- 1. execute at the open
        for sym in list(pending):
            order = pending[sym]
            j = index_by_ts.get(sym, {}).get(t)
            if j is None:
                continue  # missing bar: keep the order for the next available open
            del pending[sym]
            row = frames[sym].iloc[j]
            open_px = float(row["open"])
            if open_px <= 0 or not math.isfinite(open_px):
                continue

            if order.side == "sell":
                if sym not in positions:
                    continue
                pos, price, fee = ledger.sell(symbol=sym, price_ref=open_px, ts=t + step)
                trades.append(_trade_record(pos, price=price, fee=fee, ts=t + step,
                                            reason=order.reason, run_id=run_id, mode=mode,
                                            exit_signal="exit_long"))
                decide(t, sym, "exit_filled", order.reason, price, order.strategy, "exit")
                continue

            if sym in positions:
                continue
            equity_now = ledger.equity(marks)
            open_positions = [{"symbol": s, "cluster": _cluster(settings, s)} for s in positions]
            open_risk = [OpenRisk(symbol=s, cluster=_cluster(settings, s),
                                  notional=p.qty * p.entry_price, risk_usd=p.risk_usd,
                                  stop_pct=(p.entry_price - p.stop) / p.entry_price * 100.0)
                         for s, p in positions.items()]
            regime = RegimeState(order.regime, 0.0, 0.0, 0.5, 0.5)
            gate = risk.can_enter(
                symbol=sym, cluster=_cluster(settings, sym), equity=equity_now,
                cash=ledger.cash, open_positions=open_positions, open_risk=open_risk,
                regime=regime, breaker=breaker, cooldown_until=cooldown_until.get(sym, 0),
                now=t, bar_return_pct=0.0,
            )
            size_usd = order.size_usd
            stop_pct = max(0.0, (open_px - order.stop) / open_px)
            risk_usd = size_usd * stop_pct
            heat_left = risk.remaining_heat(equity_now, open_risk)
            if gate.approved and risk_usd > heat_left + 1e-9:
                scale = heat_left / risk_usd if risk_usd > 0 else 0.0
                size_usd *= max(0.0, min(1.0, scale))
                risk_usd *= max(0.0, min(1.0, scale))
                if size_usd < settings.costs.min_notional:
                    gate = type(gate)(False, "portfolio heat cap leaves less than the minimum order")
            if not gate.approved:
                decide(t, sym, "blocked", gate.reason, open_px, order.strategy, "entry")
                continue
            try:
                ledger.buy(
                    symbol=sym, size_usd=size_usd, open_price=open_px, ts=t,
                    bar_ts=order.signal_bar_ts, stop=order.stop, trail_atr=order.trail_atr,
                    strategy=order.strategy, version=order.strategy_version,
                    model_version=None, prob=order.prob, risk_usd=risk_usd,
                    reason=order.reason, tag=order.tag, regime=order.regime,
                    features=order.features,
                )
                decide(t, sym, "buy_filled", order.reason, open_px, order.strategy, "entry",
                       size_usd=round(size_usd, 4), stop=order.stop)
            except ValueError as exc:
                decide(t, sym, "blocked", str(exc), open_px, order.strategy, "entry")

        # ------------------------------------------- 2. manage each symbol's bar
        for sym, idx in index_by_ts.items():
            j = idx.get(t)
            if j is None:
                continue
            df = frames[sym]
            row = df.iloc[j]
            close = float(row["close"])
            last_close[sym] = close
            bar_ts = int(row["bar_ts"])

            if sym in positions:
                pos = positions[sym]
                pos.mark(float(row["high"]), float(row["low"]))
                if float(row["low"]) <= pos.stop:
                    fill_ref = min(float(row["open"]), pos.stop)
                    stopped = pos.stop
                    pos_out, price, fee = ledger.sell(symbol=sym, price_ref=fill_ref, ts=t + step)
                    trades.append(_trade_record(
                        pos_out, price=price, fee=fee, ts=t + step,
                        reason=f"stop_loss at {stopped:.6g}", run_id=run_id, mode=mode,
                        exit_signal="stop"))
                    decide(t, sym, "stop_filled", f"low {float(row['low']):.6g} <= stop {stopped:.6g}",
                           price, pos_out.strategy, "exit")
                    cooldown_until[sym] = max(
                        cooldown_until.get(sym, 0),
                        t + settings.risk.loss_cooldown_bars * step,
                    )
                    continue

            if sym not in strategies or sym not in tradable:
                continue
            pos_view = None
            if sym in positions:
                p = positions[sym]
                pos_view = PositionView(qty=p.qty, entry_price=p.entry_price,
                                        entry_ts=p.entry_ts, stop_price=p.stop,
                                        entry_bar_ts=p.entry_bar_ts, trail_atr=p.trail_atr)
            signal = strategies[sym].signal(df, j, pos_view)
            label_regime = str(row.get("market_label") or "chop")
            regime = RegimeState(label_regime, float(row.get("market_score") or 0.0),
                                 float(row.get("btc_trend") or 0.0),
                                 float(row.get("breadth") or 0.5), 0.5)

            if signal.action == "exit_long" and sym in positions:
                pending[sym] = PendingOrder(
                    side="sell", symbol=sym, size_usd=0.0, stop=0.0, trail_atr=0.0,
                    reason=signal.reason, tag=signal.tag, prob=None,
                    strategy=strategies[sym].name, strategy_version=strategies[sym].version,
                    signal_bar_ts=bar_ts, features=signal.indicators, regime=regime.label,
                )
                continue

            if sym in positions:
                pos = positions[sym]
                new_stop = pos.stop
                init_risk = pos.entry_price - pos.initial_stop
                if init_risk > 0:
                    # Break-even stop ratchet once in +1.2R profit (covers round-trip taker fees)
                    if pos.max_price >= pos.entry_price + 1.2 * init_risk:
                        be_stop = pos.entry_price * 1.002
                        new_stop = max(new_stop, be_stop)
                    # Profit-lock ratchet once in +2.2R profit (locks in at least +0.8R)
                    if pos.max_price >= pos.entry_price + 2.2 * init_risk:
                        p_lock = pos.entry_price + 0.8 * init_risk
                        new_stop = max(new_stop, p_lock)
                if pos.trail_atr > 0:
                    atr_now = float(row["atr14"])
                    if math.isfinite(atr_now):
                        new_stop = max(new_stop, close - pos.trail_atr * atr_now)
                if new_stop > pos.stop:
                    pos.stop = new_stop

            if (signal.action == "enter_long" and sym not in positions and sym not in pending
                    and not halted_dd and not breaker["daily_halted"]):
                prob = None
                gate_reason = None
                if gates and sym in gates:
                    g = gates[sym]
                    prob = g.predict(row)
                    if prob < g.threshold:
                        gate_reason = (f"model gate: P(win)={prob:.2f} < "
                                       f"threshold {g.threshold:.2f}")
                if gate_reason:
                    decide(t, sym, "blocked", gate_reason, close,
                           strategies[sym].name, "model", prob=prob)
                    continue
                if regime.label == "bear":
                    decide(t, sym, "blocked",
                           f"bear regime (score {regime.score:.2f}): long-only stays in cash",
                           close, strategies[sym].name, "regime")
                    continue
                symbol_mult = 1.0
                if store is not None:
                    symbol_mult = float(store.stats_row(sym).get("risk_multiplier", 1.0))
                sizing = risk.position_size(
                    equity=ledger.equity(marks), cash=ledger.cash, price=close,
                    stop=signal.stop_price, regime=regime, symbol_multiplier=symbol_mult,
                    prob=prob,
                )
                if not sizing.approved:
                    decide(t, sym, "blocked", sizing.reason, close,
                           strategies[sym].name, "sizing")
                    continue
                pending[sym] = PendingOrder(
                    side="buy", symbol=sym, size_usd=sizing.size_usd, stop=signal.stop_price,
                    trail_atr=signal.trail_atr, reason=signal.reason, tag=signal.tag,
                    prob=prob, strategy=strategies[sym].name,
                    strategy_version=strategies[sym].version, signal_bar_ts=bar_ts,
                    features=signal.indicators, regime=regime.label,
                )
                decide(t, sym, "buy_signal", signal.reason, close, strategies[sym].name,
                       "entry", size_usd=round(sizing.size_usd, 4), stop=signal.stop_price,
                       prob=prob)

        # ------------------------------------------------ 3. mark to market
        marks = marks_at(t)
        equity = ledger.equity(marks)
        hwm = max(hwm, equity)
        dd = (hwm - equity) / hwm * 100.0 if hwm else 0.0
        invested = sum(p.qty * marks.get(s, p.entry_price) for s, p in positions.items())
        equity_rows.append((run_id, t + step, equity, ledger.cash, invested,
                            invested / equity * 100.0 if equity else 0.0, dd, len(positions)))

    # ------------------------------------------------------------ final close
    last_t = timeline[-1] + step
    for sym in list(positions):
        pos = positions[sym]
        px = last_close.get(sym, pos.entry_price)
        pos_out, price, fee = ledger.sell(symbol=sym, price_ref=px, ts=last_t)
        trades.append(_trade_record(pos_out, price=price, fee=fee, ts=last_t,
                                    reason="end of evaluation window", run_id=run_id,
                                    mode=mode, exit_signal="forced_exit"))

    equity_df = pd.DataFrame(equity_rows, columns=[
        "run_id", "ts", "equity", "cash", "invested", "gross_exposure", "drawdown", "positions"
    ])
    total_bars = max(len(equity_df), 1)
    bars_in_market = int((equity_df["positions"] > 0).sum())
    metrics = compute_metrics(
        equity_df=equity_df[["ts", "equity"]], trades=trades, initial_equity=equity0,
        timeframe=tf, bars_in_market=bars_in_market, total_bars=total_bars,
    )
    metrics["total_fees"] = round(ledger.fees, 4)
    metrics["total_slippage_cost"] = round(ledger.slippage, 4)
    metrics["exit_mix"] = _exit_mix(trades)
    metrics["capital"] = equity0

    per_symbol = {sym: summarize_trades([t for t in trades if t.symbol == sym])
                  for sym in settings.symbols}
    per_strategy = {name: summarize_trades([t for t in trades if t.strategy == name])
                    for name in sorted({t.strategy for t in trades})}

    benchmarks = _benchmarks(settings, frames, equity0, window_start or int(equity_df["ts"].iloc[0]))
    metrics["benchmarks"] = benchmarks
    metrics["per_symbol"] = per_symbol
    metrics["per_strategy"] = per_strategy

    result = PortfolioResult(
        run_id=run_id, equity=equity_df, trades=trades, metrics=metrics,
        decisions=decisions, risk_events=risk_events, per_symbol=per_symbol,
        per_strategy=per_strategy, benchmarks=benchmarks,
        final_equity=float(equity_df["equity"].iloc[-1]),
    )

    if store is not None:
        store.save_run(
            run_id=run_id, kind="portfolio_backtest", label=label,
            start_ts=int(equity_df["ts"].iloc[0]), end_ts=int(equity_df["ts"].iloc[-1]),
            initial_equity=equity0, final_equity=result.final_equity, metrics=metrics,
            params={"universe": [s.model_dump() for s in settings.universe],
                    "risk": settings.risk.model_dump(), "costs": settings.costs.model_dump(),
                    "gates": sorted(gates) if gates else []},
            symbols=list(frames), notes=f"{len(trades)} closed trades",
        )
        store.save_equity(run_id, [tuple(r) for r in equity_df.itertuples(index=False)])
        for trade in trades:
            store.save_trade(trade)
        for ev in risk_events:
            store.db.record_risk_event(ts=ev["ts"], mode=mode, run_id=run_id, kind=ev["kind"],
                                       action=ev["action"], detail=ev["detail"],
                                       value=ev.get("value"))
    return result


def _benchmarks(settings: Settings, frames: dict[str, pd.DataFrame],
                equity0: float, start_ts: int) -> dict:
    """Equal-weight buy-and-hold across the same universe, plus BTC only."""
    cost = (settings.costs.taker_fee_bps + settings.costs.spread_bps / 2
            + settings.costs.slippage_bps) / 10_000.0
    curve_frames = []
    for sym, df in frames.items():
        if df.empty:
            continue
        d = df[["bar_ts", "close"]].copy()
        d["known"] = d["bar_ts"] + tf_seconds(settings.timeframe)
        d = d[d["known"] >= start_ts]
        if d.empty:
            continue
        d[sym] = d["close"] / float(d["close"].iloc[0]) * (1.0 - cost)
        curve_frames.append(d[["known", sym]].set_index("known"))
    if not curve_frames:
        return {}
    combined = pd.concat(curve_frames, axis=1).sort_index().ffill()
    equal = combined.mean(axis=1) * equity0
    out = {
        "equal_weight": {
            "name": "Equal-weight buy & hold (same window)",
            "final_equity": round(float(equal.iloc[-1]), 2),
            "net_return_pct": round((float(equal.iloc[-1]) / equity0 - 1) * 100, 3),
            "max_drawdown_pct": round(max_drawdown(equal.to_numpy(float))[0], 3),
        }
    }
    if "BTC/USD" in combined:
        btc = combined["BTC/USD"] * equity0
        rets = btc.pct_change().dropna()
        sharpe, _ = sharpe_sortino(rets)
        out["btc_hold"] = {
            "name": "BTC buy & hold (same window)",
            "final_equity": round(float(btc.iloc[-1]), 2),
            "net_return_pct": round((float(btc.iloc[-1]) / equity0 - 1) * 100, 3),
            "max_drawdown_pct": round(max_drawdown(btc.to_numpy(float))[0], 3),
            "sharpe": sharpe,
        }
    return out
