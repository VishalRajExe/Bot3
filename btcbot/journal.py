"""Trade journal and post-trade analysis.

The journal is the permanent record of every simulated round trip.  Analysis in
this module produces **descriptive findings**, never automatic rule changes:
a losing trade is not evidence of a bad decision, and a winning trade is not
evidence of a good one.  Findings are stored with their sample sizes so a human
can judge them, and the experiment workflow is the only path to changing a
strategy.
"""

from __future__ import annotations

import json
import logging
import uuid

import pandas as pd

from .data.timeframes import to_datetime
from .db import Database, utcnow_iso
from .execution.models import TradeRecord

log = logging.getLogger(__name__)


def row_to_trade(r) -> TradeRecord:
    return TradeRecord(
        id=r["id"], mode=r["mode"], run_id=r["run_id"], symbol=r["symbol"], side=r["side"],
        strategy=r["strategy"], strategy_version=r["strategy_version"],
        entry_ts=r["entry_ts"], entry_bar_ts=r["entry_bar_ts"], entry_price=r["entry_price"],
        entry_signal=r["entry_signal"] or "", entry_reason=r["entry_reason"] or "",
        exit_ts=r["exit_ts"], exit_price=r["exit_price"] or 0.0,
        exit_signal=r["exit_signal"] or "", exit_reason=r["exit_reason"] or "",
        qty=r["qty"], initial_stop=r["initial_stop"], intended_stop=r["intended_stop"],
        stop_distance_pct=r["stop_distance_pct"], gross_pnl=r["gross_pnl"] or 0.0,
        fees=r["fees"] or 0.0, slippage_cost=r["slippage_cost"] or 0.0,
        net_pnl=r["net_pnl"] or 0.0, return_pct=r["return_pct"] or 0.0,
        r_multiple=r["r_multiple"], mfe_pct=r["mfe_pct"] or 0.0, mae_pct=r["mae_pct"] or 0.0,
        holding_hours=r["holding_hours"] or 0.0, regime=r["regime"],
        model_version=r["model_version"],
        indicators_entry=json.loads(r["indicators_entry_json"] or "{}"),
        sentiment_entry=json.loads(r["sentiment_entry_json"] or "{}"),
        risk_events=json.loads(r["risk_events_json"] or "[]"),
        anomalies=json.loads(r["anomalies_json"] or "[]"),
        entry_order_id=r["entry_order_id"], exit_order_id=r["exit_order_id"],
    )


def load_trades(db: Database, *, mode: str | None = None, run_id: str | None = None,
                limit: int | None = None) -> list[TradeRecord]:
    clauses, params = [], []
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    sql = "SELECT * FROM trades"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY exit_ts DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [row_to_trade(r) for r in db.query(sql, params)]


def trade_to_dict(t: TradeRecord) -> dict:
    return {
        "id": t.id, "mode": t.mode, "run_id": t.run_id, "strategy": t.strategy,
        "strategy_version": t.strategy_version, "model_version": t.model_version,
        "entry_ts": t.entry_ts, "entry_time": to_datetime(t.entry_ts).isoformat(),
        "entry_price": t.entry_price, "entry_reason": t.entry_reason,
        "exit_ts": t.exit_ts,
        "exit_time": to_datetime(t.exit_ts).isoformat() if t.exit_ts else None,
        "exit_price": t.exit_price, "exit_reason": t.exit_reason, "qty": t.qty,
        "stop_distance_pct": t.stop_distance_pct, "gross_pnl": round(t.gross_pnl, 4),
        "fees": round(t.fees, 4), "slippage_cost": round(t.slippage_cost, 4),
        "net_pnl": round(t.net_pnl, 4), "return_pct": round(t.return_pct, 4),
        "r_multiple": None if t.r_multiple is None else round(t.r_multiple, 3),
        "mfe_pct": round(t.mfe_pct, 3), "mae_pct": round(t.mae_pct, 3),
        "holding_hours": round(t.holding_hours, 2), "regime": t.regime,
        "indicators_entry": t.indicators_entry, "anomalies": t.anomalies,
    }


def journal_summary(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0}
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "net_pnl": round(sum(t.net_pnl for t in trades), 2),
        "avg_win": round(sum(t.net_pnl for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(t.net_pnl for t in losses) / len(losses), 2) if losses else 0.0,
        "total_fees": round(sum(t.fees for t in trades), 2),
        "exit_reasons": pd.Series([t.exit_reason or "unknown" for t in trades])
        .value_counts().to_dict(),
        "by_regime": pd.Series([t.regime or "unknown" for t in trades]).value_counts().to_dict(),
    }


def review_trades(db: Database, trades: list[TradeRecord], *, persist: bool = True) -> list[dict]:
    """Generate evidence-based observations about losing trades.

    Categories are deliberately narrow and measured, with sample sizes attached.
    """
    findings: list[dict] = []
    losses = [t for t in trades if t.net_pnl <= 0]
    if not trades:
        return findings

    def add(category: str, finding: str, evidence: dict) -> None:
        findings.append({"category": category, "finding": finding, "evidence": evidence})

    stopped = [t for t in losses if t.exit_reason and t.exit_reason.startswith("stop-loss")]
    if stopped:
        recovery = [t for t in stopped if t.mfe_pct <= 1.0]
        if len(stopped) >= 3:
            add("stop_placement",
                f"{len(recovery)}/{len(stopped)} stopped trades never gained more than 1% "
                "before stopping; consider whether entries are late or stops too tight for "
                "current volatility. This is a hypothesis, not a proven cause.",
                {"stopped_trades": len(stopped), "no_followthrough": len(recovery)})

    wide = [t for t in losses if (t.stop_distance_pct or 0) > 0 and t.mae_pct < -0.9 * t.stop_distance_pct]
    if len(wide) >= 3:
        add("stop_distance",
            f"{len(wide)} losing trades reached at least 90% of their stop distance; stop size "
            "may be poorly matched to volatility for this strategy.",
            {"trades": len(wide)})

    if len(losses) >= 3:
        by_regime = pd.Series([t.regime or "unknown" for t in losses]).value_counts()
        top_regime = by_regime.index[0]
        if by_regime.iloc[0] >= max(3, 0.4 * len(losses)):
            add("regime_mismatch",
                f"{by_regime.iloc[0]} of {len(losses)} losing trades occurred in regime "
                f"'{top_regime}'. Test a regime filter as a strategy experiment.",
                {"regime": top_regime, "losses": int(by_regime.iloc[0])})

    fees = sum(t.fees for t in trades)
    gross = sum(t.gross_pnl for t in trades)
    if gross != 0 and abs(fees / gross) > 0.25:
        add("costs",
            f"Fees are {fees / abs(gross) * 100:.1f}% of gross P&L; the edge is cost-sensitive "
            "and higher turnover would likely destroy it.",
            {"fees": round(fees, 2), "gross_pnl": round(gross, 2)})

    short_hold = [t for t in losses if t.holding_hours < 48]
    if len(short_hold) >= 3:
        add("holding_period",
            f"{len(short_hold)} losing trades closed within 48h; check whether exits are too "
            "reactive for the strategy's timeframe.",
            {"trades": len(short_hold)})

    if persist and findings:
        db.executemany(
            "INSERT INTO trade_reviews (trade_id, created_at, category, finding, evidence_json)"
            " VALUES (?,?,?,?,?)",
            [(trades[-1].id, utcnow_iso(), f["category"], f["finding"],
              json.dumps(f["evidence"])) for f in findings],
        )
    return findings


def record_experiment(db: Database, *, name: str, hypothesis: str, baseline_run_id: str | None,
                      candidate_run_id: str | None, verdict: str, notes: str,
                      metrics: dict | None = None) -> str:
    exp_id = f"exp_{uuid.uuid4().hex[:12]}"
    db.execute(
        "INSERT INTO experiments (id, created_at, name, hypothesis, baseline_run_id,"
        " candidate_run_id, verdict, notes, metrics_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (exp_id, utcnow_iso(), name, hypothesis, baseline_run_id, candidate_run_id,
         verdict, notes, json.dumps(metrics or {}, default=str)),
    )
    return exp_id
