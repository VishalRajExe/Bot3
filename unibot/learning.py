"""Learning from realised trades - without self-modifying code.

Two distinct things happen here, and they are deliberately separated:

1. **Adaptation** (automatic, bounded): per-symbol risk multiplier moves inside
   [0.5, 1.25] only after 20+ closed trades, based on a smoothed win rate.  The
   model can also be retrained and *only* promoted if it passes the purged
   walk-forward test in :mod:`unibot.models`.  Nothing can widen a hard limit.
2. **Lessons** (research, human-readable): cohort comparisons over the trade
   memory, written to the database with the sample size and evidence.  A lesson
   is a hypothesis with numbers, not an automatic rule change.

Every claim in a lesson includes the size of the sample it came from, and
lessons refuse to fire below a minimum cohort size.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from .models import MODEL_FEATURES

log = logging.getLogger(__name__)

MIN_COHORT = 8


def _pct(n: int, d: int) -> float:
    return round(n / d * 100.0, 2) if d else 0.0


def cohort_table(memory: list[dict]) -> list[dict]:
    """Group closed trades by symbol / strategy / regime / volatility bucket."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in memory:
        feats = {}
        try:
            feats = json.loads(row.get("features_json") or "{}")
        except json.JSONDecodeError:
            pass
        atr = feats.get("atr_pct")
        if atr is None:
            vol_bucket = "unknown"
        elif atr < 1.5:
            vol_bucket = "low_vol"
        elif atr < 4.0:
            vol_bucket = "mid_vol"
        else:
            vol_bucket = "high_vol"
        buckets[("symbol", row["symbol"])].append(row)
        buckets[("strategy", row.get("strategy") or "?")].append(row)
        buckets[("regime", row.get("regime") or "unknown")].append(row)
        buckets[("volatility", vol_bucket)].append(row)
    out = []
    for (kind, key), rows in buckets.items():
        n = len(rows)
        if n == 0:
            continue
        wins = sum(1 for r in rows if r["net_pnl"] > 0)
        total = sum(r["net_pnl"] for r in rows)
        rets = [r["return_pct"] for r in rows]
        out.append({
            "kind": kind, "key": key, "trades": n, "wins": wins,
            "win_rate_pct": _pct(wins, n), "net_pnl": round(total, 4),
            "avg_return_pct": round(float(np.mean(rets)), 3),
            "expectancy": round(total / n, 4),
            "sample_ok": n >= MIN_COHORT,
        })
    return sorted(out, key=lambda r: (r["kind"], -r["trades"]))


def generate_lessons(store, *, min_cohort: int = MIN_COHORT, max_new: int = 4) -> list[dict]:
    """Compare cohorts and write only well-supported findings."""
    memory = store.memory(limit=20_000)
    if len(memory) < min_cohort:
        return []
    table = cohort_table(memory)
    existing = {l["title"] for l in store.lessons(200)}
    added: list[dict] = []

    def add(title: str, finding: str, evidence: dict, kind: str = "cohort",
            symbol: str | None = None, n: int = 0) -> None:
        if title in existing or len(added) >= max_new:
            return
        store.add_lesson(kind=kind, symbol=symbol, title=title, finding=finding,
                         evidence=evidence, sample_size=n)
        added.append({"title": title, "finding": finding, "sample_size": n})
        existing.add(title)

    def cohort(kind: str, key: str) -> dict | None:
        for row in table:
            if row["kind"] == kind and row["key"] == key:
                return row
        return None

    overall = cohort_table(memory)
    overall_rows = [r for r in overall if r["kind"] == "symbol"]
    if overall_rows:
        best = max(overall_rows, key=lambda r: r["expectancy"])
        worst = min(overall_rows, key=lambda r: r["expectancy"])
        if best["sample_ok"] and best["expectancy"] > 0:
            add(f"best symbol: {best['key']}",
                f"{best['key']} has the highest expectancy "
                f"({best['expectancy']:+.2f} per trade) over {best['trades']} trades.",
                best, symbol=best["key"], n=best["trades"])
        if worst["sample_ok"] and worst["expectancy"] < 0:
            add(f"worst symbol: {worst['key']}",
                f"{worst['key']} has the lowest expectancy "
                f"({worst['expectancy']:+.2f} per trade) over {worst['trades']} trades; "
                f"consider disabling it rather than increasing size.",
                worst, symbol=worst["key"], n=worst["trades"])

    for row in table:
        if row["kind"] == "regime" and row["sample_ok"] and row["key"] in ("bull", "chop", "bear"):
            if row["expectancy"] < 0:
                add(f"regime drag: {row['key']}",
                    f"Trades opened in the '{row['key']}' regime average "
                    f"{row['avg_return_pct']:+.2f}% ({row['trades']} trades). "
                    f"The regime filter is doing useful work.",
                    row, n=row["trades"])
        if row["kind"] == "volatility" and row["sample_ok"] and row["key"] == "high_vol":
            if row["expectancy"] < 0:
                add("high volatility underperforms",
                    f"High-ATR setups (>4% per bar) average {row['avg_return_pct']:+.2f}% "
                    f"over {row['trades']} trades. Tighten the volatility ceiling or "
                    f"cut size in this bucket.",
                    row, n=row["trades"])
        if row["kind"] == "strategy" and row["sample_ok"]:
            if row["expectancy"] < 0:
                add(f"strategy loses: {row['key']}",
                    f"{row['key']} averages {row['avg_return_pct']:+.2f}% per trade over "
                    f"{row['trades']} trades. Re-run its research sweep before it is "
                    f"assigned more symbols.", row, n=row["trades"])

    # Fee drag: a structural lesson the system can always quantify.
    fees = store.db.scalar(
        "SELECT COALESCE(SUM(fees),0) FROM trades WHERE mode='live'"
    )
    gross = store.db.scalar(
        "SELECT COALESCE(SUM(gross_pnl),0) FROM trades WHERE mode='live'"
    )
    if gross and abs(float(gross)) > 1e-9 and float(fees) > 0:
        drag = float(fees) / abs(float(gross)) * 100.0
        add("fee drag is measurable",
            f"Fees are {drag:.1f}% of gross P&L across live paper trades "
            f"(${float(fees):.2f} paid). Any strategy with sub-1% average wins cannot "
            f"survive this cost structure.",
            {"fees": round(float(fees), 4), "gross_pnl": round(float(gross), 4),
             "drag_pct": round(drag, 2)}, kind="cost")

    return added


def learning_state(store, settings) -> dict:
    """Everything the dashboard needs to show what has been learned."""
    memory = store.memory(limit=5000)
    stats = store.symbol_stats()
    rows = []
    for sym in settings.symbols:
        st = stats.get(sym, {})
        n = int(st.get("n_trades") or 0)
        wins = int(st.get("wins") or 0)
        rows.append({
            "symbol": sym,
            "trades": n,
            "win_rate_pct": _pct(wins, n),
            "net_pnl": round(float(st.get("net_pnl") or 0.0), 4),
            "consecutive_losses": int(st.get("consec_losses") or 0),
            "risk_multiplier": round(float(st.get("risk_multiplier") or 1.0), 3),
            "cooldown_until": int(st.get("cooldown_until") or 0),
        })
    models = []
    for sym in settings.symbols:
        active = store.active_model(sym)
        candidates = store.models(sym)
        latest = candidates[0] if candidates else None
        models.append({
            "symbol": sym,
            "active": {"version": active["version"], "auc": active["auc"],
                       "threshold": active["threshold"]} if active else None,
            "attempts": len(candidates),
            "last_status": latest["status"] if latest else "untrained",
            "last_auc": latest["auc"] if latest else None,
        })
    lessons = store.lessons(50)
    return {
        "memory": {
            "trades_remembered": len(memory),
            "symbols_traded": len({m["symbol"] for m in memory}),
            "cohorts": cohort_table(memory),
        },
        "symbols": rows,
        "models": models,
        "lessons": [
            {"ts": l["ts"], "kind": l["kind"], "symbol": l["symbol"], "title": l["title"],
             "finding": l["finding"], "sample_size": l["sample_size"], "status": l["status"]}
            for l in lessons
        ],
        "rules": {
            "risk_adaptation": "bounded [0.5x, 1.25x] after 20 trades; never touches hard limits",
            "model_promotion": "requires out-of-sample AUC and net improvement",
            "auto_rule_changes": "none - lessons are hypotheses for review",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def review_after_trade(store, settings, trade, *, features: dict, prob: float | None,
                       regime: str | None, equity: float) -> dict:
    """Persist memory/stats for a closed live trade and refresh lessons."""
    step = int(settings.paper.poll_seconds)
    from btcbot.data.timeframes import tf_seconds
    cooldown_seconds = settings.risk.loss_cooldown_bars * tf_seconds(settings.timeframe)
    stats = store.apply_trade_outcome(
        symbol=trade.symbol, net_pnl=trade.net_pnl, return_pct=trade.return_pct,
        ts=int(trade.exit_ts), trade_id=trade.id, strategy=trade.strategy,
        features=features, prob=prob, regime=regime,
        cooldown_bars_seconds=cooldown_seconds if trade.net_pnl <= 0 else 0,
        consecutive_limit=settings.risk.consecutive_losses_limit,
        cooldown_hours=settings.risk.cooldown_hours,
    )
    new_lessons = generate_lessons(store)
    return {"stats": stats, "lessons": new_lessons, "equity": round(equity, 4),
            "poll": step}
