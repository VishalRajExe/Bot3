"""Controlled experiments.

Rules of engagement
-------------------
* The baseline is never overwritten: every candidate gets its own run id and its
  own parameters.
* Acceptance looks at the **last 30% of the data only** (out-of-sample), requires
  a minimum trade count, and requires that drawdown does not deteriorate.
* A candidate that looks better on the full sample but not out-of-sample is
  rejected.  Results are recorded even when they are bad.
* Parameter grids are reported as sensitivity, not as an optimisation exercise.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.engine import RunResult, run_backtest
from .config import Config
from .data.ingest import CandleIngestor
from .db import Database
from .indicators import atr
from .journal import record_experiment
from .ml.train import (predictions_frame, promotion_decision, register_model,
                       walk_forward_predictions)
from .reporting import write_comparison, write_run_artifacts
from .sentiment.features import attach_fear_greed
from .strategies import build_strategy
from .strategies.base import Params, PositionView, Signal, Strategy
from .strategies.ml import MLGated

log = logging.getLogger(__name__)

OOS_FRACTION = 0.30
MIN_OOS_TRADES = 20


class FearGreedParams(Params):
    enter_below: float = 25.0
    exit_above: float = 55.0
    atr_window: int = 14
    stop_atr_mult: float = 3.0


class FearGreedContrarian(Strategy):
    """Experimental sentiment-only rule: buy extreme fear, sell greed."""

    name = "fear_greed_contrarian"
    version = "0.1.0-exp"
    default_timeframe = "1d"
    description = ("Experimental: long when the point-in-time Fear & Greed index is in extreme "
                   "fear, exit in greed. Used only to test whether sentiment carries information.")
    PARAMS = FearGreedParams

    @property
    def warmup_bars(self) -> int:
        return self.params.atr_window + 2

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df["high"], df["low"], df["close"], self.params.atr_window)
        return df

    def _signal(self, df: pd.DataFrame, i: int, position: PositionView | None) -> Signal:
        row = df.iloc[i]
        fg = self._val(row, "fg_value")
        close = float(row["close"])
        atr_val = self._val(row, "atr")
        if fg is None:
            return Signal.hold("no point-in-time sentiment available")
        ind = {"close": close, "fg_value": fg, "atr": atr_val}
        if position is None and fg <= self.params.enter_below:
            stop = close - self.params.stop_atr_mult * atr_val if atr_val else None
            return Signal("enter_long", f"extreme fear (F&G {fg:.0f} <= {self.params.enter_below})",
                          stop, ind, tag="sentiment")
        if position is not None and fg >= self.params.exit_above:
            return Signal("exit_long", f"sentiment recovered (F&G {fg:.0f} >= {self.params.exit_above})",
                          None, ind, tag="sentiment")
        return Signal.hold(f"F&G {fg:.0f}: no sentiment setup", **ind)


def _run(cfg: Config, db: Database, strategy: Strategy, df: pd.DataFrame, timeframe: str,
         notes: str, start_ts: int | None = None) -> RunResult:
    return run_backtest(cfg, db, strategy, df, timeframe=timeframe, persist=True,
                        start_ts=start_ts, notes=notes)


def _split_ts(df: pd.DataFrame, fraction: float = OOS_FRACTION) -> int:
    idx = int(len(df) * (1 - fraction))
    return int(df["bar_ts"].iloc[idx])


def evaluate_candidate(baseline_full: RunResult, candidate_full: RunResult,
                       baseline_oos: RunResult, candidate_oos: RunResult) -> tuple[str, str]:
    """Documented acceptance rule for a proposed improvement."""
    b, c = baseline_oos.metrics, candidate_oos.metrics
    if (c.get("trades") or 0) < MIN_OOS_TRADES:
        return "inconclusive", (f"out-of-sample trades {c.get('trades')} below the required "
                                f"{MIN_OOS_TRADES}; the sample is too small to judge, "
                                "regardless of the return difference")
    if (c.get("net_return_pct") or 0) <= (b.get("net_return_pct") or 0):
        return "rejected", (f"out-of-sample return {c.get('net_return_pct')}% did not beat "
                            f"baseline {b.get('net_return_pct')}%")
    if (c.get("max_drawdown_pct") or 0) > (b.get("max_drawdown_pct") or 0) + 2.0:
        return "rejected", (f"out-of-sample drawdown {c.get('max_drawdown_pct')}% is more than "
                            "2pp worse than baseline")
    if (candidate_full.metrics.get("net_return_pct") or 0) < (
            baseline_full.metrics.get("net_return_pct") or 0) - 2.0:
        return "rejected", "full-sample return is materially worse; improvement is not robust"
    return "accepted", (f"OOS return {c.get('net_return_pct')}% vs baseline "
                        f"{b.get('net_return_pct')}%, drawdown within tolerance. "
                        "Accepted for paper-trading comparison, not proven profitable.")


def run_core_suite(cfg: Config, db: Database) -> dict[str, RunResult]:
    """Backtest every registered strategy in its intended timeframe."""
    ing = CandleIngestor(cfg, db)
    data = {tf: ing.load_history(tf) for tf in cfg.data.timeframes}
    specs = [
        ("ema_std", "1d", {}, "baseline: EMA-65 std-band reconstruction"),
        ("ema_std", "4h", {}, "baseline: EMA-65 std-band on 4h"),
        ("trend_ema", "1d", {}, "trend following EMA 50/200"),
        ("breakout_donchian", "4h", {}, "Donchian breakout"),
        ("meanrev_bb_rsi", "4h", {}, "Bollinger/RSI mean reversion"),
        ("regime_adaptive", "4h", {}, "regime-aware switch"),
    ]
    results: dict[str, RunResult] = {}
    for name, tf, params, note in specs:
        if data.get(tf) is None or data[tf].empty:
            log.warning("skipping %s %s: no data", name, tf)
            continue
        strat = build_strategy(name, params, tf)
        res = _run(cfg, db, strat, data[tf], tf, note)
        results[f"{name}:{tf}"] = res
        try:
            write_run_artifacts(cfg, db, res.run_id)
        except Exception as exc:  # noqa: BLE001 - artifacts are secondary
            log.warning("artifact generation failed for %s: %s", res.run_id, exc)
    write_comparison(db)
    return results


def run_ema_std_experiments(cfg: Config, db: Database) -> list[dict]:
    ing = CandleIngestor(cfg, db)
    df = ing.load_history("1d")
    if df.empty:
        raise RuntimeError("no 1d data; run fetch first")
    oos_ts = _split_ts(df)

    base = _run(cfg, db, build_strategy("ema_std"), df, "1d", "experiment baseline")
    base_oos = _run(cfg, db, build_strategy("ema_std"), df, "1d",
                    "experiment baseline (OOS window)", start_ts=oos_ts)
    outcomes = []

    def candidate(label: str, hypothesis: str, params: dict, kind: str = "candidate") -> None:
        strat = build_strategy("ema_std", params)
        full = _run(cfg, db, strat, df, "1d", f"experiment: {label}")
        oos = _run(cfg, db, build_strategy("ema_std", params), df, "1d",
                   f"experiment: {label} (OOS window)", start_ts=oos_ts)
        verdict, notes = evaluate_candidate(base, full, base_oos, oos)
        record_experiment(db, name=label, hypothesis=hypothesis,
                          baseline_run_id=base.run_id, candidate_run_id=full.run_id,
                          verdict=verdict, notes=notes,
                          metrics={"full": full.metrics, "oos": oos.metrics,
                                   "baseline_oos": base_oos.metrics})
        outcomes.append({"experiment": label, "kind": kind, "verdict": verdict,
                         "notes": notes, "params": params,
                         "full_return": full.metrics.get("net_return_pct"),
                         "full_dd": full.metrics.get("max_drawdown_pct"),
                         "oos_return": oos.metrics.get("net_return_pct"),
                         "oos_dd": oos.metrics.get("max_drawdown_pct"),
                         "oos_trades": oos.metrics.get("trades")})
        log.info("experiment %s -> %s (%s)", label, verdict, notes)

    # Band-threshold sensitivity: reported, not cherry-picked.
    for bull, bear in ((0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (1.0, -1.0)):
        candidate(f"band_{bull}_{bear}",
                  f"EMA-65 state thresholds bull_z={bull}, bear_z={bear}",
                  {"bull_z": bull, "bear_z": bear}, kind="sensitivity")

    # A strategy without a stop cannot be deployed here: the risk manager refuses
    # to size a position with no defined invalidation level.  Recorded explicitly
    # rather than silently skipped.
    no_stop = build_strategy("ema_std", {"use_stop": False})
    no_stop_run = _run(cfg, db, no_stop, df, "1d", "experiment: pure state rule without stop")
    record_experiment(
        db, name="pure_state_no_stop",
        hypothesis="Does the pure state rule work without a protective stop?",
        baseline_run_id=base.run_id, candidate_run_id=no_stop_run.run_id,
        verdict="rejected",
        notes=("Structurally rejected: the risk manager refuses to size a position without a "
               "hard stop, so this variant cannot trade at all. This is a deliberate safety "
               "property, not a performance result."),
        metrics={"full": no_stop_run.metrics},
    )
    outcomes.append({"experiment": "pure_state_no_stop", "kind": "safety",
                     "verdict": "rejected",
                     "notes": "risk layer refuses to size an unstoppable position",
                     "params": {"use_stop": False},
                     "full_return": no_stop_run.metrics.get("net_return_pct"),
                     "full_dd": no_stop_run.metrics.get("max_drawdown_pct"),
                     "oos_return": None, "oos_dd": None, "oos_trades": 0})
    candidate("trend_filter_200", "Does requiring price above EMA200 improve risk-adjusted returns?",
              {"ema_trend_filter": 200})
    candidate("wider_stop_5atr", "Is the 3-ATR stop too tight for daily BTC volatility?",
              {"stop_atr_mult": 5.0})
    candidate("tighter_stop_2atr", "Is a 2-ATR stop better for this entry style?",
              {"stop_atr_mult": 2.0})

    # Risk-budget sensitivity uses a fresh baseline so the comparison is honest.
    for risk_pct in (1.0, 2.0):
        original = cfg.risk.risk_per_trade_pct
        cfg.risk.risk_per_trade_pct = risk_pct
        try:
            full = _run(cfg, db, build_strategy("ema_std"), df, "1d",
                        f"experiment: risk budget {risk_pct}%")
            oos = _run(cfg, db, build_strategy("ema_std"), df, "1d",
                       f"experiment: risk budget {risk_pct}% (OOS)", start_ts=oos_ts)
        finally:
            cfg.risk.risk_per_trade_pct = original
        record_experiment(db, name=f"risk_budget_{risk_pct}pct",
                          hypothesis="Does a larger risk budget improve absolute returns without "
                                     "breaching drawdown limits?",
                          baseline_run_id=base.run_id, candidate_run_id=full.run_id,
                          verdict="report_only",
                          notes=f"Risk per trade {risk_pct}% (baseline {original}%). This is a "
                                "sizing change, not a strategy edge.",
                          metrics={"full": full.metrics, "oos": oos.metrics})
        outcomes.append({"experiment": f"risk_budget_{risk_pct}pct", "kind": "sizing",
                         "verdict": "report_only", "notes": "return scales with size, not edge",
                         "params": {"risk_per_trade_pct": risk_pct},
                         "full_return": full.metrics.get("net_return_pct"),
                         "full_dd": full.metrics.get("max_drawdown_pct"),
                         "oos_return": oos.metrics.get("net_return_pct"),
                         "oos_dd": oos.metrics.get("max_drawdown_pct"),
                         "oos_trades": oos.metrics.get("trades")})

    df4 = ing.load_history("4h")
    if not df4.empty:
        oos4 = _split_ts(df4)
        full4 = _run(cfg, db, build_strategy("ema_std", {}, "4h"), df4, "4h",
                     "experiment: 4h timeframe")
        oos_4 = _run(cfg, db, build_strategy("ema_std", {}, "4h"), df4, "4h",
                     "experiment: 4h timeframe (OOS)", start_ts=oos4)
        record_experiment(db, name="timeframe_4h",
                          hypothesis="Does the same band rule work better on 4h bars?",
                          baseline_run_id=base.run_id, candidate_run_id=full4.run_id,
                          verdict="report_only",
                          notes="Different timeframe: capital efficiency and costs per year differ. "
                                "Not directly comparable to the daily baseline.",
                          metrics={"full": full4.metrics, "oos": oos_4.metrics})
        outcomes.append({"experiment": "timeframe_4h", "kind": "timeframe",
                         "verdict": "report_only", "notes": "not directly comparable",
                         "params": {}, "full_return": full4.metrics.get("net_return_pct"),
                         "full_dd": full4.metrics.get("max_drawdown_pct"),
                         "oos_return": oos_4.metrics.get("net_return_pct"),
                         "oos_dd": oos_4.metrics.get("max_drawdown_pct"),
                         "oos_trades": oos_4.metrics.get("trades")})
    return outcomes


def run_ml_experiment(cfg: Config, db: Database) -> dict:
    ing = CandleIngestor(cfg, db)
    df = ing.load_history("1d")
    if df.empty:
        raise RuntimeError("no 1d data")
    wf = walk_forward_predictions(df, cfg, horizon=5, min_train_bars=700, test_bars=120)
    version = f"gbdt_1d_h5_{datetime.now(timezone.utc):%Y%m%d}"

    preds = wf.predictions[["bar_ts", "prob_up"]].copy()
    ml_strategy = MLGated(predictions=preds, model_version=version)
    start_ts = int(preds["bar_ts"].iloc[0])
    ml_run = _run(cfg, db, ml_strategy, df, "1d",
                  f"ML-gated strategy using out-of-fold predictions from {version}",
                  start_ts=start_ts)
    base_run = _run(cfg, db, build_strategy("ema_std"), df, "1d",
                    "identity-agnostic baseline over the ML evaluation window", start_ts=start_ts)
    status, reason = promotion_decision(wf.metrics, strategy_metrics=ml_run.metrics,
                                        baseline_metrics=base_run.metrics)
    artifact = register_model(db, cfg, wf, version=version, status=status,
                              notes=f"walk-forward evaluation; gate: {reason}")
    verdict, notes = evaluate_candidate(base_run, ml_run, base_run, ml_run)
    if status == "rejected":
        verdict = "rejected"
        notes = f"model gate: {reason}. ML strategy OOS return {ml_run.metrics.get('net_return_pct')}%."

    record_experiment(db, name="ml_gbdt_h5",
                      hypothesis="Does a gradient-boosted model predicting 5-day cost-clearing "
                                 "returns add value over the EMA-65 baseline?",
                      baseline_run_id=base_run.run_id, candidate_run_id=ml_run.run_id,
                      verdict=verdict, notes=notes,
                      metrics={"model": wf.metrics, "strategy": ml_run.metrics,
                               "baseline": base_run.metrics})
    return {
        "version": version, "status": status, "gate_reason": reason,
        "artifact": artifact, "model_metrics": wf.metrics,
        "strategy_metrics": ml_run.metrics, "baseline_metrics": base_run.metrics,
        "run_id": ml_run.run_id, "baseline_run_id": base_run.run_id,
        "predictions": int(len(preds)), "verdict": verdict, "notes": notes,
    }


def run_sentiment_experiment(cfg: Config, db: Database, *, with_news: bool = False) -> dict:
    from .sentiment.fng import store_fng
    from .sentiment.news import collect as collect_news

    fetched = 0
    try:
        fetched = store_fng(db)
    except Exception as exc:  # noqa: BLE001
        log.warning("Fear & Greed fetch failed: %s", exc)
    news = collect_news(db) if with_news else {"stored": 0, "errors": {}}

    ing = CandleIngestor(cfg, db)
    raw = ing.load_history("1d")
    df = attach_fear_greed(raw, db)
    coverage = int(df["fg_value"].notna().sum())
    if coverage < 500:
        return {"available": False, "rows_fetched": fetched, "coverage_days": coverage,
                "news": news,
                "note": "Sentiment history is too short for a meaningful out-of-sample test."}

    # 1. Does sentiment add incremental predictive value to the ML model?
    without = walk_forward_predictions(df.drop(columns=["fg_value", "fg_norm", "fg_change_7",
                                                        "fg_extreme_fear", "fg_extreme_greed"]),
                                       cfg, horizon=5, min_train_bars=700, test_bars=120)
    with_fg = walk_forward_predictions(df, cfg, horizon=5, min_train_bars=700, test_bars=120)
    delta_auc = (with_fg.metrics["auc"] - without.metrics["auc"])

    # 2. Can sentiment alone trade profitably after costs?
    fg_strategy = FearGreedContrarian()
    fg_run = _run(cfg, db, fg_strategy, df, "1d", "experimental sentiment-only rule")
    base_start = int(fg_run.equity["ts"].iloc[0])
    base_run = _run(cfg, db, build_strategy("ema_std"), df, "1d",
                    "EMA-65 baseline over the sentiment evaluation window", start_ts=base_start)

    beats_baseline = (fg_run.metrics.get("net_return_pct") or 0) > (
        base_run.metrics.get("net_return_pct") or 0)
    beats_hold = (fg_run.metrics.get("net_return_pct") or 0) > (
        fg_run.benchmark["buy_hold"].get("net_return_pct") or 0)
    verdict = "rejected" if not (beats_baseline and beats_hold and delta_auc > 0) else "inconclusive"
    notes = (
        f"Fear & Greed coverage: {coverage} daily bars. Incremental OOF AUC from sentiment "
        f"features: {delta_auc:+.4f} (base {without.metrics['auc']}, with {with_fg.metrics['auc']}). "
        f"Sentiment-only rule net return {fg_run.metrics.get('net_return_pct')}% vs EMA-65 "
        f"{base_run.metrics.get('net_return_pct')}% and buy&hold "
        f"{fg_run.benchmark['buy_hold'].get('net_return_pct')}%. "
        + ("Sentiment is rejected as a trading input on this evidence and remains disabled."
           if verdict == "rejected" else
           "Evidence is mixed; sentiment stays experimental and disabled for trading.")
    )
    record_experiment(db, name="sentiment_fear_greed",
                      hypothesis="Does point-in-time Fear & Greed sentiment improve returns or "
                                 "predictive accuracy after costs?",
                      baseline_run_id=base_run.run_id, candidate_run_id=fg_run.run_id,
                      verdict=verdict, notes=notes,
                      metrics={"delta_auc": delta_auc, "without_auc": without.metrics["auc"],
                               "with_auc": with_fg.metrics["auc"],
                               "sentiment_only": fg_run.metrics, "baseline": base_run.metrics})
    return {
        "available": True, "coverage_days": coverage, "delta_auc": delta_auc,
        "without_auc": without.metrics["auc"], "with_auc": with_fg.metrics["auc"],
        "sentiment_strategy": fg_run.metrics, "baseline": base_run.metrics,
        "buy_hold": fg_run.benchmark["buy_hold"], "verdict": verdict, "notes": notes,
        "news": news, "rows_fetched": fetched,
    }


def write_experiment_report(path: str | Path, *, suite: dict, experiments: list[dict],
                            ml: dict | None = None, sentiment: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment log", "",
             "Generated " + datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "", "All numbers are simulated, net of fees/spread/slippage.", "",
             "## Strategy suite", "",
             "| Strategy | TF | Net % | CAGR % | MaxDD % | Sharpe | Trades | Win % | B&H % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, res in suite.items():
        m, b = res.metrics, res.benchmark.get("buy_hold", {})
        lines.append(
            f"| {res.strategy} | {res.timeframe} | {m.get('net_return_pct')} | {m.get('cagr_pct')} "
            f"| {m.get('max_drawdown_pct')} | {m.get('sharpe')} | {m.get('trades')} "
            f"| {m.get('win_rate_pct')} | {b.get('net_return_pct')} |"
        )
    lines += ["", "## Controlled changes vs the EMA-65 baseline", "",
              "| Experiment | Verdict | Full net % | OOS net % | OOS MaxDD % | OOS trades | Note |",
              "|---|---|---|---|---|---|---|"]
    for e in experiments:
        lines.append(
            f"| {e['experiment']} | {e['verdict']} | {e.get('full_return')} "
            f"| {e.get('oos_return')} | {e.get('oos_dd')} | {e.get('oos_trades')} "
            f"| {e.get('notes','')[:120]} |"
        )
    if ml:
        lines += ["", "## Machine learning", "",
                  f"- Model `{ml['version']}` status: **{ml['status']}** ({ml['gate_reason']})",
                  f"- Out-of-fold AUC: {ml['model_metrics']['auc']} "
                  f"on {ml['model_metrics']['predictions']} predictions "
                  f"across {ml['model_metrics']['folds']} folds",
                  f"- Strategy net return: {ml['strategy_metrics'].get('net_return_pct')}% "
                  f"vs EMA-65 {ml['baseline_metrics'].get('net_return_pct')}% over the same window",
                  f"- Verdict: **{ml['verdict']}** — {ml['notes']}"]
    if sentiment:
        lines += ["", "## Sentiment", "",
                  f"- Available: {sentiment.get('available')} "
                  f"(coverage {sentiment.get('coverage_days')} days)",
                  f"- {sentiment.get('notes', sentiment.get('note',''))}"]
    lines += ["", "## Interpretation", "",
              "An experiment is only accepted when the out-of-sample window improves after costs "
              "with no material drawdown regression. 'Accepted' means 'worth paper-trading', "
              "never 'proven profitable'."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
