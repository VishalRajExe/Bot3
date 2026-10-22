"""Per-symbol strategy research.

For every coin the sweep tests a small, fixed grid of strategy families and
parameters on a **training** slice, then evaluates each candidate once on a
held-out **test** slice.  Selection uses the test slice only, so the reported
winner is an out-of-sample result rather than the best in-sample curve.

The grid is intentionally small (10 candidates) because a large grid on 5 years
of 4h bars is an overfitting machine.  Every candidate's train and test metrics
are persisted so a reviewer can see how much of the result came from selection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot.data.timeframes import tf_seconds

from .backtest import run_portfolio_backtest
from .config import Settings, SymbolCfg
from .strategies import REGISTRY, build as build_strategy

log = logging.getLogger(__name__)

GRID: dict[str, list[dict]] = {
    "trend_breakout": [
        {"donchian": 20, "trend_ema": 100, "stop_atr": 2.5, "trail_atr": 3.0, "min_atr_pct": 1.0},
        {"donchian": 35, "trend_ema": 100, "stop_atr": 2.8, "trail_atr": 3.4, "min_atr_pct": 1.0},
        {"donchian": 55, "trend_ema": 100, "stop_atr": 3.0, "trail_atr": 4.0, "min_atr_pct": 0.8},
    ],
    "trend_pullback": [
        {"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 35.0, "stop_atr": 2.5, "trail_atr": 3.2},
        {"ema_fast": 50, "ema_slow": 200, "rsi_reclaim": 42.0, "stop_atr": 2.5, "trail_atr": 3.2},
        {"ema_fast": 20, "ema_slow": 100, "rsi_reclaim": 40.0, "stop_atr": 2.2, "trail_atr": 2.8},
    ],
    "mean_reversion": [
        {"bb_window": 20, "bb_std": 2.0, "rsi_entry": 30.0, "adx_max": 22.0, "stop_atr": 2.4},
        {"bb_window": 20, "bb_std": 2.4, "rsi_entry": 28.0, "adx_max": 20.0, "stop_atr": 2.6},
        {"bb_window": 20, "bb_std": 2.0, "rsi_entry": 32.0, "adx_max": 24.0, "stop_atr": 2.2},
    ],
}


@dataclass
class Candidate:
    symbol: str
    strategy: str
    params: dict
    train: dict
    test: dict
    score: float
    status: str

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "strategy": self.strategy, "params": self.params,
                "train": self.train, "test": self.test, "score": round(self.score, 3),
                "status": self.status}


def _score(test: dict) -> float:
    """Risk-adjusted ranking that punishes drawdown and tiny samples."""
    trades = int(test.get("trades") or 0)
    if trades < 5:
        return -999.0
    ret = float(test.get("net_return_pct") or 0.0)
    dd = max(float(test.get("max_drawdown_pct") or 0.0), 2.0)
    # both return and drawdown matter; subtract a heavy penalty for losing money
    base = ret / dd
    if ret < 0:
        base -= 1.0
    if trades < 10:
        base *= 0.6
    return base


def _symbol_only(settings: Settings, sym_cfg: SymbolCfg) -> Settings:
    risk = settings.risk.model_copy(update={"max_positions": 1, "max_per_cluster": 1,
                                            "max_position_pct": 100.0,
                                            "cash_reserve_pct": 0.0,
                                            "portfolio_heat_pct": 2.0})
    return settings.model_copy(update={"universe": [sym_cfg], "risk": risk})


def _slice_times(df: pd.DataFrame, train_frac: float = 0.65) -> tuple[int, int, int]:
    ts = df["bar_ts"].to_numpy("int64")
    warm = min(len(df) - 30, 500)
    start = int(ts[min(warm, len(ts) - 30)])
    split = int(ts[int(len(ts) * train_frac)])
    return start, split, int(ts[-1]) + tf_seconds("4h")


def sweep_symbol(settings: Settings, frame: pd.DataFrame, sym_cfg: SymbolCfg
                 ) -> tuple[Candidate, list[Candidate]]:
    if frame.empty or len(frame) < 700:
        return Candidate(sym_cfg.symbol, sym_cfg.strategy, sym_cfg.params, {}, {},
                         -999, "insufficient_data"), []
    start, split, end = _slice_times(frame)
    rows: list[Candidate] = []
    for strategy_name, grid in GRID.items():
        for params in grid:
            cfg = sym_cfg.model_copy(update={"strategy": strategy_name, "params": params})
            strat = build_strategy(cfg)
            one = _symbol_only(settings, cfg)
            try:
                train = run_portfolio_backtest(
                    one, {cfg.symbol: frame}, start_ts=start, end_ts=split,
                    initial_equity=settings.capital, mode="backtest",
                    label=f"sweep {cfg.symbol} {strategy_name}", strategies={cfg.symbol: strat},
                )
                test = run_portfolio_backtest(
                    one, {cfg.symbol: frame}, start_ts=split, end_ts=end,
                    initial_equity=settings.capital, mode="backtest",
                    label=f"sweep-test {cfg.symbol} {strategy_name}",
                    strategies={cfg.symbol: strat},
                )
            except Exception as exc:
                log.warning("candidate failed %s %s: %s", cfg.symbol, strategy_name, exc)
                continue
            rows.append(Candidate(
                symbol=cfg.symbol, strategy=strategy_name, params=params,
                train=_compact(train.metrics), test=_compact(test.metrics),
                score=_score(test.metrics),
                status="candidate",
            ))
    if not rows:
        return Candidate(sym_cfg.symbol, sym_cfg.strategy, sym_cfg.params, {}, {},
                         -999, "no_candidates"), []
    # Selection uses ONLY the held-out test slice.
    best = max(rows, key=lambda c: c.score)
    best.status = "selected"
    for r in rows:
        if r is not best:
            r.status = "rejected"
    return best, rows


def _compact(metrics: dict) -> dict:
    keys = ["net_return_pct", "max_drawdown_pct", "sharpe", "trades", "win_rate_pct",
            "profit_factor", "expectancy_per_trade", "avg_holding_hours", "total_fees",
            "exposure_pct", "calmar"]
    return {k: metrics.get(k) for k in keys}


def sweep_all(settings: Settings, frames: dict[str, pd.DataFrame], *, store=None,
              output: str = "reports/unibot/strategy_sweep") -> dict:
    results = []
    for sym_cfg in settings.universe:
        frame = frames.get(sym_cfg.symbol)
        best, rows = sweep_symbol(settings, frame, sym_cfg)
        results.append({"symbol": sym_cfg.symbol, "selected": best.as_dict(),
                        "candidates": [r.as_dict() for r in rows]})
        log.info("sweep %s -> %s %s score %.2f", sym_cfg.symbol, best.strategy,
                 best.params, best.score)
    payload = {"generated_at": pd.Timestamp.utcnow().isoformat(), "grid": GRID,
               "selection_rule": "highest test-slice return/drawdown with >=5 trades; "
                                 "train slice never used for selection",
               "results": results}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    if store is not None:
        store.db.execute(
            "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?,?,?)",
            ("unibot_strategy_sweep", json.dumps(payload), pd.Timestamp.utcnow().isoformat()),
        )
    return payload


def _markdown(payload: dict) -> str:
    lines = ["# Per-symbol strategy sweep", "",
             "Selection uses the **held-out test slice** only (last 35% of history).",
             "Train metrics are shown for context, not used to pick the winner.", "",
             "| Symbol | Selected strategy | Params | Test return | Test DD | Test trades | Win | Sharpe |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in payload["results"]:
        s = r["selected"]
        t = s["test"]
        lines.append(
            f"| {r['symbol']} | {s['strategy']} | `{json.dumps(s['params'])}` | "
            f"{_f(t.get('net_return_pct'))}% | {_f(t.get('max_drawdown_pct'))}% | "
            f"{t.get('trades')} | {_f(t.get('win_rate_pct'))}% | {_f(t.get('sharpe'))} |")
    lines += ["", "## Full grid", ""]
    for r in payload["results"]:
        lines.append(f"### {r['symbol']}")
        lines.append("")
        lines.append("| Strategy | Params | Test ret | Test DD | Trades | Score | Status |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for c in r["candidates"]:
            t = c["test"]
            lines.append(f"| {c['strategy']} | `{json.dumps(c['params'])}` | "
                         f"{_f(t.get('net_return_pct'))}% | {_f(t.get('max_drawdown_pct'))}% | "
                         f"{t.get('trades')} | {c['score']} | {c['status']} |")
        lines.append("")
    return "\n".join(lines)


def _f(v) -> str:
    return "—" if v is None else f"{float(v):.2f}"


def apply_selection(settings_path: str, payload: dict) -> Settings:
    """Write the selected strategy/params back into the config (explicit step)."""
    raw = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    selected = {r["symbol"]: r["selected"] for r in payload["results"]}
    for entry in raw["universe"]:
        sel = selected.get(entry["symbol"])
        if not sel or sel["status"] != "selected":
            continue
        entry["strategy"] = sel["strategy"]
        entry["params"] = sel["params"]
    Path(settings_path).write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return Settings.load(settings_path)
