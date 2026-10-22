"""Markdown reports written next to the evidence they describe."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings


def _f(v, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v) -> str:
    return "—" if v is None else f"{float(v):+.2f}%"


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(k, "—")) for k, _ in cols) + " |")
    return "\n".join([head, sep, *body])


def write_backtest_report(result, path: str | Path, *, label: str = "",
                          settings: Settings | None = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    m = result.metrics
    bench = result.benchmarks or m.get("benchmarks", {})
    lines = [
        f"# Portfolio backtest — {label or result.run_id}",
        "",
        f"- run id: `{result.run_id}`",
        f"- period: {m.get('period_start', '')[:10]} → {m.get('period_end', '')[:10]} "
        f"({_f(m.get('period_years'))} years)",
        f"- timeframe: {m.get('timeframe')} · capital ${_f(m.get('capital'))}",
        "",
        "## Headline",
        "",
        "| Metric | Strategy | Equal-weight hold | BTC hold |",
        "|---|---:|---:|---:|",
        f"| Net return | {_pct(m.get('net_return_pct'))} | "
        f"{_pct(bench.get('equal_weight', {}).get('net_return_pct'))} | "
        f"{_pct(bench.get('btc_hold', {}).get('net_return_pct'))} |",
        f"| Max drawdown | {_f(m.get('max_drawdown_pct'))}% | "
        f"{_f(bench.get('equal_weight', {}).get('max_drawdown_pct'))}% | "
        f"{_f(bench.get('btc_hold', {}).get('max_drawdown_pct'))}% |",
        f"| Final equity | ${_f(m.get('final_equity'))} | "
        f"${_f(bench.get('equal_weight', {}).get('final_equity'))} | "
        f"${_f(bench.get('btc_hold', {}).get('final_equity'))} |",
        "",
        "## Trading statistics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Closed trades | {m.get('trades')} |",
        f"| Win rate | {_f(m.get('win_rate_pct'))}% |",
        f"| Profit factor | {_f(m.get('profit_factor'))} |",
        f"| Expectancy / trade | ${_f(m.get('expectancy_per_trade'))} |",
        f"| Average R | {_f(m.get('avg_r_multiple'))} |",
        f"| Average hold | {_f(m.get('avg_holding_hours'))} h |",
        f"| Exposure | {_f(m.get('exposure_pct'))}% |",
        f"| Sharpe / Sortino | {_f(m.get('sharpe'))} / {_f(m.get('sortino'))} |",
        f"| Fees paid | ${_f(m.get('total_fees'))} |",
        f"| Slippage cost | ${_f(m.get('total_slippage_cost'))} |",
        f"| Exit mix | `{json.dumps(m.get('exit_mix', {}))}` |",
        "",
        "## Per symbol",
        "",
        _table([{"symbol": k, "trades": v["trades"], "win": _f(v["win_rate_pct"]),
                 "net": _f(v["net_pnl"]), "avg_ret": _f(v["avg_return_pct"]),
                 "avg_r": _f(v["avg_r"])} for k, v in result.per_symbol.items()],
               [("symbol", "s"), ("trades", "n"), ("win", "n"), ("net", "n"),
                ("avg_ret", "n"), ("avg_r", "n")]),
        "",
        "## Per strategy",
        "",
        _table([{"strategy": k, "trades": v["trades"], "win": _f(v["win_rate_pct"]),
                 "net": _f(v["net_pnl"]), "avg_ret": _f(v["avg_return_pct"])}
                for k, v in result.per_strategy.items()],
               [("strategy", "s"), ("trades", "n"), ("win", "n"), ("net", "n"),
                ("avg_ret", "n")]),
        "",
        "## Method notes",
        "",
        "- Signals use closed bars; fills happen at the next bar's open.",
        "- Stops fill at `min(open, stop)` so gaps cost more than the stop.",
        "- Fees: "
        f"{settings.costs.taker_fee_bps if settings else 25:.0f} bps taker per side, plus "
        "half-spread and 5 bps slippage on every fill.",
        "- One shared cash balance and the live risk manager govern all symbols.",
        "- Long-only spot, no leverage, no shorting.",
        "",
        f"> {m.get('sample_warning') or 'Sample size is adequate for a first read; still not a live guarantee.'}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def write_model_report(store, settings: Settings, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = store.models()
    lines = ["# Per-coin meta-labeling models", "",
             "Each model is trained on that coin's own history and only filters entries "
             "proposed by its primary strategy. A model is used for trading **only** when "
             "it passes purged walk-forward validation (AUC ≥ "
             f"{settings.models.min_auc} and positive net improvement).", "",
             "| Symbol | Strategy | Status | AUC | Brier | Threshold | OOS trades | Improvement | Artifact |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r['symbol']} | {r['strategy']} | **{r['status']}** | {_f(r['auc'], 3)} | "
            f"{_f(r['brier'], 3)} | {_f(r['threshold'])} | {r['oos_trades'] or 0} | "
            f"{_f(r['oos_improvement_pct'])} pp | `{Path(r['artifact_path'] or '').name}` |")
    rejected = [r for r in rows if r["status"] == "rejected"]
    lines += ["", "## What rejection means", "",
              f"{len(rejected)} model(s) failed validation and are **not** used to trade. "
              "They remain in the registry so the failure is reviewable rather than hidden. "
              "This is the expected outcome when a 10-candidate strategy grid is already "
              "near the information limit of 5 years of 4h bars.", ""]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def write_status_report(store, settings: Settings, path: str | Path, *,
                        extra: dict | None = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    overview = store.get_portfolio("equity_snapshot", {}) or {}
    positions = store.positions()
    regime = store.latest_regime() or {}
    recon = store.latest_recon() or {}
    trades = store.db.query(
        "SELECT COUNT(*) n, COALESCE(SUM(net_pnl),0) pnl, COALESCE(SUM(fees),0) fees"
        " FROM trades WHERE mode='live'")
    row = dict(trades[0]) if trades else {"n": 0, "pnl": 0, "fees": 0}
    lines = ["# Live paper status", "",
             f"- equity: **${_f(overview.get('equity'))}** · cash ${_f(overview.get('cash'))}",
             f"- closed trades: {row['n']} · net ${_f(row['pnl'])} · fees ${_f(row['fees'])}",
             f"- regime: **{regime.get('label', '—')}** (score {_f(regime.get('score'), 3)})",
             f"- reconciliation: **{recon.get('status', '—')}** — {recon.get('detail', '')}",
             f"- open positions: {len(positions)}", ""]
    if positions:
        lines += ["| Symbol | Qty | Entry | Stop | Strategy |", "|---|---:|---:|---:|---|"]
        for pos in positions:
            lines.append(f"| {pos['symbol']} | {_f(pos['qty'], 6)} | {_f(pos['entry_price'], 6)} | "
                         f"{_f(pos['stop_price'], 6)} | {pos['strategy']} |")
        lines.append("")
    models = store.models()
    active = [m for m in models if m["status"] == "active"]
    lines += [f"- active models: {len(active)} / {len(models)} trained", ""]
    if extra:
        lines += ["## Runtime", "", "```json", json.dumps(extra, indent=2, default=str), "```", ""]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)
