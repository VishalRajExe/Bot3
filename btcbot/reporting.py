"""Research artifacts: trade CSVs, equity curves, charts and comparison tables.

Every artifact is generated from stored simulated results.  No placeholder data
is ever written; an empty run produces an explicit "no trades" note instead of a
fabricated chart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config, PROJECT_ROOT
from .data.store import CandleStore
from .data.timeframes import to_datetime
from .db import Database
from .journal import load_trades, trade_to_dict

log = logging.getLogger(__name__)

BG = "#0b0f16"
PANEL = "#111827"
GRID = "#1f2937"
TEXT = "#e5e7eb"
UP = "#22c55e"
DOWN = "#ef4444"
ACCENT = "#38bdf8"
MUTED = "#9ca3af"


def _layout(fig: go.Figure, title: str, height: int = 620) -> go.Figure:
    fig.update_layout(
        title=title, template="plotly_dark", height=height,
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=12),
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def write_run_artifacts(cfg: Config, db: Database, run_id: str,
                        out_root: str | Path = "reports/backtests") -> dict:
    row = db.query_one("SELECT * FROM backtest_runs WHERE id=?", (run_id,))
    if row is None:
        raise ValueError(f"run {run_id} not found")
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(row["metrics_json"] or "{}")
    benchmark = json.loads(row["benchmark_json"] or "{}")
    params = json.loads(row["params_json"] or "{}")
    equity = pd.DataFrame([dict(r) for r in db.query(
        "SELECT ts, equity, cash, position_qty, price, drawdown FROM equity_curve"
        " WHERE run_id=? ORDER BY ts", (run_id,))])
    trades = load_trades(db, run_id=run_id)

    # --- data files
    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "strategy": row["strategy"], "version": row["strategy_version"],
        "timeframe": row["timeframe"], "params": params,
        "period": {"start": metrics.get("period_start"), "end": metrics.get("period_end")},
        "dataset_hash": row["dataset_hash"], "costs": json.loads(row["costs_json"] or "{}"),
        "risk": json.loads(row["risk_json"] or "{}"), "metrics": metrics,
        "benchmark": benchmark,
    }, indent=2, default=str), encoding="utf-8")

    trade_rows = [trade_to_dict(t) for t in trades]
    pd.DataFrame(trade_rows).to_csv(out / "trades.csv", index=False)
    if not equity.empty:
        eq = equity.copy()
        eq["time"] = eq["ts"].map(lambda x: to_datetime(int(x)).isoformat())
        eq[["time", "ts", "equity", "cash", "position_qty", "price", "drawdown"]].to_csv(
            out / "equity.csv", index=False
        )

    # --- price chart with signals
    store = CandleStore(db)
    df = store.load(cfg.exchange.pair, row["timeframe"])
    start_ts = row["start_ts"]
    bar_index = df.index[df["bar_ts"] >= start_ts]
    from_idx = max(int(bar_index[0]) - 120, 0) if len(bar_index) else 0
    view = df.iloc[from_idx:]
    if not view.empty:
        x = view["dt"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=x, open=view["open"], high=view["high"], low=view["low"], close=view["close"],
            name="BTC/USDT", increasing_line_color=UP, decreasing_line_color=DOWN,
            showlegend=False,
        ))
        # Strategy overlays only when the run used the EMA-65 family.
        if row["strategy"] == "ema_std":
            close = view["close"]
            ema65 = close.ewm(span=65, adjust=False, min_periods=65).mean()
            std65 = close.rolling(65, min_periods=65).std(ddof=0)
            fig.add_trace(go.Scatter(x=x, y=ema65, name="EMA 65", line=dict(color=ACCENT, width=1.4)))
            for k, dash in ((1, "dot"), (2, "dot")):
                fig.add_trace(go.Scatter(x=x, y=ema65 + k * std65, name=f"+{k}s",
                                         line=dict(color="rgba(56,189,248,0.35)", width=1, dash=dash)))
                fig.add_trace(go.Scatter(x=x, y=ema65 - k * std65, name=f"-{k}s",
                                         line=dict(color="rgba(56,189,248,0.35)", width=1, dash=dash)))
        if trade_rows:
            fig.add_trace(go.Scatter(
                x=[t["entry_time"] for t in trade_rows],
                y=[t["entry_price"] for t in trade_rows],
                mode="markers", name="entry",
                marker=dict(symbol="triangle-up", size=11, color=UP,
                            line=dict(color="#065f46", width=1)),
                text=[f"{t['entry_reason']}<br>stop {t['stop_distance_pct']}%" for t in trade_rows],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[t["exit_time"] for t in trade_rows],
                y=[t["exit_price"] for t in trade_rows],
                mode="markers", name="exit",
                marker=dict(symbol="triangle-down", size=11, color=DOWN,
                            line=dict(color="#7f1d1d", width=1)),
                text=[f"{t['exit_reason']}<br>net {t['net_pnl']:.2f} ({t['return_pct']:.2f}%)"
                      for t in trade_rows],
                hovertemplate="%{text}<extra></extra>",
            ))
        _layout(fig, f"{row['strategy']} v{row['strategy_version']} — "
                     f"{row['timeframe']} signals & simulated trades", 700)
        fig.update_xaxes(rangeslider_visible=False)
        fig.write_html(out / "chart.html", include_plotlyjs=True, full_html=True)

    # --- equity / drawdown / monthly
    if not equity.empty:
        eq_x = [to_datetime(int(t)) for t in equity["ts"]]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.58, 0.22, 0.20],
                            subplot_titles=("Equity", "Drawdown %", "Monthly returns %"))
        fig.add_trace(go.Scatter(x=eq_x, y=equity["equity"], name="strategy equity",
                                 line=dict(color=ACCENT, width=1.6)), row=1, col=1)
        bh = benchmark.get("buy_hold", {})
        if bh.get("net_return_pct") is not None:
            start_eq = float(equity["equity"].iloc[0])
            bench_curve = start_eq * (1 + bh["net_return_pct"] / 100.0) ** (
                (equity["ts"] - equity["ts"].iloc[0]) / max(equity["ts"].iloc[-1] - equity["ts"].iloc[0], 1)
            )
            fig.add_trace(go.Scatter(x=eq_x, y=bench_curve, name="BTC buy & hold (same window)",
                                     line=dict(color=MUTED, width=1.2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eq_x, y=equity["drawdown"], name="drawdown",
                                 fill="tozeroy", line=dict(color=DOWN, width=1)), row=2, col=1)
        monthly = metrics.get("monthly_returns") or []
        if monthly:
            fig.add_trace(go.Bar(x=[m["month"] for m in monthly],
                                 y=[m["return_pct"] for m in monthly],
                                 marker_color=[UP if m["return_pct"] >= 0 else DOWN for m in monthly],
                                 name="monthly return %"), row=3, col=1)
        _layout(fig, f"{row['strategy']} — equity, drawdown and monthly returns", 820)
        fig.write_html(out / "performance.html", include_plotlyjs=True, full_html=True)

    summary_path = out / "SUMMARY.md"
    bh_ret = benchmark.get("buy_hold", {}).get("net_return_pct")
    summary_path.write_text(
        f"""# {row['strategy']} v{row['strategy_version']} — {row['timeframe']}

Run `{run_id}` created {row['created_at']}.  **Backtest, not live trading.**

| Metric | Value |
|---|---|
| Period | {metrics.get('period_start')} → {metrics.get('period_end')} |
| Net return | {metrics.get('net_return_pct')}% |
| CAGR | {metrics.get('cagr_pct')}% |
| Max drawdown | {metrics.get('max_drawdown_pct')}% |
| Sharpe / Sortino | {metrics.get('sharpe')} / {metrics.get('sortino')} |
| Trades | {metrics.get('trades')} |
| Win rate | {metrics.get('win_rate_pct')}% |
| Profit factor | {metrics.get('profit_factor')} |
| Exposure | {metrics.get('exposure_pct')}% |
| Fees paid | {metrics.get('total_fees')} USDT |
| Buy & hold (same window) | {bh_ret}% |
| Buy & hold max drawdown | {benchmark.get('buy_hold', {}).get('max_drawdown_pct')}% |

Parameters: `{json.dumps(params, default=str)}`

Execution assumptions: signals on bar close, fills at next bar open; taker fee
{json.loads(row['costs_json'] or '{}').get('taker_fee_bps')} bps/side, spread
{json.loads(row['costs_json'] or '{}').get('spread_bps')} bps, slippage
{json.loads(row['costs_json'] or '{}').get('slippage_bps')} bps.  Dataset hash
`{row['dataset_hash']}`.
""",
        encoding="utf-8",
    )
    log.info("wrote artifacts for %s to %s", run_id, out)
    return {
        "dir": str(out),
        "metrics": str(out / "metrics.json"),
        "trades": str(out / "trades.csv"),
        "chart": str(out / "chart.html") if (out / "chart.html").exists() else None,
        "performance": str(out / "performance.html") if (out / "performance.html").exists() else None,
        "summary": str(summary_path),
    }


def comparison_table(db: Database, run_ids: list[str] | None = None) -> pd.DataFrame:
    sql = ("SELECT id, created_at, strategy, strategy_version, timeframe, metrics_json,"
           " benchmark_json, params_json FROM backtest_runs ORDER BY created_at")
    rows = db.query(sql)
    if run_ids:
        keep = set(run_ids)
        rows = [r for r in rows if r["id"] in keep]
    out = []
    for r in rows:
        m = json.loads(r["metrics_json"] or "{}")
        b = json.loads(r["benchmark_json"] or "{}")
        out.append({
            "run_id": r["id"],
            "strategy": r["strategy"],
            "version": r["strategy_version"],
            "timeframe": r["timeframe"],
            "net_return_pct": m.get("net_return_pct"),
            "cagr_pct": m.get("cagr_pct"),
            "max_drawdown_pct": m.get("max_drawdown_pct"),
            "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"),
            "calmar": m.get("calmar"),
            "trades": m.get("trades"),
            "win_rate_pct": m.get("win_rate_pct"),
            "profit_factor": m.get("profit_factor"),
            "exposure_pct": m.get("exposure_pct"),
            "fees": m.get("total_fees"),
            "buy_hold_return_pct": b.get("buy_hold", {}).get("net_return_pct"),
            "buy_hold_max_dd_pct": b.get("buy_hold", {}).get("max_drawdown_pct"),
            "excess_vs_bh_pct": m.get("excess_return_vs_buy_hold_pct"),
            "params": json.dumps(json.loads(r["params_json"] or "{}"), separators=(",", ":")),
        })
    return pd.DataFrame(out)


def write_comparison(db: Database, path: str | Path = "reports/backtests/comparison.md",
                     run_ids: list[str] | None = None) -> Path:
    df = comparison_table(db, run_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("# Strategy comparison\n\nNo backtest runs yet.\n", encoding="utf-8")
        return path
    cols = ["strategy", "version", "timeframe", "net_return_pct", "cagr_pct",
            "max_drawdown_pct", "sharpe", "calmar", "trades", "win_rate_pct",
            "profit_factor", "exposure_pct", "buy_hold_return_pct", "excess_vs_bh_pct"]
    header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    path.write_text(
        "# Strategy comparison (backtests)\n\n"
        "All figures are simulated net of fees, spread and slippage. "
        "Buy & hold is measured over the same window as each run.\n\n"
        + header + body +
        "\nA positive net return is not evidence of an edge. Compare against the "
        "buy-and-hold column and the drawdown before drawing conclusions.\n",
        encoding="utf-8",
    )
    df.to_csv(path.with_suffix(".csv"), index=False)
    return path
