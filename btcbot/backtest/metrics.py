"""Performance metrics.

All metrics are computed from the actual simulated equity curve and the trades
that produced it.  Nothing is annualised when the sample is too short to make
the number meaningful; those fields are left as ``None``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..execution.models import TradeRecord


def _safe(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 6)


def max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """Return (max drawdown %, index of the trough)."""
    if len(equity) == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak == 0, np.nan, peak)
    dd = np.nan_to_num(dd, nan=0.0)
    idx = int(np.argmin(dd))
    return float(-dd[idx] * 100.0), idx


def drawdown_duration_days(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    eq = equity_df["equity"].to_numpy(dtype=float)
    ts = equity_df["ts"].to_numpy(dtype=np.int64)
    peak = -np.inf
    peak_ts = ts[0]
    worst = 0
    for value, t in zip(eq, ts):
        if value >= peak:
            peak = value
            peak_ts = t
        else:
            worst = max(worst, int(t) - int(peak_ts))
    return worst / 86_400.0


def daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    """UTC-calendar daily equity returns from the bar-level curve."""
    if equity_df.empty:
        return pd.Series(dtype=float)
    df = equity_df[["ts", "equity"]].copy()
    df["day"] = df["ts"] // 86_400
    daily = df.groupby("day", sort=True)["equity"].last()
    return daily.pct_change().dropna()


def sharpe_sortino(returns: pd.Series) -> tuple[float | None, float | None]:
    if len(returns) < 20 or returns.std(ddof=1) == 0:
        return None, None
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    downside = returns[returns < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = mean / std * math.sqrt(365.0) if std > 0 else None
    sortino = mean / down_std * math.sqrt(365.0) if down_std > 0 else None
    return _safe(sharpe), _safe(sortino)


def monthly_returns(equity_df: pd.DataFrame) -> list[dict]:
    if equity_df.empty:
        return []
    df = equity_df[["ts", "equity"]].copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    monthly = df.set_index("dt")["equity"].resample("ME").last().dropna()
    if monthly.empty:
        return []
    start_value = float(equity_df["equity"].iloc[0])
    prev = start_value
    out = []
    for dt, value in monthly.items():
        ret = (float(value) - prev) / prev * 100.0 if prev else 0.0
        out.append({"month": dt.strftime("%Y-%m"), "return_pct": round(ret, 4),
                    "equity": round(float(value), 2)})
        prev = float(value)
    return out


def compute_metrics(*, equity_df: pd.DataFrame, trades: list[TradeRecord],
                    initial_equity: float, timeframe: str, bars_in_market: int,
                    total_bars: int) -> dict:
    if equity_df.empty:
        return {"error": "empty equity curve"}
    eq = equity_df["equity"].to_numpy(dtype=float)
    ts = equity_df["ts"].to_numpy(dtype=np.int64)
    final = float(eq[-1])
    start_ts, end_ts = int(ts[0]), int(ts[-1])
    years = max((end_ts - start_ts) / (365.25 * 86_400), 1e-9)
    total_return = (final / initial_equity - 1.0) * 100.0
    cagr = ((final / initial_equity) ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0
    mdd, trough_idx = max_drawdown(eq)
    daily = daily_returns(equity_df)
    sharpe, sortino = sharpe_sortino(daily)
    vol = float(daily.std(ddof=1) * math.sqrt(365.0) * 100.0) if len(daily) > 1 else None

    closed = [t for t in trades if t.exit_ts is not None]
    wins = [t for t in closed if t.net_pnl > 0]
    losses = [t for t in closed if t.net_pnl <= 0]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (None if not wins else float("inf"))
    avg_win = gross_win / len(wins) if wins else None
    avg_loss = -gross_loss / len(losses) if losses else None
    expectancy = float(np.mean([t.net_pnl for t in closed])) if closed else None
    avg_r = float(np.mean([t.r_multiple for t in closed if t.r_multiple is not None])) if closed else None
    avg_hold = float(np.mean([t.holding_hours for t in closed])) if closed else None

    by_regime: dict[str, dict] = {}
    for t in closed:
        key = t.regime or "unknown"
        bucket = by_regime.setdefault(key, {"trades": 0, "wins": 0, "net_pnl": 0.0, "return_sum": 0.0})
        bucket["trades"] += 1
        bucket["wins"] += int(t.net_pnl > 0)
        bucket["net_pnl"] += t.net_pnl
        bucket["return_sum"] += t.return_pct
    for bucket in by_regime.values():
        bucket["win_rate_pct"] = round(bucket["wins"] / bucket["trades"] * 100.0, 2)
        bucket["avg_return_pct"] = round(bucket["return_sum"] / bucket["trades"], 4)
        bucket["net_pnl"] = round(bucket["net_pnl"], 2)
        bucket.pop("return_sum")

    metrics = {
        "initial_equity": round(initial_equity, 2),
        "final_equity": round(final, 2),
        "net_return_pct": _safe(total_return),
        "cagr_pct": _safe(cagr),
        "max_drawdown_pct": _safe(mdd),
        "drawdown_duration_days": _safe(drawdown_duration_days(equity_df)),
        "volatility_pct": _safe(vol),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": _safe(cagr / mdd) if mdd > 0 else None,
        "trades": len(closed),
        "win_rate_pct": _safe(len(wins) / len(closed) * 100.0) if closed else None,
        "profit_factor": _safe(profit_factor) if profit_factor not in (None, float("inf")) else profit_factor,
        "avg_win": _safe(avg_win),
        "avg_loss": _safe(avg_loss),
        "expectancy_per_trade": _safe(expectancy),
        "avg_r_multiple": _safe(avg_r),
        "avg_holding_hours": _safe(avg_hold),
        "total_fees": round(sum(t.fees for t in closed), 2),
        "total_slippage_cost": round(sum(t.slippage_cost for t in closed), 2),
        "exposure_pct": round(bars_in_market / max(total_bars, 1) * 100.0, 2),
        "turnover_x": round(
            sum(t.qty * t.entry_price + t.qty * t.exit_price for t in closed) / initial_equity, 3
        ) if closed else 0.0,
        "period_start": pd.Timestamp(start_ts, unit="s", tz="UTC").isoformat(),
        "period_end": pd.Timestamp(end_ts, unit="s", tz="UTC").isoformat(),
        "period_years": round(years, 3),
        "timeframe": timeframe,
        "regime_breakdown": by_regime,
        "monthly_returns": monthly_returns(equity_df),
        "best_trade_pct": _safe(max((t.return_pct for t in closed), default=None)),
        "worst_trade_pct": _safe(min((t.return_pct for t in closed), default=None)),
        "sample_warning": "fewer than 10 trades: metrics are statistically weak"
        if len(closed) < 10 else None,
        "equity_trough_ts": int(ts[trough_idx]) if len(ts) else None,
    }
    return metrics


def buy_and_hold_benchmark(df: pd.DataFrame, initial_equity: float, *,
                           fee_bps: float, slippage_bps: float, spread_bps: float) -> dict:
    """Buy at the first bar's open and hold to the end, paying entry costs once."""
    if df.empty:
        return {"error": "empty data"}
    entry_ref = float(df["open"].iloc[0])
    cost = (fee_bps + slippage_bps + spread_bps / 2.0) / 10_000.0
    entry = entry_ref * (1.0 + cost)
    qty = initial_equity / (entry * (1.0 + fee_bps / 10_000.0))
    cash_left = initial_equity - qty * entry - qty * entry * fee_bps / 10_000.0
    eq = cash_left + qty * df["close"].to_numpy(dtype=float)
    curve = pd.DataFrame({
        "ts": df["bar_ts"].to_numpy(dtype=np.int64) + 0,  # close of each bar
        "equity": eq,
    })
    final = float(eq[-1])
    mdd, _ = max_drawdown(eq)
    days = max((int(df["bar_ts"].iloc[-1]) - int(df["bar_ts"].iloc[0])) / 86_400.0, 1e-9)
    years = days / 365.25
    rets = daily_returns(curve)
    sharpe, _ = sharpe_sortino(rets)
    return {
        "name": "BTC buy & hold",
        "initial_equity": round(initial_equity, 2),
        "final_equity": round(final, 2),
        "net_return_pct": _safe((final / initial_equity - 1.0) * 100.0),
        "cagr_pct": _safe(((final / initial_equity) ** (1.0 / years) - 1.0) * 100.0),
        "max_drawdown_pct": _safe(mdd),
        "sharpe": sharpe,
        "entry_price_ref": entry_ref,
        "qty": round(qty, 8),
        "assumptions": {
            "entry": "first bar open of the evaluation window",
            "costs_bps_round_trip_entry": round(cost * 10_000, 2),
            "note": "one entry, no exit, no rebalancing; spot long only",
        },
    }


def cash_benchmark(initial_equity: float) -> dict:
    return {
        "name": "Cash (no trading)",
        "initial_equity": round(initial_equity, 2),
        "final_equity": round(initial_equity, 2),
        "net_return_pct": 0.0,
        "cagr_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "note": "uninvested quote currency, ignores risk-free yield",
    }
