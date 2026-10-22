"""Tests for the unified engine's core maths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unibot.config import CostsCfg, RiskCfg, Settings, SymbolCfg
from unibot.features import add_features, attach_market_context, resample_ohlcv, \
    triple_barrier_labels
from unibot.regime import RegimeState, build_market_series, regime_at
from unibot.risk import OpenRisk, RiskManager
from unibot.strategies import BreakoutParams, MeanRevParams, TrendBreakout


def frame(n: int = 400, *, start: float = 100.0, drift: float = 0.4,
          noise: float = 1.0, seed: int = 3, tf: int = 14_400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(np.full(n, drift) + rng.normal(0, noise, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, noise * 0.4, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, noise, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, noise, n))
    ts = np.arange(n, dtype=np.int64) * tf + (1_700_000_000 // 86_400) * 86_400
    return pd.DataFrame({
        "bar_ts": ts, "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(100, 300, n), "quote_volume": rng.uniform(1e4, 3e4, n),
        "trades": rng.integers(50, 200, n), "close_time": ts + tf, "source": "test",
    })


def settings_for(symbol: str = "BTC/USD", *, strategy: str = "trend_breakout",
                 params: dict | None = None, capital: float = 100.0) -> Settings:
    return Settings(
        capital=capital,
        universe=[SymbolCfg(symbol=symbol, cluster="btc", strategy=strategy,
                            params=params or {})],
    )


# ------------------------------------------------------------------ features
def test_resample_keeps_only_full_buckets():
    df = frame(24)
    base = df.copy()
    base["dt"] = pd.to_datetime(base["bar_ts"], unit="s", utc=True)
    out = resample_ohlcv(base, "4h", "1d")
    assert len(out) == 4  # 24 x 4h == 4 full days
    assert out.iloc[0]["open"] == pytest.approx(base.iloc[0]["open"])
    assert out.iloc[0]["close"] == pytest.approx(base.iloc[5]["close"])
    assert out.iloc[0]["volume"] == pytest.approx(base.iloc[:6]["volume"].sum())

    partial = base.iloc[:18]
    out2 = resample_ohlcv(partial, "4h", "1d")
    assert len(out2) == 3  # the incomplete day is dropped, not filled


def test_features_are_causal_under_append():
    raw = frame(300)
    base = add_features(raw)
    before = base[["ema200", "rsi14", "atr_pct", "don_hi_55", "bb_lower"]].iloc[-1].to_dict()

    extra = raw.iloc[[-1]].copy()
    new_ts = int(raw.iloc[-1]["bar_ts"]) + 14_400
    extra["bar_ts"] = new_ts
    extra["close_time"] = new_ts + 14_400
    longer = add_features(pd.concat([raw, extra], ignore_index=True))
    after = longer[["ema200", "rsi14", "atr_pct", "don_hi_55", "bb_lower"]].iloc[-2].to_dict()
    for key in before:
        assert before[key] == pytest.approx(after[key], rel=1e-10), key


def test_market_context_uses_only_completed_daily_bars():
    bars = add_features(frame(50, tf=14_400))
    market = pd.DataFrame({
        "known_ts": [bars.iloc[10]["bar_ts"] + 86_400 + 60],  # known one minute after close
        "score": [0.8], "breadth": [0.7], "btc_trend": [12.0], "vol_pct": [0.3],
        "label": ["bull"],
    })
    merged = attach_market_context(bars, market)
    # the bar whose close is after known_ts sees it; earlier bars do not
    assert merged.iloc[20]["market_score"] == pytest.approx(0.8)
    assert merged.iloc[2]["market_score"] == 0.0
    assert merged.iloc[20]["market_label"] == "bull"


# -------------------------------------------------------------------- labels
def test_triple_barrier_labels_win_loss_and_timeout():
    n = 40
    df = add_features(frame(n, drift=0.0, noise=0.1))
    df["atr14"] = 1.0
    # Construct explicit paths: up, down, flat.
    df.loc[df.index[5], ["open", "high", "low", "close"]] = [100, 100.2, 99.9, 100]
    df.loc[df.index[6], ["open", "high", "low", "close"]] = [100, 100.5, 99.8, 100.4]
    df.loc[df.index[7], ["high", "low", "close"]] = [103.0, 100.3, 102.8]  # +2 ATR hit
    df.loc[df.index[8], ["high", "low", "close"]] = [101, 99.0, 99.5]
    df.loc[df.index[9], ["high", "low", "close"]] = [100, 96.0, 97.0]     # -1 ATR hit

    labels = triple_barrier_labels(df, horizon=6, upper_atr=2.0, lower_atr=1.0, fee_pct=0.5)
    assert labels.iloc[5]["outcome"] == "win"
    assert labels.iloc[5]["label"] == 1
    assert labels.iloc[8]["outcome"] == "loss"
    assert labels.iloc[8]["label"] == 0


def test_triple_barrier_stop_wins_ties():
    df = add_features(frame(20, drift=0.0, noise=0.05))
    df["atr14"] = 1.0
    df.loc[df.index[5], ["open", "high", "low", "close"]] = [100, 100, 100, 100]
    df.loc[df.index[6], ["open", "high", "low", "close"]] = [100, 110, 90, 100]  # both hit
    labels = triple_barrier_labels(df, horizon=3, upper_atr=2.0, lower_atr=1.0, fee_pct=0.0)
    assert labels.iloc[5]["outcome"] == "loss"


# --------------------------------------------------------------------- regime
def test_regime_labels_and_causality():
    daily = {}
    for sym, drift in (("BTC/USD", 0.6), ("ETH/USD", 0.5), ("SOL/USD", 0.4)):
        d = frame(500, drift=drift, noise=0.8, seed=hash(sym) % 1000, tf=86_400)
        daily[sym] = d
    market = build_market_series(daily)
    assert not market.empty
    assert set(market["label"]).issubset({"bull", "chop", "bear"})
    assert market["known_ts"].is_monotonic_increasing

    ts = int(market.iloc[250]["known_ts"])
    state = regime_at(market, ts)
    assert isinstance(state, RegimeState)
    # a time before the series starts cannot see the future
    assert regime_at(market, 0).label == "chop"


# ----------------------------------------------------------------------- risk
def test_position_size_respects_risk_cap_and_reserve():
    rm = RiskManager(RiskCfg(risk_per_trade_pct=1.0, max_position_pct=35.0,
                             cash_reserve_pct=25.0), CostsCfg())
    d = rm.position_size(equity=100, cash=100, price=100, stop=95.0,
                         regime=RegimeState("bull", 1.0, 10.0, 0.8, 0.2))
    assert d.approved
    assert d.size_usd <= 35.0
    assert d.risk_usd <= 1.0 + 1e-9
    assert d.size_usd == pytest.approx(20.0, rel=1e-6)  # $1 risk / 5% stop


def test_position_size_rejects_tight_stop_for_fees():
    rm = RiskManager(RiskCfg(), CostsCfg())
    d = rm.position_size(equity=100, cash=100, price=100, stop=99.9,
                         regime=RegimeState("bull", 1.0, 10.0, 0.8, 0.2))
    assert not d.approved
    assert "too tight" in d.reason


def test_position_size_respects_cash_reserve():
    rm = RiskManager(RiskCfg(risk_per_trade_pct=2.0, max_position_pct=50.0,
                             cash_reserve_pct=60.0), CostsCfg())
    d = rm.position_size(equity=100, cash=100, price=100, stop=96.0,
                         regime=RegimeState("bull", 1.0, 10.0, 0.8, 0.2))
    assert d.approved
    assert d.size_usd == pytest.approx(40.0, rel=1e-6)  # only 40% spendable


def test_can_enter_blocks_bear_regime_positions_and_clusters():
    rm = RiskManager(RiskCfg(max_positions=2, max_per_cluster=1), CostsCfg())
    bear = RegimeState("bear", -0.8, -20, 0.2, 0.8)
    d = rm.can_enter(symbol="ETH/USD", cluster="l1", equity=100, cash=100,
                     open_positions=[], open_risk=[], regime=bear, breaker={},
                     cooldown_until=0, now=1000, bar_return_pct=0.0)
    assert not d.approved and "bear" in d.reason

    bull = RegimeState("bull", 0.7, 8, 0.7, 0.3)
    d = rm.can_enter(symbol="SOL/USD", cluster="l1", equity=100, cash=100,
                     open_positions=[{"symbol": "ETH/USD", "cluster": "l1"}],
                     open_risk=[OpenRisk("ETH/USD", "l1", 20, 1.0, 5.0)],
                     regime=bull, breaker={}, cooldown_until=0, now=1000,
                     bar_return_pct=1.0)
    assert not d.approved and "cluster" in d.reason

    d = rm.can_enter(symbol="SOL/USD", cluster="l1", equity=100, cash=100,
                     open_positions=[{"symbol": "ETH/USD", "cluster": "l1"}],
                     open_risk=[], regime=bull, breaker={"daily_halted": True},
                     cooldown_until=0, now=1000, bar_return_pct=1.0)
    assert not d.approved and "daily" in d.reason


def test_breaker_thresholds():
    rm = RiskManager(RiskCfg(daily_loss_limit_pct=3.0, max_drawdown_pct=12.0), CostsCfg())
    b = rm.breaker_state(equity=97.0, hwm=100.0, day_start=100.0)
    assert b["daily_halted"] is True
    assert b["drawdown_halted"] is False
    b = rm.breaker_state(equity=87.0, hwm=100.0, day_start=100.0)
    assert b["drawdown_halted"] is True


# ----------------------------------------------------------------- strategies
def test_breakout_requires_new_high_and_volume():
    df = add_features(frame(300, drift=0.8, noise=0.3))
    strat = TrendBreakout({"donchian": 20, "trend_ema": 50, "min_atr_pct": 0.0,
                           "max_atr_pct": 100.0, "min_vol_ratio": 1.0})
    df.loc[df.index[-1], "close"] = df.iloc[-1]["don_hi_20"] + 1.0
    df.loc[df.index[-1], "high"] = df.iloc[-1]["close"] + 0.5
    df.loc[df.index[-1], "vol_ratio"] = 2.0
    sig = strat._signal(df, len(df) - 1, None)
    assert sig.action == "enter_long", sig.reason
    assert sig.stop_price < float(df.iloc[-1]["close"])


def test_mean_reversion_needs_range_and_stretch():
    from unibot.strategies import MeanReversion

    df = add_features(frame(200, drift=0.0, noise=0.6))
    df.loc[df.index[-1], "close"] = df.iloc[-1]["bb_lower"] - 0.5
    df.loc[df.index[-1], "rsi14"] = 25.0
    df.loc[df.index[-1], "adx14"] = 10.0
    df.loc[df.index[-1], "atr_pct"] = 2.0
    strat = MeanReversion({"min_atr_pct": 0.0, "max_atr_pct": 50.0})
    sig = strat._signal(df, len(df) - 1, None)
    assert sig.action == "enter_long"
    assert sig.stop_price < float(df.iloc[-1]["close"])

    df.loc[df.index[-1], "adx14"] = 40.0
    sig = strat._signal(df, len(df) - 1, None)
    assert sig.action == "hold"


def test_warmup_always_holds():
    df = add_features(frame(300))
    strat = TrendBreakout({"donchian": 55, "trend_ema": 100})
    assert strat.signal(df, 5, None).action == "hold"
    assert "warm-up" in strat.signal(df, 5, None).reason


def test_invalid_params_are_rejected():
    with pytest.raises(Exception):
        BreakoutParams(donchian=37)
    with pytest.raises(Exception):
        MeanRevParams(bb_std=-1)
