"""Learning, model validation, persistence and notification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unibot.config import ModelCfg, Settings, SymbolCfg
from unibot.features import add_features
from unibot.learning import cohort_table, generate_lessons, learning_state
from unibot.models import (MODEL_FEATURES, ModelArtifact, _make_classifier,
                           walk_forward_splits)
from unibot.notify import TelegramNotifier
from unibot.store import UnibotStore


def settings(tmp_path) -> Settings:
    return Settings(
        capital=100.0,
        paths={"db": str(tmp_path / "t.sqlite3"), "models": str(tmp_path / "models"),
               "reports": str(tmp_path / "reports"), "dashboard": "web/unibot"},
        universe=[SymbolCfg(symbol="BTC/USD", cluster="btc", strategy="trend_breakout")],
    )


# --------------------------------------------------------------------- models
def test_model_artifact_roundtrip_and_predict(tmp_path):
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(200, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    y = (x["rsi14"] + rng.normal(0, 0.5, 200) > 0).astype(int)
    clf = _make_classifier(ModelCfg())
    clf.fit(x, y)
    art = ModelArtifact("BTC/USD", "trend_breakout", "4h", "v1", 0.55,
                        MODEL_FEATURES, clf, "2026-01-01T00:00:00Z", 0.6)
    path = art.save(tmp_path / "m.joblib")
    loaded = ModelArtifact.load(path)
    row = x.iloc[0].to_dict()
    assert 0.0 <= loaded.predict(row) <= 1.0
    assert loaded.threshold == 0.55
    assert loaded.features == MODEL_FEATURES


def test_walk_forward_splits_are_chronological():
    cfg = ModelCfg()
    ts = []
    start = 1_600_000_000
    for day in range(0, 1500):
        ts.append(start + day * 86_400)
    data = pd.DataFrame({"bar_ts": ts, "label": np.random.default_rng(2).integers(0, 2, len(ts))})
    splits = walk_forward_splits(data, cfg)
    assert splits
    for train, test in splits:
        assert train["bar_ts"].max() < test["bar_ts"].min()
        # no future observations inside a training block
        assert len(test) > 0


def test_cohorts_require_minimum_sample():
    memory = [
        {"symbol": "BTC/USD", "strategy": "trend_breakout", "net_pnl": 1.0,
         "return_pct": 1.0, "regime": "bull", "features_json": '{"atr_pct": 1.0}'},
        {"symbol": "BTC/USD", "strategy": "trend_breakout", "net_pnl": -0.5,
         "return_pct": -0.5, "regime": "chop", "features_json": '{"atr_pct": 6.0}'},
    ]
    table = cohort_table(memory)
    assert all(row["sample_ok"] is False for row in table)
    assert {r["kind"] for r in table} == {"symbol", "strategy", "regime", "volatility"}


# -------------------------------------------------------------------- memory
def test_stats_adapt_within_bounds_and_cooldown(tmp_path):
    store = UnibotStore(tmp_path / "s.sqlite3")
    for i in range(25):
        stats = store.apply_trade_outcome(
            symbol="BTC/USD", net_pnl=-0.4, return_pct=-0.4, ts=1000 + i,
            trade_id=f"t{i}", strategy="trend_breakout", features={"rsi14": 60},
            prob=0.5, regime="chop", cooldown_bars_seconds=3600,
            consecutive_limit=99, cooldown_hours=12, min_samples_for_adaptation=20,
        )
        assert 0.5 <= stats["risk_multiplier"] <= 1.25
    assert stats["consec_losses"] == 25
    assert stats["cooldown_until"] > 0
    assert len(store.memory()) == 25
    assert store.db.scalar("SELECT COUNT(*) FROM uni_memory") == 25

    for i in range(40):
        stats = store.apply_trade_outcome(
            symbol="BTC/USD", net_pnl=1.0, return_pct=1.0, ts=2000 + i,
            trade_id=f"w{i}", strategy="trend_breakout", features={}, prob=0.6,
            regime="bull", cooldown_bars_seconds=0, consecutive_limit=3,
            cooldown_hours=12, min_samples_for_adaptation=20,
        )
        assert 0.5 <= stats["risk_multiplier"] <= 1.25


def test_consecutive_losses_trigger_cooldown(tmp_path):
    store = UnibotStore(tmp_path / "c.sqlite3")
    stats = {}
    for i in range(3):
        stats = store.apply_trade_outcome(
            symbol="ETH/USD", net_pnl=-1.0, return_pct=-1.0, ts=1000 + i,
            trade_id=f"l{i}", strategy="trend_breakout", features={}, prob=0.4,
            regime="chop", cooldown_bars_seconds=3600, consecutive_limit=3,
            cooldown_hours=12, min_samples_for_adaptation=20,
        )
    assert stats["cooldown_until"] >= 1000 + 12 * 3600


def test_learning_state_payload(tmp_path):
    store = UnibotStore(tmp_path / "l.sqlite3")
    s = settings(tmp_path)
    state = learning_state(store, s)
    assert state["memory"]["trades_remembered"] == 0
    assert state["symbols"][0]["symbol"] == "BTC/USD"
    assert state["rules"]["auto_rule_changes"].startswith("none")
    assert generate_lessons(store) == []


# ---------------------------------------------------------------- notifier
def test_notifier_disabled_logs_and_returns_false(tmp_path):
    store = UnibotStore(tmp_path / "n.sqlite3")
    n = TelegramNotifier("", "", enabled=False, store=store)
    ok = n.decision(symbol="BTC/USD", action="hold", price=100.0, reason="nothing",
                    strategy="trend_breakout", regime="bull")
    assert ok is False
    alerts = store.alerts()
    assert alerts and alerts[0]["delivered"] == 0
    assert "telegram disabled" in alerts[0]["error"]


def test_notifier_formats_messages(monkeypatch):
    n = TelegramNotifier("token", "chat", enabled=True)
    sent = []
    monkeypatch.setattr(n, "send", lambda text, **kw: sent.append((text, kw)) or True)
    n.decision(symbol="ETH/USD", action="buy_signal", price=1234.5,
               reason="broke 55-bar high", strategy="trend_breakout", regime="bull",
               size_usd=25.0, stop=1190.0, prob=0.61, equity=100.0)
    text, kw = sent[0]
    assert "BUY SIGNAL" in text and "ETH/USD" in text
    assert "broke 55-bar high" in text
    assert "0.61" in text
    assert kw["symbol"] == "ETH/USD"


def test_hold_digest_lists_every_symbol():
    n = TelegramNotifier("t", "c", enabled=True)
    captured = []
    n.send = lambda text, **kw: (captured.append(text), True)[1]
    n.hold_digest([{"symbol": "BTC/USD", "action": "hold", "reason": "below high"},
                   {"symbol": "SOL/USD", "action": "hold", "reason": "RSI mid"}],
                  equity=100.0, open_positions=0)
    all_texts = " ".join(captured)
    assert "BTC/USD" in all_texts and "SOL/USD" in all_texts
