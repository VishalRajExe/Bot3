"""API surface tests with a seeded temporary database and no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from btcbot.data.store import CandleStore
from unibot import api as api_module
from unibot.config import Credentials, Settings, SymbolCfg
from unibot.store import UnibotStore


def make_settings(tmp_path) -> Settings:
    return Settings(
        capital=100.0,
        universe=[SymbolCfg(symbol="BTC/USD", cluster="btc", strategy="trend_breakout"),
                  SymbolCfg(symbol="ETH/USD", cluster="l1", strategy="trend_pullback")],
        paths={"db": str(tmp_path / "api.sqlite3"), "models": str(tmp_path / "models"),
               "reports": str(tmp_path / "reports"), "dashboard": "web/unibot"},
    )


def seed_candles(store, symbol: str, timeframes=("1h", "1d")) -> None:
    candles = CandleStore(store.db)
    rng = np.random.default_rng(11)
    for tf, step, n in (("1h", 3600, 24 * 400), ("1d", 86_400, 400)):
        close = 30_000 + np.cumsum(rng.normal(0, 120, n))
        open_ = close + rng.normal(0, 40, n)
        high = np.maximum(open_, close) + 80
        low = np.minimum(open_, close) - 80
        ts = np.arange(n, dtype=np.int64) * step + 1_700_000_000
        df = pd.DataFrame({"bar_ts": ts, "open": open_, "high": high, "low": low,
                           "close": close, "volume": 100.0, "quote_volume": 1e6,
                           "trades": 100, "close_time": ts + step, "dt": pd.NaT,
                           "source": "test"})
        df["dt"] = pd.to_datetime(df["bar_ts"], unit="s", utc=True)
        candles.upsert(df, symbol, tf)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    store = UnibotStore(s.paths.db)
    seed_candles(store, "BTC/USD")
    seed_candles(store, "ETH/USD")
    store.save_portfolio("equity_snapshot", {"equity": 101.0, "cash": 80.0,
                                             "positions": 0, "regime": "chop",
                                             "ts": 1_700_000_000})
    store.record_decision(ts=1_700_000_000, bar_ts=1_700_000_000, mode="live",
                          symbol="BTC/USD", timeframe="4h", strategy="trend_breakout",
                          strategy_version="1", action="hold", reason="no breakout",
                          price=30_000.0, regime="chop", features={"rsi14": 50})
    store.save_regime(ts=1_700_000_000, label="chop", score=0.1, btc_trend=2.0,
                      breadth=0.5, vol_percentile=0.4, payload={})
    monkeypatch.setattr(api_module, "load_credentials",
                        lambda *a, **k: Credentials("", "", "", "", "https://paper-api.alpaca.markets"))
    app = api_module.create_app(s)
    with TestClient(app) as c:
        yield c
    store.close()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "paper"
    assert body["live_orders_enabled"] is False


def test_symbols_and_candles(client):
    r = client.get("/api/symbols")
    assert r.status_code == 200
    body = r.json()
    assert len(body["symbols"]) == 2
    assert body["symbols"][0]["price"] is not None

    r = client.get("/api/candles/BTC/USD?limit=100")
    assert r.status_code == 200
    payload = r.json()
    assert len(payload["candles"]) == 100
    c = payload["candles"][-1]
    assert c["h"] >= c["l"]
    assert "ema200" in c


def test_unknown_symbol_is_404(client):
    assert client.get("/api/candles/NOPE/USD").status_code == 404


def test_overview_tolerates_missing_broker(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["equity"] == 101.0
    assert body["paper"] is True
    assert body["regime"]["label"] == "chop"


def test_decisions_and_models(client):
    r = client.get("/api/decisions?limit=10")
    assert r.status_code == 200
    rows = r.json()["decisions"]
    assert rows[0]["symbol"] == "BTC/USD"
    assert rows[0]["features"]["rsi14"] == 50

    r = client.get("/api/models")
    assert r.status_code == 200
    assert r.json()["models"] == []


def test_backtests_empty_and_missing_run(client):
    r = client.get("/api/backtests")
    assert r.status_code == 200
    assert r.json()["runs"] == []
    assert client.get("/api/backtests/nope").status_code == 404


def test_data_and_risk(client):
    r = client.get("/api/data")
    assert r.status_code == 200
    coverage = r.json()["coverage"]
    assert len(coverage) == 2
    assert coverage[0]["1h"]["bars"] > 0

    r = client.get("/api/risk")
    assert r.status_code == 200
    assert r.json()["limits"]["risk_per_trade_pct"] == 1.0


def test_learning_endpoint(client):
    r = client.get("/api/learning")
    assert r.status_code == 200
    body = r.json()
    assert body["memory"]["trades_remembered"] == 0
    assert len(body["symbols"]) == 2
