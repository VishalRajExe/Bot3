from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from btcbot.api import create_app
from btcbot.backtest import run_backtest
from btcbot.config import Config
from btcbot.data.store import CandleStore
from btcbot.db import Database
from btcbot.strategies import build_strategy
from tests.helpers import make_candles


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data.db_path = str(tmp_path / "api.sqlite3")
    db = Database(cfg.db_path)
    df = make_candles(400, seed=6)
    CandleStore(db).upsert(df, cfg.exchange.pair, "1d")

    def boom(_cfg):
        raise RuntimeError("offline in tests")

    monkeypatch.setattr("btcbot.api.get_ticker", boom)

    strat = build_strategy("ema_std")
    run_backtest(cfg, db, strat, df, persist=True, notes="api test")

    app = create_app(cfg, db)
    with TestClient(app) as c:
        yield c
    db.close()


def test_health_declares_paper_only(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["trading_mode"] == "paper"
    assert body["live_orders_enabled"] is False
    assert body["leverage"] == 1
    assert body["backtest_runs"] >= 1


def test_market_endpoint_serves_cached_candles_and_survives_offline_ticker(client):
    body = client.get("/api/market?timeframe=1d&limit=120").json()
    assert body["symbol"] == "BTC/USDT"
    assert len(body["candles"]["close"]) == 120
    assert len(body["candles"]["time"]) == 120
    assert body["spot"]["price"] is None
    assert body["quality"]["status"] in ("ok", "warn")
    assert body["regime"] is not None


def test_runs_and_detail_are_consistent(client):
    runs = client.get("/api/runs").json()["runs"]
    assert runs
    run = runs[0]
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["metrics"]["trades"] == run["trades"]
    assert detail["metrics"]["net_return_pct"] == run["net_return_pct"]
    assert len(detail["equity"]) > 100
    assert len(detail["markers"]) == run["trades"]
    if detail["trades"]:
        t = detail["trades"][0]
        assert t["entry_price"] > 0 and t["exit_price"] > 0


def test_trades_endpoint(client):
    body = client.get("/api/trades?mode=backtest").json()
    assert "trades" in body and "summary" in body
    assert body["summary"]["trades"] == len(body["trades"])


def test_config_exposes_no_secrets(client):
    body = client.get("/api/config").json()
    assert body["trading"]["enable_live_orders"] is False
    assert "api_key" not in str(body).lower()
    assert "secret" not in str(body).lower()


def test_paper_endpoint_reports_no_session_before_start(client):
    body = client.get("/api/paper").json()
    assert "session" in body


def test_backtest_endpoint_runs(client):
    resp = client.post("/api/backtest", json={"strategy": "ema_std", "timeframe": "1d"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"].startswith("bt_")
    assert body["metrics"]["trades"] >= 0
    assert "buy_hold" in body["benchmark"]


def test_unknown_strategy_is_400(client):
    assert client.post("/api/backtest", json={"strategy": "nope"}).status_code == 400


def test_learning_and_sentiment_endpoints(client):
    learning = client.get("/api/learning").json()
    assert "experiments" in learning and "models" in learning
    sent = client.get("/api/sentiment").json()
    assert "note" in sent
