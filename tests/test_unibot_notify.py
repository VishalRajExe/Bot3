"""Tests for upgraded Telegram notifications, lifecycle events, and deduplication."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from unibot.notify import TelegramNotifier, _format_duration


def test_format_duration():
    assert _format_duration(0.25) == "15m"
    assert _format_duration(1.0) == "1h"
    assert _format_duration(2.5) == "2h 30m"
    assert _format_duration(25.0) == "1d 1h"
    assert _format_duration(49.25) == "2d 1h 15m"
    assert _format_duration(0.0) == "0m"


class MockResponse:
    def __init__(self, data: dict):
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_telegram_order_lifecycle():
    notifier = TelegramNotifier("fake_token", "fake_chat", enabled=True)
    with patch("urllib.request.urlopen", return_value=MockResponse({"ok": True})) as mock_url:
        # 1. signal
        ok_sig = notifier.signal(
            symbol="BTC/USD", action="enter_long", price=82500.0,
            reason="broke 20-bar high", strategy="trend_breakout",
            regime="bull", stop=80000.0, prob=0.68,
            indicators={"rsi14": 58.2, "vol_ratio": 1.4},
        )
        assert ok_sig
        assert mock_url.called

        # 2. order submitted
        ok_sub = notifier.order_submitted(
            symbol="BTC/USD", side="buy", qty=0.001, price=82500.0,
            size_usd=82.5, order_id="ord_test_123", reason="breakout setup",
        )
        assert ok_sub

        # 3. order filled
        ok_fill = notifier.order_filled(
            symbol="BTC/USD", side="buy", qty=0.001, fill_price=82510.0,
            notional=82.51, stop_price=80000.0, stop_order_id="stop_test_456",
            prob=0.68, strategy="trend_breakout", regime="bull", equity=100.50,
        )
        assert ok_fill

        # 4. trade closed profit
        ok_close_win = notifier.trade_closed(
            symbol="BTC/USD", entry=82510.0, exit_price=85000.0,
            net_pnl=2.45, return_pct=3.02, reason="take profit target",
            holding_hours=18.5, equity=102.95, is_stop=False,
        )
        assert ok_close_win

        # 5. trade closed stop loss
        ok_close_loss = notifier.trade_closed(
            symbol="BTC/USD", entry=82510.0, exit_price=80000.0,
            net_pnl=-2.55, return_pct=-3.04, reason="stop_loss hit",
            holding_hours=6.0, equity=97.95, is_stop=True,
        )
        assert ok_close_loss


def test_telegram_digests():
    notifier = TelegramNotifier("fake_token", "fake_chat", enabled=True)
    with patch("urllib.request.urlopen", return_value=MockResponse({"ok": True})) as mock_url:
        # positions digest
        ok_pos = notifier.positions_digest([
            {"symbol": "BTC/USD", "entry_price": 80000.0, "current_price": 82000.0,
             "stop_price": 78000.0},
        ], equity=102.5, cash=20.0)
        assert ok_pos

        # portfolio digest
        ok_port = notifier.portfolio_digest(
            equity=102.5, cash=20.0, day_pnl=2.5, open_positions=1,
            closed_trades=10, win_rate=60.0, regime="bull",
        )
        assert ok_port


def test_telegram_deduplication():
    notifier = TelegramNotifier("fake_token", "fake_chat", enabled=True, dedup_window_seconds=600)
    with patch("urllib.request.urlopen", return_value=MockResponse({"ok": True})) as mock_url:
        # First send should hit urlopen
        r1 = notifier.send("Identical warning message", kind="risk", symbol="BTC/USD")
        assert r1
        assert mock_url.call_count == 1

        # Second send within dedup window should be suppressed and return True without calling urlopen
        r2 = notifier.send("Identical warning message", kind="risk", symbol="BTC/USD")
        assert r2
        assert mock_url.call_count == 1

        # Force send bypasses deduplication
        r3 = notifier.send("Identical warning message", kind="risk", symbol="BTC/USD", force=True)
        assert r3
        assert mock_url.call_count == 2
