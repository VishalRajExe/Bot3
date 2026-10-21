from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from btcbot.data.store import (CandleFormatError, CandleStore, dataset_hash, parse_klines,
                              validate_frame)
from btcbot.data.timeframes import expected_bar_times, floor_to_bar, tf_seconds

BASE = 1_600_000_000 - (1_600_000_000 % 86_400)


def kline(ts: int, o=100.0, h=105.0, l=95.0, c=102.0, v=10.0):
    return [ts * 1000, str(o), str(h), str(l), str(c), str(v), ts * 1000 + 86_399_999,
            str(v * c), 42, str(v / 2), str(v * c / 2), "0"]


def test_parse_klines_types_and_alignment():
    raw = [kline(BASE + i * 86_400) for i in range(5)]
    df = parse_klines(raw, "1d", "binance")
    assert list(df.columns[:6]) == ["bar_ts", "open", "high", "low", "close", "volume"]
    assert df["bar_ts"].dtype == np.int64
    assert str(df["dt"].dt.tz) == "UTC"
    assert len(df) == 5


def test_parse_deduplicates_keeping_last():
    raw = [kline(BASE), kline(BASE, c=111.0)]
    df = parse_klines(raw, "1d", "binance")
    assert len(df) == 1
    assert df["close"].iloc[0] == 111.0


def test_validate_ok_and_gap_detection():
    raw = [kline(BASE + i * 86_400) for i in range(100)]
    df = parse_klines(raw, "1d", "binance")
    rep = validate_frame(df, "1d")
    assert rep.status == "ok" and rep.missing_bars == 0

    gapped = df.drop(index=[4, 5]).reset_index(drop=True)
    rep2 = validate_frame(gapped, "1d")
    assert rep2.missing_bars == 2
    assert rep2.status == "warn"


def test_validate_detects_bad_ohlc():
    raw = [kline(BASE + i * 86_400) for i in range(5)]
    raw[2][2] = "1"  # high far below low: incoherent
    df = parse_klines(raw, "1d", "binance")
    rep = validate_frame(df, "1d")
    assert rep.bad_ohlc >= 1
    assert rep.status == "fail"


def test_validate_detects_stale_data():
    raw = [kline(BASE + i * 86_400) for i in range(5)]
    df = parse_klines(raw, "1d", "binance")
    future = BASE + 30 * 86_400
    rep = validate_frame(df, "1d", now_s=future)
    assert rep.details["stale"] is True
    assert rep.status == "fail"


def test_validate_rejects_empty_and_off_grid():
    df = parse_klines([], "1d", "binance")
    assert validate_frame(df, "1d").status == "fail"
    off = parse_klines([kline(BASE + 1234)], "1d", "binance")
    with pytest.raises(CandleFormatError):
        validate_frame(off, "1d")


def test_dataset_hash_is_stable_and_sensitive():
    raw = [kline(BASE + i * 86_400, c=100 + i) for i in range(6)]
    df = parse_klines(raw, "1d", "binance")
    assert dataset_hash(df) == dataset_hash(df.copy())
    changed = df.copy()
    changed.loc[3, "close"] += 0.01
    assert dataset_hash(changed) != dataset_hash(df)


def test_store_upsert_load_and_summary(tmp_path):
    from btcbot.db import Database

    db = Database(tmp_path / "candles.sqlite3")
    store = CandleStore(db)
    raw = [kline(BASE + i * 86_400, c=100 + i) for i in range(20)]
    df = parse_klines(raw, "1d", "binance")
    assert store.upsert(df, "BTC/USDT", "1d") == 20
    loaded = store.load("BTC/USDT", "1d")
    assert len(loaded) == 20
    assert store.last_open_time("BTC/USDT", "1d") == int(df["bar_ts"].iloc[-1])
    summary = store.summary("BTC/USDT", "1d")
    assert summary["bars"] == 20
    # Re-upserting updates rather than duplicating.
    df.loc[0, "close"] = 999.0
    store.upsert(df, "BTC/USDT", "1d")
    assert store.count("BTC/USDT", "1d") == 20
    assert store.load("BTC/USDT", "1d")["close"].iloc[0] == 999.0
    db.close()


def test_timeframe_helpers():
    assert tf_seconds("4h") == 14_400
    assert floor_to_bar(BASE + 5, "1d") == BASE
    grid = expected_bar_times(BASE, BASE + 3 * 86_400, "1d")
    assert grid == [BASE, BASE + 86_400, BASE + 2 * 86_400]
