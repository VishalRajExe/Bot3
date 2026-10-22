"""Data ingestion and prepared frames.

Candles live in the shared ``candles`` table with provenance.  This module only
ever *adds* new bars for a symbol/timeframe; it never interpolates gaps and it
records a quality report after each sync so the dashboard can show the truth
about the data the bot is trading on.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from btcbot.data.store import CandleStore, validate_frame
from btcbot.data.timeframes import tf_seconds

from .config import Settings
from .features import add_features, attach_market_context, resample_ohlcv
from .market import AlpacaClient, data_quality, fetch_history, history_start_epoch
from .regime import build_market_series

log = logging.getLogger(__name__)


def sync_symbol(store, client: AlpacaClient, symbol: str, timeframe: str,
                *, start: int, force: bool = False) -> dict:
    """Incrementally sync one symbol/timeframe.  Returns a small report."""
    candles = CandleStore(store.db)
    last = candles.last_open_time(symbol, timeframe)
    step = tf_seconds(timeframe)
    if last is None or force:
        fetch_from = start
    else:
        fetch_from = last  # re-fetch the last bar to capture corrections
    if not force and last is not None and int(time.time()) - last < step:
        return {"symbol": symbol, "timeframe": timeframe, "added": 0,
                "status": "fresh", "last": last}

    df = fetch_history(client, symbol=symbol, timeframe=timeframe, start=fetch_from)
    if df.empty:
        return {"symbol": symbol, "timeframe": timeframe, "added": 0, "status": "empty"}
    report = validate_frame(df, timeframe, symbol=symbol, max_gap_bars=2)
    candles.record_quality(report)
    added = candles.upsert(df[df["bar_ts"] < int(time.time())], symbol, timeframe)
    candles.record_source("alpaca", f"{client.data_base}/v1beta3/crypto/us/bars",
                          "crypto bars (research + paper parity; 1h/1d since 2021)")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "added": added,
        "status": report.status,
        "bars": report.bars,
        "missing": report.missing_bars,
        "first": report.details.get("first"),
        "last": report.details.get("last"),
    }


def sync_all(store, settings: Settings, client: AlpacaClient, *,
             force: bool = False, symbols: list[str] | None = None,
             timeframes: tuple[str, ...] = ("1h", "1d")) -> list[dict]:
    starts = {"1h": history_start_epoch(settings.hourly_start),
              "1d": history_start_epoch(settings.history_start)}
    out: list[dict] = []
    targets = symbols or settings.symbols
    for tf in timeframes:
        if tf == "1d":
            continue
        for sym in targets:
            try:
                rep = sync_symbol(store, client, sym, tf, start=starts[tf], force=force)
            except Exception as exc:  # recorded, not swallowed
                log.exception("sync failed for %s %s", sym, tf)
                rep = {"symbol": sym, "timeframe": tf, "added": 0, "status": "error",
                       "error": str(exc)[:300]}
            out.append(rep)
            log.info("sync %s %s: %s", sym, tf, rep.get("status"))
    if "1d" in timeframes:
        for sym in targets:
            try:
                rep = sync_symbol(store, client, sym, "1d", start=starts["1d"], force=force)
            except Exception as exc:
                log.exception("sync failed for %s 1d", sym)
                rep = {"symbol": sym, "timeframe": "1d", "added": 0, "status": "error",
                       "error": str(exc)[:300]}
            out.append(rep)
            log.info("sync %s 1d: %s", sym, rep.get("status"))
    return out


def load_bars(store, symbol: str, timeframe: str) -> pd.DataFrame:
    return CandleStore(store.db).load(symbol, timeframe)


def build_market(store, settings: Settings) -> pd.DataFrame:
    daily = {s.symbol: load_bars(store, s.symbol, "1d") for s in settings.universe}
    return build_market_series(daily)


def prepare_bars(store, settings: Settings, symbol: str, market: pd.DataFrame | None = None,
                 timeframe: str | None = None) -> pd.DataFrame:
    """Decision-timeframe bars with features and causal market context."""
    target_tf = timeframe or settings.timeframe
    if target_tf == "1d":
        bars = load_bars(store, symbol, "1d")
        if bars.empty:
            base = load_bars(store, symbol, settings.base_timeframe)
            bars = resample_ohlcv(base, settings.base_timeframe, "1d") if not base.empty else base
    else:
        base = load_bars(store, symbol, settings.base_timeframe)
        if base.empty:
            return base
        if settings.base_timeframe != target_tf:
            bars = resample_ohlcv(base, settings.base_timeframe, target_tf)
        else:
            bars = base
    if bars.empty:
        return bars
    features = add_features(bars)
    if market is None:
        market = build_market(store, settings)
    return attach_market_context(features, market, target_tf)


def prepare_all(store, settings: Settings) -> dict[str, pd.DataFrame]:
    market = build_market(store, settings)
    return {s.symbol: prepare_bars(store, settings, s.symbol, market)
            for s in settings.universe}


def coverage(store, settings: Settings) -> list[dict]:
    candles = CandleStore(store.db)
    out = []
    for s in settings.universe:
        row = {"symbol": s.symbol, "strategy": s.strategy, "cluster": s.cluster,
               "trade_enabled": s.trade_enabled}
        for tf in ("1h", "1d"):
            df = candles.load(s.symbol, tf)
            row[tf] = data_quality(df, tf) if not df.empty else {"bars": 0, "status": "empty"}
        out.append(row)
    return out
