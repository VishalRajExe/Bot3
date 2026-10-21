"""Alternative.me Fear & Greed Index.

This is the only sentiment source with a long, free, structured history that we
could verify.  Attribution: data by Alternative.me (https://alternative.me/crypto/fear-and-greed-index/).

Timing rule: the daily value is stamped at 00:00 UTC for the day it describes.
To stay strictly conservative, we treat it as knowable only at the **end** of
that day (``available_ts = ts + 86400``).  A backtest can therefore never use a
sentiment value before it existed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from ..db import Database, utcnow_iso

log = logging.getLogger(__name__)

FNG_URL = "https://api.alternative.me/fng/"
SOURCE = "alternative.me:fear_greed"
ATTRIBUTION = "Fear & Greed Index by Alternative.me"


def fetch_fng(limit: int = 0, timeout: float = 15.0) -> list[dict]:
    resp = requests.get(FNG_URL, params={"limit": limit, "format": "json"},
                        timeout=timeout,
                        headers={"User-Agent": "btcbot-research/0.1"})
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload:
        raise ValueError(f"unexpected Fear & Greed payload: {str(payload)[:200]}")
    return payload["data"]


def store_fng(db: Database, data: list[dict] | None = None) -> int:
    data = data if data is not None else fetch_fng(0)
    rows = []
    for item in data:
        try:
            ts = int(item["timestamp"])
            value = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((
            SOURCE, ts, ts + 86_400, "fear_greed", value,
            str(item.get("value_classification", "")),
            json.dumps({"attribution": ATTRIBUTION}), utcnow_iso(),
        ))
    if rows:
        db.executemany(
            "INSERT OR REPLACE INTO sentiment_observations (source, ts, available_ts, kind,"
            " value, label, payload_json, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    log.info("stored %d Fear & Greed observations", len(rows))
    return len(rows)


def load_fng(db: Database, start_ts: int | None = None) -> pd.DataFrame:
    sql = ("SELECT ts, available_ts, value, label FROM sentiment_observations "
           "WHERE kind='fear_greed'")
    params: list = []
    if start_ts is not None:
        sql += " AND ts >= ?"
        params.append(int(start_ts))
    sql += " ORDER BY ts"
    rows = db.query(sql, params)
    if not rows:
        return pd.DataFrame(columns=["ts", "available_ts", "value", "label", "available_dt"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["available_dt"] = pd.to_datetime(df["available_ts"], unit="s", utc=True)
    return df


def latest_value(db: Database) -> dict | None:
    row = db.query_one(
        "SELECT ts, value, label FROM sentiment_observations WHERE kind='fear_greed'"
        " ORDER BY ts DESC LIMIT 1"
    )
    if row is None:
        return None
    return {
        "value": row["value"],
        "label": row["label"],
        "date": datetime.fromtimestamp(row["ts"], tz=timezone.utc).date().isoformat(),
        "attribution": ATTRIBUTION,
    }
