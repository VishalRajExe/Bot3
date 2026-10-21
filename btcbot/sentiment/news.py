"""Crypto news headlines via public RSS, scored with a small lexicon.

Scope and honesty
-----------------
* Only **headlines, timestamps and links** are stored, locally, for research
  context.  Article text is not copied or republished.
* Publisher terms differ and can restrict automated collection; this module is
  for personal research and keeps attribution/link per item.  See
  ``LICENSE_AUDIT.md``.  It is not used to make trading decisions by default.
* Reddit's unauthenticated JSON endpoint returned HTTP 403 during research and
  is deliberately not integrated.
* Lexicon scoring is a crude baseline, recorded as such.  No claim is made that
  it predicts returns; the research report measures that explicitly.

RSS endpoints verified reachable during development:
  * https://www.coindesk.com/arc/outboundfeeds/rss/
  * https://cointelegraph.com/rss
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

from ..db import Database, utcnow_iso

log = logging.getLogger(__name__)

FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
}

POSITIVE = {
    "surge", "surges", "rally", "rallies", "soar", "soars", "gain", "gains", "bullish",
    "record", "high", "highs", "adoption", "approval", "approved", "inflow", "inflows",
    "buy", "buying", "upgrade", "partnership", "breakout", "recover", "recovers", "boost",
    "optimistic", "growth", "accumulate", "accumulation", "demand", "etf", "institutional",
}
NEGATIVE = {
    "crash", "crashes", "plunge", "plunges", "drop", "drops", "fall", "falls", "bearish",
    "hack", "hacked", "exploit", "fraud", "scam", "lawsuit", "ban", "banned", "outflow",
    "outflows", "sell", "selling", "liquidation", "liquidations", "fear", "warning", "risk",
    "decline", "declines", "slump", "panic", "bankruptcy", "default", "regulatory", "probe",
}

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text or "").strip()


def score_headline(title: str) -> float:
    words = re.findall(r"[a-z]+", title.lower())
    if not words:
        return 0.0
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def parse_feed(xml_text: str, source: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        ts = None
        if pub:
            try:
                ts = int(datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").timestamp())
            except ValueError:
                try:
                    ts = int(datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
                             .replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                    ts = None
        if ts is None:
            ts = int(time.time())
        items.append({
            "source": source,
            "ts": ts,
            "title": title[:300],
            "link": link[:500],
            "score": score_headline(title),
            "hash": hashlib.sha256(f"{source}:{title}".encode()).hexdigest()[:16],
        })
    return items


def fetch_feed(source: str, url: str, timeout: float = 12.0) -> list[dict]:
    resp = requests.get(url, timeout=timeout,
                        headers={"User-Agent": "btcbot-research/0.1 (+personal research)"})
    resp.raise_for_status()
    return parse_feed(resp.text, source)


def collect(db: Database, feeds: dict[str, str] | None = None) -> dict:
    feeds = feeds or FEEDS
    total = 0
    errors: dict[str, str] = {}
    for source, url in feeds.items():
        try:
            items = fetch_feed(source, url)
        except Exception as exc:  # noqa: BLE001 - a dead feed must not break the system
            errors[source] = str(exc)
            log.warning("news feed %s failed: %s", source, exc)
            continue
        rows = []
        for it in items:
            rows.append((
                f"{source}:headline", it["ts"], it["ts"], "news_headline", it["score"],
                it["title"], json.dumps({"link": it["link"], "hash": it["hash"],
                                         "source": source}),
                utcnow_iso(),
            ))
        if rows:
            db.executemany(
                "INSERT OR IGNORE INTO sentiment_observations (source, ts, available_ts, kind,"
                " value, label, payload_json, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
        total += len(rows)
    return {"stored": total, "errors": errors}


def latest_headlines(db: Database, limit: int = 12) -> list[dict]:
    rows = db.query(
        "SELECT source, ts, value, label, payload_json FROM sentiment_observations"
        " WHERE kind='news_headline' ORDER BY ts DESC LIMIT ?", (int(limit),)
    )
    out = []
    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        out.append({
            "source": r["source"], "ts": r["ts"],
            "published": datetime.fromtimestamp(r["ts"], tz=timezone.utc).isoformat(),
            "score": r["value"], "title": r["label"], "link": payload.get("link"),
        })
    return out


def daily_news_features(db: Database) -> pd.DataFrame:
    rows = db.query(
        "SELECT ts, value FROM sentiment_observations WHERE kind='news_headline' ORDER BY ts"
    )
    if not rows:
        return pd.DataFrame(columns=["day", "news_count", "news_score"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["day"] = df["ts"] // 86_400
    agg = df.groupby("day").agg(news_count=("value", "size"), news_score=("value", "mean"))
    return agg.reset_index()
