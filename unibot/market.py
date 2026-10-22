"""Alpaca paper-trading and crypto market-data client.

Only the documented REST surface is used (no SDK), so the exact endpoints and
payloads are visible and testable.  Every request is retried on 429/5xx with
backoff, and the client refuses to talk to a non-paper trading host.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from btcbot.data.timeframes import tf_seconds, to_epoch

log = logging.getLogger(__name__)

ALPACA_TF = {"1h": "1Hour", "1d": "1Day", "15m": "15Min", "5m": "5Min", "1m": "1Min"}


class AlpacaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status in (429, 500, 502, 503, 504) or self.status is None


class AlpacaClient:
    def __init__(self, key: str, secret: str, *, paper_base: str, data_base: str,
                 timeout: float = 20.0) -> None:
        if not key or not secret:
            raise ValueError("Alpaca credentials are missing")
        if "paper-api" not in paper_base:
            raise ValueError("refusing a non-paper Alpaca trading endpoint")
        self.key = key
        self.secret = secret
        self.paper_base = paper_base.rstrip("/")
        self.data_base = data_base.rstrip("/")
        self.timeout = timeout
        self._last_call = 0.0

    # ------------------------------------------------------------- transport
    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
                "accept": "application/json", "content-type": "application/json"}

    def _throttle(self) -> None:
        gap = 0.32  # stays well under the 200 requests/minute basic limit
        wait = gap - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _request(self, method: str, url: str, *, params: dict | None = None,
                 body: dict | None = None, retries: int = 4) -> dict:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        payload = json.dumps(body).encode() if body is not None else None
        last: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            req = urllib.request.Request(url, data=payload, headers=self._headers(), method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                text = exc.read()[:800].decode(errors="ignore")
                err = AlpacaError(f"{method} {url.split('?')[0]} -> {exc.code}: {text}",
                                  exc.code, text)
                last = err
                if not err.retryable:
                    raise err
            except urllib.error.URLError as exc:
                last = AlpacaError(f"{method} {url.split('?')[0]} failed: {exc.reason}")
            time.sleep(min(2.0 ** attempt, 8.0))
        raise last or AlpacaError("request failed")

    # ----------------------------------------------------------------- trading
    def account(self) -> dict:
        return self._request("GET", f"{self.paper_base}/v2/account")

    def positions(self) -> list[dict]:
        return self._request("GET", f"{self.paper_base}/v2/positions")

    def orders(self, status: str = "open") -> list[dict]:
        return self._request("GET", f"{self.paper_base}/v2/orders", params={"status": status, "limit": 100})

    def assets(self) -> list[dict]:
        return self._request("GET", f"{self.paper_base}/v2/assets",
                             params={"asset_class": "crypto", "status": "active"})

    def submit_market_order(self, *, symbol: str, side: str, qty: float,
                            client_order_id: str | None = None) -> dict:
        body = {"symbol": symbol, "qty": f"{qty:.9f}", "side": side,
                "type": "market", "time_in_force": "gtc"}
        if client_order_id:
            body["client_order_id"] = client_order_id[:48]
        return self._request("POST", f"{self.paper_base}/v2/orders", body=body)

    def submit_stop_limit(self, *, symbol: str, qty: float, stop_price: float,
                          limit_price: float, client_order_id: str | None = None) -> dict:
        body = {"symbol": symbol, "qty": f"{qty:.9f}", "side": "sell",
                "type": "stop_limit", "time_in_force": "gtc",
                "stop_price": f"{stop_price:.9f}", "limit_price": f"{limit_price:.9f}"}
        if client_order_id:
            body["client_order_id"] = client_order_id[:48]
        return self._request("POST", f"{self.paper_base}/v2/orders", body=body)

    def cancel_order(self, order_id: str) -> None:
        try:
            self._request("DELETE", f"{self.paper_base}/v2/orders/{order_id}")
        except AlpacaError as exc:
            if exc.status not in (404, 422):
                raise

    def close_position(self, symbol: str) -> dict:
        return self._request("DELETE", f"{self.paper_base}/v2/positions/{urllib.parse.quote(symbol)}")

    def order(self, order_id: str) -> dict:
        return self._request("GET", f"{self.paper_base}/v2/orders/{order_id}")

    # -------------------------------------------------------------------- data
    def bars(self, symbol: str, timeframe: str, *, start: int | None = None,
             end: int | None = None, limit: int = 1000) -> pd.DataFrame:
        tf = ALPACA_TF.get(timeframe)
        if tf is None:
            raise ValueError(f"unsupported Alpaca timeframe {timeframe}")
        rows: list[dict] = []
        token: str | None = None
        step = tf_seconds(timeframe)
        while True:
            params: dict = {"symbols": symbol, "timeframe": tf, "limit": limit,
                            "sort": "asc"}
            if start is not None:
                params["start"] = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
            if end is not None:
                params["end"] = datetime.fromtimestamp(end, tz=timezone.utc).isoformat()
            if token:
                params["page_token"] = token
            data = self._request("GET", f"{self.data_base}/v1beta3/crypto/us/bars", params=params)
            chunk = (data.get("bars") or {}).get(symbol) or []
            rows.extend(chunk)
            token = data.get("next_page_token")
            if not token or not chunk:
                break
            if len(rows) > 400_000:
                break
        if not rows:
            return _empty_frame()
        df = pd.DataFrame(rows)
        df["bar_ts"] = [
            int(datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()) for t in df["t"]
        ]
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                "v": "volume", "n": "trades", "vw": "quote_volume"})
        df["quote_volume"] = df.get("quote_volume", pd.Series([float("nan")] * len(df)))
        df["trades"] = df.get("trades", pd.Series([0] * len(df)))
        df["close_time"] = df["bar_ts"] + step
        df["dt"] = pd.to_datetime(df["bar_ts"], unit="s", utc=True)
        df["source"] = "alpaca"
        cols = ["bar_ts", "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "close_time", "dt", "source"]
        out = df[cols].sort_values("bar_ts").drop_duplicates("bar_ts", keep="last")
        return out.reset_index(drop=True)

    def latest_prices(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            return {}
        data = self._request(
            "GET", f"{self.data_base}/v1beta3/crypto/us/latest/bars",
            params={"symbols": ",".join(symbols)},
        )
        out: dict[str, dict] = {}
        for sym, bar in (data.get("bars") or {}).items():
            px = bar.get("p") or bar.get("ap") or bar.get("bp")
            if px is None:
                px = bar.get("c") or bar.get("close")
            if px is None:
                continue
            out[sym] = {
                "price": float(px),
                "ts": int(datetime.fromisoformat(
                    str(bar["t"]).replace("Z", "+00:00")).timestamp()) if bar.get("t") else None,
            }
        return out


def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=["bar_ts", "open", "high", "low", "close", "volume",
                               "quote_volume", "trades", "close_time", "dt", "source"])
    df["bar_ts"] = df["bar_ts"].astype("int64")
    return df


def fetch_history(client: AlpacaClient, *, symbol: str, timeframe: str,
                  start: int, end: int | None = None) -> pd.DataFrame:
    """Fetch bars, tolerating Alpaca's 1000-bar pages and gaps."""
    df = client.bars(symbol, timeframe, start=start, end=end)
    if df.empty:
        return df
    df = df[(df["bar_ts"] >= start) & (df["close"] <= (end or 2**62))]
    return df.reset_index(drop=True)


def data_quality(df: pd.DataFrame, timeframe: str, *, now: int | None = None) -> dict:
    """Structural quality check with explicit gap accounting (no interpolation)."""
    if df.empty:
        return {"bars": 0, "status": "fail", "reason": "no data"}
    step = tf_seconds(timeframe)
    ts = df["bar_ts"].to_numpy("int64")
    gaps = int((pd.Series(ts).diff().dropna() > step).sum())
    missing = int(((pd.Series(ts).diff().dropna() // step) - 1).clip(lower=0).sum())
    bad = int(((df["high"] < df["low"]) | (df["close"] <= 0)).sum())
    zero_vol = int((df["volume"] <= 0).sum())
    last = int(ts[-1])
    stale_days = ((now or int(time.time())) - (last + step)) / 86_400.0
    status = "ok"
    if bad or missing > len(df) * 0.02:
        status = "fail"
    elif gaps or zero_vol:
        status = "warn"
    return {
        "bars": int(len(df)),
        "first": datetime.fromtimestamp(int(ts[0]), tz=timezone.utc).isoformat(),
        "last": datetime.fromtimestamp(last, tz=timezone.utc).isoformat(),
        "gaps": gaps,
        "missing_bars": missing,
        "missing_pct": round(missing / max(len(df) + missing, 1) * 100, 3),
        "bad_bars": bad,
        "zero_volume_bars": zero_vol,
        "stale_days": round(stale_days, 2),
        "status": status,
    }


def history_start_epoch(start: str) -> int:
    return to_epoch(start)


def iso_ts(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


__all__ = ["AlpacaClient", "AlpacaError", "fetch_history", "data_quality",
           "history_start_epoch", "iso_ts", "ALPACA_TF", "timedelta"]
