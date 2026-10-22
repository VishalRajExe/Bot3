"""Telegram notifications for every decision the bot makes.

Message classes:

* **signals** - buy/sell opportunity identified with indicators, target stop, and edge;
* **orders** - order submission and confirmed broker fill with price, qty, and stop;
* **exits** - take profit or stop loss filled with duration, net P&L, return %, and equity;
* **positions & hold digest** - periodic per-symbol status summary so "the bot is watching"
  is visible without noise;
* **portfolio digest** - total balance, cash, 24h P&L, win rate, and market regime;
* **risk & breakers** - circuit breakers, drawdown warnings, and broker reconcile events;
* **system events** - startup, heartbeat, and AI model training/promotion.

Delivery is best-effort: a Telegram outage is logged, never allowed to stop the
trading loop. A token-bucket limiter keeps the bot inside Telegram's limits, and
intelligent deduplication prevents repetitive alert spam.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

log = logging.getLogger(__name__)

ACTION_ICON = {
    "buy_signal": "\U0001F7E2",       # green circle
    "buy_submitted": "\U0001F535",    # blue circle
    "buy_filled": "\u2705",           # check mark
    "hold": "\u26AA",                 # white circle
    "exit_signal": "\U0001F534",      # red circle
    "exit_submitted": "\U0001F535",   # blue circle
    "exit_filled": "\U0001F4B0",      # money bag (profit)
    "stop_filled": "\U0001F6D1",      # stop sign
    "blocked": "\U0001F6AB",          # no entry
    "watch": "\U0001F440",            # eyes
    "risk": "\u26A0\uFE0F",           # warning
    "breaker": "\U0001F6A8",          # rotating beacon
    "portfolio": "\U0001F4C8",        # chart increasing
    "positions": "\U0001F4CA",        # bar chart
}


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}" if abs(value) >= 1 else f"${value:.4f}"


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}%"


def _format_duration(hours: float) -> str:
    """Format duration into days, hours, and minutes."""
    if hours < 0:
        hours = 0.0
    total_mins = int(round(hours * 60))
    days, rem_m = divmod(total_mins, 1440)
    h, m = divmod(rem_m, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or not parts:
        parts.append(f"{m}m")
    return " ".join(parts)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, *, enabled: bool = True,
                 rate_limit_per_minute: int = 18, store=None,
                 dedup_window_seconds: int = 600) -> None:
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(enabled and token and chat_id)
        self.store = store
        self.limit = max(1, rate_limit_per_minute)
        self.dedup_window = max(0, dedup_window_seconds)
        self._sent: deque[float] = deque()
        self._last_sent_hashes: dict[str, float] = {}
        self._failures = 0

    # ------------------------------------------------------------- transport
    def _allowed(self) -> bool:
        now = time.monotonic()
        while self._sent and now - self._sent[0] > 60:
            self._sent.popleft()
        return len(self._sent) < self.limit

    def send(self, text: str, *, kind: str = "info", symbol: str | None = None,
             retries: int = 3, force: bool = False) -> bool:
        if not self.enabled:
            if self.store:
                self.store.log_alert(kind=kind, symbol=symbol, text=text,
                                     delivered=False, error="telegram disabled")
            return False

        now = time.monotonic()
        dedup_key = f"{kind}:{symbol}:{hash(text)}"
        if not force and self.dedup_window > 0:
            last_time = self._last_sent_hashes.get(dedup_key)
            if last_time and (now - last_time < self.dedup_window):
                log.debug("telegram duplicate suppressed (%s %s)", kind, symbol)
                return True

        if not self._allowed():
            log.warning("telegram rate limit reached; dropping %s alert", kind)
            if self.store:
                self.store.log_alert(kind=kind, symbol=symbol, text=text,
                                     delivered=False, error="rate limited")
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text[:4000],
            "disable_web_page_preview": "true",
        }).encode()

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, data=payload, timeout=20) as resp:
                    ok = json.loads(resp.read()).get("ok", False)
                self._sent.append(time.monotonic())
                self._last_sent_hashes[dedup_key] = now
                if self.store:
                    self.store.log_alert(kind=kind, symbol=symbol, text=text, delivered=ok)
                return bool(ok)
            except Exception as exc:
                log.warning("telegram send failed (%s/%s): %s", attempt + 1, retries, exc)
                time.sleep(min(2 ** attempt, 6))

        self._failures += 1
        if self.store:
            self.store.log_alert(kind=kind, symbol=symbol, text=text, delivered=False,
                                 error="send failed")
        return False

    # --------------------------------------------------------------- signals & orders
    def signal(self, *, symbol: str, action: str, price: float, reason: str,
               strategy: str, regime: str, stop: float | None = None,
               prob: float | None = None, indicators: dict | None = None) -> bool:
        """Alert on trade opportunity identified by strategy."""
        is_buy = "buy" in action or "enter" in action
        icon = "\U0001F7E2" if is_buy else "\U0001F534"
        label = "BUY SIGNAL (ENTRY SETUP)" if is_buy else "EXIT SIGNAL"
        lines = [
            f"{icon} {label}  {symbol}",
            f"Price: {_money(price)}",
            f"Reason: {reason}",
            f"Strategy: {strategy}  |  Regime: {regime}",
        ]
        if stop:
            lines.append(f"Target Stop: {_money(stop)}")
        if prob is not None:
            lines.append(f"Model P(win): {prob:.2f}")
        if indicators:
            sample = {k: v for k, v in indicators.items()
                      if k in ("rsi14", "atr_pct", "vol_ratio", "adx14", "donchian_pos")}
            if sample:
                lines.append("Indicators: " + " · ".join(f"{k}={v}" for k, v in sample.items()))
        return self.send("\n".join(lines), kind=f"signal:{action}", symbol=symbol, force=True)

    def order_submitted(self, *, symbol: str, side: str, qty: float, price: float,
                        size_usd: float, order_id: str | None = None, reason: str = "") -> bool:
        """Alert when an order is submitted to the broker."""
        icon = "\U0001F535"  # blue circle
        lines = [
            f"{icon} ORDER SUBMITTED  {symbol}",
            f"Side: {side.upper()}  |  Est Price: {_money(price)}",
            f"Qty: {qty:.6g}  |  Notional: {_money(size_usd)}",
        ]
        if order_id:
            lines.append(f"Order ID: {order_id[:16]}...")
        if reason:
            lines.append(f"Reason: {reason}")
        return self.send("\n".join(lines), kind=f"order_submitted:{side}", symbol=symbol, force=True)

    def order_filled(self, *, symbol: str, side: str, qty: float, fill_price: float,
                     notional: float, stop_price: float | None = None,
                     stop_order_id: str | None = None, prob: float | None = None,
                     strategy: str = "", regime: str = "", equity: float | None = None) -> bool:
        """Alert when an entry order execution is confirmed by the broker."""
        icon = "\u2705"  # green check mark
        lines = [
            f"{icon} ORDER FILLED  {symbol}",
            f"Side: {side.upper()}  |  Fill Price: {_money(fill_price)}",
            f"Filled Qty: {qty:.6g}  |  Notional: {_money(notional)}",
        ]
        if stop_price:
            stop_dist = (fill_price - stop_price) / fill_price * 100.0 if fill_price else 0.0
            lines.append(f"Stop Loss: {_money(stop_price)} (-{stop_dist:.1f}%)")
        if stop_order_id:
            lines.append(f"Stop Order ID: {stop_order_id[:16]}...")
        if strategy:
            lines.append(f"Strategy: {strategy}  |  Regime: {regime}")
        if prob is not None:
            lines.append(f"Model P(win): {prob:.2f}")
        if equity is not None:
            lines.append(f"Account Equity: {_money(equity)}")
        return self.send("\n".join(lines), kind=f"order_filled:{side}", symbol=symbol, force=True)

    def decision(self, *, symbol: str, action: str, price: float, reason: str,
                 strategy: str, regime: str, size_usd: float | None = None,
                 stop: float | None = None, prob: float | None = None,
                 equity: float | None = None) -> bool:
        """Unified decision message (backward-compatible)."""
        icon = ACTION_ICON.get(action, "\u2139\uFE0F")
        label = action.replace("_", " ").upper()
        lines = [f"{icon} {label}  {symbol}",
                 f"Price: {_money(price)}",
                 f"Why: {reason}",
                 f"Strategy: {strategy}  |  Regime: {regime}"]
        if size_usd:
            lines.append(f"Size: {_money(size_usd)}")
        if stop:
            lines.append(f"Stop: {_money(stop)}")
        if prob is not None:
            lines.append(f"Model P(win): {prob:.2f}")
        if equity is not None:
            lines.append(f"Equity: {_money(equity)}")
        return self.send("\n".join(lines), kind=f"decision:{action}", symbol=symbol, force=True)

    def trade_closed(self, *, symbol: str, entry: float, exit_price: float, net_pnl: float,
                     return_pct: float, reason: str, holding_hours: float,
                     equity: float | None = None, is_stop: bool = False) -> bool:
        """Comprehensive exit confirmation with duration, PnL and account equity."""
        if is_stop or "stop" in reason.lower() or net_pnl < 0:
            icon = "\U0001F6D1"  # stop sign
            title = "STOP LOSS TRIGGERED" if ("stop" in reason.lower() or is_stop) else "TRADE CLOSED (LOSS)"
        else:
            icon = "\U0001F4B0"  # money bag
            title = "TAKE PROFIT / CLOSED (PROFIT)"

        dur = _format_duration(holding_hours)
        lines = [
            f"{icon} {title}  {symbol}",
            f"Entry: {_money(entry)} -> Exit: {_money(exit_price)}",
            f"Net P&L: {_money(net_pnl)} ({_pct(return_pct)})",
            f"Held: {dur}  |  Exit: {reason}",
        ]
        if equity is not None:
            lines.append(f"Account Equity: {_money(equity)}")
        return self.send("\n".join(lines), kind="trade_closed", symbol=symbol, force=True)

    # ----------------------------------------------------------- digests & status
    def positions_digest(self, positions: list[dict], *, equity: float | None = None,
                         cash: float | None = None) -> bool:
        """Digest of currently open positions and unrealized P&L."""
        if not positions:
            return False
        icon = "\U0001F4CA"  # bar chart
        lines = [f"{icon} OPEN POSITIONS ({len(positions)})"]
        for p in positions:
            sym = p["symbol"]
            entry = float(p.get("entry_price") or 0)
            cur = float(p.get("current_price") or entry)
            unrl = (cur - entry) / entry * 100.0 if entry else 0.0
            stop = float(p.get("stop_price") or 0)
            p_icon = "\U0001F7E2" if unrl >= 0 else "\U0001F534"
            stop_str = f" | stop {_money(stop)}" if stop else ""
            lines.append(f"{p_icon} {sym}: {_money(entry)} -> {_money(cur)} ({_pct(unrl)}){stop_str}")
        if equity is not None:
            eq_str = f"Equity: {_money(equity)}"
            if cash is not None:
                eq_str += f" | Cash: {_money(cash)}"
            lines.append(eq_str)
        return self.send("\n".join(lines), kind="positions_digest")

    def portfolio_digest(self, *, equity: float, cash: float, day_pnl: float | None = None,
                         open_positions: int = 0, closed_trades: int = 0,
                         win_rate: float | None = None, regime: str = "chop") -> bool:
        """Periodic portfolio balance and performance digest."""
        icon = "\U0001F4C8"  # chart increasing
        lines = [
            f"{icon} PORTFOLIO SUMMARY",
            f"Total Equity: {_money(equity)}  |  Cash: {_money(cash)}",
        ]
        if day_pnl is not None:
            day_pnl_pct = (day_pnl / (equity - day_pnl) * 100.0) if (equity - day_pnl) > 0 else 0.0
            lines.append(f"Day P&L: {_money(day_pnl)} ({_pct(day_pnl_pct)})")
        invested = max(0.0, equity - cash)
        inv_pct = (invested / equity * 100.0) if equity > 0 else 0.0
        lines.append(f"Positions: {open_positions} ({inv_pct:.1f}% invested)  |  Regime: {regime}")
        if closed_trades > 0 and win_rate is not None:
            lines.append(f"Closed Trades: {closed_trades}  |  Win Rate: {win_rate:.1f}%")
        return self.send("\n".join(lines), kind="portfolio_digest")

    def hold_digest(self, rows: list[dict], *, equity: float | None = None,
                    open_positions: int = 0) -> bool:
        """Watching digest summarizing symbols evaluated without new entries."""
        if not rows:
            return False
        lines = ["\U0001F440 WATCHING (no trade this cycle)"]
        for r in rows:
            lines.append(f"{r['symbol']}: {r['action'].replace('_', ' ')} - {r['reason'][:120]}")
        if equity is not None:
            lines.append(f"Equity: {_money(equity)} | Open positions: {open_positions}")
        return self.send("\n".join(lines), kind="hold_digest")

    def model_update(self, symbol: str, status: str, auc: float | None,
                     improvement: float | None, threshold: float | None) -> bool:
        """Alert on AI model walk-forward validation and promotion."""
        icon = "\U0001F9E0"  # brain
        lines = [f"{icon} MODEL {status.upper()}  {symbol}",
                 f"Out-of-sample AUC: {auc:.3f}" if auc is not None else "AUC: n/a",
                 f"Improvement: {improvement:+.3f} pp" if improvement is not None else
                 "Improvement: n/a"]
        if threshold is not None:
            lines.append(f"Gate Threshold: {threshold:.2f}")
        if status != "active":
            lines.append("Not used for trading (failed validation)")
        return self.send("\n".join(lines), kind="model", symbol=symbol, force=True)

    def risk(self, title: str, detail: str, symbol: str | None = None) -> bool:
        """Risk warning or circuit breaker alert."""
        icon = "\U0001F6A8" if "breaker" in title.lower() or "halt" in title.lower() else "\u26A0\uFE0F"
        return self.send(f"{icon} {title}\n{detail}", kind="risk", symbol=symbol, force=True)

    def startup(self, *, equity: float, symbols: int, mode: str) -> bool:
        """Bot online startup notification."""
        return self.send(
            f"\U0001F680 UniBot online ({mode})\n"
            f"Equity: {_money(equity)} | {symbols} markets | Paper only",
            kind="startup",
            force=True,
        )

    def heartbeat(self, *, equity: float, open_positions: int, regime: str,
                  trades: int) -> bool:
        """Periodic system heartbeat."""
        return self.send(
            f"\u2764\uFE0F heartbeat\nEquity: {_money(equity)} | Positions: {open_positions} | "
            f"Regime: {regime} | Closed trades: {trades}",
            kind="heartbeat",
        )
