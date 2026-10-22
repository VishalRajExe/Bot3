"""Command line entry points: `python -m unibot <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from btcbot.data.timeframes import to_epoch

from .backtest import run_portfolio_backtest
from .config import Settings, load_credentials
from .datafeed import coverage, prepare_all, sync_all
from .market import AlpacaClient
from .models import load_active_gates, train_all
from .report import write_backtest_report, write_model_report, write_status_report
from .store import UnibotStore

log = logging.getLogger("unibot")

# Held open for the lifetime of the process so the OS keeps the lock.
_ROLE_LOCKS: dict[str, socket.socket] = {}
_LOCK_PORTS = {"paper": 8611, "serve": 8612}


def acquire_role_lock(role: str) -> bool:
    """Take an exclusive loopback port as a single-instance mutex.

    Two paper loops would place duplicate orders, so a second process must
    refuse to start.  A bound socket is used instead of a lock file because
    child processes can inherit an open file handle (and its lock) but cannot
    share a listening socket.
    """
    port = _LOCK_PORTS[role]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
    except OSError:
        sock.close()
        return False
    _ROLE_LOCKS[role] = sock
    return True


def setup_logging(level: str = "INFO") -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)
        file = RotatingFileHandler("logs/unibot.log", maxBytes=8_000_000,
                                   backupCount=5, encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)


def _client(settings: Settings) -> AlpacaClient:
    creds = load_credentials(".env")
    if not creds.alpaca_configured:
        raise SystemExit("Alpaca credentials missing from .env")
    return AlpacaClient(creds.alpaca_key, creds.alpaca_secret,
                        paper_base=creds.alpaca_base_url,
                        data_base=settings.alpaca.data_base_url)


def cmd_sync(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    reports = sync_all(store, settings, _client(settings), force=args.force)
    for r in reports:
        print(json.dumps(r, default=str))
    bad = [r for r in reports if r.get("status") in ("error", "empty", "fail")]
    return 1 if bad else 0


def cmd_backtest(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    frames = prepare_all(store, settings)
    start = to_epoch(args.start) if args.start else None
    end = to_epoch(args.end) if args.end else None
    gates = load_active_gates(store, settings) if args.gates else None
    result = run_portfolio_backtest(
        settings, frames, start_ts=start, end_ts=end, store=store,
        label=args.label or "portfolio baseline", gates=gates,
    )
    path = write_backtest_report(result, Path(settings.paths.reports) / "backtest.md",
                                 label=args.label or "portfolio baseline", settings=settings)
    m = result.metrics
    print(json.dumps({
        "run_id": result.run_id,
        "return_pct": m.get("net_return_pct"),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
        "trades": m.get("trades"),
        "win_rate_pct": m.get("win_rate_pct"),
        "profit_factor": m.get("profit_factor"),
        "sharpe": m.get("sharpe"),
        "final_equity": m.get("final_equity"),
        "benchmarks": result.benchmarks,
        "gates": sorted(gates) if gates else [],
        "report": path,
    }, indent=2, default=str))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .research import apply_selection, sweep_all

    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    frames = prepare_all(store, settings)
    payload = sweep_all(settings, frames, store=store)
    for r in payload["results"]:
        s = r["selected"]
        t = s["test"]
        print(f"{r['symbol']:10s} {s['strategy']:16s} ret {t.get('net_return_pct')}% "
              f"dd {t.get('max_drawdown_pct')}% trades {t.get('trades')} score {s['score']}")
    if args.apply:
        apply_selection(args.config, payload)
        print("applied selection to", args.config)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    frames = prepare_all(store, settings)
    results = train_all(store, settings, frames)
    for r in results:
        print(f"{r.get('symbol'):10s} {r.get('status'):12s} auc={r.get('auc')} "
              f"improvement={r.get('oos_improvement')} {r.get('reason', '')}")
    write_model_report(store, settings, Path(settings.paths.reports) / "models.md")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    from .live import LiveEngine

    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    engine = LiveEngine(settings, store)

    def _stop(signum, frame):  # noqa: ANN001
        log.info("shutdown requested")
        engine.stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if args.once:
        result = engine.run_cycle()
        print(json.dumps({"equity": result["equity"], "regime": result["regime"],
                          "decisions": result["decisions"], "recon": result["recon"]}, indent=2))
        return 0
    if not acquire_role_lock("paper"):
        log.warning("another paper loop is running (lock port %s) - exiting",
                    _LOCK_PORTS["paper"])
        return 0
    engine.run_forever()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .api import create_app

    settings = Settings.load(args.config)
    if not acquire_role_lock("serve"):
        log.warning("another API server is running (lock port %s) - exiting",
                    _LOCK_PORTS["serve"])
        return 0
    uvicorn.run(create_app(settings), host=settings.api.host, port=settings.api.port,
                log_level="info")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    snap = store.get_portfolio("equity_snapshot", {}) or {}
    regime = store.latest_regime() or {}
    recon = store.latest_recon() or {}
    trades = store.db.query(
        "SELECT COUNT(*) n, COALESCE(SUM(net_pnl),0) pnl FROM trades WHERE mode='live'")
    row = dict(trades[0]) if trades else {"n": 0, "pnl": 0}
    print(f"equity          ${float(snap.get('equity') or settings.capital):,.2f}")
    print(f"cash            ${float(snap.get('cash') or settings.capital):,.2f}")
    print(f"positions       {len(store.positions())}")
    print(f"regime          {regime.get('label', '—')} (score {regime.get('score', '—')})")
    print(f"reconcile       {recon.get('status', '—')}")
    print(f"closed trades   {row['n']} (net ${float(row['pnl']):,.2f})")
    print(f"active models   {sum(1 for m in store.models() if m['status'] == 'active')}")
    print(f"last cycle      {snap.get('ts')} ({int(time.time()) - int(snap.get('ts') or 0)}s ago)")
    write_status_report(store, settings, Path(settings.paths.reports) / "status.md")
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    from .notify import TelegramNotifier

    settings = Settings.load(args.config)
    creds = load_credentials(".env")
    notifier = TelegramNotifier(creds.telegram_token, creds.telegram_chat_id,
                                enabled=True)
    ok = notifier.send("\U0001F9EA Test from `unibot notify-test`.\n"
                       "This channel will receive every BUY, SELL, HOLD digest, model "
                       "promotion and risk event.", kind="test")
    print("delivered" if ok else "failed - check token/chat id")
    return 0 if ok else 1


def cmd_data(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    print(json.dumps(coverage(store, settings), indent=2, default=str))
    return 0


def cmd_reset_breaker(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    state = store.get_portfolio("breakers", {}) or {}
    state["drawdown_halted"] = False
    state["equity"] = state.get("hwm")
    store.save_portfolio("breakers", state)
    store.db.record_risk_event(ts=int(time.time()), mode="live", run_id=None,
                               kind="manual_reset", action="allow_new_entries",
                               detail="operator reset drawdown latch", value=state)
    print("drawdown latch reset; high-water mark kept at", state.get("hwm"))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    store = UnibotStore(settings.paths.db)
    snap = store.get_portfolio("equity_snapshot", {}) or {}
    path = write_status_report(store, settings,
                               Path(settings.paths.reports) / "status.md",
                               extra={"snapshot": snap, "positions": store.positions(),
                                      "regime": store.latest_regime()})
    print("wrote", path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="unibot", description="Unified multi-asset paper bot")
    p.add_argument("--config", default="config/unibot.json")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sync", help="download/refresh candles")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("backtest", help="run the portfolio backtest")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--label")
    s.add_argument("--gates", action="store_true", help="apply active ML gates")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("sweep", help="per-symbol strategy/parameter sweep")
    s.add_argument("--apply", action="store_true", help="write winners into the config")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("train", help="train per-symbol meta-label models")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("paper", help="run the live paper loop")
    s.add_argument("--once", action="store_true", help="one cycle then exit")
    s.set_defaults(func=cmd_paper)

    s = sub.add_parser("serve", help="dashboard API")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("status", help="print current state")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("data", help="data coverage report")
    s.set_defaults(func=cmd_data)

    s = sub.add_parser("notify-test", help="send a Telegram test message")
    s.set_defaults(func=cmd_notify)

    s = sub.add_parser("reset-breaker", help="clear the drawdown latch")
    s.set_defaults(func=cmd_reset_breaker)

    s = sub.add_parser("report", help="write the status report")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
