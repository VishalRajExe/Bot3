"""Command line interface.

    python -m btcbot.cli init
    python -m btcbot.cli fetch
    python -m btcbot.cli status
    python -m btcbot.cli backtest --strategy ema_std --timeframe 1d
    python -m btcbot.cli compare
    python -m btcbot.cli experiments [--ml] [--sentiment]
    python -m btcbot.cli sentiment [--news]
    python -m btcbot.cli ml
    python -m btcbot.cli serve [--port 8601] [--paper]
    python -m btcbot.cli paper [--replay-bars 400] [--once]
    python -m btcbot.cli journal [--mode paper]
    python -m btcbot.cli report --run <run_id> | --all
    python -m btcbot.cli export
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from . import __version__
from .backtest.engine import run_backtest
from .config import Config, load_config
from .data.exchange import MarketDataError
from .data.ingest import CandleIngestor
from .data.timeframes import parse_iso_date, to_datetime
from .db import Database
from .journal import load_trades, journal_summary, review_trades, trade_to_dict
from .logging_setup import setup_logging
from .paper import PaperTrader
from .reporting import write_comparison, write_run_artifacts
from .strategies import STRATEGIES, build_strategy, list_strategies

log = logging.getLogger("btcbot.cli")


def _open(cfg: Config) -> Database:
    return Database(cfg.db_path)


def cmd_init(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    print(f"Config: {Path('config/config.json').resolve()}")
    print(f"Database: {cfg.db_path}")
    print(f"Schema version: {db.scalar('SELECT value FROM schema_meta WHERE key=?', ('schema_version',))}")
    print(f"Symbol: {cfg.exchange.pair} | mode: {cfg.trading.mode} | live orders: "
          f"{cfg.trading.enable_live_orders}")
    db.close()
    return 0


def cmd_fetch(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    ing = CandleIngestor(cfg, db)
    ok = True
    for tf in (args.timeframe or cfg.data.timeframes):
        try:
            rep = ing.sync(tf, force_full=args.force)
            print(f"{tf}: {rep.bars} bars, status={rep.status}, missing={rep.missing_bars}, "
                  f"range {to_datetime(rep.first_ts).date() if rep.first_ts else '-'} .. "
                  f"{to_datetime(rep.last_ts).date() if rep.last_ts else '-'}")
        except MarketDataError as exc:
            ok = False
            print(f"{tf}: FAILED: {exc}")
    db.close()
    return 0 if ok else 1


def cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    print(f"BTC research system v{__version__} | symbol {cfg.exchange.pair} | "
          f"mode {cfg.trading.mode} | live orders {cfg.trading.enable_live_orders}")
    for tf in cfg.data.timeframes:
        s = db.query_one(
            "SELECT COUNT(*) n, MIN(open_time) a, MAX(open_time) b FROM candles"
            " WHERE symbol=? AND timeframe=?", (cfg.exchange.pair, tf))
        if s and s["n"]:
            from .data.timeframes import tf_seconds
            age = (time.time() - (s["b"] + tf_seconds(tf))) / 3600
            print(f"  data {tf}: {s['n']} bars, {to_datetime(s['a']).date()} .. "
                  f"{to_datetime(s['b']).date()} ({age:.1f}h since close)")
        else:
            print(f"  data {tf}: none cached")
    runs = db.query("SELECT id, strategy, timeframe, metrics_json, created_at FROM backtest_runs"
                    " ORDER BY created_at DESC LIMIT 5")
    print(f"  backtest runs: {db.scalar('SELECT COUNT(*) FROM backtest_runs')} "
          f"(latest {runs[0]['created_at'] if runs else 'none'})")
    sessions = db.query("SELECT id, mode, status, strategy, started_at FROM paper_sessions"
                        " ORDER BY started_at DESC LIMIT 3")
    for s in sessions:
        print(f"  paper {s['id']}: {s['status']} ({s['mode']}, {s['strategy']}) started {s['started_at']}")
    if not runs and not sessions:
        print("  no research or paper activity yet")
    db.close()
    return 0


def cmd_backtest(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    ing = CandleIngestor(cfg, db)
    tf = args.timeframe or STRATEGIES[args.strategy]().default_timeframe
    df = ing.load_history(tf)
    if df.empty:
        print("No candles cached. Run `python -m btcbot.cli fetch` first.")
        return 1
    params = json.loads(args.params) if args.params else {}
    strat = build_strategy(args.strategy, params, tf)
    res = run_backtest(cfg, db, strat, df, timeframe=tf, initial_equity=args.equity,
                       persist=not args.no_persist,
                       start_ts=parse_iso_date(args.start) if args.start else None,
                       end_ts=parse_iso_date(args.end) if args.end else None,
                       notes=args.notes)
    m = res.metrics
    bh = res.benchmark["buy_hold"]
    print(f"\n{res.strategy} v{res.strategy_version} [{res.timeframe}] run {res.run_id}")
    print(f"  period        {m.get('period_start')} .. {m.get('period_end')} "
          f"({m.get('period_years')}y)")
    print(f"  net return    {m.get('net_return_pct')}%   CAGR {m.get('cagr_pct')}%")
    print(f"  max drawdown  {m.get('max_drawdown_pct')}%   duration {m.get('drawdown_duration_days')}d")
    print(f"  sharpe/sortino {m.get('sharpe')} / {m.get('sortino')}   calmar {m.get('calmar')}")
    print(f"  trades        {m.get('trades')}   win rate {m.get('win_rate_pct')}%   "
          f"PF {m.get('profit_factor')}")
    print(f"  exposure      {m.get('exposure_pct')}%   fees {m.get('total_fees')} USDT")
    print(f"  buy & hold    {bh.get('net_return_pct')}%   max DD {bh.get('max_drawdown_pct')}%")
    print(f"  excess vs B&H {m.get('excess_return_vs_buy_hold_pct')}%")
    if m.get("sample_warning"):
        print(f"  NOTE: {m['sample_warning']}")
    if res.persisted and args.report:
        paths = write_run_artifacts(cfg, db, res.run_id)
        print(f"  artifacts     {paths['dir']}")
    db.close()
    return 0


def cmd_compare(args: argparse.Namespace, cfg: Config) -> int:
    from .experiments import run_core_suite

    db = _open(cfg)
    suite = run_core_suite(cfg, db)
    if not suite:
        print("No strategies could be run; fetch data first.")
        return 1
    print(f"\n{'strategy':22} {'tf':4} {'net%':>9} {'cagr%':>7} {'maxDD%':>7} {'sharpe':>7} "
          f"{'trades':>7} {'B&H%':>10}")
    for res in suite.values():
        m = res.metrics
        print(f"{res.strategy:22} {res.timeframe:4} {m.get('net_return_pct', 0):>9.2f} "
              f"{m.get('cagr_pct', 0):>7.2f} {m.get('max_drawdown_pct', 0):>7.2f} "
              f"{(m.get('sharpe') or 0):>7.2f} {m.get('trades', 0):>7} "
              f"{res.benchmark['buy_hold'].get('net_return_pct', 0):>10.2f}")
    print("\nComparison written to reports/backtests/comparison.md")
    db.close()
    return 0


def cmd_experiments(args: argparse.Namespace, cfg: Config) -> int:
    from .experiments import (run_core_suite, run_ema_std_experiments, run_ml_experiment,
                              run_sentiment_experiment, write_experiment_report)

    db = _open(cfg)
    print("Running strategy suite ...")
    suite = run_core_suite(cfg, db)
    print("Running EMA-65 controlled experiments ...")
    experiments = run_ema_std_experiments(cfg, db)
    ml = None
    sentiment = None
    if args.ml:
        print("Running ML walk-forward experiment ...")
        ml = run_ml_experiment(cfg, db)
    if args.sentiment:
        print("Running sentiment experiment ...")
        sentiment = run_sentiment_experiment(cfg, db)
    path = write_experiment_report("reports/experiments.md", suite=suite,
                                   experiments=experiments, ml=ml, sentiment=sentiment)
    print(f"Experiment log: {path}")
    for e in experiments:
        print(f"  {e['experiment']:24} {e['verdict']:12} full {e.get('full_return')}% "
              f"OOS {e.get('oos_return')}%")
    db.close()
    return 0


def cmd_sentiment(args: argparse.Namespace, cfg: Config) -> int:
    from .sentiment.fng import latest_value, store_fng
    from .sentiment.news import collect, latest_headlines

    db = _open(cfg)
    try:
        n = store_fng(db)
        print(f"Fear & Greed: stored {n} daily observations")
    except Exception as exc:  # noqa: BLE001
        print(f"Fear & Greed fetch failed: {exc}")
    if args.news:
        res = collect(db)
        print(f"News: {res['stored']} headlines stored, errors={res['errors']}")
    latest = latest_value(db)
    if latest:
        print(f"  latest: {latest['value']} ({latest['label']}) on {latest['date']}")
    for h in latest_headlines(db, 5):
        print(f"  [{h['source']}] {h['published'][:16]} {h['title'][:90]}")
    db.close()
    return 0


def cmd_ml(args: argparse.Namespace, cfg: Config) -> int:
    from .experiments import run_ml_experiment

    db = _open(cfg)
    result = run_ml_experiment(cfg, db)
    print(f"Model {result['version']}: {result['status']} ({result['gate_reason']})")
    mm = result["model_metrics"]
    print(f"  OOF AUC {mm['auc']} over {mm['predictions']} predictions, {mm['folds']} folds")
    print(f"  calibration: {mm['calibration'][:3]}")
    print(f"  top features: {mm['feature_importance'][:5]}")
    print(f"  strategy return {result['strategy_metrics'].get('net_return_pct')}% vs baseline "
          f"{result['baseline_metrics'].get('net_return_pct')}% on the same window")
    print(f"  verdict: {result['verdict']} - {result['notes']}")
    db.close()
    return 0


def cmd_paper(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    trader = PaperTrader(cfg, db)
    print(f"Paper session {trader.session_id} | strategy {trader.strategy.name} "
          f"v{trader.strategy.version} | {trader.timeframe} | mode {trader.mode}")
    print("PAPER TRADING ONLY: no exchange orders are sent. Real-money execution is disabled.")
    if args.replay_bars:
        info = trader.run_replay(args.replay_bars)
        print(f"Replayed {info['bars']} closed bars: {info['trades']} simulated trades, "
              f"net {info['net_return_pct']}% (historical replay, not live performance)")
    if args.once:
        print(json.dumps(trader.poll_once(), indent=2, default=str))
        db.close()
        return 0
    print(f"Polling every {cfg.paper.poll_seconds}s. Press Ctrl+C to stop safely.")
    try:
        while True:
            result = trader.poll_once()
            print(f"{to_datetime(int(time.time())):%H:%M:%S} {json.dumps(result, default=str)[:200]}")
            time.sleep(max(cfg.paper.poll_seconds, 5))
    except KeyboardInterrupt:
        print("\nStopping paper session (state persisted).")
        trader.set_status("stopped")
    finally:
        db.close()
    return 0


def cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    import uvicorn

    if args.port:
        cfg.api.port = args.port
    os.environ.setdefault("BTCBOT_CONFIG", str(Path("config/config.json").resolve()))
    print(f"Dashboard: http://{cfg.api.host}:{cfg.api.port}")
    print("Paper trading only. No real orders can be placed.")
    if args.paper:
        db = _open(cfg)
        trader = PaperTrader(cfg, db)
        if args.replay_bars:
            info = trader.run_replay(args.replay_bars)
            print(f"Replayed {info['bars']} bars for demonstration: {info['trades']} trades")
        from .paper import PaperService
        service = PaperService(trader)
        service.tick()
        service.start()
        print(f"Paper loop started for session {trader.session_id}")
    uvicorn.run("btcbot.api:app", host=cfg.api.host, port=cfg.api.port, log_level="info")
    return 0


def cmd_journal(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    trades = load_trades(db, mode=args.mode, limit=args.limit)
    summary = journal_summary(trades)
    print(f"Journal ({args.mode or 'all modes'}): {summary}")
    for t in trades[: args.limit]:
        print(f"  {to_datetime(t.entry_ts).date()} -> {to_datetime(t.exit_ts).date()} "
              f"{t.strategy:20} {t.return_pct:>7.2f}% net {t.net_pnl:>9.2f} "
              f"({t.exit_reason[:34]})")
    findings = review_trades(db, trades, persist=False)
    if findings:
        print("\nFindings (descriptive, not automatic changes):")
        for f in findings:
            print(f"  [{f['category']}] {f['finding']}")
    db.close()
    return 0


def cmd_report(args: argparse.Namespace, cfg: Config) -> int:
    db = _open(cfg)
    if args.run:
        paths = write_run_artifacts(cfg, db, args.run)
        print(json.dumps(paths, indent=2))
    else:
        runs = db.query("SELECT id FROM backtest_runs ORDER BY created_at")
        for r in runs:
            try:
                paths = write_run_artifacts(cfg, db, r["id"])
                print(f"{r['id']}: {paths['dir']}")
            except Exception as exc:  # noqa: BLE001
                print(f"{r['id']}: failed: {exc}")
        print(write_comparison(db))
    db.close()
    return 0


def cmd_export(args: argparse.Namespace, cfg: Config) -> int:
    """Export a consolidated research JSON bundle."""
    db = _open(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = [dict(r) for r in db.query("SELECT * FROM backtest_runs ORDER BY created_at")]
    for r in runs:
        r["metrics"] = json.loads(r.pop("metrics_json") or "{}")
        r["benchmark"] = json.loads(r.pop("benchmark_json") or "{}")
        r["params"] = json.loads(r.pop("params_json") or "{}")
    payload = {
        "generated_at": to_datetime(int(time.time())).isoformat(),
        "system": {"version": __version__, "symbol": cfg.exchange.pair,
                   "mode": cfg.trading.mode, "live_orders": cfg.trading.enable_live_orders},
        "runs": runs,
        "experiments": [dict(r) for r in db.query("SELECT * FROM experiments")],
        "models": [dict(r) for r in db.query("SELECT * FROM models")],
        "paper_sessions": [dict(r) for r in db.query("SELECT * FROM paper_sessions")],
        "trades": [trade_to_dict(t) for t in load_trades(db, limit=5000)],
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Exported {len(runs)} runs and {len(payload['trades'])} trades to {out}")
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="btcbot", description="BTC/USDT research and paper trading")
    p.add_argument("--config", default=None, help="path to config JSON")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create/validate config and database")

    f = sub.add_parser("fetch", help="download missing historical candles")
    f.add_argument("--timeframe", action="append", help="limit to one timeframe (repeatable)")
    f.add_argument("--force", action="store_true", help="re-download from the configured start date")

    sub.add_parser("status", help="show data, run and session status")

    b = sub.add_parser("backtest", help="run one backtest")
    b.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    b.add_argument("--timeframe", default=None)
    b.add_argument("--params", default=None, help="JSON object of strategy parameters")
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--equity", type=float, default=None)
    b.add_argument("--notes", default="")
    b.add_argument("--no-persist", action="store_true")
    b.add_argument("--report", action="store_true", help="write charts/CSV artifacts")

    sub.add_parser("compare", help="run the full strategy suite and write a comparison")

    e = sub.add_parser("experiments", help="run controlled experiments")
    e.add_argument("--ml", action="store_true")
    e.add_argument("--sentiment", action="store_true")

    s = sub.add_parser("sentiment", help="refresh sentiment data")
    s.add_argument("--news", action="store_true", help="also fetch RSS headlines")

    sub.add_parser("ml", help="train and evaluate the ML model")

    sv = sub.add_parser("serve", help="start the dashboard/API")
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--paper", action="store_true", help="also start the paper loop")
    sv.add_argument("--replay-bars", type=int, default=0)

    pa = sub.add_parser("paper", help="run paper trading in the foreground")
    pa.add_argument("--replay-bars", type=int, default=0)
    pa.add_argument("--once", action="store_true", help="single poll, then exit")

    j = sub.add_parser("journal", help="show the trade journal")
    j.add_argument("--mode", default=None, choices=["paper", "backtest"])
    j.add_argument("--limit", type=int, default=25)

    r = sub.add_parser("report", help="write research artifacts")
    r.add_argument("--run", default=None)

    ex = sub.add_parser("export", help="export a JSON research bundle")
    ex.add_argument("--out", default="reports/export.json")

    sub.add_parser("strategies", help="list available strategies")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.log_level:
        cfg.log_level = args.log_level
    setup_logging(cfg.log_level)
    handlers = {
        "init": cmd_init, "fetch": cmd_fetch, "status": cmd_status, "backtest": cmd_backtest,
        "compare": cmd_compare, "experiments": cmd_experiments, "sentiment": cmd_sentiment,
        "ml": cmd_ml, "serve": cmd_serve, "paper": cmd_paper, "journal": cmd_journal,
        "report": cmd_report, "export": cmd_export,
    }
    if args.command == "strategies":
        for s in list_strategies():
            print(f"{s['name']:22} v{s['version']:8} {s['timeframe']:3} {s['description']}")
        return 0
    try:
        return handlers[args.command](args, cfg)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
