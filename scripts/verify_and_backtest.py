"""Full portfolio backtest and data coverage check."""
import sys
import time
sys.path.insert(0, ".")

from unibot.config import Settings, load_credentials
from unibot.market import AlpacaClient
from unibot.store import UnibotStore
from unibot.datafeed import sync_all, coverage, prepare_all, build_market
from unibot.backtest import run_portfolio_backtest

settings = Settings.load("config/unibot.json")
creds = load_credentials(".env")
client = AlpacaClient(
    creds.alpaca_key, creds.alpaca_secret,
    paper_base=creds.alpaca_base_url,
    data_base=settings.alpaca.data_base_url,
)
store = UnibotStore(settings.paths.db)

# Sync latest data
print("Syncing data...")
t0 = time.time()
results = sync_all(store, settings, client, timeframes=("1h", "1d"))
print(f"Sync done in {time.time()-t0:.1f}s")
for r in results:
    sym = r.get("symbol","?"); tf = r.get("timeframe","?")
    bars = r.get("bars",0); status = r.get("status","?")
    print(f"  {sym:12s} {tf:4s}: {bars:6d} bars  status={status}")

print("\nData coverage:")
cov = coverage(store, settings)
for row in cov:
    sym = row["symbol"]
    h1 = row.get("1h", {}); d1 = row.get("1d", {})
    ok = row.get("trade_enabled", False)
    print(f"  {'[ON]' if ok else '[off]'} {sym:12s}  "
          f"1h:{h1.get('bars',0):5d}bars({h1.get('status','?')})  "
          f"1d:{d1.get('bars',0):5d}bars({d1.get('status','?')})")

print("\nPreparing feature frames...")
frames = prepare_all(store, settings)
for sym, df in frames.items():
    print(f"  {sym:12s}: {len(df)} bars  cols={list(df.columns[:8])}")

print("\nRunning full portfolio backtest (2021-01-01 to now)...")
t0 = time.time()
result = run_portfolio_backtest(
    settings=settings,
    frames=frames,
    store=store,
    run_id="init_portfolio",
    label="Initial portfolio backtest – all symbols, 4h, 2021 onwards",
)
elapsed = time.time() - t0
m = result.metrics

print(f"\n{'='*60}")
print(f"Portfolio backtest complete in {elapsed:.1f}s")
print(f"{'='*60}")
print(f"Run ID         : {result.run_id}")
print(f"Trades         : {m.get('n_trades', 0)}")
print(f"Net return     : {m.get('net_return_pct', 0.0):+.2f}%")
print(f"Annualized     : {m.get('annualized_return_pct', 0.0):+.2f}%")
print(f"Max drawdown   : {m.get('max_drawdown_pct', 0.0):.2f}%")
print(f"Sharpe         : {m.get('sharpe', 0.0):.3f}")
print(f"Sortino        : {m.get('sortino', 0.0):.3f}")
print(f"Win rate       : {m.get('win_rate_pct', 0.0):.1f}%")
print(f"Profit factor  : {m.get('profit_factor', 0.0):.2f}")
print(f"Total fees     : ${m.get('total_fees', 0.0):.4f}")
print(f"Exposure%      : {m.get('exposure_pct', 0.0):.1f}%")

benches = m.get("benchmarks", {})
if benches:
    print("\nBenchmarks (same window, after entry costs):")
    for name, b in benches.items():
        print(f"  {b.get('name','?'):50s}  "
              f"return={b.get('net_return_pct',0.0):+.2f}%  "
              f"dd={b.get('max_drawdown_pct',0.0):.2f}%")

print("\nPer-symbol summary:")
ps = m.get("per_symbol", {})
for sym, s in sorted(ps.items()):
    if s.get("n_trades", 0) == 0:
        continue
    print(f"  {sym:12s}: trades={s['n_trades']:3d}  "
          f"net={s.get('net_pnl',0.0):+.4f}  "
          f"wr={s.get('win_rate_pct',0.0):.0f}%  "
          f"pf={s.get('profit_factor',0.0):.2f}")

print("\nPer-strategy summary:")
pstr = m.get("per_strategy", {})
for strat, s in sorted(pstr.items()):
    if s.get("n_trades", 0) == 0:
        continue
    print(f"  {strat:20s}: trades={s['n_trades']:3d}  "
          f"net={s.get('net_pnl',0.0):+.4f}  "
          f"wr={s.get('win_rate_pct',0.0):.0f}%")

print("\nAll results saved to DB. Start the dashboard to view charts.")
