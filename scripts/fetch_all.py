"""Fetch historical data for all universe symbols."""
import sys
import time
sys.path.insert(0, ".")

from unibot.config import Settings, load_credentials
from unibot.market import AlpacaClient
from unibot.store import UnibotStore
from unibot.datafeed import sync_all

settings = Settings.load("config/unibot.json")
creds = load_credentials(".env")
client = AlpacaClient(
    creds.alpaca_key, creds.alpaca_secret,
    paper_base=creds.alpaca_base_url,
    data_base=settings.alpaca.data_base_url,
)
store = UnibotStore(settings.paths.db)

print("Starting data sync for all symbols...")
t0 = time.time()
result = sync_all(store, settings, client, timeframes=("1h", "1d"))
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
for sym, info in result.items():
    for tf, r in info.items():
        bars = r.get("bars", 0)
        status = r.get("status", "?")
        print(f"  {sym} {tf}: bars={bars} status={status}")
