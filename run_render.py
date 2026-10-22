"""Render deployment entrypoint.

Runs both the continuous autonomous trading worker and the web dashboard server
within a single unified Render service.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sys
import threading
import time
import urllib.request
import uvicorn

from unibot.config import Settings
from unibot.store import UnibotStore
from unibot.live import LiveEngine
from unibot.api import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("render_entrypoint")


def _run_worker(settings: Settings, store: UnibotStore) -> None:
    """Continuous paper trading worker loop."""
    log.info("Initializing LiveEngine for paper trading loop...")
    time.sleep(2.0)  # brief pause while server binds port
    try:
        engine = LiveEngine(settings, store)
        log.info("LiveEngine initialized successfully. Entering run_forever loop...")
        engine.run_forever()
    except Exception as exc:
        log.exception("Fatal error in paper trading worker: %s", exc)


def _keepalive(port: int) -> None:
    """Lightweight self-ping every 10 minutes to maintain activity on free tier."""
    time.sleep(30.0)
    url = f"http://127.0.0.1:{port}/api/health"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                log.debug("Keepalive self-ping ok: %s", resp.status)
        except Exception as exc:
            log.debug("Keepalive ping skipped: %s", exc)
        time.sleep(600)  # 10 minutes


def main() -> None:
    port = int(os.environ.get("PORT", "10000"))
    
    # Ensure all runtime directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("reports/unibot", exist_ok=True)

    config_path = "config/unibot.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    settings = Settings.load(config_path)

    # Decompress pre-seeded research database if sqlite file is absent or empty
    db_path = settings.paths.db
    gz_path = db_path + ".gz"
    if (not os.path.exists(db_path) or os.path.getsize(db_path) < 1024 * 1024) and os.path.exists(gz_path):
        log.info("Unpacking pre-seeded database %s (%d bytes) -> %s ...",
                 gz_path, os.path.getsize(gz_path), db_path)
        with gzip.open(gz_path, "rb") as f_in:
            with open(db_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        log.info("Pre-seeded database extracted successfully (%d bytes).", os.path.getsize(db_path))

    store = UnibotStore(settings.paths.db)

    # Launch autonomous trading engine in background daemon thread
    worker = threading.Thread(target=_run_worker, args=(settings, store), daemon=True, name="TradingWorker")
    worker.start()
    log.info("Started background trading worker thread.")

    # Launch keepalive thread
    keepalive_thread = threading.Thread(target=_keepalive, args=(port,), daemon=True, name="Keepalive")
    keepalive_thread.start()

    # Start FastAPI dashboard
    log.info("Starting web dashboard on 0.0.0.0:%d", port)
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
