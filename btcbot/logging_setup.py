"""Structured logging to stdout and a rotating file."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"


def setup_logging(level: str = "INFO", log_dir: str | Path = "logs", filename: str = "btcbot.log") -> None:
    root = logging.getLogger()
    if getattr(root, "_btcbot_configured", False):
        return
    root.setLevel(level.upper())
    fmt = logging.Formatter(LOG_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(path / filename, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Third-party noise reduction.
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._btcbot_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
