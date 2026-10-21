from __future__ import annotations

import pytest

from btcbot.config import Config
from btcbot.db import Database

from tests.helpers import make_candles


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.data.db_path = str(tmp_path / "test.sqlite3")
    c.paper.initial_equity = 10_000.0
    c.backtest.initial_equity = 10_000.0
    return c


@pytest.fixture
def db(cfg) -> Database:
    database = Database(cfg.db_path)
    yield database
    database.close()


@pytest.fixture
def candles():
    return make_candles(600, seed=11)
