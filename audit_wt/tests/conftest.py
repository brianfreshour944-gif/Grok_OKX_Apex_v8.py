# tests/conftest.py — shared test fixtures.
#
# config.py requires APCA_API_KEY_ID/APCA_API_SECRET_KEY at import time (it
# constructs a real alpaca-py TradingClient/CryptoHistoricalDataClient as
# module-level singletons) and every other module in this repo imports from
# config. These must be set before ANY repo module is imported, so this
# happens at conftest collection time, before test modules import anything.
#
# DATABASE_URL is deliberately left unset: every database.py function is a
# documented no-op when it's absent, which is the simplest safe way to keep
# these tests from needing a real Postgres instance.
import os

os.environ.setdefault("APCA_API_KEY_ID", "test-key")
os.environ.setdefault("APCA_API_SECRET_KEY", "test-secret")
os.environ.pop("DATABASE_URL", None)

import asyncio
from unittest.mock import MagicMock

import pytest


def run_async(coro):
    """Run a coroutine to completion without needing the pytest-asyncio plugin."""
    return asyncio.run(coro)


@pytest.fixture
def mock_trading_client(monkeypatch):
    """
    Replaces the trading_client singleton with a MagicMock everywhere it has
    already been imported by name (portfolio.py, orders.py both do
    `from config import trading_client`, which binds a local reference in
    each module's namespace -- patching config.trading_client alone would
    NOT affect those already-bound references).
    """
    import config
    import portfolio
    import orders

    mock = MagicMock()
    monkeypatch.setattr(config, "trading_client", mock)
    monkeypatch.setattr(portfolio, "trading_client", mock)
    monkeypatch.setattr(orders, "trading_client", mock)
    return mock
