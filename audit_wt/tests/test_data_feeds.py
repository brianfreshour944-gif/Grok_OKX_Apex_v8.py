# tests/test_data_feeds.py — scan_stable_assets(), with data_client mocked
# out (no real Alpaca API calls, no network).

from unittest.mock import MagicMock

import pytest

from data_feeds import scan_stable_assets
from conftest import run_async


def _fake_bar(volume, close):
    return MagicMock(volume=volume, close=close)


def _mock_data_client(volumes: dict, raise_for: set = frozenset()):
    """volumes: {symbol: (volume, close)}. raise_for: symbols whose fetch should fail."""
    def fake_get_crypto_bars(req):
        symbol = req.symbol_or_symbols
        if symbol in raise_for:
            raise RuntimeError(f"no data for {symbol}")
        result = MagicMock()
        vol, close = volumes[symbol]
        result.data = {symbol: [_fake_bar(vol, close)]}
        return result
    return MagicMock(get_crypto_bars=fake_get_crypto_bars)


def test_ranks_by_dollar_volume_descending(monkeypatch):
    import data_feeds
    volumes = {
        "BTC/USD": (100, 50000),   # $5,000,000
        "ETH/USD": (1000, 2000),  # $2,000,000
        "SOL/USD": (10000, 100),  # $1,000,000
    }
    monkeypatch.setattr(data_feeds, "data_client", _mock_data_client(volumes))

    result = run_async(scan_stable_assets(limit_scope=10, candidates=list(volumes.keys())))

    assert result == ["BTC/USD", "ETH/USD", "SOL/USD"]


def test_respects_limit_scope():
    import data_feeds as df_module
    volumes = {"BTC/USD": (100, 50000), "ETH/USD": (1000, 2000), "SOL/USD": (10000, 100)}
    import unittest.mock as um
    with um.patch.object(df_module, "data_client", _mock_data_client(volumes)):
        result = run_async(scan_stable_assets(limit_scope=2, candidates=list(volumes.keys())))
    assert result == ["BTC/USD", "ETH/USD"]


def test_restricts_to_given_candidates_only(monkeypatch):
    """This is the whole point of the `candidates` param: the live bot passes
    DYNAMIC_UNIVERSE_CANDIDATES (10 symbols the model was trained on) instead
    of the full 24-symbol DEFAULT_SCAN_CANDIDATES."""
    import data_feeds
    seen = []

    def fake_get_crypto_bars(req):
        symbol = req.symbol_or_symbols
        seen.append(symbol)
        result = MagicMock()
        result.data = {symbol: [_fake_bar(100, 100)]}
        return result

    monkeypatch.setattr(data_feeds, "data_client", MagicMock(get_crypto_bars=fake_get_crypto_bars))

    run_async(scan_stable_assets(limit_scope=10, candidates=["BTC/USD", "ETH/USD"]))

    assert set(seen) == {"BTC/USD", "ETH/USD"}


def test_falls_back_to_safe_default_when_everything_fails(monkeypatch):
    import data_feeds
    monkeypatch.setattr(
        data_feeds, "data_client",
        MagicMock(get_crypto_bars=MagicMock(side_effect=RuntimeError("API down"))),
    )

    result = run_async(scan_stable_assets(limit_scope=5, candidates=["BTC/USD", "ETH/USD"]))

    assert result == ["BTC/USD", "ETH/USD", "SOL/USD"]


def test_one_bad_symbol_does_not_abort_the_whole_scan(monkeypatch):
    import data_feeds
    volumes = {"BTC/USD": (100, 50000), "SOL/USD": (100, 100)}
    monkeypatch.setattr(data_feeds, "data_client", _mock_data_client(volumes, raise_for={"ETH/USD"}))

    result = run_async(scan_stable_assets(limit_scope=5, candidates=["BTC/USD", "ETH/USD", "SOL/USD"]))

    assert set(result) == {"BTC/USD", "SOL/USD"}


def test_default_candidates_used_when_none_given(monkeypatch):
    import data_feeds
    seen = []

    def fake_get_crypto_bars(req):
        seen.append(req.symbol_or_symbols)
        raise RuntimeError("just counting calls")

    monkeypatch.setattr(data_feeds, "data_client", MagicMock(get_crypto_bars=fake_get_crypto_bars))

    run_async(scan_stable_assets(limit_scope=5))  # no candidates passed

    assert set(seen) == set(data_feeds.DEFAULT_SCAN_CANDIDATES)
