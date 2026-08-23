# tests/test_portfolio.py — position sizing math and Alpaca-client-facing
# portfolio helpers (mocked, no network).

from unittest.mock import MagicMock

import pytest

from portfolio import (
    normalize_symbol, calculate_kelly_multiplier,
    sell_largest_position, swap_weakest_position, sync_existing_positions,
    sell_retry_cooldown,
)


def test_normalize_symbol():
    assert normalize_symbol("BTC/USD") == "BTCUSD"
    assert normalize_symbol("ETHUSD") == "ETHUSD"


# ── Kelly multiplier ──

def test_kelly_multiplier_negative_edge_returns_minimum():
    # Low win probability relative to reward/risk -> negative Kelly fraction.
    mult = calculate_kelly_multiplier(signal_prob=0.4, profit_target_pct=0.01, stop_loss_pct=0.05)
    assert mult == 0.5


def test_kelly_multiplier_clamped_to_max():
    mult = calculate_kelly_multiplier(signal_prob=0.99, profit_target_pct=0.10, stop_loss_pct=0.01)
    assert mult == 3.0


def test_kelly_multiplier_zero_stop_loss_returns_neutral():
    assert calculate_kelly_multiplier(signal_prob=0.7, profit_target_pct=0.02, stop_loss_pct=0.0) == 1.0


def test_kelly_multiplier_always_within_bounds():
    for p in [0.3, 0.45, 0.5, 0.51, 0.6, 0.75, 0.9]:
        mult = calculate_kelly_multiplier(signal_prob=p, profit_target_pct=0.02, stop_loss_pct=0.03)
        assert 0.5 <= mult <= 3.0


# ── sell_largest_position (mocked Alpaca client) ──

def test_sell_largest_position_sells_biggest_by_market_value(mock_trading_client):
    sell_retry_cooldown.clear()
    small = MagicMock(symbol="ETHUSD", market_value="1000", qty="1.0", current_price="2000")
    big = MagicMock(symbol="BTCUSD", market_value="5000", qty="0.1", current_price="50000")
    mock_trading_client.get_all_positions.return_value = [small, big]

    sell_largest_position()

    assert mock_trading_client.submit_order.called
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.symbol == "BTCUSD"


def test_sell_largest_position_no_positions_is_a_noop(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = []
    sell_largest_position()  # must not raise
    assert not mock_trading_client.submit_order.called


def test_sell_largest_position_respects_retry_cooldown(mock_trading_client):
    sell_retry_cooldown.clear()
    import time
    pos = MagicMock(symbol="BTCUSD", market_value="5000", qty="0.1", current_price="50000")
    mock_trading_client.get_all_positions.return_value = [pos]
    sell_retry_cooldown["BTCUSD"] = time.time()  # just failed a moment ago

    sell_largest_position()

    assert not mock_trading_client.submit_order.called
    sell_retry_cooldown.clear()


# ── swap_weakest_position ──

def test_swap_weakest_position_sells_the_lowest_signal_holding(mock_trading_client, monkeypatch):
    import portfolio
    monkeypatch.setattr(portfolio, "SYMBOLS", ["BTC/USD", "ETH/USD"])
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(symbol="BTCUSD", qty="0.1", current_price="50000", market_value="5000"),
        MagicMock(symbol="ETHUSD", qty="1.0", current_price="2000", market_value="2000"),
    ]
    latest_signals = {"BTC/USD": 0.55, "ETH/USD": 0.40}  # ETH is weakest

    sold_value = swap_weakest_position("SOL/USD", new_signal=0.90, latest_signals=latest_signals, threshold=0.05)

    assert sold_value == pytest.approx(2000.0)
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.symbol == "ETHUSD"


def test_swap_weakest_position_does_nothing_below_threshold(mock_trading_client, monkeypatch):
    import portfolio
    monkeypatch.setattr(portfolio, "SYMBOLS", ["BTC/USD"])
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(symbol="BTCUSD", qty="0.1", current_price="50000", market_value="5000"),
    ]
    latest_signals = {"BTC/USD": 0.55}

    sold_value = swap_weakest_position("SOL/USD", new_signal=0.56, latest_signals=latest_signals, threshold=0.05)

    assert sold_value == 0.0
    assert not mock_trading_client.submit_order.called


# ── sync_existing_positions ──

def test_sync_existing_positions_seeds_missing_state(mock_trading_client, monkeypatch):
    import portfolio
    monkeypatch.setattr(portfolio, "SYMBOLS", ["BTC/USD"])
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(symbol="BTCUSD", qty=0.1, avg_entry_price=50000.0, market_value=5100.0, current_price=51000.0),
    ]
    entry_time, highest_prices = {}, {}

    sync_existing_positions(entry_time, highest_prices)

    assert "BTC/USD" in entry_time
    assert highest_prices["BTC/USD"] == pytest.approx(51000.0)


def test_sync_existing_positions_cleans_up_closed_positions(mock_trading_client, monkeypatch):
    import portfolio
    monkeypatch.setattr(portfolio, "SYMBOLS", ["BTC/USD", "ETH/USD"])
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(symbol="BTCUSD", qty=0.1, avg_entry_price=50000.0, market_value=5100.0, current_price=51000.0),
    ]
    entry_time = {"BTC/USD": 123.0, "ETH/USD": 456.0}
    highest_prices = {"BTC/USD": 51000.0, "ETH/USD": 2000.0}

    sync_existing_positions(entry_time, highest_prices)

    assert "ETH/USD" not in entry_time
    assert "ETH/USD" not in highest_prices
    assert entry_time["BTC/USD"] == 123.0  # untouched, was already tracked
