# tests/test_portfolio.py — position sizing math and Alpaca-client-facing
# portfolio helpers (mocked, no network).

from unittest.mock import MagicMock

import pytest

from portfolio import (
    normalize_symbol, denormalize_symbol, calculate_kelly_multiplier,
    sell_largest_position, swap_weakest_position, sync_existing_positions,
    sell_retry_cooldown, pending_exit_until, has_pending_exit, mark_pending_exit,
)
from conftest import run_async


@pytest.fixture(autouse=True)
def _clear_portfolio_module_state():
    """
    sell_retry_cooldown and pending_exit_until are module-level dicts in
    portfolio.py, not per-call state -- without clearing them between tests,
    a symbol marked pending in one test (e.g. after a successful sell)
    silently short-circuits a later test's sell_largest_position()/
    swap_weakest_position() call for the same symbol, since it looks like
    a sell is already in flight. This bit exactly once while adding
    pending_exit_until: two tests started failing for a reason that had
    nothing to do with what they were testing.
    """
    sell_retry_cooldown.clear()
    pending_exit_until.clear()
    yield
    sell_retry_cooldown.clear()
    pending_exit_until.clear()


def test_normalize_symbol():
    assert normalize_symbol("BTC/USD") == "BTCUSD"
    assert normalize_symbol("ETHUSD") == "ETHUSD"


def test_denormalize_symbol():
    assert denormalize_symbol("BTCUSD") == "BTC/USD"
    assert denormalize_symbol("DOGEUSD") == "DOGE/USD"


@pytest.mark.parametrize("symbol", ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"])
def test_normalize_denormalize_are_inverses(symbol):
    """
    This is what lets a held position survive its symbol rotating out of the
    dynamic universe: denormalize_symbol() reconstructs the standard symbol
    structurally, without needing to look it up in any fixed candidate list.
    """
    assert denormalize_symbol(normalize_symbol(symbol)) == symbol


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

def _mock_position(symbol, market_value, qty, current_price, avg_entry_price):
    return MagicMock(
        symbol=symbol, market_value=market_value, qty=qty,
        current_price=current_price, avg_entry_price=avg_entry_price,
    )


def _mock_filled_order(order_id="oid", filled_avg_price=None, commission="0.0"):
    return MagicMock(id=order_id, filled_avg_price=filled_avg_price, commission=commission)


def test_sell_largest_position_sells_biggest_by_market_value(mock_trading_client):
    small = _mock_position("ETHUSD", "1000", "1.0", "2000", "1900")
    big = _mock_position("BTCUSD", "5000", "0.1", "50000", "48000")
    mock_trading_client.get_all_positions.return_value = [small, big]
    mock_trading_client.submit_order.return_value = _mock_filled_order()
    mock_trading_client.get_order_by_id.return_value = _mock_filled_order(filled_avg_price="50000")

    run_async(sell_largest_position())

    assert mock_trading_client.submit_order.called
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    # Must be the slash format ('BTC/USD'), matching every other order
    # submission path -- not Alpaca's raw slash-less position.symbol
    # ('BTCUSD'), which risks a format-mismatch rejection on the exchange.
    assert order_data.symbol == "BTC/USD"


def test_sell_largest_position_no_positions_is_a_noop(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = []
    run_async(sell_largest_position())  # must not raise
    assert not mock_trading_client.submit_order.called


def test_sell_largest_position_respects_retry_cooldown(mock_trading_client):
    import time
    pos = _mock_position("BTCUSD", "5000", "0.1", "50000", "48000")
    mock_trading_client.get_all_positions.return_value = [pos]
    sell_retry_cooldown["BTC/USD"] = time.time()  # just failed a moment ago

    run_async(sell_largest_position())

    assert not mock_trading_client.submit_order.called


def test_sell_largest_position_sets_retry_cooldown_on_failure(mock_trading_client):
    pos = _mock_position("BTCUSD", "5000", "0.1", "50000", "48000")
    mock_trading_client.get_all_positions.return_value = [pos]
    mock_trading_client.submit_order.side_effect = RuntimeError("exchange rejected order")

    run_async(sell_largest_position())

    assert "BTC/USD" in sell_retry_cooldown


# ── pending-exit guard: prevents duplicate SELLs on a still-unfilled position ──

def test_has_pending_exit_false_by_default():
    assert has_pending_exit("BTC/USD") is False


def test_has_pending_exit_true_immediately_after_marking():
    mark_pending_exit("BTC/USD")
    assert has_pending_exit("BTC/USD") is True


def test_sell_largest_position_skips_when_a_sell_is_already_pending(mock_trading_client):
    """
    Without this guard: has_position stays True every cycle while a limit
    sell remains unfilled (Alpaca still reports the position as fully
    held), so the same exit condition re-fires and submits a duplicate
    full-size SELL on top of the still-pending one.
    """
    pos = _mock_position("BTCUSD", "5000", "0.1", "50000", "48000")
    mock_trading_client.get_all_positions.return_value = [pos]
    mark_pending_exit("BTC/USD")

    run_async(sell_largest_position())

    assert not mock_trading_client.submit_order.called


def test_sell_largest_position_marks_pending_exit_on_success(mock_trading_client):
    pos = _mock_position("BTCUSD", "5000", "0.1", "50000", "48000")
    mock_trading_client.get_all_positions.return_value = [pos]
    mock_trading_client.submit_order.return_value = _mock_filled_order()
    mock_trading_client.get_order_by_id.return_value = _mock_filled_order(filled_avg_price="50000")

    run_async(sell_largest_position())

    assert has_pending_exit("BTC/USD")


def test_swap_weakest_position_skips_when_a_sell_is_already_pending(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = [
        _mock_position("BTCUSD", "5000", "0.1", "50000", "48000"),
    ]
    mark_pending_exit("BTC/USD")

    sold_value = run_async(
        swap_weakest_position("SOL/USD", new_signal=0.90, latest_signals={"BTC/USD": 0.40}, threshold=0.05)
    )

    assert sold_value == 0.0
    assert not mock_trading_client.submit_order.called


def test_sell_largest_position_records_realized_pnl(mock_trading_client):
    """The whole point of routing this through place_order() -- unlike the
    old direct trading_client.submit_order() call, this sell must now show
    up with a realized PnL, same as any other exit."""
    pos = _mock_position("BTCUSD", "5000", "0.1", "50000", "48000")
    mock_trading_client.get_all_positions.return_value = [pos]
    mock_trading_client.submit_order.return_value = _mock_filled_order()
    mock_trading_client.get_order_by_id.return_value = _mock_filled_order(filled_avg_price="50000")

    import orders
    recorded = {}
    import unittest.mock as um
    with um.patch.object(orders, "record_trade", lambda *a, **kw: recorded.update(kw)):
        run_async(sell_largest_position())

    assert recorded["realized_pnl"] == pytest.approx((50000.0 - 48000.0) * 0.1)


# ── swap_weakest_position ──

def test_swap_weakest_position_sells_the_lowest_signal_holding(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = [
        _mock_position("BTCUSD", "5000", "0.1", "50000", "48000"),
        _mock_position("ETHUSD", "2000", "1.0", "2000", "1900"),
    ]
    mock_trading_client.submit_order.return_value = _mock_filled_order()
    mock_trading_client.get_order_by_id.return_value = _mock_filled_order(filled_avg_price="2000")
    latest_signals = {"BTC/USD": 0.55, "ETH/USD": 0.40}  # ETH is weakest

    sold_value = run_async(swap_weakest_position("SOL/USD", new_signal=0.90, latest_signals=latest_signals, threshold=0.05))

    assert sold_value == pytest.approx(2000.0)
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.symbol == "ETH/USD"


def test_swap_weakest_position_does_nothing_below_threshold(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = [
        _mock_position("BTCUSD", "5000", "0.1", "50000", "48000"),
    ]
    latest_signals = {"BTC/USD": 0.55}

    sold_value = run_async(swap_weakest_position("SOL/USD", new_signal=0.56, latest_signals=latest_signals, threshold=0.05))

    assert sold_value == 0.0
    assert not mock_trading_client.submit_order.called


def test_swap_weakest_position_returns_zero_on_sell_failure(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = [
        _mock_position("BTCUSD", "5000", "0.1", "50000", "48000"),
    ]
    mock_trading_client.submit_order.side_effect = RuntimeError("exchange rejected order")
    latest_signals = {"BTC/USD": 0.40}

    sold_value = run_async(swap_weakest_position("SOL/USD", new_signal=0.90, latest_signals=latest_signals, threshold=0.05))

    assert sold_value == 0.0


# ── sync_existing_positions ──

def test_sync_existing_positions_seeds_missing_state(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(symbol="BTCUSD", qty=0.1, avg_entry_price=50000.0, market_value=5100.0, current_price=51000.0),
    ]
    entry_time, highest_prices = {}, {}

    sync_existing_positions(entry_time, highest_prices)

    assert "BTC/USD" in entry_time
    assert highest_prices["BTC/USD"] == pytest.approx(51000.0)


def test_sync_existing_positions_cleans_up_closed_positions(mock_trading_client):
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(symbol="BTCUSD", qty=0.1, avg_entry_price=50000.0, market_value=5100.0, current_price=51000.0),
    ]
    entry_time = {"BTC/USD": 123.0, "ETH/USD": 456.0}
    highest_prices = {"BTC/USD": 51000.0, "ETH/USD": 2000.0}

    sync_existing_positions(entry_time, highest_prices)

    assert "ETH/USD" not in entry_time
    assert "ETH/USD" not in highest_prices
    assert entry_time["BTC/USD"] == 123.0  # untouched, was already tracked
