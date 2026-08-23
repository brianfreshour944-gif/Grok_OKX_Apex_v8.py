# tests/test_orders.py — order submission (mocked Alpaca client, no network,
# no real DB since DATABASE_URL is unset in tests).

from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import OrderSide

from orders import _sanitize_price, place_order
from conftest import run_async


# ── _sanitize_price ──

@pytest.mark.parametrize("price,expected_places", [
    (238.723456, 2),
    (0.05123456, 4),
    (0.00012345, 6),
    (0.0000001234, 8),
])
def test_sanitize_price_rounds_down_to_expected_precision(price, expected_places):
    result = _sanitize_price(price)
    s = f"{result:.10f}".rstrip("0")
    decimals = len(s.split(".")[1]) if "." in s else 0
    assert decimals <= expected_places


def test_sanitize_price_rounds_down_not_to_nearest():
    # 238.729 at 2dp should floor to 238.72, not round to 238.73.
    assert _sanitize_price(238.729) == 238.72


# ── place_order ──

def test_place_order_buy_success(mock_trading_client):
    fake_order = MagicMock(id="order-1")
    fake_filled = MagicMock(id="order-1", filled_avg_price="100.05", commission="0.01")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.return_value = fake_filled

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is True
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.symbol == "BTC/USD"
    assert order_data.side == OrderSide.BUY
    # BUY limit is 0.1% above the reference price
    assert order_data.limit_price == pytest.approx(100.1, abs=0.01)


def test_place_order_sell_prices_below_market(mock_trading_client):
    fake_order = MagicMock(id="order-2")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.return_value = MagicMock(id="order-2", filled_avg_price=None, commission=None)

    result = run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.01, price=100.0))

    assert result is True
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.side == OrderSide.SELL
    # SELL limit is 0.1% below the reference price
    assert order_data.limit_price == pytest.approx(99.9, abs=0.01)


def test_place_order_returns_false_on_submit_failure(mock_trading_client):
    mock_trading_client.submit_order.side_effect = RuntimeError("network error")

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is False


def test_place_order_still_succeeds_if_fill_lookup_fails(mock_trading_client):
    """Order submission succeeding is what matters; a failure to fetch fill
    details afterward (e.g. transient API hiccup) must not be treated as a
    failed trade -- it already executed on the exchange."""
    fake_order = MagicMock(id="order-3")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.side_effect = RuntimeError("transient")

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is True
