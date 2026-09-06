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
    fake_filled = MagicMock(id="order-1", filled_avg_price="100.05")
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
    mock_trading_client.get_order_by_id.return_value = MagicMock(id="order-2", filled_avg_price=None)

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


# ── Realized PnL pass-through ──

def test_sell_with_avg_entry_and_fill_records_realized_pnl(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    fake_order = MagicMock(id="order-4")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-4", filled_avg_price="110.0",
    )

    result = run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9, avg_entry=100.0))

    assert result is True
    # NOTE: alpaca-py 0.33.0 Order model has no 'commission' field, so actual_fee
    # stays 0.0 — realized PnL is gross, not net of fees.
    assert recorded["realized_pnl"] == pytest.approx(20.0)   # (110-100)*2 - 0.0 fee
    assert recorded["realized_pnl_pct"] == pytest.approx(0.10)  # 10% vs avg_entry


def test_buy_never_records_realized_pnl(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-5")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-5", filled_avg_price="100.1",
    )

    run_async(place_order("BTC/USD", OrderSide.BUY, qty=1.0, price=100.0, avg_entry=95.0))

    assert recorded["realized_pnl"] is None
    assert recorded["realized_pnl_pct"] is None


def test_sell_without_avg_entry_does_not_record_realized_pnl(mock_trading_client, monkeypatch):
    """swap_weakest_position/sell_largest_position don't go through place_order
    at all today (a separate known gap), but any SELL that omits avg_entry --
    e.g. because the caller has no position data -- must not fabricate a
    realized PnL number rather than silently guessing at one."""
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-6")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-6", filled_avg_price="110.0",
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9))  # no avg_entry

    assert recorded["realized_pnl"] is None
    assert recorded["realized_pnl_pct"] is None


def test_sell_without_a_fill_price_does_not_record_realized_pnl(mock_trading_client, monkeypatch):
    """Must not compute realized PnL against the limit price when the actual
    fill price is unknown -- that would conflate slippage with PnL."""
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-7")
    mock_trading_client.get_order_by_id.side_effect = RuntimeError("transient")

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9, avg_entry=100.0))

    assert recorded["realized_pnl"] is None
    assert recorded["realized_pnl_pct"] is None
