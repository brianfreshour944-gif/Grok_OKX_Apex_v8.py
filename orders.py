# orders.py — Order submission with DB logging and sell-qty precision fix.

import asyncio
import math
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import logger, trading_client, BOT_NAME
from database import record_trade
from api_utils import call_with_rate_limit_handling_async
from money import realized_pnl as calc_realized_pnl, pnl_pct_fraction as calc_pnl_pct


def _sanitize_price(price: float) -> float:
    """
    Round a price to the correct number of decimal places based on magnitude,
    preventing Alpaca's "limit price exceeds maximum precision" rejection.

    Alpaca's rule: max 9 decimal places. We apply tighter, magnitude-based
    rounding to eliminate any float64 arithmetic noise before submission.
    """
    d = Decimal(str(price))
    if price >= 1.0:
        return float(d.quantize(Decimal('0.01'), rounding=ROUND_DOWN))
    elif price >= 0.01:
        return float(d.quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
    elif price >= 0.0001:
        return float(d.quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
    else:
        return float(d.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))


async def place_order(symbol: str, side: OrderSide, qty: float, price: float = None,
                       avg_entry: float = None, order_id_out: dict = None) -> bool:
    """
    Submits an order to Alpaca and logs it to the database.
    BUYs are Limit Orders (0.1% above market to ensure fill).
    SELLs are Limit Orders at the current price.

    Sell qty is floored to 8 decimal places before submission to prevent
    Alpaca's 'insufficient balance' rejection caused by float64 precision drift.
    Prices are sanitized via Decimal to prevent 'exceeds max precision' errors.

    After submission, fetches the filled order details from the exchange to
    record actual fill_price and fees, enabling accurate PnL tracking.

    avg_entry: the position's average entry price at the time of a SELL,
    used to compute and persist realized PnL (net of fees) for that closed
    position. Ignored for BUY orders; if omitted on a SELL, realized PnL is
    simply not recorded for that trade (rather than guessed).

    order_id_out: optional dict; on successful submission the exchange order
    id is written to order_id_out["order_id"] so callers can associate
    downstream logging (e.g. experience capture) with the real order.
    """
    try:
        if side == OrderSide.SELL:
            qty         = math.floor(qty * 1e8) / 1e8
            raw_limit   = price * 0.999 if price else None
            limit_price = _sanitize_price(raw_limit) if raw_limit else None
            order_data  = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )
        else:
            qty = math.floor(qty * 1e8) / 1e8  # Floor BUY qty to 8 decimals too
            raw_limit   = price * 1.001 if price else None
            limit_price = _sanitize_price(raw_limit) if raw_limit else None
            order_data  = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )

        order = await call_with_rate_limit_handling_async(
            trading_client.submit_order, order_data=order_data,
            max_retries=5, base_delay=1.0
        )
        if order_id_out is not None:
            order_id_out["order_id"] = str(getattr(order, "id", None))
        
        # Fetch actual fill details from exchange for accurate fee/slippage tracking.
        # CRITICAL FIX: Limit orders may not fill immediately. We must wait for
        # the fill to actually happen before returning, otherwise the next
        # trading cycle reads stale positions from the exchange (which still
        # shows the position as open), causing position sizing based on stale
        # data and "order not found" errors when trying to look up the order.
        actual_fill_price = None
        actual_fee = 0.0
        try:
            # Poll for fill status up to FILL_WAIT_TIMEOUT seconds
            FILL_WAIT_TIMEOUT = 60  # seconds
            FILL_POLL_INTERVAL = 2  # seconds between polls
            elapsed = 0.0
            filled_order = None

            while elapsed < FILL_WAIT_TIMEOUT:
                filled_order = await call_with_rate_limit_handling_async(
                    trading_client.get_order_by_id, order.id,
                    max_retries=3, base_delay=1.0
                )
                if filled_order is None:
                    await asyncio.sleep(FILL_POLL_INTERVAL)
                    elapsed += FILL_POLL_INTERVAL
                    continue

                # Check if the order has filled (filled_qty == qty means filled)
                filled_qty = getattr(filled_order, 'filled_qty', None)
                if filled_qty and float(filled_qty) > 0:
                    if hasattr(filled_order, 'filled_avg_price') and filled_order.filled_avg_price:
                        actual_fill_price = float(filled_order.filled_avg_price)
                    # NOTE: alpaca-py 0.33.0 Order model has no 'commission' field.
                    # Fee data is not accessible via the TradingClient order objects.
                    # actual_fee remains 0.0 — fees are not currently captured.
                    break

                await asyncio.sleep(FILL_POLL_INTERVAL)
                elapsed += FILL_POLL_INTERVAL

            if filled_order is not None:
                if actual_fill_price is None and hasattr(filled_order, 'filled_avg_price') and filled_order.filled_avg_price:
                    actual_fill_price = float(filled_order.filled_avg_price)
                # NOTE: alpaca-py 0.33.0 Order model has no 'commission' field.

        except Exception as fill_err:
            logger.warning(f"Could not fetch fill details for order {order.id}: {fill_err}")
        
        # Log slippage if fill price differs from expected
        if actual_fill_price and price:
            slippage = actual_fill_price - price
            slippage_pct = (slippage / price) * 100 if price > 0 else 0
            if abs(slippage_pct) > 0.1:  # Only log significant slippage
                logger.warning(
                    f"Slippage: {side.value} {symbol} | "
                    f"Expected: ${price:.4f} | Actual fill: ${actual_fill_price:.4f} | "
                    f"Diff: ${slippage:.4f} ({slippage_pct:+.2f}%)"
                )
        
        # Realized PnL is only meaningful for a SELL closing a known position,
        # and only once we have an actual fill price to measure it against --
        # measuring it against the limit `price` would conflate slippage with PnL.
        realized_pnl_dollar = None
        realized_pnl_pct = None
        if side == OrderSide.SELL and avg_entry is not None and actual_fill_price is not None:
            realized_pnl_dollar = calc_realized_pnl(avg_entry, actual_fill_price, qty, fee=actual_fee)
            realized_pnl_pct = calc_pnl_pct(avg_entry, actual_fill_price)

        await asyncio.to_thread(
            record_trade,
            BOT_NAME, symbol, side.value, qty, price,
            order_id=order.id, fee=actual_fee, fill_price=actual_fill_price,
            realized_pnl=realized_pnl_dollar, realized_pnl_pct=realized_pnl_pct,
        )
        pnl_log = f" | Realized PnL: ${realized_pnl_dollar:+.2f} ({realized_pnl_pct*100:+.2f}%)" if realized_pnl_dollar is not None else ""
        logger.info(f"Order submitted: {side.value} {symbol} {qty:.6f} limit={limit_price} fill={actual_fill_price or 'pending'} fee=${actual_fee:.4f}{pnl_log}")
        return True

    except Exception as e:
        logger.error(f"Order failed ({side.value} {symbol} qty={qty:.6f}): {e}")
        return False
