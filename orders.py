# orders.py — Order submission with DB logging and sell-qty precision fix.

import math
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import logger, trading_client, BOT_NAME
from database import record_trade


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


async def place_order(symbol: str, side: OrderSide, qty: float, price: float = None) -> bool:
    """
    Submits an order to Alpaca and logs it to the database.
    BUYs are Limit Orders (0.1% above market to ensure fill).
    SELLs are Limit Orders at the current price.

    Sell qty is floored to 8 decimal places before submission to prevent
    Alpaca's 'insufficient balance' rejection caused by float64 precision drift.
    Prices are sanitized via Decimal to prevent 'exceeds max precision' errors.
    """
    try:
        if side == OrderSide.SELL:
            qty         = math.floor(qty * 1e8) / 1e8
            limit_price = _sanitize_price(price) if price else None
            order_data  = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )
        else:
            raw_limit   = price * 1.001 if price else None
            limit_price = _sanitize_price(raw_limit) if raw_limit else None
            order_data  = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )

        order = trading_client.submit_order(order_data=order_data)
        record_trade(BOT_NAME, symbol, side.value, qty, price, order_id=order.id)
        logger.info(f"✅ Order submitted: {side.value} {symbol} {qty:.6f} limit={limit_price}")
        return True

    except Exception as e:
        logger.error(f"❌ Order failed ({side.value} {symbol} qty={qty:.6f}): {e}")
        return False
