# orders.py — Order submission with DB logging and sell-qty precision fix.

import math

from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import logger, trading_client, BOT_NAME
from database import record_trade


async def place_order(symbol: str, side: OrderSide, qty: float, price: float = None) -> bool:
    """
    Submits an order to Alpaca and logs it to the database.
    BUYs are Market Orders. SELLs are Limit Orders.

    Sell qty is floored to 8 decimal places before submission to prevent
    Alpaca's 'insufficient balance' rejection caused by float64 precision drift:
        float('22966090.330189045') -> 22966090.330189046  (1 ULP too high)
    """
    try:
        if side == OrderSide.SELL:
            qty = math.floor(qty * 1e8) / 1e8
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=price
            )
        else:
            limit_price = price * 1.001 if price else None
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )

        order = trading_client.submit_order(order_data=order_data)
        record_trade(BOT_NAME, symbol, side.value, qty, price, order_id=order.id)
        logger.info(f"✅ Order submitted: {side.value} {symbol} {qty:.6f} limit={price if side == OrderSide.SELL else limit_price}")
        return True

    except Exception as e:
        logger.error(f"❌ Order failed ({side.value} {symbol} qty={qty:.6f}): {e}")
        return False
