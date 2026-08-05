# portfolio.py — Alpaca position management: fetch, sync, and force-sell helpers.

import asyncio
import time

from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import (
    logger, trading_client, SYMBOLS, MIN_POSITION_USD, HEARTBEAT_PATH
)

# Per-symbol sell retry cooldown: prevents spam when settlement is delayed.
sell_retry_cooldown: dict = {}


def normalize_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' -> 'BTCUSD' to match Alpaca's position key format."""
    return symbol.replace("/", "")


def get_all_positions() -> dict:
    """
    Returns {alpaca_symbol: {qty, avg_entry, market_value, current_price}}
    for every open Alpaca position.
    """
    try:
        positions = trading_client.get_all_positions()
        return {
            p.symbol: {
                "qty":           float(p.qty),
                "avg_entry":     float(p.avg_entry_price),
                "market_value":  float(p.market_value),
                "current_price": float(p.current_price),
            }
            for p in positions
        }
    except Exception as e:
        logger.error(f"get_all_positions failed: {e}")
        return {}


def get_buying_power() -> float:
    """Returns current account buying power, or 0.0 on failure."""
    try:
        return float(trading_client.get_account().buying_power)
    except Exception as e:
        logger.error(f"Buying power fetch failed: {e}")
        return 0.0


async def get_all_positions_async() -> dict:
    """Async wrapper for get_all_positions to avoid blocking the event loop."""
    return await asyncio.to_thread(get_all_positions)


async def get_buying_power_async() -> float:
    """Async wrapper for get_buying_power to avoid blocking the event loop."""
    return await asyncio.to_thread(get_buying_power)


def sell_largest_position() -> None:
    """
    Force-sells the largest open position (by market value) when the
    portfolio cap is exceeded. Backs off for 5 minutes after a failed attempt
    to avoid spamming Alpaca when cash hasn't settled.
    """
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            return

        largest = max(positions, key=lambda p: float(p.market_value))
        now     = time.time()

        if largest.symbol in sell_retry_cooldown:
            elapsed = now - sell_retry_cooldown[largest.symbol]
            if elapsed < 300:
                logger.warning(
                    f"⏳ Sell retry cooldown for {largest.symbol} "
                    f"({300 - elapsed:.0f}s remaining)"
                )
                return

        logger.warning(
            f"📉 Cap exceeded — force selling {largest.symbol} "
            f"${float(largest.market_value):.2f}"
        )
        try:
            trading_client.submit_order(
                order_data=LimitOrderRequest(
                    symbol=largest.symbol,
                    qty=float(largest.qty),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    limit_price=float(largest.current_price)
                )
            )
            sell_retry_cooldown.pop(largest.symbol, None)
        except Exception as sell_err:
            sell_retry_cooldown[largest.symbol] = now
            logger.error(f"Sell order failed (will retry later): {sell_err}")

    except Exception as e:
        logger.error(f"sell_largest_position failed: {e}")


def sync_existing_positions(entry_time: dict, highest_prices: dict) -> None:
    """
    On startup, populate entry_time/highest_prices for positions already open in Alpaca.
    Also cleans up stale state for symbols that are no longer held.
    Does not overwrite entries already being tracked for active positions.
    """
    logger.info("🔍 Scanning existing positions on startup...")
    positions = get_all_positions()
    
    # Get set of currently held symbols (keys from the positions dict)
    current_alpaca_positions = set(positions.keys()) if positions else set()
    
    # Clean up stale state for symbols no longer held in Alpaca
    for sym in list(entry_time.keys()):
        alpaca_sym = normalize_symbol(sym)
        if alpaca_sym not in current_alpaca_positions:
            logger.info(f"🧹 Cleaning up stale state for {sym} (not currently held)")
            entry_time.pop(sym, None)
            highest_prices.pop(sym, None)
    
    if not positions:
        logger.info("No existing positions found.")
        return

    for alpaca_sym, data in positions.items():
        for sym in SYMBOLS:
            if normalize_symbol(sym) == alpaca_sym:
                if sym not in entry_time:
                    entry_time[sym] = time.time()
                if sym not in highest_prices:
                    highest_prices[sym] = data["avg_entry"]
                logger.info(
                    f"♻️  Restored: {sym} | qty={data['qty']:.6f} | "
                    f"avg_entry=${data['avg_entry']:.4f}"
                )
                break


def swap_weakest_position(new_symbol: str, new_signal: float, latest_signals: dict, threshold: float = 0.05) -> float:
    """
    Evaluates currently open positions. If there's a held position with a signal significantly 
    weaker than the new_signal, it force-sells that position to free up capital.

    Returns the market value (USD notional) of the position sold, so callers can
    decrement their own running portfolio-value tracking accordingly. Returns
    0.0 if no swap was executed (nothing to swap, threshold not met, or the
    sell order failed).
    """
    try:
        positions = get_all_positions()
        if not positions:
            return 0.0

        weakest_sym = None
        weakest_signal = float('inf')
        weakest_qty = 0.0
        weakest_price = 0.0
        weakest_value = 0.0

        for alpaca_sym, p_data in positions.items():
            # Reverse map from alpaca_sym to standard symbol
            held_sym = next((s for s in SYMBOLS if normalize_symbol(s) == alpaca_sym), None)
            if not held_sym:
                continue

            # Fallback to 0.5 if we don't have a recent signal for the held asset
            held_signal = latest_signals.get(held_sym, 0.5)

            if held_signal < weakest_signal:
                weakest_signal = held_signal
                weakest_sym = alpaca_sym
                weakest_qty = p_data['qty']
                weakest_price = p_data['current_price']
                weakest_value = p_data['market_value']

        if weakest_sym and (new_signal - weakest_signal) >= threshold:
            logger.warning(
                f"🔄 SWAP TRIGGERED: Selling {weakest_sym} (signal {weakest_signal:.4f}) "
                f"to make room for {new_symbol} (signal {new_signal:.4f})"
            )
            
            # Execute limit sell for the weakest asset
            try:
                from alpaca.trading.requests import LimitOrderRequest
                trading_client.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=weakest_sym,
                        qty=float(weakest_qty),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        limit_price=float(weakest_price)
                    )
                )
                return float(weakest_value)
            except Exception as e:
                logger.error(f"Failed to execute swap sell for {weakest_sym}: {e}")
                return 0.0

    except Exception as e:
        logger.error(f"swap_weakest_position failed: {e}")
        
    return 0.0

def cancel_stale_orders(timeout_minutes=3):
    """
    Finds and cancels any open orders that have been sitting unfilled for longer than timeout_minutes.
    This frees up buying power that gets locked by Limit Order Entries.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from datetime import datetime, timezone
        
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(req)
        now = datetime.now(timezone.utc)
        
        for order in open_orders:
            order_age = (now - order.created_at).total_seconds() / 60.0
            if order_age > timeout_minutes:
                logger.info(f"🗑️ Canceling stale unfilled order {order.id} for {order.symbol} (Age: {order_age:.1f}m)")
                trading_client.cancel_order_by_id(order.id)
    except Exception as e:
        logger.error(f"Failed to cancel stale orders: {e}")


async def cancel_stale_orders_async(timeout_minutes=3):
    """Async wrapper for cancel_stale_orders to avoid blocking the event loop."""
    await asyncio.to_thread(cancel_stale_orders, timeout_minutes)

def calculate_kelly_multiplier(signal_prob: float, profit_target_pct: float, stop_loss_pct: float) -> float:
    """
    Dynamically calculates a Half-Kelly multiplier based on the ML model's win probability.
    W = signal_prob (the model's confidence in the trade)
    R = Reward/Risk ratio (profit_target / stop_loss)
    Returns a multiplier (e.g., 0.5 to 3.0) to scale the BASE_RISK_PERCENT.
    """
    if stop_loss_pct <= 0 or profit_target_pct <= 0:
        return 1.0
        
    w = signal_prob
    r = profit_target_pct / stop_loss_pct
    
    # Kelly Formula: K = W - ((1 - W) / R)
    kelly_fraction = w - ((1.0 - w) / r)
    
    # If the edge is technically negative (Kelly < 0), we default to a minimal base multiplier 
    # instead of 0, assuming the user's hard BUY_SIGNAL threshold already filters bad trades.
    if kelly_fraction <= 0:
        return 0.5
        
    # We map the Kelly fraction to a conservative multiplier (Half-Kelly approach)
    # A 10% Kelly fraction (0.10) is huge. We'll map K=0.0 -> 0.5x, K=0.10 -> 2.0x, K>=0.20 -> 3.0x
    multiplier = 0.5 + (kelly_fraction * 15.0)
    
    # Cap the multiplier to prevent over-leveraging
    return max(0.5, min(multiplier, 3.0))

def write_heartbeat():
    """Writes a timestamp to a file so external monitors (like Coolify) can verify the bot is alive."""
    import os
    from datetime import datetime, timezone
    try:
        # Ensure directory exists if using /tmp or similar
        os.makedirs(os.path.dirname(HEARTBEAT_PATH) or ".", exist_ok=True)
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception as e:
        logger.error(f"Heartbeat write failed: {e}")
