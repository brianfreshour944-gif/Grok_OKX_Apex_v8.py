# portfolio.py — Alpaca position management: fetch, sync, and force-sell helpers.

import asyncio
import time

from alpaca.trading.enums import OrderSide

from config import (
    logger, trading_client, HEARTBEAT_PATH, MAX_HOLD_HOURS
)
from orders import place_order


# --- Portfolio State Class ---
class PortfolioState:
    """Encapsulates portfolio-related state to avoid module-level globals."""
    
    def __init__(self):
        # Per-symbol sell retry cooldown: prevents spam when settlement is delayed.
        self.sell_retry_cooldown: dict = {}
        
        # Per-symbol "a sell was just submitted for this position, don't submit
        # another yet" guard, shared by every sell path (the main exit loop,
        # sell_largest_position, swap_weakest_position).
        self.PENDING_EXIT_SECONDS = 240
        self.pending_exit_until: dict = {}

    def has_pending_exit(self, symbol: str) -> bool:
        return time.time() < self.pending_exit_until.get(symbol, 0.0)

    def mark_pending_exit(self, symbol: str) -> None:
        self.pending_exit_until[symbol] = time.time() + self.PENDING_EXIT_SECONDS


# Global instance for backward compatibility
_portfolio_state = PortfolioState()


def has_pending_exit(symbol: str) -> bool:
    return _portfolio_state.has_pending_exit(symbol)


def mark_pending_exit(symbol: str) -> None:
    _portfolio_state.mark_pending_exit(symbol)


def normalize_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' -> 'BTCUSD' to match Alpaca's position key format."""
    return symbol.replace("/", "")


def denormalize_symbol(alpaca_sym: str) -> str:
    """
    Convert 'BTCUSD' -> 'BTC/USD'. Assumes a 3-letter USD quote currency,
    true for every pair this bot has ever traded.

    Used instead of reverse-searching a fixed symbol list (e.g. SYMBOLS or
    DYNAMIC_UNIVERSE_CANDIDATES) so that a position held in a symbol that
    has since rotated OUT of the current dynamic universe is still
    recognized and kept under exit management, rather than silently
    dropped from monitoring because it's no longer in "the list".
    """
    return f"{alpaca_sym[:-3]}/{alpaca_sym[-3:]}"


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


def _select_largest_position_to_sell() -> dict | None:
    """
    Picks the largest open position (by market value) to force-sell when the
    portfolio cap is exceeded. Read-only -- does not submit anything. Returns
    None if there's nothing to sell or the symbol is still in its 5-minute
    retry-cooldown window (avoids spamming Alpaca when cash hasn't settled).
    """
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            return None

        largest = max(positions, key=lambda p: float(p.market_value))
        # Denormalize here: Alpaca's position.symbol is slash-less ('BTCUSD'),
        # but every other order submission path in this bot (place_order via
        # main_bot.py's regular buy/sell) uses the slash format ('BTC/USD').
        # Submitting the raw position symbol directly (as this function used
        # to do) risked the exchange rejecting the format mismatch on every
        # forced de-risk sell -- align on the format already proven to work.
        symbol  = denormalize_symbol(largest.symbol)
        now     = time.time()

        if symbol in _portfolio_state.sell_retry_cooldown:
            elapsed = now - _portfolio_state.sell_retry_cooldown[symbol]
            if elapsed < 300:
                logger.warning(
                    f"⏳ Sell retry cooldown for {symbol} "
                    f"({300 - elapsed:.0f}s remaining)"
                )
                return None

        return {
            "symbol":    symbol,
            "qty":       float(largest.qty),
            "price":     float(largest.current_price),
            "avg_entry": float(largest.avg_entry_price),
        }

    except Exception as e:
        status_code = getattr(e, 'status_code', None)
        if status_code == 429:
            logger.warning(f"Rate limited during sell_largest_position. Will retry next cycle: {e}")
        else:
            logger.error(f"sell_largest_position failed: {e}")
        return None


async def sell_largest_position() -> None:
    """
    Force-sells the largest open position (by market value) when the
    portfolio cap is exceeded, via place_order() so it gets the same
    rate-limit handling, DB logging, and realized-PnL tracking as every
    other sell in the bot. Backs off for 5 minutes after a failed attempt.
    """
    candidate = await asyncio.to_thread(_select_largest_position_to_sell)
    if candidate is None:
        return

    if has_pending_exit(candidate["symbol"]):
        logger.info(f"⏳ {candidate['symbol']}: a sell is already pending, skipping duplicate force-sell")
        return

    logger.warning(
        f"📉 Cap exceeded — force selling {candidate['symbol']} "
        f"${candidate['qty'] * candidate['price']:.2f}"
    )
    success = await place_order(
        candidate["symbol"], OrderSide.SELL, candidate["qty"], candidate["price"],
        avg_entry=candidate["avg_entry"],
    )
    if success:
        _portfolio_state.sell_retry_cooldown.pop(candidate["symbol"], None)
        mark_pending_exit(candidate["symbol"])
    else:
        _portfolio_state.sell_retry_cooldown[candidate["symbol"]] = time.time()


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
        sym = denormalize_symbol(alpaca_sym)
        if sym not in entry_time:
            # Crash-recovery: do NOT seed entry_time at "now". A position held
            # 3.5h pre-crash would otherwise get a fresh hold clock, re-arming
            # the time-decay stop-loss (which halves the stop after 2h held)
            # and delaying max-hold. When the true open time is unknown
            # (bot_state row lost / DB down), assume mid-life: half of
            # MAX_HOLD_HOURS elapsed. This biases toward the tighter, more
            # conservative stop rather than the widest one.
            entry_time[sym] = time.time() - (MAX_HOLD_HOURS * 3600 * 0.5)
            logger.info(f"   Seeded entry_time for {sym} (was missing after restart; "
                        f"assumed {MAX_HOLD_HOURS/2:.1f}h held)")
        if sym not in highest_prices:
            # Seed highest_prices from current exchange price so the trailing
            # stop baseline reflects the actual market price at restart,
            # not just the original entry. Using max() ensures we never
            # set a peak below current price (which would trigger an
            # immediate false trailing-stop sell).
            highest_prices[sym] = max(data["avg_entry"], data["current_price"])
            logger.info(f"   Seeded highest_prices for {sym} from exchange (price={data['current_price']:.4f})")
        logger.info(
            f"Renewing: {sym} | qty={data['qty']:.6f} | "
            f"avg_entry=${data['avg_entry']:.4f}"
        )


def _select_weakest_position_to_swap(new_symbol: str, new_signal: float, latest_signals: dict, threshold: float):
    """
    Evaluates currently open positions. If there's a held position with a signal
    significantly weaker than new_signal, returns the fields needed to sell it
    (symbol, qty, price, avg_entry, market_value). Read-only -- does not submit
    anything. Returns None if there's nothing to swap or threshold isn't met.
    """
    try:
        positions = get_all_positions()
        if not positions:
            return None

        weakest_sym = None
        weakest_signal = float('inf')
        weakest_data = None

        for alpaca_sym, p_data in positions.items():
            held_sym = denormalize_symbol(alpaca_sym)

            # Fallback to 0.5 if we don't have a recent signal for the held asset
            held_signal = latest_signals.get(held_sym, 0.5)

            if held_signal < weakest_signal:
                weakest_signal = held_signal
                # Keep the slash-formatted symbol, not the raw alpaca_sym --
                # same format-consistency reasoning as in
                # _select_largest_position_to_sell above.
                weakest_sym = held_sym
                weakest_data = p_data

        if weakest_sym and (new_signal - weakest_signal) >= threshold:
            logger.warning(
                f"🔄 SWAP TRIGGERED: Selling {weakest_sym} (signal {weakest_signal:.4f}) "
                f"to make room for {new_symbol} (signal {new_signal:.4f})"
            )
            return {
                "symbol":       weakest_sym,
                "qty":          float(weakest_data['qty']),
                "price":        float(weakest_data['current_price']),
                "avg_entry":    float(weakest_data['avg_entry']),
                "market_value": float(weakest_data['market_value']),
            }

    except Exception as e:
        logger.error(f"swap_weakest_position failed: {e}")

    return None


async def swap_weakest_position(new_symbol: str, new_signal: float, latest_signals: dict, threshold: float = 0.05) -> float:
    """
    Force-sells the currently held position with the weakest signal (if it's
    significantly weaker than new_signal) to free up capital, via place_order()
    so it gets the same rate-limit handling, DB logging, and realized-PnL
    tracking as every other sell in the bot.

    Returns the market value (USD notional) of the position sold, so callers can
    decrement their own running portfolio-value tracking accordingly. Returns
    0.0 if no swap was executed (nothing to swap, threshold not met, or the
    sell order failed).
    """
    candidate = await asyncio.to_thread(
        _select_weakest_position_to_swap, new_symbol, new_signal, latest_signals, threshold
    )
    if candidate is None:
        return 0.0

    if has_pending_exit(candidate["symbol"]):
        logger.info(f"⏳ {candidate['symbol']}: a sell is already pending, skipping duplicate swap-sell")
        return 0.0

    success = await place_order(
        candidate["symbol"], OrderSide.SELL, candidate["qty"], candidate["price"],
        avg_entry=candidate["avg_entry"],
    )
    if success:
        mark_pending_exit(candidate["symbol"])
        return candidate["market_value"]
    logger.error(f"Failed to execute swap sell for {candidate['symbol']}")
    return 0.0

def cancel_stale_orders(timeout_minutes=3):
    """
    Finds and cancels any open orders that have been sitting unfilled for longer than timeout_minutes.
    This frees up buying power that gets locked by Limit Order Entries.
    Uses rate-limit-aware API calls with exponential backoff.
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
                logger.info(f"Canceling stale unfilled order {order.id} for {order.symbol} (Age: {order_age:.1f}m)")
                trading_client.cancel_order_by_id(order.id)
    except Exception as e:
        # Check if it's a rate limit error and back off
        status_code = getattr(e, 'status_code', None)
        if status_code == 429:
            logger.warning(f"Rate limited during cancel_stale_orders. Will retry next cycle: {e}")
        else:
            logger.error(f"Failed to cancel stale orders: {e}")


async def cancel_stale_orders_async(timeout_minutes=3):
    """Async wrapper for cancel_stale_orders to avoid blocking the event loop."""
    await asyncio.to_thread(cancel_stale_orders, timeout_minutes)


def fetch_open_sell_symbols() -> set:
    """
    Returns the set of Alpaca-format symbols ('BTCUSD') that currently have an
    OPEN SELL order on the exchange. Used at startup so that, after a crash,
    every held position with a live sell gets a pending-exit guard — the
    in-memory pending_exit_until dict does not survive a restart, and without
    this the bot would submit a duplicate full-size SELL on its first cycle.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus, OrderSide as _OS
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return {
            o.symbol for o in trading_client.get_orders(req)
            if o.side == _OS.SELL
        }
    except Exception as e:
        logger.warning(f"fetch_open_sell_symbols failed (non-fatal): {e}")
        return set()


def fetch_recent_closed_orders(hours: float = 24.0) -> list:
    """
    Returns closed orders submitted within the last `hours` hours. Used by the
    startup trade-log reconciliation: a crash between submit_order() success
    and the DB write loses that trade row permanently unless it is back-filled
    from exchange order history.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from datetime import datetime, timedelta, timezone
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=datetime.now(timezone.utc) - timedelta(hours=hours),
        )
        return list(trading_client.get_orders(req))
    except Exception as e:
        logger.warning(f"fetch_recent_closed_orders failed (non-fatal): {e}")
        return []

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
