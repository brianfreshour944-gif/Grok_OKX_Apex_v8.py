# portfolio.py — Alpaca position management: fetch, sync, and force-sell helpers.

import time

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import (
    logger, trading_client, SYMBOLS, MIN_POSITION_USD
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
                order_data=MarketOrderRequest(
                    symbol=largest.symbol,
                    qty=float(largest.qty),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )
            )
            sell_retry_cooldown.pop(largest.symbol, None)
        except Exception as sell_err:
            sell_retry_cooldown[largest.symbol] = now
            logger.error(f"Sell order failed (will retry later): {sell_err}")

    except Exception as e:
        logger.error(f"sell_largest_position failed: {e}")


def sync_existing_positions(entry_time: dict) -> None:
    """
    On startup, populate entry_time for any positions already open in Alpaca.
    Does not overwrite entries already being tracked.
    """
    logger.info("🔍 Scanning existing positions on startup...")
    positions = get_all_positions()
    if not positions:
        logger.info("No existing positions found.")
        return

    for alpaca_sym, data in positions.items():
        for sym in SYMBOLS:
            if normalize_symbol(sym) == alpaca_sym:
                if sym not in entry_time:
                    entry_time[sym] = time.time()
                logger.info(
                    f"♻️  Restored: {sym} | qty={data['qty']:.6f} | "
                    f"avg_entry=${data['avg_entry']:.4f}"
                )
                break
