# database.py — PostgreSQL trade and equity logging.
# Silently skips all DB calls when DATABASE_URL is not set.
# Uses context managers to prevent connection leaks on exceptions.

import os
import psycopg2
from config import logger, BOT_NAME


def _init_bot_status_table(cur):
    """Create bot_status table if it doesn't exist."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bot_status (
            bot_name TEXT PRIMARY KEY,
            starting_equity NUMERIC,
            live_equity NUMERIC,
            live_equity_updated_at TIMESTAMP,
            last_update TIMESTAMP
        )
    """)

def _init_bot_state_table(cur):
    """Create bot_state table if it doesn't exist."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bot_state (
            symbol TEXT PRIMARY KEY,
            cooldown_until REAL,
            entry_time REAL,
            latest_signal REAL,
            highest_price REAL
        )
    """)

def _init_tables(cur):
    """Create required tables if they don't exist. Called once at startup."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_history (
            id        SERIAL PRIMARY KEY,
            bot_name  TEXT    NOT NULL,
            equity    NUMERIC NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            bot_name TEXT,
            exchange TEXT,
            symbol TEXT,
            side TEXT,
            price REAL,
            quantity REAL,
            value REAL,
            fee REAL,
            fill_price REAL,
            order_id TEXT,
            timestamp TIMESTAMP
        )
    """)
    # Migration: add fill_price column if it doesn't exist (for older DBs)
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'trades' AND column_name = 'fill_price'
            ) THEN
                ALTER TABLE trades ADD COLUMN fill_price REAL;
            END IF;
        END $$;
    """)
    # Migration: add realized_pnl / realized_pnl_pct columns if they don't exist.
    # Only populated on SELL rows (net of fees) -- NULL on BUY rows and on SELLs
    # where the caller didn't have an avg_entry to compute against.
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'trades' AND column_name = 'realized_pnl'
            ) THEN
                ALTER TABLE trades ADD COLUMN realized_pnl REAL;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'trades' AND column_name = 'realized_pnl_pct'
            ) THEN
                ALTER TABLE trades ADD COLUMN realized_pnl_pct REAL;
            END IF;
        END $$;
    """)


def init_db():
    """
    Run all DDL once at startup. Safe to call multiple times (IF NOT EXISTS).
    No-op when DATABASE_URL is not set.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                _init_tables(cur)
                _init_bot_status_table(cur)
                _init_bot_state_table(cur)
            conn.commit()
        logger.info("📘 DB: tables initialised")
    except Exception as e:
        logger.warning(f"⚠️ DB init failed (non-fatal): {e}")


def vacuum_full_if_needed():
    """
    Run VACUUM (ANALYZE) on tables that may have bloated from frequent
    UPSERTs (bot_state) and high-volume inserts (trades, equity_history).

    CRITICAL: PostgreSQL's VACUUM cannot run inside a transaction block.
    psycopg2's default context manager 'with psycopg2.connect(...)' starts a
    transaction, so calling VACUUM inside it raises:
      'cannot execute VACUUM from within a transaction block'

    Fix: set conn.autocommit = True so VACUUM runs as its own statement,
    not wrapped in BEGIN/COMMIT.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return
    try:
        with psycopg2.connect(db_url) as conn:
            conn.autocommit = True  # must be outside any transaction block
            with conn.cursor() as cur:
                cur.execute("VACUUM (ANALYZE) bot_state")
                cur.execute("VACUUM (ANALYZE) trades")
                cur.execute("VACUUM (ANALYZE) equity_history")
                cur.execute("VACUUM (ANALYZE) bot_status")
        logger.info("📘 DB: vacuum completed")
    except Exception as e:
        logger.warning(f"⚠️ DB vacuum failed (non-fatal): {e}")


def save_bot_state(cooldown_until, entry_time, latest_signals, highest_prices):
    """
    Save the bot's state to the database.

    Iterates the union of keys across all four dicts rather than a fixed
    symbol list -- with a dynamic trading universe, the set of symbols with
    state to persist changes over time and isn't known statically.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return
    all_symbols = set(cooldown_until) | set(entry_time) | set(latest_signals) | set(highest_prices)
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                for symbol in all_symbols:
                    cur.execute("""
                        INSERT INTO bot_state (symbol, cooldown_until, entry_time, latest_signal, highest_price)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                            cooldown_until = EXCLUDED.cooldown_until,
                            entry_time = EXCLUDED.entry_time,
                            latest_signal = EXCLUDED.latest_signal,
                            highest_price = EXCLUDED.highest_price
                    """, (
                        symbol,
                        cooldown_until.get(symbol),
                        entry_time.get(symbol),
                        latest_signals.get(symbol),
                        highest_prices.get(symbol)
                    ))
            conn.commit()
    except Exception as e:
        logger.error(f"DB Error saving bot state: {e}")

def load_bot_state():
    """Load the bot's state from the database."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return {}, {}, {}, {}
    
    cooldown_until = {}
    entry_time = {}
    latest_signals = {}
    highest_prices = {}

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, cooldown_until, entry_time, latest_signal, highest_price FROM bot_state")
                for row in cur.fetchall():
                    symbol, cd, et, ls, hp = row
                    if cd is not None:
                        cooldown_until[symbol] = cd
                    if et is not None:
                        entry_time[symbol] = et
                    if ls is not None:
                        latest_signals[symbol] = ls
                    if hp is not None:
                        highest_prices[symbol] = hp
        logger.info("📘 DB: Bot state loaded from database.")
    except Exception as e:
        logger.error(f"DB Error loading bot state: {e}")
    
    return cooldown_until, entry_time, latest_signals, highest_prices


def record_trade(bot_name, symbol, side, qty, price, order_id=None, fee=0.0, fill_price=None,
                  realized_pnl=None, realized_pnl_pct=None):
    """
    Log a completed trade to the trades table.

    realized_pnl / realized_pnl_pct are only meaningful for SELL rows that
    close out a position against a known avg_entry (see orders.py); they are
    left NULL for BUY rows and for SELLs where no avg_entry was available.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.debug("DATABASE_URL not set - skipping trade DB log")
        return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                value = (fill_price or price or 0.0) * qty
                actual_fill_price = fill_price if fill_price else (price or 0.0)
                cur.execute("""
                    INSERT INTO trades
                        (bot_name, exchange, symbol, side, price, quantity,
                         value, fee, fill_price, order_id, realized_pnl, realized_pnl_pct, timestamp)
                    VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (bot_name, symbol, side, price or 0.0, qty, value,
                      float(fee), float(actual_fill_price),
                      str(order_id) if order_id else None,
                      float(realized_pnl) if realized_pnl is not None else None,
                      float(realized_pnl_pct) if realized_pnl_pct is not None else None))
            conn.commit()
            pnl_str = f" | Realized PnL: ${realized_pnl:.2f}" if realized_pnl is not None else ""
            logger.info(f"Recorded trade: {side} {symbol} | Qty: {qty:.6f} | Price: {price} | Fill: {actual_fill_price} | Fee: ${float(fee):.4f}{pnl_str}")
    except Exception as e:
        logger.error(f"DB Error: {e}")


def backfill_trade_if_missing(order) -> bool:
    """
    Crash-recovery reconciliation: if an exchange order has no row in the
    trades table (the process died between submit_order() success and the
    record_trade() write), insert one from the exchange's own order data.
    Returns True if a row was inserted, False otherwise. Never raises.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return False
    order_id = str(getattr(order, "id", "") or "")
    if not order_id:
        return False
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM trades WHERE order_id = %s LIMIT 1",
                            (order_id,))
                if cur.fetchone() is not None:
                    return False  # already recorded — nothing to back-fill

                qty = float(getattr(order, "filled_qty", None)
                            or getattr(order, "qty", 0) or 0)
                fill_price = getattr(order, "filled_avg_price", None)
                fill_price = float(fill_price) if fill_price else None
                # NOTE: alpaca-py 0.33.0 Order model has no 'commission' field;
                # fees are not available from order objects.
                fee = float(getattr(order, "commission", None) or 0.0)
                side = str(getattr(order, "side", "") or "")
                symbol = str(getattr(order, "symbol", "") or "")
                created = getattr(order, "created_at", None)
                value = (fill_price or 0.0) * qty
                cur.execute("""
                    INSERT INTO trades
                        (bot_name, exchange, symbol, side, price, quantity,
                         value, fee, fill_price, order_id, timestamp)
                    VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
                """, (bot_name, symbol, side, fill_price or 0.0, qty, value,
                      fee, fill_price, order_id, created))
            conn.commit()
        logger.info(f"🔁 Back-filled missing trade row from exchange order {order_id} "
                    f"({side} {symbol} qty={qty})")
        return True
    except Exception as e:
        logger.error(f"backfill_trade_if_missing failed for order {order_id}: {e}")
        return False


def get_realized_pnl_summary(bot_name):
    """
    Returns {'total_realized_pnl': float, 'total_fees': float, 'closed_trades': int}
    summed across all SELL trades with a recorded realized_pnl for this bot,
    or None if DATABASE_URL is not set or the query fails.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.debug("DATABASE_URL not set - skipping realized PnL summary")
        return None
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(realized_pnl), 0), COALESCE(SUM(fee), 0), COUNT(*)
                    FROM trades
                    WHERE bot_name = %s AND side = 'sell' AND realized_pnl IS NOT NULL
                """, (bot_name,))
                total_pnl, total_fees, closed_trades = cur.fetchone()
        return {
            "total_realized_pnl": float(total_pnl),
            "total_fees": float(total_fees),
            "closed_trades": int(closed_trades),
        }
    except Exception as e:
        logger.error(f"DB Error fetching realized PnL summary: {e}")
        return None


def report_equity(bot_name, equity):
    """Log current equity to the equity_history table."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.debug("DATABASE_URL not set — skipping equity DB log")
        return False
    try:
        from decimal import Decimal
        eq_val = float(Decimal(str(equity)))
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO equity_history (bot_name, equity, timestamp)
                    VALUES (%s, %s, NOW())
                """, (bot_name, eq_val))
            conn.commit()
        logger.debug(f"Equity reported: ${eq_val:,.2f}")
        return True
    except Exception as e:
        logger.error(f"Equity reporting failed: {e}")
        return False
