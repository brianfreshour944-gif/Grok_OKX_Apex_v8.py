# database.py — PostgreSQL trade and equity logging.
# Silently skips all DB calls when DATABASE_URL is not set.
# Uses context managers to prevent connection leaks on exceptions.

import os
import psycopg2
from config import logger, BOT_NAME


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
            conn.commit()
        logger.info("📘 DB: tables initialised")
    except Exception as e:
        logger.warning(f"⚠️ DB init failed (non-fatal): {e}")


def record_trade(bot_name, symbol, side, qty, price, pnl_pct=None, order_id=None):
    """Log a completed trade to the trades table."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.debug("DATABASE_URL not set — skipping trade DB log")
        return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                value = (price or 0.0) * qty
                cur.execute("""
                    INSERT INTO trades
                        (bot_name, exchange, symbol, side, price, quantity,
                         value, fee, order_id, timestamp)
                    VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, 0, %s, NOW())
                """, (bot_name, symbol, side, price or 0.0, qty, value,
                      str(order_id) if order_id else None))
            conn.commit()
        logger.info(f"📘 DB: {side} {symbol} | Qty: {qty:.6f} | Price: {price}")
    except Exception as e:
        logger.error(f"DB Error: {e}")


def report_equity(bot_name, equity):
    """Log current equity to the equity_history table."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.debug("DATABASE_URL not set — skipping equity DB log")
        return False
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO equity_history (bot_name, equity, timestamp)
                    VALUES (%s, %s, NOW())
                """, (bot_name, float(equity)))
            conn.commit()
        logger.debug(f"📊 Equity reported: ${equity:,.2f}")
        return True
    except Exception as e:
        logger.error(f"Equity reporting failed: {e}")
        return False
