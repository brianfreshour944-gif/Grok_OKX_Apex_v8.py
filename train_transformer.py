#!/usr/bin/env python3
import asyncio
import logging
import os
import time
import psycopg2
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from ml_predictor import GrokGQA_Transformer, FEATURE_COLS

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION ---
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v8")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
ORDER_AMOUNT = 50.0 
# Caps removed: Allowing the bot to use your full available account balance
MAX_SINGLE_TRADE_USD = 1000.0 

MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SEQUENCE_LEN = 32

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()
cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}

# [Feature Engineering classes remain unchanged - safe to keep your existing definitions]
def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Keep your existing implementation)
    return df[FEATURE_COLS]

class SafeMLPredictor:
    # ... (Keep your existing implementation)
    pass

def execute_trade(bot_name, symbol, side, qty):
    """Submit order to Alpaca using IOC to ensure immediate execution."""
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol, 
                qty=qty, 
                side=side, 
                time_in_force=TimeInForce.IOC # Changed to IOC for immediate fill/cancel
            )
        )
        logger.info(f"✅ Placed {side.value} order for {symbol} | Order ID: {order.id}")

        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (bot_name, symbol, side, price, quantity, timestamp)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (bot_name, symbol, side.value, float(order.filled_avg_price or 0.0), qty))
        conn.commit()
        cur.close()
        conn.close()
        return order
    except Exception as e:
        logger.error(f"Trade execution failed for {symbol}: {e}")
        return None

async def run_trading_mode(bot_name):
    global cooldown_until
    predictor = SafeMLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("Starting trading loop with full account access...")

    while True:
        try:
            # STEP 1: Fetch REAL-TIME Buying Power from Alpaca
            account = trading_client.get_account()
            actual_buying_power = float(account.buying_power)
            
            for symbol in SYMBOLS:
                now = time.time()
                if now < cooldown_until.get(symbol, 0.0):
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None: continue

                signal = predictor.predict(df)
                current_price = df['close'].iloc[-1]

                # Check position
                has_position = False
                qty_held = 0.0
                try:
                    pos_symbol = symbol.replace("/", "")
                    position = trading_client.get_position(pos_symbol)
                    has_position = True
                    qty_held = float(position.qty)
                except:
                    has_position = False

                # SELL logic
                if has_position and signal < 0.61:
                    logger.info(f"🔻 SELL {symbol}")
                    if execute_trade(bot_name, symbol, OrderSide.SELL, qty_held):
                        cooldown_until[symbol] = now + 3600
                    continue

                # BUY logic
                if not has_position and signal > 0.63:
                    if actual_buying_power < ORDER_AMOUNT:
                        logger.warning(f"🚫 BUY suppressed: Only ${actual_buying_power:.2f} available.")
                        continue

                    qty = ORDER_AMOUNT / current_price
                    logger.info(f"🎯 BUY signal for {symbol}")
                    
                    if execute_trade(bot_name, symbol, OrderSide.BUY, qty):
                        cooldown_until[symbol] = now + 600
                        # Refresh buying power after trade
                        actual_buying_power -= ORDER_AMOUNT

                await asyncio.sleep(2)
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_trading_mode(BOT_NAME))
