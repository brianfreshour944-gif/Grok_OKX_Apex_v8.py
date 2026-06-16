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

# ========================= CONFIG =========================
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v8")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]

ORDER_AMOUNT = 50.0
MAX_SINGLE_TRADE_USD = 100.0
SEQUENCE_LEN = 32

MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()


cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}
pending_orders = {}  # <-- NEW: prevents double buys/sells

# ========================= FEATURES =========================
def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df = df.copy()
    df['returns'] = df['close'].pct_change().fillna(0.0)
    df['vol_14'] = df['returns'].rolling(14).std().fillna(0.0)

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9).mean()
    df['macd'] = (macd_line - signal_line).fillna(0.0)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean().fillna(0.0)

    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_width'] = ((sma + 2*std) - (sma - 2*std)) / sma
    df['bb_width'] = df['bb_width'].fillna(0.0)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df[FEATURE_COLS]

# ========================= MODEL =========================
class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        self.model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS),
            seq_len=seq_len,
            embed_dim=128,
            num_layers=8,
            num_q_heads=16,
            num_kv_heads=4,
            dropout=0.1
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        scaler_path = os.path.join(os.path.dirname(model_path), "feature_scaler.pkl")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    def predict(self, df):
        df = safe_add_features(df)
        data = df.tail(self.seq_len).values.astype(np.float32)

        if len(data) < self.seq_len:
            return 0.5

        if self.scaler:
            data = self.scaler.transform(data)

        x = torch.tensor(data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(x).item())

# ========================= DB LOGGING =========================
async def record_trade(bot_name, symbol, side, qty, order_id):
    """Wait for fill, then log to PostgreSQL."""
    try:
        # Wait for fill
        for _ in range(20):
            order = trading_client.get_order_by_id(order_id)
            if order.filled_avg_price:
                break
            await asyncio.sleep(1)

        price = float(order.filled_avg_price or 0.0)

        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (bot_name, symbol, side, price, quantity, order_id, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (
            bot_name,
            symbol,
            side,
            price,
            qty,
            str(order_id)   # <-- FIXED HERE
        ))
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"📘 Logged trade: {side} {symbol} @ {price}")

    except Exception as e:
        logger.error(f"DB logging failed: {e}")






# ========================= ORDER EXECUTION =========================
async def place_order(bot_name, symbol, side, qty):
    """Submit order and track it safely."""
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
        )

        pending_orders[symbol] = order.id
        logger.info(f"🟦 Submitted {side} order for {symbol} | ID: {order.id}")

        await record_trade(bot_name, symbol, side.value, qty, order.id)

        # Clear pending state
        pending_orders.pop(symbol, None)

        return True

    except Exception as e:
        logger.error(f"Order failed: {e}")
        return False

# ========================= OHLCV =========================
async def get_clean_ohlcv_dataframe(symbol):
    end = datetime.now()
    start = end - timedelta(hours=6)

    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=500
    )

    bars = data_client.get_crypto_bars(req).data.get(symbol, [])
    if len(bars) < SEQUENCE_LEN:
        return None

    df = pd.DataFrame([{
        "timestamp": b.timestamp,
        "open": float(b.open or 0),
        "high": float(b.high or 0),
        "low": float(b.low or 0),
        "close": float(b.close or 0),
        "volume": float(b.volume or 0)
    } for b in bars])

    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_localize(None)

    df = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).fillna(0)

    return df.tail(SEQUENCE_LEN)

# ========================= MAIN LOOP =========================
async def run_trading_mode(bot_name):
    predictor = SafeMLPredictor(MODEL_PATH, SEQUENCE_LEN)
    logger.info("🚀 Bot started")

    while True:
        try:
            for symbol in SYMBOLS:
                now = time.time()

                # Skip if order pending
                if symbol in pending_orders:
                    logger.info(f"⏳ Pending order for {symbol}, skipping")
                    continue

                # Cooldown
                if now < cooldown_until[symbol]:
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue

                signal = predictor.predict(df)
                price = df["close"].iloc[-1]

                # Detect position
                pos_symbol = symbol.replace("/", "")
                try:
                    pos = trading_client.get_position(pos_symbol)
                    qty_held = float(pos.qty)
                    has_position = qty_held > 0
                except:
                    has_position = False
                    qty_held = 0

                # SELL
                if has_position and signal < 0.61:
                    logger.info(f"🔻 SELL {symbol} @ {price} (signal={signal:.3f})")
                    await place_order(bot_name, symbol, OrderSide.SELL, qty_held)
                    cooldown_until[symbol] = now + 3600
                    continue

                # BUY
                if not has_position and signal > 0.63:
                    qty = ORDER_AMOUNT / price
                    trade_value = qty * price

                    if trade_value > MAX_SINGLE_TRADE_USD:
                        continue

                    # Alpaca buying power
                    account = trading_client.get_account()
                    buying_power = float(account.buying_power)

                    if trade_value > buying_power:
                        continue

                    logger.info(f"🟢 BUY {symbol} @ {price} (signal={signal:.3f})")
                    await place_order(bot_name, symbol, OrderSide.BUY, qty)
                    cooldown_until[symbol] = now + 600

                await asyncio.sleep(2)

            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_trading_mode(BOT_NAME))





