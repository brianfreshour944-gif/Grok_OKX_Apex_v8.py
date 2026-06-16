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

# Only import the model architecture and constants, NOT the predictor
from ml_predictor import GrokGQA_Transformer, FEATURE_COLS

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION ---
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v8")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
ORDER_AMOUNT = 50.0
MAX_SINGLE_TRADE_USD = 100.0  # Safety cap per order
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SEQUENCE_LEN = 32

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()
cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}

# ---------- SAFE FEATURE ENGINEERING ----------
def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all features with aggressive None -> 0 conversion."""
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df = df.copy()

    df['returns'] = df['close'].pct_change().fillna(0.0)
    df['vol_14'] = df['returns'].rolling(window=14).std().fillna(0.0)

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi.fillna(50.0).replace([np.inf, -np.inf], 50.0)

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd'] = macd_line - signal_line
    df['macd'] = df['macd'].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0.0)

    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    df['bb_width'] = (upper - lower) / sma
    df['bb_width'] = df['bb_width'].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            df[col] = df[col].replace([np.inf, -np.inf], 0.0)

    return df[FEATURE_COLS]

# ---------- LOCAL SAFE PREDICTOR ----------
class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.input_dim = len(FEATURE_COLS)
        self.model = GrokGQA_Transformer(
            input_dim=self.input_dim, seq_len=seq_len,
            embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        logger.info(f"✅ Model weights loaded from {model_path}")
        scaler_path = os.path.join(os.path.dirname(model_path), 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
            logger.warning("No scaler found; predictions will be unnormalized")

    def predict(self, df: pd.DataFrame) -> float:
        try:
            df = df.copy()
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            df = df.map(lambda x: 0.0 if x is None else x)

            df_features = safe_add_features(df)
            data = df_features[FEATURE_COLS].tail(self.seq_len).values.astype(np.float32)
            if len(data) < self.seq_len:
                logger.warning(f"Insufficient rows after feature engineering: {len(data)}")
                return 0.5

            if self.scaler is not None:
                data = self.scaler.transform(data).astype(np.float32)

            x = torch.tensor(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(x).item()
            return float(pred)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5

# ---------- HELPER FUNCTIONS ----------
def get_total_portfolio_value():
    """Returns the total dollar value of all crypto positions."""
    try:
        positions = trading_client.get_all_positions()
        return sum(float(p.market_value) for p in positions)
    except Exception as e:
        logger.error(f"Error fetching portfolio value: {e}")
        return None

def sell_largest_position():
    """
    Finds and sells the largest position by market value.
    Returns True if a sell was placed, False otherwise.
    """
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            return False
        largest = max(positions, key=lambda p: float(p.market_value))
        symbol_to_sell = largest.symbol
        qty_to_sell = float(largest.qty)
        logger.warning(
            f"📉 Selling largest position: "
            f"{symbol_to_sell} qty={qty_to_sell:.6f} (${float(largest.market_value):.2f})"
        )
        execute_trade(BOT_NAME, symbol_to_sell, OrderSide.SELL, qty_to_sell)
        return True
    except Exception as e:
        logger.error(f"Error in sell_largest_position: {e}")
        return False

def execute_trade(bot_name, symbol, side, qty):
    """Submit order to Alpaca and record trade in database."""
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC
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

async def get_clean_ohlcv_dataframe(symbol):
    """Fetch 5-min OHLCV data for the last 6 hours."""
    end = datetime.now()
    start = end - timedelta(hours=6)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=500
    )
    bars = data_client.get_crypto_bars(request).data.get(symbol, [])
    if len(bars) < SEQUENCE_LEN:
        logger.warning(f"Insufficient minute bars for {symbol}: {len(bars)}")
        return None

    data = []
    for b in bars:
        data.append({
            'timestamp': b.timestamp,
            'open':   float(b.open)   if b.open   is not None else 0.0,
            'high':   float(b.high)   if b.high   is not None else 0.0,
            'low':    float(b.low)    if b.low    is not None else 0.0,
            'close':  float(b.close)  if b.close  is not None else 0.0,
            'volume': float(b.volume) if b.volume is not None else 0.0,
        })
    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ohlc_5 = df.resample('5min').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    })

    for col in ['open', 'high', 'low', 'close', 'volume']:
        ohlc_5[col] = pd.to_numeric(ohlc_5[col], errors='coerce').fillna(0.0)
        ohlc_5[col] = ohlc_5[col].replace([np.inf, -np.inf], 0.0)

    if len(ohlc_5) < SEQUENCE_LEN:
        logger.warning(f"Not enough 5-min bars for {symbol}: {len(ohlc_5)}")
        return None

    return ohlc_5.iloc[-SEQUENCE_LEN:].astype(float)

# --- MAIN TRADING LOOP ---
async def run_trading_mode(bot_name):
    global cooldown_until
    predictor = SafeMLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("SafeMLPredictor loaded. Starting trading loop...")

    while True:
        try:
            total_value = get_total_portfolio_value()
            if total_value is None:
                logger.warning("Could not fetch portfolio value — skipping entire cycle.")
                await asyncio.sleep(30)
                continue

            for symbol in SYMBOLS:
                now = time.time()

                if now < cooldown_until.get(symbol, 0.0):
                    remaining = cooldown_until[symbol] - now
                    logger.debug(f"⏳ {symbol} in cooldown ({remaining:.0f}s remaining)")
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue

                signal = predictor.predict(df)
                current_price = df['close'].iloc[-1]

                has_position = False
                qty_held = 0.0
                try:
                    pos_symbol = symbol.replace("/", "")
                    position = trading_client.get_position(pos_symbol)
                    if float(position.qty) > 0:
                        has_position = True
                        qty_held = float(position.qty)
                except Exception:
                    has_position = False

                # SELL logic
                if has_position and signal < 0.61:
                    logger.info(f"🔻 SELL signal for {symbol} at {current_price:.2f} (signal={signal:.3f})")
                    if execute_trade(bot_name, symbol, OrderSide.SELL, qty_held):
                        cooldown_until[symbol] = now + 3600
                    continue

                # BUY logic
                if not has_position and signal > 0.63:
                    qty = ORDER_AMOUNT / current_price
                    trade_value = qty * current_price

                    if trade_value > MAX_SINGLE_TRADE_USD:
                        logger.error(
                            f"❌ Trade too large (${trade_value:.2f} > ${MAX_SINGLE_TRADE_USD:.2f}). "
                            f"Aborting BUY for {symbol}"
                        )
                        continue

                    # Use Alpaca buying power instead of fixed portfolio cap
                    try:
                        account = trading_client.get_account()
                        buying_power = float(account.buying_power)
                    except Exception as e:
                        logger.error(f"Error fetching account buying power: {e}")
                        continue

                    if trade_value > buying_power:
                        logger.warning(
                            f"🚫 BUY blocked for {symbol}: trade value ${trade_value:.2f} "
                            f"exceeds Alpaca buying power ${buying_power:.2f}"
                        )
                        continue

                    logger.info(f"🎯 BUY signal for {symbol} at {current_price:.2f} (signal={signal:.3f})")
                    if execute_trade(bot_name, symbol, OrderSide.BUY, qty):
                        cooldown_until[symbol] = now + 600

                await asyncio.sleep(2)

            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_trading_mode(BOT_NAME))




