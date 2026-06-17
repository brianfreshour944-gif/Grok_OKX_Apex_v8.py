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
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]

ACCOUNT_BASE = float(os.getenv("ACCOUNT_BASE", 10000))
BASE_RISK_PERCENT = 0.008
MIN_CONFIDENCE = 58
MAX_SINGLE_TRADE_USD = 150
COOLDOWN_BUY = 600
COOLDOWN_SELL = 1800

SEQUENCE_LEN = 32
MODEL_PATH = "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}
trade_history = []

# ========================= POSTGRESQL =========================
def record_trade(bot_name, symbol, side, qty, price, pnl_pct=None):
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        # Updated to avoid missing column error
        cur.execute("""
            INSERT INTO trades (bot_name, symbol, side, price, quantity, pnl_pct, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (bot_name, symbol, side, price, qty, pnl_pct))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"📘 DB Log: {side} {symbol} | Qty: {qty}")
    except Exception as e:
        logger.error(f"PostgreSQL error: {e}")

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

predictor = SafeMLPredictor(MODEL_PATH, SEQUENCE_LEN)

# ========================= FEATURES + REGIME =========================
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
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean().fillna(0.0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df[FEATURE_COLS]

def compute_regime_and_trend(df: pd.DataFrame):
    try:
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        price = df['close'].iloc[-1]
        atr_pct = (atr / price) * 100
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        trend = "up" if price > ema50 else "down"
        regime = "wild" if atr_pct > 4 else "normal" if atr_pct > 2 else "quiet"
        return regime, trend, round(atr_pct, 2)
    except:
        return "normal", "neutral", 2.0

# ========================= MAIN LOOP =========================
async def run_trading_mode():
    logger.info("🚀 Grok Apex Ironclad Bot v8 - Full Fixed Version Started")
    while True:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)

            for symbol in SYMBOLS:
                now = time.time()
                if now < cooldown_until.get(symbol, 0):
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue

                regime, trend, atr_pct = compute_regime_and_trend(df)
                signal = predictor.predict(df)
                price = df["close"].iloc[-1]

                try:
                    pos = trading_client.get_position(symbol.replace("/", ""))
                    has_position = float(pos.qty) > 0
                    qty_held = float(pos.qty)
                except:
                    has_position = False
                    qty_held = 0

                if has_position and signal < 0.58:
                    logger.info(f"🔻 SELL {symbol} @ {price:.2f}")
                    await place_order(symbol, OrderSide.SELL, qty_held)
                    cooldown_until[symbol] = now + COOLDOWN_SELL

                elif not has_position and signal > (MIN_CONFIDENCE / 100.0):
                    risk_usd = equity * BASE_RISK_PERCENT
                    qty = risk_usd / price
                    if qty * price > MAX_SINGLE_TRADE_USD:
                        qty = MAX_SINGLE_TRADE_USD / price

                    logger.info(f"🟢 BUY {symbol} @ {price:.2f} | Regime: {regime} | Signal: {signal:.3f}")
                    await place_order(symbol, OrderSide.BUY, qty)
                    cooldown_until[symbol] = now + COOLDOWN_BUY

                await asyncio.sleep(1.5)
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)

async def place_order(symbol, side, qty):
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
        )
        # Fixed: record_trade is sync, so no 'await'
        record_trade(BOT_NAME, symbol, side.value, qty, None)
        logger.info(f"✅ Order submitted: {side} {symbol} {qty}")
        return True
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return False

async def get_clean_ohlcv_dataframe(symbol):
    try:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=600
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
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).fillna(0)
        return df.tail(SEQUENCE_LEN)
    except Exception as e:
        logger.error(f"Data error {symbol}: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(run_trading_mode())





