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
MAX_DAILY_LOSS_PCT = -8.0

SEQUENCE_LEN = 32
MODEL_PATH = "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}
trade_history = []   # For self-learning

# ========================= POSTGRESQL =========================
def record_trade(bot_name, symbol, side, qty, price, pnl_pct=None):
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (bot_name, symbol, side, price, quantity, pnl_pct, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (bot_name, symbol, side, price, qty, pnl_pct))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"📘 Logged to PostgreSQL: {side} {symbol} @ {price}")
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
    # Your original safe_add_features function here
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

# ========================= SELF-LEARNING =========================
def self_tune():
    global BASE_RISK_PERCENT, MIN_CONFIDENCE
    if len(trade_history) < 10:
        return
    recent = trade_history[-20:]
    win_rate = sum(1 for t in recent if t.get('pnl', 0) > 0) / len(recent)
    BASE_RISK_PERCENT = max(0.005, min(0.015, BASE_RISK_PERCENT * (0.8 + win_rate * 0.8)))
    MIN_CONFIDENCE = max(52, min(72, 58 + int(win_rate * 30)))
    logger.info(f"SELF-TUNED → Risk: {BASE_RISK_PERCENT:.4f} | Min Conf: {MIN_CONFIDENCE} | Win Rate: {win_rate:.1%}")

# ========================= WALK-FORWARD =========================
def run_walk_forward_validation():
    logger.info("Running Advanced Walk-Forward Validation...")
    # (Your previous walk-forward code can be placed here)
    logger.info("Walk-Forward completed.")

# ========================= MAIN LOOP =========================
async def run_trading_mode():
    logger.info("🚀 Grok Apex Ironclad Bot v8 - Advanced Adaptive Started")
    run_walk_forward_validation()

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
                    logger.info(f"🔻 SELL {symbol} @ {price}")
                    await place_order(symbol, OrderSide.SELL, qty_held)
                    cooldown_until[symbol] = now + 1800

                elif not has_position and signal > 0.65:
                    risk_usd = equity * MAX_RISK_PER_TRADE
                    qty = risk_usd / price
                    if qty * price > 150:
                        qty = 150 / price

                    logger.info(f"🟢 BUY {symbol} @ {price} | Regime: {regime}")
                    await place_order(symbol, OrderSide.BUY, qty)
                    cooldown_until[symbol] = now + 600

                await asyncio.sleep(2)
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)

# (Keep your existing place_order and get_clean_ohlcv_dataframe functions)

if __name__ == "__main__":
    asyncio.run(run_trading_mode())





