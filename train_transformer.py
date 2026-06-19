#!/usr/bin/env python3

import asyncio
import logging
import os
import time
import math
import psycopg2
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from ml_predictor import GrokGQA_Transformer, FEATURE_COLS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("bot_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========================= CONFIG =========================
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v9_Final")
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

ACCOUNT_BASE          = float(os.getenv("ACCOUNT_BASE", 10000))
BASE_RISK_PERCENT     = 0.006
MAX_SINGLE_TRADE_USD  = 120.0
MAX_DRAWDOWN_STOP     = -10.0

MAX_PORTFOLIO_VALUE   = 190.0
MAX_OPEN_POSITIONS    = 2
MAX_HOLD_HOURS        = 4.0
PROFIT_TARGET_PCT     = 0.025
STOP_LOSS_PCT         = 0.03
BUY_SIGNAL            = 0.62
SELL_SIGNAL           = 0.52

SEQUENCE_LEN = 32
MODEL_PATH = "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}
entry_time     = {}
start_equity   = None

# ========================= UTILS =========================

def record_trade(bot_name, symbol, side, qty, price, pnl_pct=None, order_id=None):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("No DATABASE_URL found. Skipping DB log.")
        return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                value = (price or 0.0) * float(qty)
                cur.execute("""
                    INSERT INTO trades
                    (bot_name, exchange, symbol, side, price, quantity, value, fee, order_id, pnl_pct, timestamp)
                    VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, 0, %s, %s, NOW())
                """, (bot_name, symbol, side, float(price or 0), float(qty), value, str(order_id), pnl_pct))
                conn.commit()
    except Exception as e:
        logger.error(f"DB Insert Error: {e}")

def normalize_symbol(symbol):
    return symbol.replace("-", "/")

def get_buying_power():
    try:
        acc = trading_client.get_account()
        return float(acc.non_marginable_buying_power)
    except Exception as e:
        logger.error(f"Error fetching buying power: {e}")
        return 0.0

# ========================= FEATURE ENG =========================

def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns'] = df['close'].pct_change().fillna(0)
    df['vol_14']  = df['returns'].rolling(14).std().fillna(0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean().fillna(0)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)

    return df[FEATURE_COLS]

# ========================= MODEL WRAPPER =========================

class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        try:
            self.model = GrokGQA_Transformer(
                input_dim=len(FEATURE_COLS), seq_len=seq_len,
                embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1
            ).to(self.device)
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()

            scaler_path = os.path.join(os.path.dirname(model_path), "feature_scaler.pkl")
            self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            logger.info(f"✅ Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ Model Init Error: {e}")
            raise

    def predict(self, df):
        try:
            feat_df = safe_add_features(df)
            data = feat_df.tail(self.seq_len).values.astype(np.float32)
            if len(data) < self.seq_len:
                return 0.5
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            if self.scaler:
                data = self.scaler.transform(data)
            x = torch.tensor(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return float(torch.sigmoid(self.model(x)).item())
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5

predictor = SafeMLPredictor(MODEL_PATH, SEQUENCE_LEN)

# ========================= CORE LOGIC =========================

async def get_clean_ohlcv(symbol):
    try:
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=200)
        bars = data_client.get_crypto_bars(req).data.get(symbol, [])
        if not bars:
            return None
        df = pd.DataFrame([{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close, "volume": b.volume} for b in bars])
        df.set_index("timestamp", inplace=True)
        df = df.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                                      "close": "last", "volume": "sum"}).dropna()
        return df
    except Exception as e:
        logger.error(f"Data Fetch Error {symbol}: {e}")
        return None

# ---------- FLOOR FUNCTION AND UPDATED place_order_safe ----------
def floor_to(value: float, decimals: int) -> float:
    q = Decimal(str(value)).quantize(Decimal('1e-{}'.format(decimals)), rounding=ROUND_DOWN)
    return float(q)

async def place_order_safe(symbol, side, qty, price):
    try:
        clean_qty = float(qty)
        if clean_qty <= 0:
            return False

        if side == OrderSide.SELL:
            try:
                positions = trading_client.get_all_positions()
                alpaca_sym = symbol.replace("/", "")
                available_qty = 0.0
                for p in positions:
                    if p.symbol == alpaca_sym:
                        available_qty = float(p.qty)
                        break
                max_safe_qty = floor_to(available_qty, 6)
                max_safe_qty = max(0.0, max_safe_qty - 1e-7)
                clean_qty = min(clean_qty, max_safe_qty)
                if clean_qty <= 0:
                    logger.error(f"Sell skipped for {symbol}: dust-only position ({available_qty})")
                    return False
            except Exception as e:
                logger.error(f"Dust‑safe SELL check failed ({symbol}): {e}", exc_info=True)
                clean_qty = floor_to(clean_qty * 0.999, 6)
                if clean_qty <= 0:
                    return False
        else:
            clean_qty = floor_to(clean_qty, 6)
            if clean_qty <= 0:
                return False

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=clean_qty,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        order = trading_client.submit_order(order_data)
        record_trade(BOT_NAME, symbol, side.value, clean_qty, price, order_id=order.id)
        return True
    except Exception as e:
        logger.error(f"Order Execution Failed ({symbol} {side}): {e}", exc_info=True)
        return False

# ===================== MAIN TRADING LOOP =====================

async def run_trading_mode():
    global start_equity
    logger.info("🔥 Starting Grok Apex Ironclad...")

    try:
        acc = trading_client.get_account()
        start_equity = float(acc.equity)
    except:
        start_equity = ACCOUNT_BASE

    while True:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
            drawdown = (equity - start_equity) / start_equity * 100

            if drawdown < MAX_DRAWDOWN_STOP:
                logger.critical(f"🛑 KILLSWITCH: Drawdown {drawdown:.2f}%")
                break

            positions = {p.symbol: p for p in trading_client.get_all_positions()}
            open_count = len(positions)
            total_val = sum(float(p.market_value) for p in positions.values())
            buying_power = get_buying_power()

            logger.info(f"Heartbeat | Equity: ${equity:.2f} | Pos: {open_count} | Value: ${total_val:.2f}")

            # ===========================================================
            # DUST CLEANUP – close positions below $1.00 market value
            # ===========================================================
            DUST_VALUE_THRESHOLD = 1.00   # adjust as needed

            for symbol in SYMBOLS:
                alpaca_sym = symbol.replace("/", "")
                if alpaca_sym in positions:
                    pos = positions[alpaca_sym]
                    market_value = float(pos.market_value)
                    if market_value < DUST_VALUE_THRESHOLD:
                        try:
                            trading_client.close_position(alpaca_sym)
                            logger.info(f"🧹 Closed dust position: {alpaca_sym} (${market_value:.4f})")
                            # Remove from dict so main logic won't see it
                            del positions[alpaca_sym]
                            # Update total_val and open_count (recalc after loop)
                        except Exception as e:
                            logger.warning(f"Could not close dust position {alpaca_sym}: {e}")
            # Recalculate open_count and total_val after dust removal
            open_count = len(positions)
            total_val = sum(float(p.market_value) for p in positions.values())
            # ===========================================================

            # 2. Iterate Symbols (main trading logic)
            for symbol in SYMBOLS:
                alpaca_sym = symbol.replace("/", "")
                df = await get_clean_ohlcv(symbol)
                if df is None or len(df) < SEQUENCE_LEN:
                    continue

                price = df['close'].iloc[-1]
                signal = predictor.predict(df)

                if alpaca_sym in positions:
                    pos = positions[alpaca_sym]
                    qty = float(pos.qty)
                    avg_entry = float(pos.avg_entry_price)
                    pnl = (price - avg_entry) / avg_entry

                    if symbol not in entry_time:
                        entry_time[symbol] = time.time()

                    held_hrs = (time.time() - entry_time[symbol]) / 3600

                    exit_triggered = False
                    reason = ""
                    if pnl >= PROFIT_TARGET_PCT:
                        reason = "Target Hit"
                        exit_triggered = True
                    elif pnl <= -STOP_LOSS_PCT:
                        reason = "Stop Loss"
                        exit_triggered = True
                    elif held_hrs >= MAX_HOLD_HOURS:
                        reason = "Time Limit"
                        exit_triggered = True
                    elif signal < SELL_SIGNAL:
                        reason = "Weak Signal"
                        exit_triggered = True

                    if exit_triggered:
                        logger.info(f"🔴 SELLING {symbol} | {reason} | PnL: {pnl*100:.2f}%")
                        if await place_order_safe(symbol, OrderSide.SELL, qty, price):
                            entry_time.pop(symbol, None)
                            cooldown_until[symbol] = time.time() + 1800

                else:
                    if time.time() < cooldown_until.get(symbol, 0):
                        continue

                    if signal > BUY_SIGNAL and open_count < MAX_OPEN_POSITIONS:
                        if total_val + MAX_SINGLE_TRADE_USD <= MAX_PORTFOLIO_VALUE:
                            risk_amt = min(equity * BASE_RISK_PERCENT, MAX_SINGLE_TRADE_USD)
                            if buying_power > risk_amt:
                                qty_to_buy = risk_amt / price
                                logger.info(f"🟢 BUYING {symbol} | Signal: {signal:.3f} | Amt: ${risk_amt:.2f}")
                                if await place_order_safe(symbol, OrderSide.BUY, qty_to_buy, price):
                                    entry_time[symbol] = time.time()
                                    open_count += 1
                                    total_val += risk_amt

                await asyncio.sleep(1)

            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_trading_mode())







