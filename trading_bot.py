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
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from ml_predictor import GrokGQA_Transformer, FEATURE_COLS

load_dotenv()

# --- FIX: log level is now configurable via env var. Set LOG_LEVEL=DEBUG in
# your .env to see the per-symbol rejection reasons that were previously
# silently swallowed (this was the root cause of the bot looking "idle" --
# it *was* evaluating every asset every cycle, it just never told you why
# each one failed the entry signal check). ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= CONFIG =========================
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v9_CuttingEdge")
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

ACCOUNT_BASE          = float(os.getenv("ACCOUNT_BASE", 10000))
BASE_RISK_PERCENT     = 0.006
MAX_SINGLE_TRADE_USD  = 120
MAX_DRAWDOWN_STOP     = -10.0

# --- NEW: position / risk management ---
MAX_PORTFOLIO_VALUE   = 190.0   # Hard cap on total $ held across all symbols
MAX_OPEN_POSITIONS    = 2       # Don't hold more than this many symbols at once
MAX_HOLD_HOURS        = 4.0     # Force-sell after this long regardless of signal

# --- FIX: BUY_SIGNAL / SELL_SIGNAL are now overridable via env vars so you
# can tune the entry threshold without redeploying code. The bot appeared
# permanently "idle" (Positions: 0/10, Value: $0.00) because BUY_SIGNAL=0.62
# is a high bar for the model's raw sigmoid output -- if the model's
# predictions rarely/never cross that threshold, NO asset will ever qualify
# for entry, no matter how many are scanned each cycle. Lower BUY_SIGNAL via
# env var (e.g. 0.55) to confirm/fix, and pair with LOG_LEVEL=DEBUG below to
# see each symbol's actual signal value every cycle. ---
PROFIT_TARGET_PCT     = 0.02    # Sell if up 2%
STOP_LOSS_PCT         = 0.03    # Sell if down 3%
# --- TEMP DIAGNOSTIC: lowered from 0.62 to 0.51 (just above neutral coin-flip)
# to prove the model is alive and producing a valid statistical edge. Once you
# confirm trades fire and see real signal values in the logs, raise this back
# up gradually (0.55, 0.58, ...) to find the confidence/frequency sweet spot. ---
BUY_SIGNAL            = float(os.getenv("BUY_SIGNAL", 0.51))
SELL_SIGNAL           = float(os.getenv("SELL_SIGNAL", 0.45))  # FIX: widened gap from BUY_SIGNAL (was 0.55) to reduce
                                 # exits on model noise right around the 0.5 midpoint


# --- NEW: fixes for dust-order bug and signal-exit whipsaw (see log analysis) ---
MIN_POSITION_USD              = 5.0   # ignore/never re-sell positions worth less than this
MIN_HOLD_HOURS_BEFORE_SIGNAL  = 0.5   # don't let a weak-signal reading exit a position
                                       # until it's been held at least this long

SEQUENCE_LEN = 32
MODEL_PATH = "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}
entry_time     = {}   # symbol -> timestamp, tracks how long a position has been held
latest_signals = {}
highest_prices = {}   # symbol -> highest price seen while holding
start_equity   = None

# --- Track sell retry attempts to prevent spam ---
sell_retry_cooldown = {}


# ========================= SAFE POSTGRESQL =========================
def record_trade(bot_name, symbol, side, qty, price, pnl_pct=None, order_id=None):
    """
    Logs a trade to the trades table.
    FIXED:
    - Removed pnl_pct from INSERT (column doesn't exist in schema)
    - Now includes exchange, value, fee, order_id to match table schema
    - Avoids silent insert failures. price is always passed in (was None before).
    """
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        value = (price or 0.0) * qty
        cur.execute("""
            INSERT INTO trades
                (bot_name, exchange, symbol, side, price, quantity,
                 value, fee, order_id, timestamp)
            VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, 0, %s, NOW())
        """, (bot_name, symbol, side, price or 0.0, qty, value,
              str(order_id) if order_id else None))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"📘 DB: {side} {symbol} | Qty: {qty:.6f} | Price: {price}")
    except Exception as e:
        logger.error(f"DB Error: {e}")

# ========================= EQUITY REPORTING =========================
def report_equity(bot_name, equity):
    """
    FIXED: Explicitly report current equity to the database.
    This was the missing piece causing equity not to land in the database.
    Creates equity_history table if it doesn't exist and logs equity values.
    """
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()

        # Create equity_history table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id SERIAL PRIMARY KEY,
                bot_name TEXT NOT NULL,
                equity NUMERIC NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert equity report
        cur.execute("""
            INSERT INTO equity_history (bot_name, equity, timestamp)
            VALUES (%s, %s, NOW())
        """, (bot_name, float(equity)))

        conn.commit()
        cur.close()
        conn.close()
        logger.debug(f"📊 Equity reported: ${equity:,.2f}")
        return True
    except Exception as e:
        logger.error(f"Equity reporting failed: {e}")
        return False


# ========================= FEATURES =========================
# Centralized microstructure feature engineering (lagging indicators removed).
# Delegates to feature_engineering.add_features so the bot and the model always
# share the exact same feature math. Guarantees vwap / trade_count exist.
def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    from feature_engineering import add_features as _add_features
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    # Microstructure features need vwap + trade_count; default them if absent.
    for col in ['vwap', 'trade_count']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return _add_features(df)


def compute_regime_and_trend(df: pd.DataFrame):
    try:
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low']  - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr     = tr.rolling(14).mean().iloc[-1]
        price   = df['close'].iloc[-1]
        atr_pct = (atr / price) * 100 if price > 0 else 0.0
        
        # Volatility-Adjusted Moving Average (VAMA)
        base_span = 50
        baseline_vol = 1.5
        
        if atr_pct > 0:
            vol_ratio = atr_pct / baseline_vol
            vol_ratio = max(0.5, min(vol_ratio, 5.0))
            dynamic_span = max(10, int(base_span / vol_ratio))
        else:
            dynamic_span = base_span
            
        adaptive_ma   = df['close'].ewm(span=dynamic_span).mean().iloc[-1]
        trend   = "up" if price > adaptive_ma else "down"
        regime  = "wild" if atr_pct > 4.0 else "normal" if atr_pct > 2.0 else "quiet"
        return regime, trend, round(atr_pct, 2)
    except Exception:
        return "normal", "neutral", 2.0


# ========================= MODEL =========================
class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        try:
            df_feat = safe_add_features(df.copy())
            data    = df_feat.tail(self.seq_len).values.astype(np.float32)
            if len(data) < self.seq_len:
                return 0.5

            # FIX: hard sanitize before scaler to prevent the
            # "Input X contains infinity" crash seen in logs
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = np.clip(data, -1e6, 1e6)

            if self.scaler:
                data = self.scaler.transform(data).astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            x = torch.tensor(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return float(self.model(x).item())
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5


predictor = SafeMLPredictor(MODEL_PATH, SEQUENCE_LEN)


# ========================= PORTFOLIO HELPERS =========================
def get_all_positions():
    """Returns dict of Alpaca symbol -> position info."""
    try:
        positions = trading_client.get_all_positions()
        result = {}
        for p in positions:
            result[p.symbol] = {
                'qty':            float(p.qty),
                'avg_entry':      float(p.avg_entry_price),
                'market_value':   float(p.market_value),
                'current_price':  float(p.current_price),
            }
        return result
    except Exception as e:
        logger.error(f"get_all_positions failed: {e}")
        return {}

def normalize_symbol(symbol):
    return symbol.replace("/", "")

def get_buying_power():
    try:
        return float(trading_client.get_account().buying_power)
    except Exception as e:
        logger.error(f"Buying power fetch failed: {e}")
        return 0.0

def sell_largest_position():
    """
    FIXED: Add retry cooldown to prevent spam when settlement is delayed.
    Alpaca may reject sells with "insufficient balance" if cash hasn't settled yet.
    This function now backs off for 5 minutes after a failed sell attempt.
    """
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            return
        largest = max(positions, key=lambda p: float(p.market_value))
        
        now = time.time()
        
        # Check if we've recently tried to sell this position and it failed
        if largest.symbol in sell_retry_cooldown:
            last_attempt = sell_retry_cooldown[largest.symbol]
            time_since_attempt = now - last_attempt
            if time_since_attempt < 300:  # 5 minute cooldown
                logger.warning(
                    f"⏳ Sell retry cooldown for {largest.symbol} "
                    f"({300 - time_since_attempt:.0f}s remaining)")
                return
        
        logger.warning(
            f"📉 Cap exceeded — force selling {largest.symbol} "
            f"${float(largest.market_value):.2f}")
        
        try:
            trading_client.submit_order(
                order_data=LimitOrderRequest(
                    symbol=largest.symbol, qty=float(largest.qty),
                    side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                    limit_price=float(largest.current_price)
                )
            )
            # Clear cooldown on successful submit
            sell_retry_cooldown.pop(largest.symbol, None)
        except Exception as sell_error:
            # Record this failed attempt with timestamp
            sell_retry_cooldown[largest.symbol] = now
            logger.error(f"Sell order failed (will retry later): {sell_error}")
            
    except Exception as e:
        logger.error(f"sell_largest_position failed: {e}")

def swap_weakest_position(new_symbol: str, new_signal: float, latest_signals: dict, threshold: float = 0.05) -> bool:
    """
    Evaluates currently open positions. If there's a held position with a signal significantly 
    weaker than the new_signal, it force-sells that position to free up capital and returns True.
    """
    try:
        positions = get_all_positions()
        if not positions:
            return False

        weakest_sym = None
        weakest_signal = float('inf')
        weakest_qty = 0.0
        weakest_price = 0.0

        for alpaca_sym, p_data in positions.items():
            held_sym = next((s for s in SYMBOLS if normalize_symbol(s) == alpaca_sym), None)
            if not held_sym:
                continue

            held_signal = latest_signals.get(held_sym, 0.5)

            if held_signal < weakest_signal:
                weakest_signal = held_signal
                weakest_sym = alpaca_sym
                weakest_qty = p_data['qty']
                weakest_price = p_data['current_price']

        if weakest_sym and (new_signal - weakest_signal) >= threshold:
            logger.warning(
                f"🔄 SWAP TRIGGERED: Selling {weakest_sym} (signal {weakest_signal:.4f}) "
                f"to make room for {new_symbol} (signal {new_signal:.4f})"
            )
            
            try:
                trading_client.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=weakest_sym,
                        qty=float(weakest_qty),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        limit_price=float(weakest_price)
                    )
                )
                return True
            except Exception as e:
                logger.error(f"Failed to execute swap sell for {weakest_sym}: {e}")
                return False

    except Exception as e:
        logger.error(f"swap_weakest_position failed: {e}")
        
    return False

def cancel_stale_orders(timeout_minutes=3):
    """
    Finds and cancels any open orders that have been sitting unfilled for longer than timeout_minutes.
    This frees up buying power that gets locked by Limit Order Entries.
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
                logger.info(f"🗑️ Canceling stale unfilled order {order.id} for {order.symbol} (Age: {order_age:.1f}m)")
                trading_client.cancel_order_by_id(order.id)
    except Exception as e:
        logger.error(f"Failed to cancel stale orders: {e}")

def calculate_kelly_multiplier(signal_prob: float, profit_target_pct: float, stop_loss_pct: float) -> float:
    if stop_loss_pct <= 0 or profit_target_pct <= 0:
        return 1.0
        
    w = signal_prob
    r = profit_target_pct / stop_loss_pct
    kelly_fraction = w - ((1.0 - w) / r)
    
    if kelly_fraction <= 0:
        return 0.5
        
    multiplier = 0.5 + (kelly_fraction * 15.0)
    return max(0.5, min(multiplier, 3.0))

# ========================= STARTUP SYNC =========================
def sync_existing_positions():
    """
    On startup, populate entry_time for any positions already held.
    FIXED: Don't overwrite existing entry_time if position was already being tracked.
    """
    logger.info("🔍 Scanning existing positions on startup...")
    positions = get_all_positions()
    if not positions:
        logger.info("No existing positions found.")
        return
    for alpaca_sym, data in positions.items():
        for sym in SYMBOLS:
            if normalize_symbol(sym) == alpaca_sym:
                # Only set entry_time if not already tracking this position
                if sym not in entry_time:
                    entry_time[sym] = time.time()
                logger.info(
                    f"♻️  Restored: {sym} | qty={data['qty']:.6f} | "
                    f"avg_entry=${data['avg_entry']:.4f}")
                break


# ========================= MAIN LOOP =========================
async def run_trading_mode():
    global start_equity
    sync_existing_positions()
    logger.info("🚀 Grok Apex Ironclad Bot v9 - Cutting Edge with DOGE Started")

    # --- NEW: Sync starting_equity in database with actual Alpaca equity ---
    try:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    # Ensure the column exists
                    cur.execute("ALTER TABLE bot_status ADD COLUMN IF NOT EXISTS starting_equity NUMERIC")
                    # Update starting_equity to the current real equity
                    cur.execute("""
                        INSERT INTO bot_status (bot_name, starting_equity, live_equity, live_equity_updated_at, last_update)
                        VALUES (%s, %s, %s, NOW(), NOW())
                        ON CONFLICT (bot_name) DO UPDATE
                        SET starting_equity = EXCLUDED.starting_equity,
                            live_equity = EXCLUDED.live_equity,
                            live_equity_updated_at = NOW(),
                            last_update = NOW()
                    """, (BOT_NAME, float(start_equity), float(start_equity)))
                    conn.commit()
                    logger.info(f"✅ Synced starting_equity to ${start_equity:.2f} in database")
    except Exception as e:
        logger.warning(f"⚠️ Could not sync starting_equity: {e}")
    # --- End of new code ---

    while True:
        try:
            cancel_stale_orders(timeout_minutes=3)
            
            account = trading_client.get_account()
            equity  = float(account.equity)
            if start_equity is None:
                start_equity = equity

            # FIXED: Report equity to database - this was missing
            report_equity(BOT_NAME, equity)

            drawdown = (equity - start_equity) / start_equity * 100
            if drawdown < MAX_DRAWDOWN_STOP:
                logger.error("🚨 MAX DRAWDOWN HIT - Stopping trading")
                break

            # --- NEW: fetch positions once per cycle for cap enforcement ---
            current_positions = get_all_positions()
            open_count         = len(current_positions)
            total_value        = sum(p['market_value'] for p in current_positions.values())
            buying_power       = get_buying_power()

            logger.info(
                f"📊 Cycle | Positions: {open_count}/{MAX_OPEN_POSITIONS} | "
                f"Value: ${total_value:.2f} | BP: ${buying_power:.2f} | "
                f"Drawdown: {drawdown:.2f}%")

            buys_allowed             = True
            running_portfolio_value  = total_value

            if total_value >= MAX_PORTFOLIO_VALUE:
                logger.warning(
                    f"📉 Portfolio ${total_value:.2f} >= cap ${MAX_PORTFOLIO_VALUE:.2f}")
                sell_largest_position()
                buys_allowed = False

            if open_count >= MAX_OPEN_POSITIONS:
                logger.info(
                    f"🛑 Max positions reached ({open_count}/{MAX_OPEN_POSITIONS}). "
                    f"Holding off on buys.")
                buys_allowed = False

            now = time.time()
            symbols_to_process = []
            for sym in SYMBOLS:
                cooldown_until.setdefault(sym, 0.0)
                if now >= cooldown_until.get(sym, 0):
                    symbols_to_process.append(sym)

            # PARALLEL FETCH: Fetch all OHLCV dataframes concurrently
            fetch_tasks = [get_clean_ohlcv_dataframe(sym) for sym in symbols_to_process]
            dfs = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for symbol, df in zip(symbols_to_process, dfs):
                if df is None or isinstance(df, Exception):
                    if isinstance(df, Exception):
                        logger.error(f"Failed to fetch data for {symbol}: {df}")
                    continue

                alpaca_sym = normalize_symbol(symbol)

                regime, trend, atr_pct = compute_regime_and_trend(df)
                signal = predictor.predict(df)
                latest_signals[symbol] = signal
                price  = df["close"].iloc[-1]

                # --- STEP 1 DIAGNOSTIC: always-visible (INFO level) proof that
                # the model is alive and what it's actually outputting for this
                # asset, vs. the current buy threshold. This is the single line
                # requested to confirm the model is producing valid numbers
                # before touching any more strategy logic. ---
                logger.info(f"📈 Signal: {symbol} = {signal:.4f} (Buy threshold: {BUY_SIGNAL})")

                # --- FIX: DEBUG-level visibility into every symbol's raw signal
                # each cycle. Run with LOG_LEVEL=DEBUG to see exactly why the
                # bot is or isn't entering a trade on each asset. ---
                logger.debug(
                    f"🔎 {symbol} | signal={signal:.4f} (buy>{BUY_SIGNAL}, sell<{SELL_SIGNAL}) | "
                    f"price=${price:.4f} | regime={regime} | trend={trend} | atr%={atr_pct}")


                if price <= 0:
                    logger.debug(f"❌ {symbol} rejected: invalid price ({price})")
                    continue


                # --- FIXED: use cached positions dict instead of per-symbol
                # get_position() call which had no fallback on failure ---
                pos_data     = current_positions.get(alpaca_sym)
                # FIX: require a minimum dollar value, not just qty > 0, so leftover
                # dust from rounding drift after a sell doesn't get treated as a
                # real open position and re-sold every cycle (see log: repeated
                # sub-cent SOL/ETH/BTC "sell" orders right after the real sell)
                has_position = (
                    pos_data is not None
                    and pos_data['qty'] > 0
                    and pos_data['market_value'] >= MIN_POSITION_USD
                )
                qty_held     = pos_data['qty']       if has_position else 0.0
                avg_entry    = pos_data['avg_entry']  if has_position else 0.0

                # ---------------- EXIT LOGIC ----------------
                if has_position:
                    pnl_pct     = (price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                    held_hours  = (now - entry_time.get(symbol, now)) / 3600
                    exit_reason = None
                    
                    # Trailing Stop-Loss Peak Tracking
                    highest_seen = highest_prices.get(symbol, avg_entry)
                    if price > highest_seen:
                        highest_prices[symbol] = price
                        highest_seen = price

                    if highest_seen > avg_entry * (1 + PROFIT_TARGET_PCT):
                        # Trailing Mode Active (e.g. 1% trailing stop from peak)
                        trailing_stop_price = highest_seen * 0.99
                        if price <= trailing_stop_price:
                            exit_reason = f"📉 Trailing Stop triggered (Peak: ${highest_seen:.2f}, PnL: {pnl_pct*100:.2f}%)"
                    elif pnl_pct <= -STOP_LOSS_PCT:
                        exit_reason = f"🛑 Stop loss ({pnl_pct*100:.2f}%)"
                    elif held_hours >= MAX_HOLD_HOURS:
                        exit_reason = f"⏰ Max hold time ({held_hours:.1f}h)"
                    elif held_hours >= MIN_HOLD_HOURS_BEFORE_SIGNAL and signal < SELL_SIGNAL:
                        exit_reason = f"📉 Signal weak ({signal:.3f})"

                    if exit_reason:
                        logger.info(
                            f"{exit_reason} — SELL {symbol} @ {price:.2f} | "
                            f"Regime: {regime}")
                        success = await place_order(symbol, OrderSide.SELL, qty_held, price)
                        if success:
                            cooldown_until[symbol] = now + 1800
                            entry_time.pop(symbol, None)
                            highest_prices.pop(symbol, None)
                    else:
                        logger.info(
                            f"📌 Holding {symbol} | Entry: ${avg_entry:.4f} | "
                            f"Now: ${price:.4f} | Peak: ${highest_seen:.4f} | "
                            f"PnL: {pnl_pct*100:+.2f}% | "
                            f"Held: {held_hours:.1f}h | Signal: {signal:.3f}")
                    await asyncio.sleep(2)
                    continue

                # ---------------- ENTRY LOGIC ----------------
                if signal > BUY_SIGNAL:

                    if not buys_allowed:
                        if swap_weakest_position(symbol, signal, latest_signals):
                            logger.info(f"✅ Swapped weak position to make room for {symbol}. Proceeding with buy.")
                            buys_allowed = True
                            open_count = max(0, open_count - 1)
                        else:
                            logger.info(f"🚫 BUY suppressed for {symbol} (cap/position limit and no weak swap found)")
                            await asyncio.sleep(2)
                            continue

                    # Dynamic Position Sizing using Kelly Criterion
                    kelly_mult = calculate_kelly_multiplier(
                        signal_prob=signal,
                        profit_target_pct=PROFIT_TARGET_PCT,
                        stop_loss_pct=STOP_LOSS_PCT
                    )
                    
                    risk_usd = equity * BASE_RISK_PERCENT * kelly_mult
                    qty      = risk_usd / price
                    if qty * price > MAX_SINGLE_TRADE_USD:
                        qty = MAX_SINGLE_TRADE_USD / price
                    trade_value = qty * price

                    # --- NEW: headroom check ---
                    headroom = MAX_PORTFOLIO_VALUE - running_portfolio_value
                    if headroom < trade_value:
                        logger.warning(
                            f"🚫 BUY blocked {symbol}: only ${headroom:.2f} headroom "
                            f"(need ${trade_value:.2f})")
                        buys_allowed = False
                        await asyncio.sleep(2)
                        continue

                    # --- NEW: buying power check, prevents repeated
                    # "insufficient balance" failures seen in logs ---
                    if buying_power < trade_value:
                        logger.warning(
                            f"🚫 BUY blocked {symbol}: buying power ${buying_power:.2f} "
                            f"< trade ${trade_value:.2f}")
                        buys_allowed = False
                        await asyncio.sleep(2)
                        continue

                    logger.info(
                        f"🟢 BUY {symbol} @ {price:.2f} | Regime: {regime} | "
                        f"Signal: {signal:.3f} | Positions: {open_count}/{MAX_OPEN_POSITIONS}")
                    success = await place_order(symbol, OrderSide.BUY, qty, price)

                    if success:
                        cooldown_until[symbol]   = now + 900
                        entry_time[symbol]        = now
                        running_portfolio_value  += trade_value
                        open_count               += 1
                        if open_count >= MAX_OPEN_POSITIONS:
                            logger.info("🔒 Max positions reached — no more buys this cycle.")
                            buys_allowed = False
                else:
                    # --- FIX: explicit rejection-reason logging as requested.
                    # This is the single line that reveals, every 78-second
                    # cycle, exactly why each of the 18 scanned assets was
                    # NOT bought: the model's signal simply never crossed
                    # BUY_SIGNAL. Run with LOG_LEVEL=DEBUG to see it. ---
                    entry_condition   = signal > BUY_SIGNAL
                    rejection_reason  = (
                        f"signal {signal:.4f} <= BUY_SIGNAL threshold {BUY_SIGNAL}"
                    )
                    if not entry_condition:
                        logger.debug(f"❌ {symbol} rejected: {rejection_reason}")

                await asyncio.sleep(2)


            await asyncio.sleep(40)

        except Exception as e:
            logger.error(f"Critical loop error: {e}")
            await asyncio.sleep(30)


async def place_order(symbol, side, qty, price=None):
    """FIXED: now passes price and order.id into record_trade instead of None."""
    import math
    try:
        if side == OrderSide.SELL:
            qty = math.floor(qty * 1e8) / 1e8
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=price
            )
        else:
            limit_price = price * 1.001 if price else None
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )
            
        order = trading_client.submit_order(order_data=order_data)
        record_trade(BOT_NAME, symbol, side.value, qty, price, order_id=order.id)
        logger.info(f"✅ Order submitted: {side.value} {symbol} {qty:.6f} limit={price if side == OrderSide.SELL else limit_price}")
        return True
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return False


async def get_clean_ohlcv_dataframe(symbol):
    try:
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=600)
        bars = data_client.get_crypto_bars(req).data.get(symbol, [])
        if len(bars) < SEQUENCE_LEN:
            return None
        df = pd.DataFrame([{
            "timestamp": b.timestamp,
            "open":   float(b.open   or 0),
            "high":   float(b.high   or 0),
            "low":    float(b.low    or 0),
            "close":  float(b.close  or 0),
            "volume": float(b.volume or 0),
            "vwap":   float(getattr(b, "vwap", 0) or 0),
            "trade_count": float(getattr(b, "trade_count", 0) or 0)
        } for b in bars])
        df.set_index("timestamp", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.resample("5min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum", "vwap": "last", "trade_count": "sum"
        }).fillna(0)
        # FIX: drop zero-price candles that can break indicator math
        df = df[df['close'] > 0]
        if len(df) < SEQUENCE_LEN:
            return None
        return df.tail(SEQUENCE_LEN)
    except Exception as e:
        logger.error(f"Data fetch error {symbol}: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(run_trading_mode())
