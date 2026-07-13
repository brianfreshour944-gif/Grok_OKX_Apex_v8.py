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

from ml_predictor import GrokGQA_Transformer
from feature_engineering import add_features, FEATURE_COLS

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ========================= CONFIG =========================
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v9_CuttingEdge")
# Seed list used at startup (for cooldown_until / sync_existing_positions);
# refreshed every cycle by `scan_stable_assets()`.
SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD", "BCH/USD",
    "LINK/USD", "UNI/USD", "AVAX/USD", "DOT/USD", "AAVE/USD", "ADA/USD",
    "SHIB/USD", "ATOM/USD", "GRT/USD", "MKR/USD", "COMP/USD", "NEAR/USD"
]

ACCOUNT_BASE          = float(os.getenv("ACCOUNT_BASE", 10000))
BASE_RISK_PERCENT     = 0.02    # UPDATED: cap each individual position at 2% of
                                 # current equity, so most of the account stays
                                 # in reserve. This is the primary sizing rule --
                                 # see calculate_adjusted_risk(), which multiplies
                                 # this against the ACTUAL current equity every
                                 # cycle (not a fixed starting balance), so the
                                 # 2% cap tracks the account as it grows/shrinks.
MAX_SINGLE_TRADE_USD  = 100000  # UPDATED: generous absolute backstop only --
                                 # not meant to bind in normal operation. The
                                 # real per-trade ceiling is the 2% equity rule
                                 # above; this just guards against a runaway
                                 # equity value ever producing an absurd order.
MAX_DRAWDOWN_STOP     = -10.0

# --- NEW: position / risk management ---
# MAX_PORTFOLIO_VALUE is now computed dynamically each cycle from current
# equity (equity * BASE_RISK_PERCENT * MAX_OPEN_POSITIONS) instead of this
# fixed dollar figure, so the total-exposure cap scales with the account
# instead of going stale as equity changes. See run_trading_mode().
MAX_OPEN_POSITIONS    = 10      # Don't hold more than this many symbols at once
MAX_HOLD_HOURS        = 4.0     # Force-sell after this long regardless of signal
PROFIT_TARGET_PCT     = 0.02    # Sell if up 2%
STOP_LOSS_PCT         = 0.03    # Sell if down 3%
BUY_SIGNAL            = 0.62
SELL_SIGNAL           = 0.45    # FIX: widened gap from BUY_SIGNAL (was 0.55) to reduce
                                 # exits on model noise right around the 0.5 midpoint

# --- NEW: fixes for dust-order bug and signal-exit whipsaw (see log analysis) ---
MIN_POSITION_USD              = 5.0   # ignore/never re-sell positions worth less than this
MIN_ORDER_USD                  = 10.0  # Alpaca's minimum crypto order notional; skip buys under this
MIN_HOLD_HOURS_BEFORE_SIGNAL  = 0.5   # don't let a weak-signal reading exit a position
                                       # until it's been held at least this long

SEQUENCE_LEN = 32
MODEL_PATH = "grok_gqa_v9_best.pth"

# --- NEW: regime-adaptive thresholds ---
# compute_regime_and_trend() already classifies each symbol's volatility every
# cycle ("wild" / "normal" / "quiet" by ATR%), but until now that label was
# only logged, never used to change the entry/exit decision itself -- BUY/SELL
# thresholds and profit/stop levels were flat constants regardless of regime.
#
# "normal" intentionally matches the original BUY_SIGNAL/SELL_SIGNAL/
# PROFIT_TARGET_PCT/STOP_LOSS_PCT constants above 1:1, so a symbol sitting in
# the normal band trades exactly as it did before this change.
REGIME_PARAMS = {
    # Wild swings: demand more conviction to enter (avoid buying noise),
    # give it more room before stopping/taking profit (avoid getting
    # shaken out by normal volatility in that regime).
    "wild": {
        "buy_signal":        0.68,
        "sell_signal":       0.42,
        "profit_target_pct": 0.03,
        "stop_loss_pct":     0.045,
    },
    # Matches the pre-existing flat constants.
    "normal": {
        "buy_signal":        BUY_SIGNAL,
        "sell_signal":       SELL_SIGNAL,
        "profit_target_pct": PROFIT_TARGET_PCT,
        "stop_loss_pct":     STOP_LOSS_PCT,
    },
    # Quiet chop: accept a weaker signal since big moves are unlikely
    # anyway, but take profit/cut losses sooner since there's less room
    # for a move to develop before it reverses.
    "quiet": {
        "buy_signal":        0.58,
        "sell_signal":       0.47,
        "profit_target_pct": 0.015,
        "stop_loss_pct":     0.02,
    },
}

def get_regime_params(regime: str) -> dict:
    """Returns the threshold set for a regime, falling back to 'normal' for
    any unrecognized label (e.g. the "normal"/"neutral"/2.0 fallback that
    compute_regime_and_trend() returns on error)."""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS["normal"])

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

# --- TEMP DEBUG: confirm what actually loaded, without logging the real
# secret. Compare key_len/secret_len and the last4 against the values shown
# in the Alpaca dashboard, then remove this block. ---
if API_KEY and API_SECRET:
    logger.info(
        f"🔑 Credential check — key_len={len(API_KEY)} key_last4={API_KEY[-4:]} | "
        f"secret_len={len(API_SECRET)} secret_last4={API_SECRET[-4:]} | paper={PAPER}"
    )
else:
    logger.error(
        f"🔑 Credential check — APCA_API_KEY_ID present={bool(API_KEY)}, "
        f"APCA_API_SECRET_KEY present={bool(API_SECRET)}. "
        f"One or both env vars are missing on this Coolify app."
    )

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}
entry_time     = {}   # symbol -> timestamp, tracks how long a position has been held
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
# NOTE: feature computation now comes from the shared feature_engineering.py
# module (imported above), the same one trading_env.py uses for training and
# ml_predictor.py uses for backtest_engine.py. This file previously had its
# own diverged safe_add_features() that (a) never computed 'bb_width' at all
# -- silently feeding the model a constant 0.0 for that input on every live
# prediction -- and (b) computed rsi/macd/atr with different formulas than
# training used. That's a real train/serve skew bug, not just a style
# difference: the model was making live decisions on inputs that didn't
# match what it learned on. Removed in favor of the single shared pipeline.

# -------------------------------------------------
# Dynamic Stable‑Asset Scanner
# -------------------------------------------------
async def scan_stable_assets(limit_scope: int = 35) -> list:
    """
    Dynamically scans Alpaca for the top crypto assets by 24‑hour dollar volume.
    Returns a list of symbols (e.g., "BTC/USD") limited to `limit_scope`.
    """
    try:
        # Broad candidate list – can be expanded by querying Alpaca assets in production
        candidates = [
            "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD", "BCH/USD",
            "LINK/USD", "UNI/USD", "AVAX/USD", "DOT/USD", "AAVE/USD", "ADA/USD",
            "SHIB/USD", "ATOM/USD", "GRT/USD", "MKR/USD", "COMP/USD", "NEAR/USD",
            "XRP/USD", "BAT/USD", "CRV/USD", "SUSHI/USD", "XTZ/USD", "YFI/USD"
        ]

        volume_data = []
        now = datetime.now()
        start_time = now - timedelta(days=1)

        for symbol in candidates:
            try:
                req = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_time
                )
                bars = data_client.get_crypto_bars(req).data.get(symbol, [])
                if bars:
                    day_bar = bars[-1]
                    dollar_volume = float(day_bar.volume) * float(day_bar.close)
                    volume_data.append({"symbol": symbol, "volume": dollar_volume})
            except Exception:
                continue

        if not volume_data:
            return ["BTC/USD", "ETH/USD", "SOL/USD"]  # safe fallback

        df_vol = pd.DataFrame(volume_data)
        df_vol = df_vol.sort_values(by="volume", ascending=False)
        top_symbols = df_vol["symbol"].head(limit_scope).tolist()
        logger.info(f"🔍 Scanner selected {len(top_symbols)} high‑volume assets.")
        return top_symbols
    except Exception as e:
        logger.error(f"Scanner exception: {e}")
        return ["BTC/USD", "ETH/USD", "SOL/USD"]


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
        ema50   = df['close'].ewm(span=50).mean().iloc[-1]
        trend   = "up" if price > ema50 else "down"
        regime  = "wild" if atr_pct > 4.0 else "normal" if atr_pct > 2.0 else "quiet"
        return regime, trend, round(atr_pct, 2)
    except Exception:
        return "normal", "neutral", 2.0

# -------------------------------------------------
# Volatility‑Adjusted Risk Sizing
# -------------------------------------------------
def calculate_adjusted_risk(equity: float, base_risk_percent: float, atr_pct: float) -> float:
    """
    Returns a dollar risk amount that is scaled down when the asset's ATR%
    exceeds a baseline volatility threshold.

    NOTE: `atr_pct` arrives already expressed as a percentage number (e.g.
    9.00 means "9%", 0.09 means "0.09%") -- see compute_regime_and_trend(),
    which does `(atr/price)*100`. baseline_vol must be in that SAME scale.
    """
    baseline_vol = 1.5  # 1.5% ATR baseline (matches atr_pct's percentage-number scale)
    if atr_pct > baseline_vol and atr_pct > 0:
        vol_scaler = baseline_vol / atr_pct
        adjusted = equity * base_risk_percent * vol_scaler
        logger.info(f"⚠️ High volatility (ATR%={atr_pct:.2f}%) – scaling risk by {vol_scaler:.2f}x")
    else:
        adjusted = equity * base_risk_percent
    # Enforce hard cap
    return min(adjusted, MAX_SINGLE_TRADE_USD)


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
            df_feat = add_features(df.copy())
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
            logger.error(f"Prediction error: {e}", exc_info=True)
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
                order_data=MarketOrderRequest(
                    symbol=largest.symbol, qty=float(largest.qty),
                    side=OrderSide.SELL, time_in_force=TimeInForce.GTC
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

            # UPDATED: total exposure cap now scales with current equity --
            # at most MAX_OPEN_POSITIONS positions, each capped at
            # BASE_RISK_PERCENT (2%) of equity, so this tracks the account
            # instead of a stale fixed dollar figure.
            max_portfolio_value = equity * BASE_RISK_PERCENT * MAX_OPEN_POSITIONS

            logger.info(
                f"📊 Cycle | Positions: {open_count}/{MAX_OPEN_POSITIONS} | "
                f"Value: ${total_value:.2f} | BP: ${buying_power:.2f} | "
                f"Drawdown: {drawdown:.2f}% | Portfolio cap: ${max_portfolio_value:.2f}")

            buys_allowed             = True
            running_portfolio_value  = total_value

            if total_value >= max_portfolio_value:
                logger.warning(
                    f"📉 Portfolio ${total_value:.2f} >= cap ${max_portfolio_value:.2f}")
                sell_largest_position()
                buys_allowed = False

            if open_count >= MAX_OPEN_POSITIONS:
                logger.info(
                    f"🛑 Max positions reached ({open_count}/{MAX_OPEN_POSITIONS}). "
                    f"Holding off on buys.")
                buys_allowed = False

            # -------------------------------------------------
            # Main trading loop – now uses dynamic asset scanner
            # -------------------------------------------------
            # Dynamically fetch high‑volume, stable assets each cycle
            active_symbols = await scan_stable_assets(limit_scope=18)

            # Ensure cooldown entries exist for any newly discovered symbols
            for sym in active_symbols:
                if sym not in cooldown_until:
                    cooldown_until[sym] = 0.0

            for symbol in active_symbols:
                now        = time.time()
                alpaca_sym = normalize_symbol(symbol)

                if now < cooldown_until.get(symbol, 0):
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue

                regime, trend, atr_pct = compute_regime_and_trend(df)
                regime_params = get_regime_params(regime)
                signal = predictor.predict(df)
                price  = df["close"].iloc[-1]

                if price <= 0:
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

                    if pnl_pct >= regime_params["profit_target_pct"]:
                        exit_reason = f"🎯 Profit target ({pnl_pct*100:.2f}%) [{regime}]"
                    elif pnl_pct <= -regime_params["stop_loss_pct"]:
                        exit_reason = f"🛑 Stop loss ({pnl_pct*100:.2f}%) [{regime}]"
                    elif held_hours >= MAX_HOLD_HOURS:
                        exit_reason = f"⏰ Max hold time ({held_hours:.1f}h)"
                    elif held_hours >= MIN_HOLD_HOURS_BEFORE_SIGNAL and signal < regime_params["sell_signal"]:
                        exit_reason = f"📉 Signal weak ({signal:.3f}) [{regime}]"

                    if exit_reason:
                        logger.info(
                            f"{exit_reason} — SELL {symbol} @ {price:.2f} | "
                            f"Regime: {regime}")
                        success = await place_order(symbol, OrderSide.SELL, qty_held, price)
                        if success:
                            cooldown_until[symbol] = now + 1800
                            entry_time.pop(symbol, None)
                    else:
                        logger.info(
                            f"📌 Holding {symbol} | Entry: ${avg_entry:.4f} | "
                            f"Now: ${price:.4f} | PnL: {pnl_pct*100:+.2f}% | "
                            f"Held: {held_hours:.1f}h | Signal: {signal:.3f}")
                    await asyncio.sleep(2)
                    continue

                # ---------------- ENTRY LOGIC ----------------
                if signal > regime_params["buy_signal"]:

                    if not buys_allowed:
                        logger.info(f"🚫 BUY suppressed for {symbol} (cap/position limit)")
                        await asyncio.sleep(2)
                        continue

                    # Volatility‑adjusted risk sizing
                    adjusted_risk = calculate_adjusted_risk(equity, BASE_RISK_PERCENT, atr_pct)
                    qty = adjusted_risk / price
                    trade_value = qty * price

                    # --- NEW: minimum-order-size check, so a heavily
                    # vol-scaled-down trade never gets submitted only to be
                    # rejected by Alpaca's $10 crypto order minimum ---
                    if trade_value < MIN_ORDER_USD:
                        logger.info(
                            f"🚫 BUY skipped {symbol}: sized trade ${trade_value:.2f} "
                            f"below Alpaca's ${MIN_ORDER_USD:.2f} order minimum "
                            f"(regime: {regime}, ATR%: {atr_pct:.2f}%)")
                        await asyncio.sleep(2)
                        continue

                    # --- NEW: headroom check ---
                    headroom = max_portfolio_value - running_portfolio_value
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

                await asyncio.sleep(2)

            await asyncio.sleep(40)

        except Exception as e:
            logger.error(f"Critical loop error: {e}")
            await asyncio.sleep(30)


async def place_order(symbol, side, qty, price=None):
    """FIXED: now passes price and order.id into record_trade instead of None."""
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
        )
        record_trade(BOT_NAME, symbol, side.value, qty, price, order_id=order.id)
        logger.info(f"✅ Order submitted: {side.value} {symbol} {qty:.6f}")
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
            "volume": float(b.volume or 0)
        } for b in bars])
        df.set_index("timestamp", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.resample("5min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
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
