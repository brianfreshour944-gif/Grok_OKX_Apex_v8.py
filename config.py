# config.py — All configuration, constants, and shared Alpaca clients.
# Every other module imports from here to avoid circular imports.
#
# Environment variables are injected by Coolify directly into the container
# process — os.getenv() reads them natively. No .env file or load_dotenv()
# is needed or used.

import logging
import os

from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient

# ── Logging ────────────────────────────────────────────────────────────────────
# .env.example documents LOG_LEVEL as configurable; it previously wasn't --
# config.py hardcoded logging.INFO regardless of what was set. Now actually
# reads it, defaulting to INFO to preserve prior behavior when unset.
_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL_NAME, logging.INFO),
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

BOT_VERSION = "2026-07-13-r2"
logger.info(f"🔖 Bot code version: {BOT_VERSION}")

# ── Identity ───────────────────────────────────────────────────────────────────
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v9_CuttingEdge")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── Magic Numbers & Paths ──────────────────────────────────────────────────────
COOLDOWN_SECONDS_BUY  = 900
COOLDOWN_SECONDS_SELL = 1800
SLEEP_PER_LOOP        = 40
HEARTBEAT_PATH        = os.getenv("HEARTBEAT_PATH", "/tmp/bot_heartbeat.txt")

# ── Universe ───────────────────────────────────────────────────────────────────
# SYMBOLS is the historical static universe. main_bot.py no longer uses it to
# decide what to trade (see DYNAMIC_UNIVERSE_CANDIDATES below) -- it's kept
# here because train_transformer.py's own TRAIN_SYMBOLS and several
# diagnostic/backtest scripts still reference it directly.
SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD"
]

# The model (grok_gqa_v9_best.pth) was trained on exactly these 10 symbols
# via train_transformer.py's TRAIN_SYMBOLS -- keep the two lists in sync.
# The live bot's dynamic universe is restricted to this pool rather than
# data_feeds.scan_stable_assets()'s full 24-symbol candidate list, because
# feature_scaler.pkl was only fit on these symbols' feature distributions;
# trading an asset the scaler never saw risks out-of-distribution inputs.
DYNAMIC_UNIVERSE_CANDIDATES = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
    "LTC/USD", "AVAX/USD", "LINK/USD", "ADA/USD",
    "BCH/USD", "DOT/USD",
]
UNIVERSE_SIZE            = 5     # how many of the above are actively open to new entries at once
UNIVERSE_REFRESH_SECONDS = 3600  # rescan cadence for 24h-volume ranking (1 hour)

# ── Risk / sizing ──────────────────────────────────────────────────────────────
ACCOUNT_BASE         = float(os.getenv("ACCOUNT_BASE", 10000))
BASE_RISK_PERCENT    = 0.02     # 2% of equity per position
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", 5000))  # absolute backstop
MAX_DRAWDOWN_STOP    = -10.0    # % drawdown at which trading halts

# ── Position management ────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS           = 10
MAX_HOLD_HOURS               = 4.0
PROFIT_TARGET_PCT            = 0.02
STOP_LOSS_PCT                = 0.03
BUY_SIGNAL                   = float(os.getenv("BUY_SIGNAL", 0.51))
SELL_SIGNAL                  = float(os.getenv("SELL_SIGNAL", 0.45))
MIN_POSITION_USD             = 5.0   # ignore dust positions below this
MIN_ORDER_USD                = 10.0  # Alpaca minimum crypto order notional
MIN_HOLD_HOURS_BEFORE_SIGNAL = 0.5   # hold at least this long before signal-exit

# Trailing stop: distance from peak scales with realized volatility (ATR%)
# instead of a fixed 1%. trailing_stop_pct = clamp(atr_pct/100 * multiplier,
# min, max). At atr_pct=2.0 (the "normal" baseline in regime.py) this
# multiplier reproduces the old fixed 1% exactly, so quiet/normal-regime
# behavior is unchanged; it only widens the stop in genuinely volatile
# markets (giving a real trade room to breathe) and tightens it in genuinely
# quiet ones (without going below the floor, which would chop out of a
# winning position on pure noise).
TRAILING_STOP_ATR_MULTIPLIER = 0.5
MIN_TRAILING_STOP_PCT        = 0.005  # 0.5% floor
MAX_TRAILING_STOP_PCT        = 0.03   # 3% ceiling

# ── Model / data ───────────────────────────────────────────────────────────────
SEQUENCE_LEN = 32
MODEL_PATH   = "grok_gqa_v9_best.pth"

# ── Experience capture & model bake-off pipeline ───────────────────────────────
# Step 0: decision-time feature vectors + outcomes (JSONL, append-only).
EXPERIENCE_LOG_PATH = os.getenv("EXPERIENCE_LOG_PATH", "live_experiences.jsonl")
# Step 3: per-cycle challenger-vs-champion shadow predictions.
SHADOW_LOG_PATH     = os.getenv("SHADOW_LOG_PATH", "shadow_predictions.jsonl")
# Steps 2/4: GBT challenger artifact produced by train_gbt_baseline.py;
# evaluated by promotion_gate.py. The live champion is still MODEL_PATH --
# a challenger only becomes champion via an explicit MODEL_PATH change,
# which SafeMLPredictor's hot-reload picks up without a restart.
GBT_CHALLENGER_PATH = os.getenv("GBT_CHALLENGER_PATH", "gbt_challenger.joblib")
# Label horizon shared by trainer/gate: 6 x 15-min bars = 90 min, matching
# train_transformer.TARGET_HORIZON so champion and challenger are compared
# on identical targets.
GBT_HORIZON_BARS    = int(os.getenv("GBT_HORIZON_BARS", "6"))

# ── Regime-adaptive thresholds ─────────────────────────────────────────────────
# NOTE: buy/sell signals are anchored to BUY_SIGNAL / SELL_SIGNAL so that
# lowering BUY_SIGNAL (e.g. 0.62 -> 0.51 for diagnostics) takes effect in
# EVERY regime, not just "normal". Original relative offsets preserved:
#   quiet = BUY_SIGNAL - 0.04  (more eager in low-vol markets)
#   wild  = BUY_SIGNAL + 0.06  (more conservative in high-vol markets)
#   quiet sell = SELL_SIGNAL + 0.02, wild sell = SELL_SIGNAL - 0.03
def get_regime_params(regime: str, base_buy=None, base_sell=None) -> dict:
    """Return threshold set for a regime computed dynamically to support runtime config changes."""
    # If not explicitly passed, read the current live variables from the module
    if base_buy is None:
        base_buy = BUY_SIGNAL
    if base_sell is None:
        base_sell = SELL_SIGNAL
        
    offsets = {
        "wild":   {"buy_offset":  0.06, "sell_offset": -0.03, "tp": 0.03, "sl": 0.045},
        "normal": {"buy_offset":  0.00, "sell_offset":  0.00, "tp": PROFIT_TARGET_PCT, "sl": STOP_LOSS_PCT},
        "quiet":  {"buy_offset": -0.04, "sell_offset":  0.02, "tp": 0.015, "sl": 0.02},
    }
    
    off = offsets.get(regime, offsets["normal"])
    
    return {
        "buy_signal":        base_buy + off["buy_offset"],
        "sell_signal":       base_sell + off["sell_offset"],
        "profit_target_pct": off["tp"],
        "stop_loss_pct":     off["sl"],
    }


def fmt_price(p: float) -> str:
    """Format a price with enough decimal places to be meaningful.
    e.g. 0.0000084 -> '0.00000840', 238.72 -> '238.72'"""
    if p == 0:
        return "0.00"
    if p >= 1:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{p:.4f}"
    if p >= 0.0001:
        return f"{p:.6f}"
    return f"{p:.8f}"


# ── Alpaca clients (singletons, initialized once on import) ────────────────────
def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"🛑 Required env var {name} is missing")
    return val

API_KEY    = _require_env("APCA_API_KEY_ID")
API_SECRET = _require_env("APCA_API_SECRET_KEY")
PAPER      = os.getenv("APCA_API_PAPER", "true").lower() == "true"

logger.info(f"🔑 Alpaca client initialized (paper={PAPER})")

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client    = CryptoHistoricalDataClient()
