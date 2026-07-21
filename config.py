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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

BOT_VERSION = "2026-07-13-r2"
logger.info(f"🔖 Bot code version: {BOT_VERSION}")

# ── Identity ───────────────────────────────────────────────────────────────────
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v9_CuttingEdge")

# ── Universe ───────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD"
]

# ── Risk / sizing ──────────────────────────────────────────────────────────────
ACCOUNT_BASE         = float(os.getenv("ACCOUNT_BASE", 10000))
BASE_RISK_PERCENT    = 0.02     # 2% of equity per position
MAX_SINGLE_TRADE_USD = 100_000  # absolute backstop
MAX_DRAWDOWN_STOP    = -10.0    # % drawdown at which trading halts

# ── Position management ────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS           = 10
MAX_HOLD_HOURS               = 4.0
PROFIT_TARGET_PCT            = 0.02
STOP_LOSS_PCT                = 0.03
BUY_SIGNAL                   = 0.51
SELL_SIGNAL                  = 0.45
MIN_POSITION_USD             = 5.0   # ignore dust positions below this
MIN_ORDER_USD                = 10.0  # Alpaca minimum crypto order notional
MIN_HOLD_HOURS_BEFORE_SIGNAL = 0.5   # hold at least this long before signal-exit

# ── Model / data ───────────────────────────────────────────────────────────────
SEQUENCE_LEN = 32
MODEL_PATH   = "grok_gqa_v9_best.pth"

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
API_KEY    = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER      = os.getenv("APCA_API_PAPER", "true").lower() == "true"

if API_KEY and API_SECRET:
    logger.info(
        f"🔑 Credential check — key_len={len(API_KEY)} key_last4={API_KEY[-4:]} | "
        f"secret_len={len(API_SECRET)} secret_last4={API_SECRET[-4:]} | paper={PAPER}"
    )
else:
    logger.error(
        "🔑 Credential check — APCA_API_KEY_ID or APCA_API_SECRET_KEY missing."
    )

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client    = CryptoHistoricalDataClient()
