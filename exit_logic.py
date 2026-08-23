# exit_logic.py — Pure position-exit decision logic, extracted from
# main_bot.py's trading loop so it can be unit tested directly (no Alpaca
# client, no DB, no asyncio event loop needed to exercise it).
#
# This is exactly the code path that had the pnl_pct unit-mismatch bug
# (safe_pct_change() returns a percentage but was compared against a
# fractional stop_loss_pct) — it was untestable in isolation before this
# extraction, which is how that bug shipped across several "Fix ..."
# commits without being caught.

from dataclasses import dataclass
from typing import Optional

from config import fmt_price
from money import safe_pct_change


@dataclass
class ExitDecision:
    exit_reason: Optional[str]  # None if the position should keep being held
    pnl_pct: float              # fraction, e.g. -0.03 == -3% (NOT x100)
    highest_seen: float         # possibly-updated peak price
    dynamic_sl_pct: float       # fraction, the stop-loss threshold actually applied this cycle


def evaluate_exit(
    *,
    avg_entry: float,
    price: float,
    highest_seen: float,
    held_hours: float,
    signal: float,
    regime: str,
    profit_target_pct: float,
    stop_loss_pct: float,
    sell_signal: float,
    max_hold_hours: float,
    min_hold_hours_before_signal: float,
) -> ExitDecision:
    """
    Decides whether an open position should be exited this cycle.

    All *_pct arguments are fractions (0.03 == 3%), matching config.py's
    convention (BUY/SELL/PROFIT_TARGET/STOP_LOSS are all fractions there).
    pnl_pct is computed via money.safe_pct_change(), which returns a
    percentage already scaled by 100 -- it is divided back to a fraction
    here so it is directly comparable to stop_loss_pct/profit_target_pct
    and so displaying it as `pnl_pct * 100` gives the real percentage.

    Priority order (first match wins), matching the original live logic:
      1. Trailing stop (once price has run up past profit_target_pct from entry)
      2. Time-decay stop loss (tightens the stop the longer the position is held)
      3. Max hold time
      4. Weak signal (only once min_hold_hours_before_signal has elapsed)
    """
    pnl_pct = safe_pct_change(avg_entry, price) / 100.0 if avg_entry > 0 else 0.0

    if price > highest_seen:
        highest_seen = price

    # Time-Decay Stop Loss Logic
    dynamic_sl_pct = stop_loss_pct
    if held_hours >= 2.0:
        dynamic_sl_pct = stop_loss_pct * 0.5   # Halve the stop loss
    elif held_hours >= 1.0:
        dynamic_sl_pct = stop_loss_pct * 0.75  # Tighten by 25%

    exit_reason = None

    if highest_seen > avg_entry * (1 + profit_target_pct):
        # Trailing Mode Active (1% trailing stop from peak)
        trailing_stop_price = highest_seen * 0.99
        if price <= trailing_stop_price:
            exit_reason = (
                f"📉 Trailing Stop triggered (Peak: ${fmt_price(highest_seen)}, "
                f"PnL: {pnl_pct*100:.2f}%) [{regime}]"
            )
    elif pnl_pct <= -dynamic_sl_pct:
        exit_reason = (
            f"🛑 Time-Decay Stop loss ({pnl_pct*100:.2f}% <= "
            f"-{dynamic_sl_pct*100:.2f}%) [{regime}]"
        )
    elif held_hours >= max_hold_hours:
        exit_reason = f"⏰ Max hold time ({held_hours:.1f}h)"
    elif held_hours >= min_hold_hours_before_signal and signal < sell_signal:
        exit_reason = f"📉 Signal weak ({signal:.3f}) [{regime}]"

    return ExitDecision(
        exit_reason=exit_reason,
        pnl_pct=pnl_pct,
        highest_seen=highest_seen,
        dynamic_sl_pct=dynamic_sl_pct,
    )
