# money.py — Decimal-based money utilities to prevent float precision drift.
# All money-related calculations use Decimal for exact arithmetic.
# Callers pass floats in and get floats back, but intermediate math
# is performed in Decimal precision.

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, getcontext

getcontext().prec = 28

# Common quantization for 8 decimal places (crypto asset qty precision)
QUANT_8DP  = Decimal("0.00000001")
QUANT_4DP  = Decimal("0.0001")
QUANT_2DP  = Decimal("0.01")


def to_dec(val) -> Decimal:
    """Convert any numeric input to Decimal safely."""
    if isinstance(val, Decimal):
        return val
    if isinstance(val, float):
        return Decimal(str(val))
    if isinstance(val, str):
        return Decimal(val)
    return Decimal(str(val))


def money(value, places: Decimal = QUANT_2DP) -> float:
    """Convert to Decimal, quantize to standard money precision, return float."""
    return float(to_dec(value).quantize(places, rounding=ROUND_HALF_UP))


def qty(value) -> float:
    """Floor a quantity to 8 decimal places (crypto minimum precision)."""
    return float(to_dec(value).quantize(QUANT_8DP, rounding=ROUND_DOWN))


def mul(a, b) -> float:
    """Multiply two money values with Decimal precision."""
    return float(to_dec(a) * to_dec(b))


def div(numerator, denominator) -> float:
    """Divide two money values with Decimal precision."""
    num = to_dec(numerator)
    den = to_dec(denominator)
    if den == 0:
        raise ZeroDivisionError("division by zero in money.div")
    return float(num / den)


def pct_change_x100(old_price, new_price) -> float:
    """
    Calculate percentage change: (new - old) / old, Decimal precision.

    Returns a PERCENTAGE already scaled by 100 (e.g. 3.0 for a 3% move) --
    NOT a fraction. See pnl_pct_fraction() below for the fraction-scale
    equivalent. This repo shipped a real bug once (commit 9610523) from
    conflating the two scales; the differing names are meant to make the
    scale unmistakable at every call site rather than relying on
    documentation alone.
    """
    old_d = to_dec(old_price)
    new_d = to_dec(new_price)
    if old_d == 0:
        return 0.0
    return float(((new_d - old_d) / old_d) * Decimal("100"))


def weighted_avg(total_value, total_qty) -> float:
    """Calculate weighted average entry price from total value and qty."""
    val = to_dec(total_value)
    qty_d = to_dec(total_qty)
    if qty_d == 0:
        return 0.0
    return float(val / qty_d)


def pnl_dollar(avg_entry, current_price, qty) -> float:
    """Calculate unrealized PnL in dollars: (price - avg_entry) * qty."""
    return float((to_dec(current_price) - to_dec(avg_entry)) * to_dec(qty))


def realized_pnl(avg_entry, exit_price, qty, fee=0.0) -> float:
    """
    Calculate realized PnL in dollars for a closed (sold) position, net of fees:
    (exit_price - avg_entry) * qty - fee.
    """
    gross = (to_dec(exit_price) - to_dec(avg_entry)) * to_dec(qty)
    return float(gross - to_dec(fee))


def pnl_pct_fraction(avg_entry, current_price) -> float:
    """
    Calculate unrealized PnL as a FRACTION: (price - avg_entry) / avg_entry
    (e.g. 0.03 for a 3% move) -- NOT a percentage. See pct_change_x100()
    above for the x100-scale equivalent; the two are easy to conflate, which
    is why they're named differently rather than both called "pct_change".
    """
    old_d = to_dec(avg_entry)
    new_d = to_dec(current_price)
    if old_d == 0:
        return 0.0
    return float((new_d - old_d) / old_d)