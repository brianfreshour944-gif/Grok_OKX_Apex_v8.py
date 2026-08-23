# tests/test_money.py — money.py Decimal-precision utilities.

import pytest

from money import (
    to_dec, mul, div, safe_pct_change, weighted_avg,
    pnl_dollar, pnl_pct, qty, money,
)


def test_safe_pct_change_and_pnl_pct_have_different_scales():
    """
    safe_pct_change() returns a PERCENTAGE (already x100); pnl_pct() returns
    a FRACTION. They are not interchangeable -- confusing the two is exactly
    the bug fixed in main_bot.py's exit logic (see exit_logic.py /
    tests/test_exit_logic.py). Pinning this down so a future refactor can't
    silently make the two consistent with each other and reintroduce the
    100x mismatch at whichever call site still assumes the old scale.
    """
    assert safe_pct_change(100, 105) == pytest.approx(5.0)
    assert pnl_pct(100, 105) == pytest.approx(0.05)
    assert safe_pct_change(100, 105) == pytest.approx(pnl_pct(100, 105) * 100)


def test_safe_pct_change_zero_old_price_returns_zero():
    assert safe_pct_change(0, 105) == 0.0


def test_pnl_pct_zero_avg_entry_returns_zero():
    assert pnl_pct(0, 105) == 0.0


def test_div_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        div(10, 0)


def test_weighted_avg_zero_qty_returns_zero():
    assert weighted_avg(1000, 0) == 0.0


def test_weighted_avg_basic():
    # 0.5 BTC @ 50000 + 0.3 BTC @ 52000
    total_value = 50000 * 0.5 + 52000 * 0.3
    total_qty = 0.8
    assert weighted_avg(total_value, total_qty) == pytest.approx(50750.0)


def test_pnl_dollar_basic():
    assert pnl_dollar(avg_entry=100, current_price=110, qty=2) == pytest.approx(20.0)


def test_qty_floors_to_eight_decimals():
    # Floors (not rounds) -- must never round UP and oversell.
    assert qty(1.123456789) == pytest.approx(1.12345678)


def test_money_rounds_half_up_to_two_decimals():
    # money() converts via Decimal(str(value)) before quantizing, so this is
    # exact ROUND_HALF_UP on '1.005' -- not subject to 1.005's binary float
    # noise (which would make naive round(1.005, 2) give 1.0).
    assert money(1.005) == pytest.approx(1.01)
    assert money(10.126) == pytest.approx(10.13)


def test_mul_and_div_are_inverse():
    a, b = 123.456, 7.89
    assert div(mul(a, b), b) == pytest.approx(a, rel=1e-9)


def test_to_dec_accepts_float_str_and_decimal():
    from decimal import Decimal
    assert to_dec(1.5) == Decimal("1.5")
    assert to_dec("1.5") == Decimal("1.5")
    assert to_dec(Decimal("1.5")) == Decimal("1.5")
