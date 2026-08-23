# tests/test_exit_logic.py — regression coverage for main_bot.py's position
# exit decisions (stop loss, trailing stop, max hold, weak signal).
#
# This module exists specifically because of a real, shipped bug: pnl_pct
# was compared against a fractional stop_loss_pct while itself being a
# percentage (x100), which fired the stop loss on ~0.03% noise instead of a
# real 3% loss. That code was inline in main_bot.py's 400-line async loop
# and had zero unit coverage -- the repo's own financial_audit.py claimed
# "[PASS] PnL calculation uses Decimal-based safe_pct_change" without ever
# actually calling the real function. These tests exercise the real
# evaluate_exit() with realistic config values so a similar unit mistake
# fails CI instead of shipping.

import pytest

from config import PROFIT_TARGET_PCT, STOP_LOSS_PCT, SELL_SIGNAL, MAX_HOLD_HOURS, MIN_HOLD_HOURS_BEFORE_SIGNAL
from exit_logic import evaluate_exit


def _evaluate(**overrides):
    """evaluate_exit() with sane defaults for a flat, freshly-opened, in-the-money-neutral position."""
    defaults = dict(
        avg_entry=100.0,
        price=100.0,
        highest_seen=100.0,
        held_hours=0.0,
        signal=0.5,
        regime="normal",
        profit_target_pct=PROFIT_TARGET_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        sell_signal=SELL_SIGNAL,
        max_hold_hours=MAX_HOLD_HOURS,
        min_hold_hours_before_signal=MIN_HOLD_HOURS_BEFORE_SIGNAL,
    )
    defaults.update(overrides)
    return evaluate_exit(**defaults)


# ── The actual regression: unit mismatch between pnl_pct and stop_loss_pct ──

def test_tiny_noise_dip_does_not_trigger_stop_loss():
    """A 0.03% dip must NOT trip a 3% stop loss (this is exactly the bug that shipped)."""
    decision = _evaluate(avg_entry=100.0, price=99.97, held_hours=0.0)
    assert decision.exit_reason is None


def test_real_three_percent_drop_triggers_stop_loss():
    decision = _evaluate(avg_entry=100.0, price=96.9, held_hours=0.0)  # -3.1%
    assert decision.exit_reason is not None
    assert "Stop loss" in decision.exit_reason


def test_pnl_pct_is_a_fraction_not_a_percentage():
    """A 5% move must report pnl_pct ~= 0.05, never ~= 5.0."""
    decision = _evaluate(avg_entry=100.0, price=105.0)
    assert decision.pnl_pct == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize("pct_move", [-0.1, -1.0, -2.9, -5.0, -20.0])
def test_pnl_pct_matches_actual_price_move_at_various_magnitudes(pct_move):
    """pnl_pct must always equal the real fractional price change, at any magnitude."""
    price = 100.0 * (1 + pct_move / 100.0)
    decision = _evaluate(avg_entry=100.0, price=price, held_hours=0.0)
    assert decision.pnl_pct == pytest.approx(pct_move / 100.0, rel=1e-6)


# ── Time-decay stop loss tightening ──

def test_stop_loss_tightens_after_one_hour():
    # 2.6% drop would NOT trip the full 3% stop loss...
    decision = _evaluate(avg_entry=100.0, price=97.4, held_hours=0.5)
    assert decision.exit_reason is None
    # ...but DOES trip the 0.75x-tightened 2.25% stop after 1h held.
    decision = _evaluate(avg_entry=100.0, price=97.4, held_hours=1.5)
    assert decision.exit_reason is not None
    assert "Stop loss" in decision.exit_reason


def test_stop_loss_halves_after_two_hours():
    # 2% drop trips the 0.5x-tightened 1.5% stop after 2h held.
    decision = _evaluate(avg_entry=100.0, price=98.0, held_hours=2.5)
    assert decision.exit_reason is not None
    assert decision.dynamic_sl_pct == pytest.approx(STOP_LOSS_PCT * 0.5)


# ── Trailing stop ──

def test_trailing_stop_triggers_after_profit_target_and_pullback():
    # Price ran up 10% (past the 2% profit target), peaked, then pulled back
    # more than 1% off the peak.
    decision = _evaluate(
        avg_entry=100.0, price=108.8, highest_seen=110.0, held_hours=0.5,
    )
    assert decision.exit_reason is not None
    assert "Trailing Stop" in decision.exit_reason


def test_trailing_stop_does_not_trigger_within_one_percent_of_peak():
    decision = _evaluate(
        avg_entry=100.0, price=109.5, highest_seen=110.0, held_hours=0.5,
    )
    assert decision.exit_reason is None


def test_highest_seen_updates_when_new_price_exceeds_peak():
    decision = _evaluate(avg_entry=100.0, price=112.0, highest_seen=110.0, held_hours=0.5)
    assert decision.highest_seen == 112.0


def test_highest_seen_does_not_regress_when_price_drops():
    decision = _evaluate(avg_entry=100.0, price=105.0, highest_seen=110.0, held_hours=0.5)
    assert decision.highest_seen == 110.0


# ── Max hold time ──

def test_max_hold_time_triggers_when_flat():
    decision = _evaluate(avg_entry=100.0, price=100.5, held_hours=MAX_HOLD_HOURS + 0.1, signal=0.5)
    assert decision.exit_reason is not None
    assert "Max hold time" in decision.exit_reason


def test_max_hold_time_does_not_trigger_early():
    decision = _evaluate(avg_entry=100.0, price=100.5, held_hours=MAX_HOLD_HOURS - 0.1, signal=0.5)
    assert decision.exit_reason is None


# ── Weak-signal exit, gated by minimum hold time ──

def test_weak_signal_does_not_exit_before_min_hold():
    decision = _evaluate(
        avg_entry=100.0, price=100.5,
        held_hours=MIN_HOLD_HOURS_BEFORE_SIGNAL - 0.1,
        signal=SELL_SIGNAL - 0.05,
    )
    assert decision.exit_reason is None


def test_weak_signal_exits_after_min_hold():
    decision = _evaluate(
        avg_entry=100.0, price=100.5,
        held_hours=MIN_HOLD_HOURS_BEFORE_SIGNAL + 0.1,
        signal=SELL_SIGNAL - 0.05,
    )
    assert decision.exit_reason is not None
    assert "Signal weak" in decision.exit_reason


def test_strong_signal_keeps_holding():
    decision = _evaluate(
        avg_entry=100.0, price=100.5,
        held_hours=MIN_HOLD_HOURS_BEFORE_SIGNAL + 0.1,
        signal=SELL_SIGNAL + 0.1,
    )
    assert decision.exit_reason is None
