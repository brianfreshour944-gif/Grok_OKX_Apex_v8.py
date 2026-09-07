"""
Stress Test: 3-10% Market Drops
================================
Diagnoses how the bot's exit logic (exit_logic.evaluate_exit) handles various
market drop magnitudes and shapes. Generates synthetic OHLCV data, simulates
price drops of 3%, 5%, 7%, 10% (and intermediate values), then runs the REAL
exit logic against the simulated price path.

Key scenarios:
  1. Gradual drop — tests trailing stop and time-decay stop behavior
  2. Flash drop — tests whether stops can react fast enough
  3. Drop then recovery — tests early-exit vs. holding-thru-the-valley
  4. Position at various hold times — tests the time-decay stop tightening
  5. Portfolio-level correlated crash — tests max_portfolio_value cap interaction

Run: python stress_test_market_drops.py
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config import (
    PROFIT_TARGET_PCT, STOP_LOSS_PCT, SELL_SIGNAL,
    MAX_HOLD_HOURS, MIN_HOLD_HOURS_BEFORE_SIGNAL,
    TRAILING_STOP_ATR_MULTIPLIER, MIN_TRAILING_STOP_PCT, MAX_TRAILING_STOP_PCT,
    get_regime_params, fmt_price,
)
from exit_logic import evaluate_exit, ExitDecision
from regime import compute_regime_and_trend, calculate_adjusted_risk
from money import pct_change_x100


# ── Synthetic OHLCV generator ──────────────────────────────────────────────

def make_synthetic_df(
    start_price: float = 100.0,
    drop_pct: float = -5.0,
    duration_steps: int = 40,
    drop_at_step: int = 10,
    recovery: bool = False,
    noise_pct: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame simulating a market drop.

    Args:
        start_price: Initial price
        drop_pct: Total drop percentage (e.g., -5.0 for -5%)
        duration_steps: Total number of timesteps (bars)
        drop_at_step: At which step the drop begins
        recovery: If True, price recovers after the drop
        noise_pct: Per-step random noise as percentage of price
        seed: RNG seed for reproducibility

    Returns:
        DataFrame with columns: open, high, low, close, volume, vwap, trade_count
    """
    rng = np.random.default_rng(seed)

    rows = []
    price = start_price
    peak_price = start_price

    for i in range(duration_steps):
        # Determine the "target" price at this step
        if i < drop_at_step:
            target = start_price  # flat before drop
        elif i < drop_at_step + 5:
            # Drop phase: linearly interpolate to target drop
            progress = (i - drop_at_step) / 5
            target = start_price * (1 + drop_pct / 100 * progress)
        elif recovery and i >= drop_at_step + 15:
            # Recovery phase: back to start_price
            recovery_progress = (i - drop_at_step - 15) / 10
            target = price + (start_price - price) * min(1.0, recovery_progress)
        else:
            target = start_price * (1 + drop_pct / 100)

        # Add noise
        noise = rng.normal(0, noise_pct / 100 * price)
        price = max(target + noise, 0.01)

        if price > peak_price:
            peak_price = price

        # Generate OHLC around the price
        spread = price * 0.003  # 0.3% typical spread
        open_p = price + rng.normal(0, spread * 0.5)
        close_p = price + rng.normal(0, spread * 0.5)
        high_p = max(open_p, close_p) + rng.uniform(0, spread)
        low_p = min(open_p, close_p) - rng.uniform(0, spread)

        volume = rng.uniform(800, 1200)  # base volume
        if i >= drop_at_step and i < drop_at_step + 5:
            volume *= 1.5  # higher volume during drop

        rows.append({
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "vwap": (open_p + high_p + low_p + close_p) / 4,
            "trade_count": volume / 100,
        })

    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(start=len(df), stop=0, step=-1)  # descending like real data
    return df


def simulate_price_path(
    start_price: float,
    price_path: List[float],
    held_hours: List[float],
) -> List[ExitDecision]:
    """
    Run evaluate_exit() over a synthetic price path, updating highest_seen
    and returning all decisions.

    Args:
        start_price: Entry price
        price_path: List of prices at each step
        held_hours: Corresponding hold times at each step

    Returns:
        List of ExitDecision objects
    """
    decisions = []
    highest_seen = start_price

    for i, price in enumerate(price_path):
        # Determine regime from a synthetic df
        df = make_synthetic_df(start_price=start_price, drop_pct=-1.0)

        regime, trend, atr_pct = compute_regime_and_trend(df)
        regime_params = get_regime_params(regime)

        decision = evaluate_exit(
            avg_entry=start_price,
            price=price,
            highest_seen=highest_seen,
            held_hours=held_hours[i],
            signal=0.50,
            regime=regime,
            atr_pct=atr_pct,
            profit_target_pct=regime_params["profit_target_pct"],
            stop_loss_pct=regime_params["stop_loss_pct"],
            sell_signal=regime_params["sell_signal"],
            max_hold_hours=MAX_HOLD_HOURS,
            min_hold_hours_before_signal=MIN_HOLD_HOURS_BEFORE_SIGNAL,
            trailing_stop_atr_multiplier=TRAILING_STOP_ATR_MULTIPLIER,
            min_trailing_stop_pct=MIN_TRAILING_STOP_PCT,
            max_trailing_stop_pct=MAX_TRAILING_STOP_PCT,
        )
        decisions.append(decision)
        highest_seen = decision.highest_seen

    return decisions


# ── Test scenarios ──────────────────────────────────────────────────────────

@dataclass
class DropTestResult:
    drop_pct: float
    scenario: str
    final_exit: Optional[ExitDecision]
    exit_step: Optional[int]
    exit_price: Optional[float]
    max_pnl_pct: float
    min_pnl_pct: float
    passed: bool
    note: str = ""


def test_single_position_drop(
    drop_pct: float,
    scenario_name: str,
    held_hours: float = 0.5,
    noise: float = 0.5,
) -> DropTestResult:
    """
    Test a single position going through a specific drop magnitude.
    Returns the result with exit info.
    """
    start_price = 100.0

    # Build a price path: flat for 10 steps, then drop over 20 steps
    n_flat = 10
    n_drop = 30
    total_steps = n_flat + n_drop

    price_path = []
    hours_path = []

    for i in range(total_steps):
        if i < n_flat:
            price_path.append(start_price * (1 + np.random.default_rng(42 + i).normal(0, noise / 1000)))
            hours_path.append(held_hours + i * 0.02)
        else:
            progress = (i - n_flat) / n_drop
            target = start_price * (1 + drop_pct / 100 * progress)
            noise_val = np.random.default_rng(42 + i).normal(0, noise / 100 * target)
            price_path.append(target + noise_val)
            hours_path.append(held_hours + i * 0.02)

    decisions = simulate_price_path(start_price, price_path, hours_path)

    exit_step = None
    exit_price = None
    final_decision = None

    for i, d in enumerate(decisions):
        if d.exit_reason is not None:
            exit_step = i
            exit_price = price_path[i]
            final_decision = d
            break

    if final_decision is None:
        final_decision = decisions[-1]

    pnls = [d.pnl_pct for d in decisions]
    max_pnl = max(pnls)
    min_pnl = min(pnls)

    return DropTestResult(
        drop_pct=drop_pct,
        scenario=scenario_name,
        final_exit=final_decision,
        exit_step=exit_step,
        exit_price=exit_price,
        max_pnl_pct=max_pnl,
        min_pnl_pct=min_pnl,
        passed=True,  # We're diagnosing, not passing/failing
        note=f"Exit at step {exit_step}, price {fmt_price(exit_price) if exit_price else 'N/A'}",
    )


def test_drop_range(
    drops: List[float],
    held_hours: float = 0.5,
    noise: float = 0.3,
) -> List[DropTestResult]:
    """Run tests across a range of drop magnitudes."""
    results = []
    for drop in drops:
        result = test_single_position_drop(
            drop_pct=drop,
            scenario_name=f"Single position, {drop:+.1f}% drop, held {held_hours}h",
            held_hours=held_hours,
            noise=noise,
        )
        results.append(result)
    return results


def test_hold_time_sensitivity():
    """
    Test how the time-decay stop loss triggers at different hold times
    for the same drop magnitude.
    """
    print("\n" + "=" * 70)
    print("TEST: Hold-time sensitivity to 5% single-sided drop")
    print("=" * 70)

    drop_pct = -5.0
    hold_times = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    results = []

    for ht in hold_times:
        r = test_single_position_drop(
            drop_pct=drop_pct,
            scenario_name=f"Hold {ht:.1f}h",
            held_hours=ht,
            noise=0.2,
        )
        results.append(r)
        exit_info = f"  exited at {r.exit_step} steps, price={fmt_price(r.exit_price) if r.exit_price else 'N/A'}"
        if r.final_exit.exit_reason:
            exit_info = f"  → {r.final_exit.exit_reason}"
        print(f"  Held {ht:.1f}h: min_pnl={r.min_pnl_pct*100:+.2f}%  {exit_info}")

    return results


def test_flash_crash():
    """
    Test a rapid drop over a few steps — can stops react fast enough?
    """
    print("\n" + "=" * 70)
    print("TEST: Flash crash (5% drop in 3 steps, 0.3% noise)")
    print("=" * 70)

    start_price = 100.0
    drop_pct = -5.0
    n_flat = 5
    n_drop = 3
    total = n_flat + n_drop

    price_path = []
    hours_path = []
    for i in range(total):
        if i < n_flat:
            price_path.append(start_price)
            hours_path.append(0.5 + i * 0.02)
        else:
            progress = (i - n_flat + 1) / n_drop
            target = start_price * (1 + drop_pct / 100 * progress)
            price_path.append(target)
            hours_path.append(0.5 + i * 0.02)

    decisions = simulate_price_path(start_price, price_path, hours_path)

    for i, (price, d) in enumerate(zip(price_path, decisions)):
        marker = f"  EXIT: {d.exit_reason}" if d.exit_reason else ""
        print(f"  Step {i}: price={fmt_price(price)}  pnl={d.pnl_pct*100:+.2f}%  "
              f"trailing_stop={d.trailing_stop_pct*100:.2f}%  {marker}")

    return decisions


def test_drop_then_recovery():
    """
    Test: -10% crash then 24h recovery. Does the bot exit on the stop
    or hold through and catch the bounce?
    """
    print("\n" + "=" * 70)
    print("TEST: 10% crash then recovery (bot exits early vs. holds through valley)")
    print("=" * 70)

    start_price = 100.0
    # 30 steps of drop, 60 steps of recovery
    total = 90
    price_path = []
    hours_path = []

    for i in range(total):
        hours_path.append(i * 0.25)
        if i < 15:
            # Drop phase: 0 to -10%
            progress = i / 15
            factor = 1 + (-10.0 / 100) * progress
        elif i < 75:
            # Recovery phase: -10% back to 0% (+0.25% per step)
            progress = (i - 15) / 60
            factor = 0.90 + 0.10 * progress
        else:
            factor = 1.0

        price_path.append(start_price * factor)

    decisions = simulate_price_path(start_price, price_path, hours_path)

    exited = False
    for i, (price, d) in enumerate(zip(price_path, decisions)):
        if d.exit_reason and not exited:
            print(f"  Step {i:3d} ({hours_path[i]:.1f}h): price={fmt_price(price)} "
                  f"→ {d.exit_reason}")
            exited = True
        elif not exited and i % 10 == 0:
            print(f"  Step {i:3d} ({hours_path[i]:.1f}h): price={fmt_price(price)} "
                  f"pnl={d.pnl_pct*100:+.2f}% ({d.exit_reason or 'holding'})")

    if not exited:
        final = decisions[-1]
        print(f"  No exit triggered. Final pnl={final.pnl_pct*100:+.2f}%")

    return decisions


def test_trailing_stop_vs_moving_stop():
    """
    Test: position is up 8%, then drops. Trailing stop (scaled by ATR)
    should catch the drop, while time-decay stop is irrelevant (was +8%,
    stop_loss is -3%).
    """
    print("\n" + "=" * 70)
    print("TEST: Trailing stop catches 8%→drawdown after profit")
    print("=" * 70)

    start_price = 100.0
    price_path = [100, 102, 105, 108, 110, 109, 107, 105, 103, 101, 99, 97, 95]
    hours_path = [0.1 * i for i in range(len(price_path))]

    decisions = simulate_price_path(start_price, price_path, hours_path)

    for i, (price, d) in enumerate(zip(price_path, decisions)):
        trailing_stop_price = d.highest_seen * (1 - d.trailing_stop_pct) if d.highest_seen > 0 else 0
        marker = f"  EXIT: {d.exit_reason}" if d.exit_reason else ""
        print(f"  Step {i}: price={fmt_price(price)}  highest={fmt_price(d.highest_seen)}  "
              f"trailing_stop={fmt_price(trailing_stop_price)}  pnl={d.pnl_pct*100:+.2f}%  {marker}")

    return decisions


def test_portfolio_correlated_crash():
    """
    Test: all positions drop 7% simultaneously.
    How does the portfolio cap (MAX_OPEN_POSITIONS, position value limits)
    interact with mass exits?
    """
    print("\n" + "=" * 70)
    print("TEST: Portfolio-level correlated 7% crash (5 positions)")
    print("=" * 70)

    start_price = 100.0
    drop_pct = -7.0
    n_positions = 5

    start_prices = [100, 200, 50, 300, 80]
    results = []

    for idx, sp in enumerate(start_prices):
        price_path = []
        n_flat = 8
        n_drop = 20

        for i in range(n_flat + n_drop):
            if i < n_flat:
                price_path.append(sp * (1 + np.random.default_rng(100 + idx + i).normal(0, 0.3 / 1000)))
            else:
                progress = (i - n_flat) / n_drop
                target = sp * (1 + drop_pct / 100 * progress)
                noise = np.random.default_rng(100 + idx + i).normal(0, 0.3 / 100 * target)
                price_path.append(target + noise)

        hours_path = [0.5 + i * 0.05 for i in range(len(price_path))]
        decisions = simulate_price_path(sp, price_path, hours_path)

        exit_step = None
        exit_pnl = None
        for i, d in enumerate(decisions):
            if d.exit_reason:
                exit_step = i
                exit_pnl = d.pnl_pct
                break

        final = decisions[-1]
        if exit_pnl is None:
            exit_pnl = final.pnl_pct
        results.append((idx + 1, exit_step, exit_pnl))

        print(f"  Position {idx+1} (entry={fmt_price(sp):>7s}): "
              f"exited at step {exit_step}, final pnl={exit_pnl*100:+.2f}%")

    worst_pnl = min(r[2] for r in results)
    print(f"\n  Worst position PnL: {worst_pnl*100:+.2f}%")
    print(f"  Average position PnL: {sum(r[2] for r in results) / len(results) * 100:+.2f}%")

    return results


def test_atr_volatility_impact():
    """
    Test: same 5% drop but at different ATR levels — the trailing stop
    scales with ATR, so higher volatility = wider stop = later exit.
    """
    print("\n" + "=" * 70)
    print("TEST: ATR volatility impact on 5% drop exit timing")
    print("=" * 70)

    start_price = 100.0
    drop_pct = -5.0
    n_flat = 8
    n_drop = 20

    price_path = []
    for i in range(n_flat + n_drop):
        if i < n_flat:
            price_path.append(start_price)
        else:
            progress = (i - n_flat) / n_drop
            target = start_price * (1 + drop_pct / 100 * progress)
            price_path.append(target)

    hours_path = [0.5 + i * 0.05 for i in range(len(price_path))]

    # Test with different ATR levels
    atr_levels = [1.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    for atr in atr_levels:
        decisions = []
        highest_seen = start_price

        for i, price in enumerate(price_path):
            # Synthetic df for regime
            df = make_synthetic_df(start_price=start_price, drop_pct=-1.0)
            regime, trend, _ = compute_regime_and_trend(df)
            # Override ATR
            regime_params = get_regime_params(regime)

            d = evaluate_exit(
                avg_entry=start_price,
                price=price,
                highest_seen=highest_seen,
                held_hours=hours_path[i],
                signal=0.50,
                regime=regime,
                atr_pct=atr,
                profit_target_pct=regime_params["profit_target_pct"],
                stop_loss_pct=regime_params["stop_loss_pct"],
                sell_signal=regime_params["sell_signal"],
                max_hold_hours=MAX_HOLD_HOURS,
                min_hold_hours_before_signal=MIN_HOLD_HOURS_BEFORE_SIGNAL,
                trailing_stop_atr_multiplier=TRAILING_STOP_ATR_MULTIPLIER,
                min_trailing_stop_pct=MIN_TRAILING_STOP_PCT,
                max_trailing_stop_pct=MAX_TRAILING_STOP_PCT,
            )
            decisions.append(d)
            highest_seen = d.highest_seen

            if d.exit_reason:
                print(f"  ATR={atr:5.1f}%: exited at step {i}, price={fmt_price(price)}  "
                      f"trailing_stop={d.trailing_stop_pct*100:.2f}%  {d.exit_reason}")
                break
        else:
            final = decisions[-1]
            print(f"  ATR={atr:5.1f}%: NO EXIT, final pnl={final.pnl_pct*100:+.2f}%  "
                  f"trailing_stop={decisions[-1].trailing_stop_pct*100:.2f}%")


# ── Main runner ─────────────────────────────────────────────────────────────

def print_result_table(results: List[DropTestResult]):
    """Pretty-print a table of results."""
    print(f"\n{'Drop %':>8} | {'Scenario':<45} | {'Exit Step':>10} | {'Exit Price':>10} | {'Min PnL':>8} | {'Max PnL':>8}")
    print("-" * 100)
    for r in results:
        exit_step = str(r.exit_step) if r.exit_step is not None else "N/A"
        exit_price = fmt_price(r.exit_price) if r.exit_price else "N/A"
        print(f"{r.drop_pct:>7.1f}% | {r.scenario:<45} | {exit_step:>10} | {exit_price:>10} | "
              f"{r.min_pnl_pct*100:>7.2f}% | {r.max_pnl_pct*100:>7.2f}%")


def main():
    print("=" * 70)
    print(" STRESS TEST: 3-10% Market Drop Diagnosis")
    print("=" * 70)

    # Test 1: Drop range at different hold times
    print("\n--- Drop range at 0.5h held ---")
    drops = [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
    results_05h = test_drop_range(drops, held_hours=0.5, noise=0.3)
    print_result_table(results_05h)

    print("\n--- Drop range at 2.0h held (time-decay stop is tighter) ---")
    results_2h = test_drop_range(drops, held_hours=2.0, noise=0.3)
    print_result_table(results_2h)

    print("\n--- Drop range at 4.0h held (max hold time approaching) ---")
    results_4h = test_drop_range(drops, held_hours=4.0, noise=0.3)
    print_result_table(results_4h)

    # Test 2: Hold-time sensitivity
    test_hold_time_sensitivity()

    # Test 3: Flash crash
    test_flash_crash()

    # Test 4: Drop then recovery
    test_drop_then_recovery()

    # Test 5: Trailing stop vs moving stop
    test_trailing_stop_vs_moving_stop()

    # Test 6: Portfolio correlated crash
    test_portfolio_correlated_crash()

    # Test 7: ATR impact
    test_atr_volatility_impact()

    # Summary analysis
    print("\n" + "=" * 70)
    print(" SUMMARY: Exit behavior across drop magnitudes (0.5h held)")
    print("=" * 70)

    for r in results_05h:
        exit_type = "NO EXIT"
        if r.exit_step is not None:
            if r.final_exit.exit_reason and "Trailing" in r.final_exit.exit_reason:
                exit_type = "TRAILING STOP"
            elif r.final_exit.exit_reason and "Stop loss" in r.final_exit.exit_reason:
                exit_type = "TIME-DECAY STOP"
            elif r.final_exit.exit_reason and "Max hold" in r.final_exit.exit_reason:
                exit_type = "MAX HOLD"
            elif r.final_exit.exit_reason and "Signal" in r.final_exit.exit_reason:
                exit_type = "SIGNAL EXIT"
            elif r.final_exit.exit_reason:
                exit_type = "OTHER"

        print(f"  {r.drop_pct:>7.1f}% drop: exited={exit_type:<18} at step={r.exit_step}  "
              f"min_pnl={r.min_pnl_pct*100:>7.2f}%")

    # Key insights
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)

    # At what drop % does the bot typically exit?
    for results_list, label in [(results_05h, "0.5h held"), (results_2h, "2.0h held"), (results_4h, "4.0h held")]:
        print(f"\n  [{label}]")
        for r in results_list:
            if r.exit_step is not None:
                print(f"    {r.drop_pct:>7.1f}% → exits at step {r.exit_step}")
            else:
                print(f"    {r.drop_pct:>7.1f}% → NO EXIT (held through full drop)")

    # Trailing stop effectiveness
    print("\n  Trailing stop analysis (from trailing_stop_vs_moving test):")
    decisions = test_trailing_stop_vs_moving_stop()
    for i, d in enumerate(decisions):
        if d.exit_reason and 'Trailing' in d.exit_reason:
            print(f"    Trailing stop triggered at step {i}, pnl={d.pnl_pct*100:+.2f}%, "
                  f"stop={d.trailing_stop_pct*100:.2f}%")
            break

    print("\n  ATR sensitivity (5% drop):")
    test_atr_volatility_impact()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
