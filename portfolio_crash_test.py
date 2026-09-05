"""
Portfolio-level correlated crash test.

Scenario-1 (from your earlier single-position test) checks how ONE position
behaves as price falls. This script checks how the WHOLE PORTFOLIO behaves
when ALL positions drop by the same percentage at the same time (i.e. the
market-wide, fully-correlated crash your bot currently has no defense against).

It reuses the same per-position exit logic (time-decay stop, trailing stop,
max hold, signal decay) so results are apples-to-apples with your earlier
single-position table, then aggregates PnL across the portfolio and checks
max portfolio drawdown against a threshold.

Wiring in your real bot logic:
    The `check_exit()` below mirrors your real exit_logic.evaluate_exit()
    signature. To use the real function, uncomment the import and replace the
    stub body. Everything else (crash generator, portfolio aggregation,
    drawdown, pass/fail, report) works unchanged as long as check_exit()
    keeps the same signature and return shape.
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics


# ---------------------------------------------------------------------------
# 0. CONFIG — mirrors config.py constants your bot actually uses
# ---------------------------------------------------------------------------

MAX_HOLD_HOURS             = 4.0
MIN_HOLD_HOURS_BEFORE_SIGNAL = 0.5
STOP_LOSS_PCT              = 0.03    # 3%
PROFIT_TARGET_PCT          = 0.06    # 6%
SELL_SIGNAL                = 0.40    # trigger signal-exit when signal drops below this
TRAILING_STOP_ATR_MULTIPLIER   = 0.5
MIN_TRAILING_STOP_PCT        = 0.005  # 0.5%
MAX_TRAILING_STOP_PCT        = 0.03   # 3%

# Regime offsets (from config.py get_regime_params)
REGIME_OFFSETS = {
    "wild":   {"tp": 0.03, "sl": 0.045, "sell_offset": -0.03},
    "normal": {"tp": PROFIT_TARGET_PCT, "sl": STOP_LOSS_PCT, "sell_offset": 0},
    "quiet":  {"tp": 0.015, "sl": 0.02, "sell_offset": 0.02},
}


# ---------------------------------------------------------------------------
# 1. PER-POSITION EXIT LOGIC
#    This mirrors your real exit_logic.evaluate_exit().
#    Set USE_REAL_EXIT_LOGIC = True to import the real one instead.
# ---------------------------------------------------------------------------

USE_REAL_EXIT_LOGIC = False  # flip to True to use your real evaluate_exit

if USE_REAL_EXIT_LOGIC:
    from exit_logic import evaluate_exit as _real_evaluate_exit
    from money import pct_change_x100

    def check_exit(
        hours_held: float,
        pnl_pct: float,        # fractional PnL, e.g. -0.03 == -3%
        avg_entry: float,
        price: float,
        highest_seen: float,
        signal: float,
        regime: str = "normal",
        atr_pct: float = 1.5,
    ) -> Optional[str]:
        """Delegates to your real exit_logic.evaluate_exit()."""
        params = REGIME_OFFSETS.get(regime, REGIME_OFFSETS["normal"])
        decision = _real_evaluate_exit(
            avg_entry=avg_entry,
            price=price,
            highest_seen=highest_seen,
            held_hours=hours_held,
            signal=signal,
            regime=regime,
            atr_pct=atr_pct,
            profit_target_pct=params["tp"],
            stop_loss_pct=params["sl"],
            sell_signal=SELL_SIGNAL + params["sell_offset"],
            max_hold_hours=MAX_HOLD_HOURS,
            min_hold_hours_before_signal=MIN_HOLD_HOURS_BEFORE_SIGNAL,
            trailing_stop_atr_multiplier=TRAILING_STOP_ATR_MULTIPLIER,
            min_trailing_stop_pct=MIN_TRAILING_STOP_PCT,
            max_trailing_stop_pct=MAX_TRAILING_STOP_PCT,
        )
        return decision.exit_reason
else:
    def time_decay_stop_pct(hours_held: float, stop_loss_pct: float = STOP_LOSS_PCT) -> float:
        """Stop-loss tightens over time: -3% -> -2.25% @1h -> -1.5% @2h+."""
        if hours_held >= 2.0:
            return stop_loss_pct * 0.5
        if hours_held >= 1.0:
            return stop_loss_pct * 0.75
        return stop_loss_pct

    def check_exit(
        hours_held: float,
        pnl_pct: float,
        avg_entry: float = 100.0,
        price: float = 100.0,
        highest_seen: float = 100.0,
        signal: float = 0.50,
        regime: str = "normal",
        atr_pct: float = 1.5,
    ) -> Optional[str]:
        """
        Returns exit reason string if the position should be closed this step,
        else None. pnl_pct is fractional (e.g. -0.03 == -3% loss).
        """
        params = REGIME_OFFSETS.get(regime, REGIME_OFFSETS["normal"])
        stop_loss_pct = params["sl"]

        stop = time_decay_stop_pct(hours_held, stop_loss_pct)
        if pnl_pct <= -stop:
            return f"STOP_LOSS (time-decay at -{stop*100:.1f}%)"

        if hours_held >= MIN_HOLD_HOURS_BEFORE_SIGNAL and signal < SELL_SIGNAL + params["sell_offset"]:
            return "SIGNAL_EXIT"

        if hours_held >= MAX_HOLD_HOURS:
            return "MAX_HOLD"

        return None


# ---------------------------------------------------------------------------
# 2. PORTFOLIO / CRASH MODEL
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    weight: float          # fraction of portfolio capital, e.g. 0.10 = 10%
    entry_price: float = 100.0
    hours_held: float = 0.0
    closed: bool = False
    exit_reason: Optional[str] = None
    exit_pnl_pct: Optional[float] = None
    pnl_history: list = field(default_factory=list)  # (hours, pnl_pct)


def run_correlated_crash(
    positions: list[Position],
    crash_schedule: list[tuple[float, float]],
    # crash_schedule: list of (hours_elapsed, market_drop_pct_at_that_point)
    # e.g. [(0.5, -2), (1.0, -3), (1.5, -5), (2.0, -8), ...]
    fill_delay_steps: int = 0,
    # fill_delay_steps > 0 simulates order-execution lag: once an exit signal
    # fires at step i, the position doesn't actually flatten (and stops
    # accruing further loss beyond that) until `fill_delay_steps` steps
    # later, using that later step's price. Set to 0 for instant fills.
):
    """
    Applies the SAME drop % to every open position at every step
    (fully correlated crash: no diversification benefit). Positions that
    have already exited stop tracking; their PnL is frozen at exit.
    """
    pending_exit_idx = {}  # pos.symbol -> index in schedule when exit signal fired

    for i, (hours_elapsed, market_drop_pct) in enumerate(crash_schedule):
        for pos in positions:
            if pos.closed:
                continue
            pos.hours_held = hours_elapsed
            # crash_schedule uses percentage values (e.g. -5 for -5%); check_exit
            # expects fractional PnL (e.g. -0.05)
            pnl_pct = market_drop_pct / 100.0
            pos.pnl_history.append((hours_elapsed, market_drop_pct))

            # if an exit signal already fired earlier and its delay has elapsed, fill now
            if pos.symbol in pending_exit_idx and i >= pending_exit_idx[pos.symbol] + fill_delay_steps:
                pos.closed = True
                pos.exit_pnl_pct = market_drop_pct  # filled at THIS step's (worse) price
                continue

            if pos.symbol not in pending_exit_idx:
                reason = check_exit(
                    hours_held=pos.hours_held,
                    pnl_pct=pnl_pct,
                    avg_entry=pos.entry_price,
                    price=pos.entry_price * (1 + pnl_pct),
                    highest_seen=max(pos.entry_price, pos.entry_price * (1 + pnl_pct)),
                    signal=0.50,
                    regime="normal",
                    atr_pct=1.5,
                )
                if reason:
                    pos.exit_reason = reason
                    pending_exit_idx[pos.symbol] = i
                    if fill_delay_steps == 0:
                        pos.closed = True
                        pos.exit_pnl_pct = market_drop_pct

    # Any position never triggered an exit within the schedule: mark as
    # "still open" at the final drop level (worst case for the report).
    final_drop = crash_schedule[-1][1]
    for pos in positions:
        if not pos.closed:
            pos.exit_pnl_pct = final_drop
            pos.exit_reason = "STILL OPEN at end of schedule"

    return positions


def portfolio_equity_curve(positions: list[Position], crash_schedule: list[tuple[float, float]]):
    """
    Builds portfolio-level equity (%) over time, weighting each position's
    PnL at each timestep by its capital weight. Once a position has exited,
    its contribution is frozen at its exit PnL (capital is 'safe' / in cash
    from that point on, earning 0%).
    """
    curve = []
    for hours_elapsed, _ in crash_schedule:
        total_pnl_pct = 0.0
        for pos in positions:
            # find pnl at or before this timestep
            pnl_at_t = None
            for h, p in pos.pnl_history:
                if h <= hours_elapsed:
                    pnl_at_t = p
            if pos.closed and pos.exit_pnl_pct is not None:
                # once closed, frozen at exit PnL for anything after exit time
                exit_time = next((h for h, p in pos.pnl_history if p == pos.exit_pnl_pct), pos.pnl_history[-1][0])
                if hours_elapsed >= exit_time:
                    pnl_at_t = pos.exit_pnl_pct
            if pnl_at_t is None:
                pnl_at_t = 0.0
            total_pnl_pct += pos.weight * pnl_at_t
        curve.append((hours_elapsed, total_pnl_pct))
    return curve


def max_drawdown(curve: list[tuple[float, float]]) -> float:
    """Max drawdown of the portfolio equity curve, in percent (positive number)."""
    peak = 0.0
    worst = 0.0
    for _, pnl in curve:
        peak = max(peak, pnl)
        drawdown = peak - pnl
        worst = max(worst, drawdown)
    return worst


# ---------------------------------------------------------------------------
# 3. TEST HARNESS
# ---------------------------------------------------------------------------

def make_equal_weighted_portfolio(symbols: list[str]) -> list[Position]:
    n = len(symbols)
    return [Position(symbol=s, weight=1.0 / n) for s in symbols]


def run_test(
    symbols: list[str],
    crash_schedule: list[tuple[float, float]],
    max_drawdown_threshold_pct: float,
    label: str,
    weights: Optional[list[float]] = None,
    fill_delay_steps: int = 0,
):
    if weights:
        positions = [Position(symbol=s, weight=w) for s, w in zip(symbols, weights)]
    else:
        positions = make_equal_weighted_portfolio(symbols)
    positions = run_correlated_crash(positions, crash_schedule, fill_delay_steps=fill_delay_steps)
    curve = portfolio_equity_curve(positions, crash_schedule)
    dd = max_drawdown(curve)
    passed = dd <= max_drawdown_threshold_pct

    print(f"\n{'='*70}\nTEST: {label}\n{'='*70}")
    print(f"{'Symbol':<8}{'Weight':>8}{'Exit @ hr':>12}{'Exit PnL':>12}   Reason")
    for pos in positions:
        exit_hr = next((h for h, p in pos.pnl_history if p == pos.exit_pnl_pct), crash_schedule[-1][0])
        print(f"{pos.symbol:<8}{pos.weight*100:>7.1f}%{exit_hr:>11.2f}h{pos.exit_pnl_pct:>11.2f}%   {pos.exit_reason}")

    print(f"\nPortfolio equity curve (hours, cumulative PnL %):")
    for h, p in curve:
        bar = "#" * int(abs(p))
        print(f"  t={h:>4.1f}h  {p:>7.2f}%  {bar}")

    print(f"\nMax portfolio drawdown: {dd:.2f}%")
    print(f"Threshold:              {max_drawdown_threshold_pct:.2f}%")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return {"label": label, "max_drawdown": dd, "threshold": max_drawdown_threshold_pct, "passed": passed, "positions": positions, "curve": curve}


# ---------------------------------------------------------------------------
# 4. SCENARIOS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    symbols_10 = [f"ASSET{i}" for i in range(1, 11)]  # 10-position equal-weight portfolio

    # Scenario A: same progressive crash as your Scenario 1, but ALL 10
    # positions drop together instead of testing one in isolation.
    progressive_schedule = [
        (0.5, -2), (1.0, -3), (1.5, -5), (2.0, -8),
        (2.5, -10), (3.0, -15), (3.5, -20), (4.0, -25),
    ]

    # Scenario B: fast/flash crash - same total drop as scenario A's early
    # steps but compressed into minutes instead of hours, to test whether
    # time-decay stops can react fast enough.
    flash_schedule = [
        (0.05, -5), (0.10, -10), (0.15, -15), (0.20, -20),
    ]

    results = []
    results.append(run_test(
        symbols_10, progressive_schedule,
        max_drawdown_threshold_pct=10.0,
        label="10-position equal-weight, correlated progressive crash (hours)",
    ))
    results.append(run_test(
        symbols_10, flash_schedule,
        max_drawdown_threshold_pct=10.0,
        label="10-position equal-weight, correlated FLASH crash (minutes)",
    ))

    # Scenario C: STRESS TEST - concentrated portfolio (one position is 40%
    # of capital) + 2-step order-fill delay (simulates slippage / exchange
    # lag during a fast, correlated sell-off, when everyone is trying to
    # exit at once and fills don't happen instantly).
    concentrated_weights = [0.40, 0.10, 0.10, 0.10, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04]
    results.append(run_test(
        symbols_10, flash_schedule,
        max_drawdown_threshold_pct=10.0,
        label="STRESS: 40%-concentrated portfolio, flash crash, 2-step fill delay",
        weights=concentrated_weights,
        fill_delay_steps=2,
    ))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['label']:<60} DD={r['max_drawdown']:>6.2f}%  {status}")
