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

--------------------------------------------------------------------------
HOW TO WIRE IN YOUR REAL BOT LOGIC
--------------------------------------------------------------------------
Replace `check_exit()` below with a call into your actual bot's exit/stop
logic. Everything else (crash generator, portfolio aggregation, drawdown,
pass/fail, report) will work unchanged as long as `check_exit()` keeps the
same signature and return shape.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# TOGGLE: flip this to True once real_check_exit() below is wired up to
# your actual bot's stop-loss / exit logic. While False, the placeholder
# logic (time-decay stop, signal decay, max hold) is used instead.
# ---------------------------------------------------------------------------
USE_REAL_EXIT_LOGIC = False


# ---------------------------------------------------------------------------
# 1. PER-POSITION EXIT LOGIC (mirrors your Scenario-1 rules)
#    Swap this out for your real bot's stop-loss / exit function.
# ---------------------------------------------------------------------------

def time_decay_stop_pct(hours_held: float) -> float:
    """Stop-loss tightens over time: -3% -> -2.25% @1h -> -1.5% @2h+."""
    if hours_held >= 2.0:
        return -1.5
    if hours_held >= 1.0:
        return -2.25
    return -3.0


def signal_decay(hours_held: float, initial_signal: float = 0.60) -> float:
    """Rough model of ML signal decaying as a position sours (from your data)."""
    return max(0.0, initial_signal - 0.05 * (hours_held / 0.5))


def placeholder_check_exit(hours_held: float, pnl_pct: float, trend: str = "down") -> Optional[str]:
    """
    Returns exit reason string if the position should be closed this step,
    else None. pnl_pct is the position's PnL in percent (negative = loss).
    """
    stop = time_decay_stop_pct(hours_held)
    if pnl_pct <= stop:
        return f"STOP_LOSS (time-decay {stop:.2f}%)"

    sig = signal_decay(hours_held)
    if hours_held >= 0.5 and sig < 0.45:
        return "SIGNAL_EXIT (ML signal < 0.45)"

    if hours_held >= 4.0:
        return "MAX_HOLD (4h forced exit)"

    return None


def real_check_exit(hours_held: float, pnl_pct: float, trend: str = "down") -> Optional[str]:
    """
    *** FILL THIS IN WITH YOUR ACTUAL BOT'S EXIT LOGIC ***

    Same contract as placeholder_check_exit(): given how long the position
    has been held (hours) and its current PnL in percent, return a string
    describing why to exit (e.g. "STOP_LOSS", "SIGNAL_EXIT", "MAX_HOLD"),
    or None if it should stay open.

    If your real function needs more inputs than hours_held/pnl_pct
    (e.g. ATR, ML signal value, regime, trend), add them as parameters
    here AND thread them through in run_correlated_crash() below where
    check_exit(...) is called.

    Example of wiring in an imported bot module:

        from my_bot.risk import get_exit_signal

        def real_check_exit(hours_held, pnl_pct, trend="down"):
            decision = get_exit_signal(
                time_held_hours=hours_held,
                unrealized_pnl_pct=pnl_pct,
                trend=trend,
            )
            return decision.reason if decision.should_exit else None
    """
    raise NotImplementedError(
        "real_check_exit() is not wired up yet. Fill in this function with "
        "your bot's actual exit logic, then it will be used automatically "
        "since USE_REAL_EXIT_LOGIC = True."
    )


def check_exit(hours_held: float, pnl_pct: float, trend: str = "down") -> Optional[str]:
    """Dispatches to real or placeholder exit logic based on the toggle above."""
    if USE_REAL_EXIT_LOGIC:
        return real_check_exit(hours_held, pnl_pct, trend=trend)
    return placeholder_check_exit(hours_held, pnl_pct, trend=trend)


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
            pnl_pct = market_drop_pct  # fully correlated: bot's move = market's move
            pos.pnl_history.append((hours_elapsed, pnl_pct))

            # if an exit signal already fired earlier and its delay has elapsed, fill now
            if pos.symbol in pending_exit_idx and i >= pending_exit_idx[pos.symbol] + fill_delay_steps:
                pos.closed = True
                pos.exit_pnl_pct = pnl_pct  # filled at THIS step's (worse) price
                continue

            if pos.symbol not in pending_exit_idx:
                reason = check_exit(pos.hours_held, pnl_pct)
                if reason:
                    pos.exit_reason = reason
                    pending_exit_idx[pos.symbol] = i
                    if fill_delay_steps == 0:
                        pos.closed = True
                        pos.exit_pnl_pct = pnl_pct

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

    # Scenario D: 10% DROP THEN 24h RECOVERY (matches your recent real market)
    # Crash hits -10% in 30min, then slowly recovers over 24h. Tests whether
    # bot exits too early on the stop-loss and misses the recovery, vs. holding
    # through the valley.
    drop_then_recovery_schedule = [
        (0.01, -2.0),
        (0.02, -4.0),
        (0.03, -6.0),
        (0.04, -8.0),
        (0.05, -10.0),
        (1.0,  -9.0),
        (2.0,  -7.0),
        (4.0,  -5.0),
        (8.0,  -3.0),
        (12.0, -1.5),
        (24.0, -0.5),
    ]
    results.append(run_test(
        symbols_10, drop_then_recovery_schedule,
        max_drawdown_threshold_pct=12.0,
        label="10% drop then 24h recovery (your recent market pattern)",
    ))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['label']:<60} DD={r['max_drawdown']:>6.2f}%  {status}")
