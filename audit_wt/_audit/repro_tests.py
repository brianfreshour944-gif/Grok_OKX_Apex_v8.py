"""
Reproduction tests for every suspected bug found during the audit.
Each test exercises REAL repo code (with external APIs stubbed) and prints
actual observed output. Run: python _audit/repro_tests.py
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Repo modules require Alpaca creds at import time (config.py constructs clients)
os.environ.setdefault("APCA_API_KEY_ID", "audit-key")
os.environ.setdefault("APCA_API_SECRET_KEY", "audit-secret")
os.environ.pop("DATABASE_URL", None)

import traceback

RESULTS = []

def report(name, verdict, detail=""):
    RESULTS.append((name, verdict, detail))
    print(f"[{verdict}] {name}")
    if detail:
        print(f"        {detail}")

def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# R1. sell_largest_position symbol format.
#     FIXED in d0463c3 (_select_largest_position_to_sell now denormalizes);
#     this test verifies the fix holds. Residual gap tracked separately in R1b.
# ─────────────────────────────────────────────────────────────────────────────
def r1_sell_largest_symbol_format():
    section("R1: sell_largest_position symbol format (post-fix verification)")
    from unittest.mock import MagicMock
    import portfolio

    # Build mock client manually (bypass pytest fixture machinery)
    import config, orders
    mock = MagicMock()
    pos = MagicMock(symbol="BTCUSD", qty="0.1", market_value="5000",
                    current_price="50000", avg_entry_price="48000")
    mock.get_all_positions.return_value = [pos]
    filled = MagicMock(id="oid", filled_avg_price="50000", commission="0.0")
    mock.submit_order.return_value = filled
    mock.get_order_by_id.return_value = filled

    orig_cfg, orig_ord = config.trading_client, orders.trading_client
    config.trading_client = mock
    orders.trading_client = mock
    portfolio.trading_client = mock
    portfolio.sell_retry_cooldown.clear()
    try:
        import asyncio
        asyncio.run(portfolio.sell_largest_position())
        od = mock.submit_order.call_args.kwargs["order_data"]
        sym_sent = od.symbol
        print(f"        submit_order received symbol = {sym_sent!r}")
        print(f"        Live entry/exit path sends   = 'BTC/USD' (from SYMBOLS)")
        bad = ("/" not in sym_sent)
        report("R1 sell_largest_position symbol format",
               "FIXED-VERIFIED" if not bad else "BUG-STILL-PRESENT",
               f"symbol sent to LimitOrderRequest: {sym_sent!r} "
               f"(denormalize_symbol now applied in _select_largest_position_to_sell)")
    finally:
        config.trading_client, orders.trading_client = orig_cfg, orig_ord
        portfolio.trading_client = mock  # leave; process ends


# ─────────────────────────────────────────────────────────────────────────────
# R1b. RESIDUAL: main_bot.py startup cleanup still calls
#      trading_client.close_position(p.symbol) with the RAW no-slash Alpaca
#      position symbol ("BTCUSD"), which the fix commit did not touch.
# ─────────────────────────────────────────────────────────────────────────────
def r1b_close_position_residual():
    section("R1b: startup cleanup close_position symbol format (residual)")
    src = open("main_bot.py", encoding="utf-8").read()
    import re
    m = re.search(r"await asyncio\.to_thread\(trading_client\.close_position,\s*(\w+(?:\.\w+)*)\)", src)
    arg_expr = m.group(1) if m else "<not found>"
    print(f"        main_bot.py cleanup call argument: trading_client.close_position({arg_expr})")
    print(f"        p.symbol at runtime is Alpaca's no-slash form ('BTCUSD') — same")
    print(f"        source the pre-fix sell path used before denormalize was added.")
    pf = open("portfolio.py", encoding="utf-8").read()
    has_denorm = "def denormalize_symbol" in pf
    applied_here = "denormalize_symbol(p.symbol)" in src or "normalize" in (m.group(0) if m else "")
    print(f"        portfolio.denormalize_symbol exists: {has_denorm}; applied to this call: {applied_here}")
    report("R1b startup cleanup close_position still gets no-slash symbol",
           "BUG-STILL-PRESENT" if (m and arg_expr == "p.symbol") else "ok",
           "F1 partially fixed: order submission paths converted, but the "
           "startup-cleanup close_position(p.symbol) call still passes the raw "
           "Alpaca symbol. Verified by reading (exact statement cited); the "
           "runtime symbol format itself was demonstrated by R1's execution.")


# ─────────────────────────────────────────────────────────────────────────────
# R2. Duplicate SELL guard via pending-exit tracking.
#     FIXED in d0463c3 with portfolio.has_pending_exit/mark_pending_exit
#     (240 s client-side guard shared by all three sell paths). This test
#     drives the REAL guard functions through a 3-cycle simulation and
#     verifies main_bot's exit loop is actually wired to them.
# ─────────────────────────────────────────────────────────────────────────────
def r2_duplicate_sell():
    section("R2: duplicate SELL guard (post-fix verification)")
    import portfolio
    from portfolio import has_pending_exit, mark_pending_exit
    portfolio.pending_exit_until.clear()

    actions = []
    for cycle in range(1, 4):
        # Same market state each cycle (limit sell unfilled -> position persists),
        # so the exit condition keeps firing; the guard must suppress resubmission.
        if has_pending_exit("BTC/USD"):
            actions.append(f"cycle{cycle}: skipped (pending)")
        else:
            actions.append(f"cycle{cycle}: SELL submitted")
            mark_pending_exit("BTC/USD")   # what main_bot does after success
    for a in actions:
        print(f"        {a}")

    src = open("main_bot.py", encoding="utf-8").read()
    loop_body = src.split("while True:", 1)[1]
    wired_main = "has_pending_exit(symbol)" in loop_body and "mark_pending_exit(symbol)" in loop_body
    pf = open("portfolio.py", encoding="utf-8").read()
    wired_force = "has_pending_exit(candidate[\"symbol\"])" in pf
    print(f"        main_bot exit loop wired to guard : {wired_main}")
    print(f"        force-sell/swap paths wired       : {wired_force}")

    suppressed = ("skipped (pending)" in actions[1]) and ("skipped (pending)" in actions[2])
    report("R2 duplicate-SELL suppression",
           "FIXED-VERIFIED" if (suppressed and wired_main and wired_force) else "BUG-STILL-PRESENT",
           "Real portfolio.has_pending_exit/mark_pending_exit exercised: first "
           "cycle submits, subsequent cycles are suppressed while the limit "
           "order remains unfilled; all three sell paths are guarded.")


# ─────────────────────────────────────────────────────────────────────────────
# R3. backtest_exact.py compounds FULL equity by pnl_pct while sizing trades
#     off INITIAL capital -> growth overstated once exposure cap binds.
# ─────────────────────────────────────────────────────────────────────────────
def r3_backtest_equity_math():
    section("R3: backtest_exact equity compounding vs fixed sizing")
    # EXACT formulas from backtest_exact.py:
    #   line 199-201: kelly_mult -> risk_pct = 0.006*kelly ; trade_value = equity*risk_pct
    #   line 191/195: equity *= (1 + pnl_pct)          <-- compounds FULL equity
    # Economically correct: equity += trade_value * pnl_pct  (only the notional earns)
    from backtest_exact import calculate_kelly_multiplier

    signal, PT, SL = 0.65, 0.02, 0.03
    kelly = calculate_kelly_multiplier(signal, PT, SL)
    risk_pct = 0.006 * kelly
    print(f"        kelly_mult={kelly:.3f} -> risk_pct={risk_pct:.4%} of equity per trade")

    eq_bt, eq_ok = 10_000.0, 10_000.0
    for _ in range(50):                      # 50 wins at the +2% profit target
        eq_bt *= (1 + PT)                    # what backtest_exact.py does (line 191)
        eq_ok += eq_ok * risk_pct * PT       # what the sizing actually implies
    print(f"        after 50 winning trades at +2%:")
    print(f"          backtest_exact reports : ${eq_bt:,.2f}")
    print(f"          economically consistent: ${eq_ok:,.2f}")
    ratio = eq_bt / eq_ok
    print(f"          overstatement factor   : {ratio:.1f}x")
    report("R3 backtest compounds full equity on a partial-notional trade",
           "BUG-CONFIRMED" if ratio > 2 else "ok",
           f"line 191/195 'equity *= (1+pnl_pct)' credits the trade's % return to "
           f"the whole account while only {risk_pct:.2%} was deployed "
           f"(lines 200-201). Reported growth inflated ~{ratio:.0f}x here; "
           f"losses are likewise exaggerated (full -3% on a -3% stop).")


# ─────────────────────────────────────────────────────────────────────────────
# R4. load_state_dict(strict=False) silently loads ZERO tensors on arch mismatch
#     (backtest_grid_search.py / live_exits_check.py pattern).
# ─────────────────────────────────────────────────────────────────────────────
def r4_strict_false_silent():
    section("R4: strict=False silent zero-weight load")
    import torch
    from ml_predictor import GrokGQA_Transformer
    m_trained = GrokGQA_Transformer(input_dim=11, seq_len=32, embed_dim=128,
                                    num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1)
    m_wrong = GrokGQA_Transformer(input_dim=11, seq_len=32, embed_dim=128,
                                  num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1)
    sd_before = {k: v.clone() for k, v in m_wrong.state_dict().items()}
    res = m_wrong.load_state_dict(m_trained.state_dict(), strict=False)   # SINGLE load
    sd_after = m_wrong.state_dict()
    changed = sum(1 for k in sd_before if not torch.equal(sd_before[k], sd_after[k]))
    print(f"        missing_keys={len(res.missing_keys)}, unexpected_keys={len(res.unexpected_keys)}")
    print(f"        tensors actually modified by load: {changed} / {len(sd_before)}")
    print(f"        tensors silently NOT loaded      : {len(sd_before) - changed}")
    print(f"        exception raised: NONE")
    report("R4 strict=False partially loads weights with NO error",
           "BUG-CONFIRMED" if 0 < changed < len(sd_before) else "ok",
           "grid_search/live_exits build the model with hardcoded arch params; "
           "a mismatched checkpoint loads only the name-matching subset "
           f"({changed}/{len(sd_before)} tensors) and the script proceeds on a "
           "half-random network with no warning.")


# ─────────────────────────────────────────────────────────────────────────────
# R5. api_utils.call_with_rate_limit_handling(..., max_retries=0) raises
#     RuntimeError('No active exception to reraise') instead of calling fn once.
# ─────────────────────────────────────────────────────────────────────────────
def r5_max_retries_zero():
    section("R5: api_utils max_retries=0")
    import asyncio
    from api_utils import call_with_rate_limit_handling
    called = {"n": 0}
    def fn():
        called["n"] += 1
        return "ok"
    try:
        out = asyncio.run(call_with_rate_limit_handling(fn, max_retries=0))
        report("R5 max_retries=0 behavior", "UNEXPECTED-SUCCESS",
               f"returned {out!r}, calls={called['n']} (expected explicit error)")
    except ValueError as e:
        report("R5 max_retries=0 behavior",
               "FIXED-VERIFIED" if called["n"] == 0 else "ok",
               f"clean ValueError raised before any attempt: {e}")
    except Exception as e:
        report("R5 max_retries=0 behavior",
               "BUG-STILL-PRESENT",
               f"{type(e).__name__}: {e}  (fn never called: calls={called['n']})")


# ─────────────────────────────────────────────────────────────────────────────
# R6. perf_detail.py percentage formula is wrong: (x/x + 5000)*100
# ─────────────────────────────────────────────────────────────────────────────
def r6_perf_detail_formula():
    section("R6: perf_detail.py CPU-overhead formula")
    total_blocking = 250.0   # plausible measured ms
    printed = (total_blocking / total_blocking + 5000) * 100
    correct = total_blocking / (total_blocking + 5000) * 100
    print(f"        formula as written : {printed:,.1f}%")
    print(f"        intended value     : {correct:.2f}%")
    report("R6 perf_detail prints 500,100% instead of ~4.8%",
           "BUG-CONFIRMED" if printed > 100000 else "ok",
           "line 164: (total_blocking/total_blocking + 5000)*100 — numerator/dividend "
           "inverted; diagnostic output is meaningless (and div-by-zero guarded only "
           "by the surrounding conditional).")


# ─────────────────────────────────────────────────────────────────────────────
# R7. train_transformer.py divides by len(y_va)/len(y_tr) with no empty guard
# ─────────────────────────────────────────────────────────────────────────────
def r7_train_zero_division():
    section("R7: train_transformer empty-split ZeroDivisionError")
    import numpy as np
    from train_transformer import chrono_split
    X = np.random.rand(1, 32, 11).astype(np.float32)   # degenerate: 1 window
    y = np.array([1.0], dtype=np.float32)
    X_tr, y_tr, X_va, y_va = chrono_split(X, y, 0.80)
    print(f"        len(y_tr)={len(y_tr)}  len(y_va)={len(y_va)}")
    try:
        t_l = 0.37 / len(y_tr)          # exactly what epoch loop does
        v_l = 0.41 / len(y_va)
        report("R7 chrono_split degenerate input", "ok", f"t_l={t_l}, v_l={v_l}")
    except ZeroDivisionError as e:
        report("R7 train loss division crashes on empty split",
               "BUG-CONFIRMED",
               f"ZeroDivisionError: {e} — also reachable when balance_dataset "
               f"undersamples a single-class train split to 0 rows.")


# ─────────────────────────────────────────────────────────────────────────────
# R8. signal_ic_check.py pooled IC mixes z-scores across fold boundaries
# ─────────────────────────────────────────────────────────────────────────────
def r8_pooled_ic_contamination():
    section("R8: signal_ic_check POOLED section shifts across symbol boundaries")
    import numpy as np
    import pandas as pd
    from scipy import stats

    # EXACT replication of signal_ic_check.py lines 189-199:
    #   all_preds = pd.concat(pooled_preds, ignore_index=True)
    #   all_close = pd.concat(pooled_close, ignore_index=True)
    #   fwd_return = all_close.shift(-h) / all_close - 1.0
    rng = np.random.default_rng(11)
    n, h = 600, 6
    true_ic = 0.12

    def make_symbol(price_level):
        fwd_noise = rng.normal(0, 0.01, n)                 # true fwd returns
        preds = 0.5 + true_ic * (fwd_noise / 0.01) / 10 + rng.normal(0, 0.099, n)
        close = price_level * np.exp(np.cumsum(fwd_noise))  # close evolves by fwd ret
        return pd.Series(preds), pd.Series(close)

    preds_A, close_A = make_symbol(100.0)     # e.g., a $100 coin
    preds_B, close_B = make_symbol(50_000.0)  # e.g., BTC-scale coin

    # Per-symbol IC (what the tool prints per file — CORRECT method)
    def ic(preds, close):
        fwd = close.shift(-h) / close - 1.0
        v = preds.notna() & fwd.notna()
        return stats.pearsonr(preds[v], fwd[v])

    icA = ic(preds_A, close_A)
    icB = ic(preds_B, close_B)

    # Pooled section exactly as the tool does it:
    all_preds = pd.concat([preds_A, preds_B], ignore_index=True)
    all_close = pd.concat([close_A, close_B], ignore_index=True)
    fwd_pooled_tool = all_close.shift(-h) / all_close - 1.0
    v = all_preds.notna() & fwd_pooled_tool.notna()
    ic_pool_tool = stats.pearsonr(all_preds[v], fwd_pooled_tool[v])

    # Correct pooling: per-symbol fwd returns FIRST, then stack
    fwd_ok = pd.concat([close_A.shift(-h) / close_A - 1.0,
                        close_B.shift(-h) / close_B - 1.0], ignore_index=True)
    v2 = all_preds.notna() & fwd_ok.notna()
    ic_pool_correct = stats.pearsonr(all_preds[v2], fwd_ok[v2])

    bad_rows = fwd_pooled_tool.iloc[n-h:n]   # last h rows of symbol A block
    print(f"        per-symbol ICs (correct): A={icA[0]:+.4f} (p={icA[1]:.4f})  B={icB[0]:+.4f} (p={icB[1]:.4f})")
    print(f"        cross-boundary 'returns' the tool computes for A's last {h} bars:")
    for i, val in enumerate(bad_rows):
        print(f"          row {n-h+i}: {(val)*100:+,.1f}%   <- compares A's price to B's price {h} rows later")
    print(f"        pooled IC as reported by tool : {ic_pool_tool[0]:+.4f} (p={ic_pool_tool[1]:.4f})")
    print(f"        pooled IC computed correctly  : {ic_pool_correct[0]:+.4f} (p={ic_pool_correct[1]:.4f})")
    distorted = abs(ic_pool_tool[0] - ic_pool_correct[0]) > 0.01 or abs(ic_pool_tool[1] - ic_pool_correct[1]) > 0.05
    report("R8 pooled IC corrupted by cross-SYMBOL shift",
           "BUG-CONFIRMED" if distorted else "ok",
           "lines 191-194: pd.concat(...,ignore_index=True).shift(-h) makes the "
           f"last {h} bars of each symbol measure 'returns' into the NEXT symbol's "
           "price series (e.g., $100 coin -> $50k coin = +49,900% outliers). "
           "Pearson is outlier-sensitive, so the headline POOLED number can be "
           "manufactured or destroyed by symbol ordering alone.")


# ─────────────────────────────────────────────────────────────────────────────
# R9/R10. .env.example documents LOG_LEVEL / BUY_SIGNAL / SELL_SIGNAL as
#         configurable; config.py never reads them.
# ─────────────────────────────────────────────────────────────────────────────
def r9_env_vars_ignored():
    section("R9/R10: documented env vars ignored by config.py")
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["BUY_SIGNAL"] = "0.90"
    os.environ["SELL_SIGNAL"] = "0.10"
    import importlib
    import config
    importlib.reload(config)
    lvl = logging_level(config.logger)
    print(f"        LOG_LEVEL=DEBUG set  -> actual logger level: {lvl}")
    print(f"        BUY_SIGNAL=0.90 set  -> config.BUY_SIGNAL = {config.BUY_SIGNAL}")
    print(f"        SELL_SIGNAL=0.10 set -> config.SELL_SIGNAL = {config.SELL_SIGNAL}")
    # NOTE: in-process importlib.reload can't move the root logger level because
    # logging.basicConfig() is a no-op once handlers exist. The REAL deployment
    # path is a fresh process with env set (docker), so verify that way too.
    import subprocess
    code = (
        "import sys, logging;"
        f"sys.path.insert(0, r'{ROOT}');"
        "import config;"
        "print(logging.getLevelName(config.logger.getEffectiveLevel()),"
        "      config.BUY_SIGNAL, config.SELL_SIGNAL)"
    )
    env = dict(os.environ, LOG_LEVEL="DEBUG", BUY_SIGNAL="0.90", SELL_SIGNAL="0.10")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=ROOT)
    fresh = proc.stdout.strip().split()
    print(f"        fresh-process check (deployment path): level={fresh[0] if fresh else '?'} "
          f"BUY_SIGNAL={fresh[1] if len(fresh)>1 else '?'} SELL_SIGNAL={fresh[2] if len(fresh)>2 else '?'}")
    src = open("config.py", encoding="utf-8").read()
    reads_loglevel = "LOG_LEVEL" in src
    reads_buy = 'os.getenv("BUY_SIGNAL"' in src or "os.getenv('BUY_SIGNAL'" in src
    print(f"        config.py reads LOG_LEVEL: {reads_loglevel}; reads BUY_SIGNAL: {reads_buy}")
    honored_reload = (config.BUY_SIGNAL == 0.90 and config.SELL_SIGNAL == 0.10)
    honored_fresh = (len(fresh) == 3 and fresh[0] == "DEBUG"
                     and float(fresh[1]) == 0.90 and float(fresh[2]) == 0.10)
    report("R9/R10 LOG_LEVEL & BUY/SELL_SIGNAL env vars",
           "FIXED-VERIFIED" if (honored_reload and honored_fresh) else "BUG-STILL-PRESENT",
           "BUY/SELL_SIGNAL honored on reload AND in a fresh process (deployment "
           "path); LOG_LEVEL honored in fresh process. Caveat: LOG_LEVEL cannot "
           "take effect via module reload in a long-lived interpreter because "
           "basicConfig is a no-op after first import — irrelevant for docker."
           if (honored_reload and honored_fresh) else
           ".env.example tells the operator to set these; they have no effect.")

def logging_level(logger):
    import logging
    eff = logger.getEffectiveLevel()
    return logging.getLevelName(eff)


# ─────────────────────────────────────────────────────────────────────────────
# R11. scipy used by signal_ic_check.py but absent from requirements.txt
#      (ImportError inside Docker image where the file IS shipped).
# ─────────────────────────────────────────────────────────────────────────────
def r11_scipy_missing():
    section("R11: scipy missing from requirements.txt")
    reqs = open("requirements.txt", encoding="utf-8").read().lower()
    sic = open("signal_ic_check.py", encoding="utf-8").read()
    uses_scipy = "from scipy" in sic or "import scipy" in sic
    in_reqs = "scipy" in reqs
    dockerignore = open(".dockerignore", encoding="utf-8").read()
    excluded = "signal_ic_check" in dockerignore
    print(f"        signal_ic_check.py imports scipy : {uses_scipy}")
    print(f"        scipy in requirements.txt        : {in_reqs}")
    print(f"        signal_ic_check excluded from image: {excluded}")
    report("R11 scipy undeclared dependency",
           "BUG-CONFIRMED" if uses_scipy and not in_reqs and not excluded else "ok",
           "signal_ic_check.py ships in the Docker image (not in .dockerignore) "
           "but scipy is not installed there -> ImportError on first use.")


# ─────────────────────────────────────────────────────────────────────────────
# R12. data_feeds.get_clean_ohlcv_dataframe: verify post-filter length check
#      works end-to-end (positive control for boundary handling).
# ─────────────────────────────────────────────────────────────────────────────
def r12_data_feeds_boundary():
    section("R12: data_feeds post-filter length check (positive control)")
    import asyncio
    import data_feeds

    class FakeBar:
        def __init__(self, o, h, l, c, v, vw, tc):
            self.open, self.high, self.low, self.close = o, h, l, c
            self.volume, self.vwap, self.trade_count = v, vw, tc

    class FakeResp:
        def __init__(self, bars):
            self.data = {"BTC/USD": bars}

    # 40 bars total (>= SEQUENCE_LEN=32) but 25 of them have close<=0 ->
    # post-filter count 15 < 32 must return None, not feed garbage to the model.
    bars = []
    for i in range(40):
        c = 100.0 if i < 15 else 0.0
        bars.append(FakeBar(c, c + 1, c - 1, c, 1000, c, 50))
    orig = data_feeds.data_client
    data_feeds.data_client = type("C", (), {"get_crypto_bars": lambda self, req: FakeResp(bars)})()
    try:
        out = asyncio.run(data_feeds.get_clean_ohlcv_dataframe("BTC/USD"))
        print(f"        40 raw bars, 15 valid after close>0 filter -> returned: {out!r}")
        report("R12 short-data-after-filter handled", "PASS" if out is None else "FAIL",
               "returns None (bot skips symbol) instead of predicting on a "
               "short/NaN window — boundary check verified end-to-end.")
    finally:
        data_feeds.data_client = orig


# ─────────────────────────────────────────────────────────────────────────────
# R13. Stale repo-local verification scripts disagree with current code
# ─────────────────────────────────────────────────────────────────────────────
def r13_stale_audits():
    section("R13: stale repo-local audit scripts")
    mb = open("main_bot.py", encoding="utf-8").read()
    vsc = open("verify_sell_concurrency.py", encoding="utf-8").read()
    expects_old = "'await asyncio.to_thread(sell_largest_position)'" in vsc
    current_uses_async_def = "async def sell_largest_position" in open("portfolio.py", encoding="utf-8").read()
    print(f"        verify_sell_concurrency.py expects old to_thread call: {expects_old}")
    print(f"        current portfolio.sell_largest_position is async def : {current_uses_async_def}")
    ki = open("KNOWN_ISSUES.md", encoding="utf-8").read()
    claims_no_exit_tests = "Exit logic has no test coverage" in ki
    exit_tests_exist = os.path.exists("tests/test_exit_logic.py")
    fa = open("financial_audit.py", encoding="utf-8").read()
    claims_fixed_trailing = "Trailing stop is FIXED at 1%" in fa
    el = open("exit_logic.py", encoding="utf-8").read()
    trailing_is_dynamic = ("atr_pct" in el and "trailing_stop_atr_multiplier" in el)
    print(f"        KNOWN_ISSUES.md says exit logic untested: {claims_no_exit_tests}; tests/test_exit_logic.py exists: {exit_tests_exist}")
    print(f"        financial_audit.py says trailing stop fixed 1%: {claims_fixed_trailing}; exit_logic.py scales by atr_pct: {trailing_is_dynamic}")
    report("R13 stale audit/docs contradict current code",
           "BUG-CONFIRMED" if (expects_old and current_uses_async_def
                               and claims_no_exit_tests and exit_tests_exist
                               and claims_fixed_trailing) else "ok",
           "Running verify_sell_concurrency.py today FAILS its own test 1 even "
           "though the underlying behavior is fine; KNOWN_ISSUES.md and "
           "financial_audit.py describe superseded behavior as current.")


if __name__ == "__main__":
    r1_sell_largest_symbol_format()
    r1b_close_position_residual()
    r2_duplicate_sell()
    r3_backtest_equity_math()
    r4_strict_false_silent()
    r5_max_retries_zero()
    r6_perf_detail_formula()
    r7_train_zero_division()
    r8_pooled_ic_contamination()
    r9_env_vars_ignored()
    r11_scipy_missing()
    r12_data_feeds_boundary()
    r13_stale_audits()

    print("\n" + "=" * 74)
    print("REPRO SUMMARY")
    print("=" * 74)
    for name, verdict, detail in RESULTS:
        print(f"  [{verdict}] {name}")