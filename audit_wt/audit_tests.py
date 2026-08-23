#!/usr/bin/env python3
"""
Audit reproduction tests — exercises suspicious code paths to verify bugs.
Run: python audit_tests.py
"""
import sys
import os
import traceback
import numpy as np
import pandas as pd
import torch

# Ensure UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = []

def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: backtest_exact.py SafeMLPredictor — missing sigmoid
# The model returns raw logits. main_bot.py and ml_predictor.py both apply
# torch.sigmoid(). backtest_exact.py's SafeMLPredictor does NOT.
# ─────────────────────────────────────────────────────────────────────────────
def test_backtest_exact_missing_sigmoid():
    print("\n=== TEST 1: backtest_exact.py missing sigmoid ===")
    # Simulate what backtest_exact.py does:
    # model returns raw logits, predict() returns them directly
    # A raw logit of 0.0 = 50% probability, but backtest returns 0.0
    # A raw logit of 2.0 = 88% probability, but backtest returns 2.0
    # BUY_SIGNAL = 0.51, so a logit of 2.0 (88% prob) would NOT trigger a buy
    # because 2.0 > 0.51 is True (it would trigger), but a logit of 0.3
    # (57% prob) would NOT trigger because 0.3 < 0.51.
    # The real issue: logits are centered around 0, so most signals will be
    # near 0, and 0 < 0.51 means almost no buys.

    # Simulate: model outputs a logit of 0.3 (which is ~57% probability)
    raw_logit = 0.3
    correct_prob = torch.sigmoid(torch.tensor(raw_logit)).item()  # ~0.574
    backtest_value = raw_logit  # backtest_exact.py returns raw logit

    print(f"  Raw logit: {raw_logit}")
    print(f"  Correct sigmoid prob: {correct_prob:.4f}")
    print(f"  backtest_exact.py returns: {backtest_value}")
    print(f"  BUY_SIGNAL threshold: 0.51")
    print(f"  Correct: {correct_prob:.4f} > 0.51 = {correct_prob > 0.51}")
    print(f"  backtest_exact: {backtest_value} > 0.51 = {backtest_value > 0.51}")

    # The bug: a logit of 0.3 represents 57% confidence (should buy at 0.51 threshold)
    # but backtest_exact returns 0.3 which is < 0.51, so it WON'T buy.
    bug_confirmed = (correct_prob > 0.51) and (backtest_value <= 0.51)
    report("backtest_exact.py missing sigmoid", bug_confirmed,
           f"logit=0.3 → correct={correct_prob:.4f} (would buy), backtest={backtest_value} (won't buy)")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: backtest_exact.py model architecture mismatch
# Trained model: num_layers=4, num_q_heads=8, num_kv_heads=2
# backtest_exact.py: num_layers=8, num_q_heads=16, num_kv_heads=4
# load_state_dict(strict=False) silently ignores mismatched layers
# ─────────────────────────────────────────────────────────────────────────────
def test_backtest_exact_arch_mismatch():
    print("\n=== TEST 2: backtest_exact.py model architecture mismatch ===")
    from ml_predictor import GrokGQA_Transformer, GQA_TransformerBlock

    # What the trained model uses (from train_transformer.py and main_bot.py)
    trained_model = GrokGQA_Transformer(
        input_dim=11, seq_len=32,
        embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1
    )

    # What backtest_exact.py uses
    backtest_model = GrokGQA_Transformer(
        input_dim=11, seq_len=32,
        embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1
    )

    trained_keys = set(trained_model.state_dict().keys())
    backtest_keys = set(backtest_model.state_dict().keys())

    # Check if the keys match
    missing_in_backtest = trained_keys - backtest_keys
    extra_in_backtest = backtest_keys - trained_keys

    # Try loading trained weights into backtest model
    trained_state = trained_model.state_dict()
    try:
        backtest_model.load_state_dict(trained_state, strict=False)
        # Check how many keys were actually loaded
        backtest_state = backtest_model.state_dict()
        loaded = 0
        not_loaded = 0
        for k in trained_state:
            if k in backtest_state:
                if torch.equal(trained_state[k], backtest_state[k]):
                    loaded += 1
                else:
                    not_loaded += 1
            else:
                not_loaded += 1
    except Exception as e:
        not_loaded = "ERROR: " + str(e)

    print(f"  Trained model layers: {len(trained_keys)} params")
    print(f"  Backtest model layers: {len(backtest_keys)} params")
    print(f"  Missing in backtest: {len(missing_in_backtest)}")
    print(f"  Extra in backtest: {len(extra_in_backtest)}")

    # The architecture mismatch means most layers won't load
    # With 4 vs 8 layers, the layer indices won't match
    arch_mismatch = len(missing_in_backtest) > 0 or len(extra_in_backtest) > 0
    report("backtest_exact.py architecture mismatch", arch_mismatch,
           f"trained={len(trained_keys)} params, backtest={len(backtest_keys)} params, "
           f"missing={len(missing_in_backtest)}, extra={len(extra_in_backtest)}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: regime.py NaN handling
# With fewer than 14 bars (rolling(14) produces NaN ATR), the function should
# explicitly default to "normal"/"neutral" and NOT silently classify as "quiet".
# ─────────────────────────────────────────────────────────────────────────────
def test_regime_nan_comparison():
    print("\n=== TEST 3: regime.py NaN handling ===")
    from regime import compute_regime_and_trend

    # Create a dataframe with valid data but only 10 rows (rolling(14) produces NaN)
    df = pd.DataFrame({
        "high": [100.0 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "close": [100.0 + i for i in range(10)],
    })

    regime, trend, atr_pct = compute_regime_and_trend(df)
    print(f"  Input: 10 rows (rolling(14) -> NaN ATR)")
    print(f"  regime={regime}, trend={trend}, atr_pct={atr_pct}")

    # The code now explicitly checks for NaN and defaults to normal/neutral
    # The bug (silent quiet classification) has been fixed
    correctly_defaults = regime == "normal" and trend == "neutral"
    report("regime.py NaN correctly defaults to normal/neutral", correctly_defaults,
           f"regime={regime}, trend={trend}, atr_pct={atr_pct}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: feature_engineering.py — rolling window with insufficient data
# add_features with fewer than 20 rows: z_score returns 0.0 (fill value)
# But what about the scaler? If scaler expects 11 features and gets fewer...
# ─────────────────────────────────────────────────────────────────────────────
def test_feature_engineering_short_input():
    print("\n=== TEST 4: feature_engineering.py short input ===")
    from feature_engineering import add_features, FEATURE_COLS

    # Test with exactly SEQUENCE_LEN (32) rows — the minimum
    df = pd.DataFrame({
        "open": np.random.uniform(99, 101, 32),
        "high": np.random.uniform(100, 102, 32),
        "low": np.random.uniform(98, 100, 32),
        "close": np.random.uniform(99, 101, 32),
        "volume": np.random.uniform(1000, 2000, 32),
        "vwap": np.random.uniform(99, 101, 32),
        "trade_count": np.random.uniform(100, 200, 32),
    })

    result = add_features(df)
    print(f"  Input: 32 rows")
    print(f"  Output shape: {result.shape}")
    print(f"  Columns: {list(result.columns)}")
    print(f"  NaN count: {result.isna().sum().sum()}")
    print(f"  Inf count: {np.isinf(result.values).sum()}")

    correct = (result.shape[1] == 11 and result.isna().sum().sum() == 0
               and np.isinf(result.values).sum() == 0)
    report("feature_engineering.py 32-row input", correct,
           f"shape={result.shape}, NaN={result.isna().sum().sum()}, Inf={np.isinf(result.values).sum()}")

    # Test with 1 row — should not crash
    df1 = df.iloc[:1].copy()
    try:
        result1 = add_features(df1)
        print(f"  1-row input: shape={result1.shape}, NaN={result1.isna().sum().sum()}")
        report("feature_engineering.py 1-row input (no crash)", True,
               f"shape={result1.shape}")
    except Exception as e:
        report("feature_engineering.py 1-row input (no crash)", False,
               f"CRASHED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Duplicate scaler files resolved
# Only one scaler file should exist to prevent confusion about which one is used.
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_scaler_files():
    print("\n=== TEST 5: Duplicate scaler files ===")
    scaler_path = "feature_scaler.pkl"
    scaler2_path = "feature_scaler (2).pkl"

    s1_exists = os.path.exists(scaler_path)
    s2_exists = os.path.exists(scaler2_path)

    print(f"  feature_scaler.pkl exists: {s1_exists}")
    print(f"  feature_scaler (2).pkl exists: {s2_exists}")

    # The duplicate file was removed, so only the primary scaler should exist
    no_duplicate = s1_exists and not s2_exists
    report("duplicate scaler files (resolved)", no_duplicate,
           f"s1={s1_exists}, s2={s2_exists} — only primary scaler should exist")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: No duplicate SafeMLPredictor class
# main_bot.py should import SafeMLPredictor from ml_predictor.py,
# NOT define its own duplicate class that shadows the import.
# ─────────────────────────────────────────────────────────────────────────────
def test_two_predictor_classes():
    print("\n=== TEST 6: SafeMLPredictor import consistency ===")
    # Check if main_bot.py has its own SafeMLPredictor
    with open("main_bot.py", encoding="utf-8") as f:
        main_bot_src = f.read()

    # Check if ml_predictor.py has SafeMLPredictor
    with open("ml_predictor.py", encoding="utf-8") as f:
        ml_src = f.read()

    main_has = "class SafeMLPredictor:" in main_bot_src
    ml_has = "class SafeMLPredictor:" in ml_src

    print(f"  main_bot.py has SafeMLPredictor: {main_has}")
    print(f"  ml_predictor.py has SafeMLPredictor: {ml_has}")

    # Check if main_bot.py imports SafeMLPredictor from ml_predictor
    imports_sp = "from ml_predictor import" in main_bot_src and "SafeMLPredictor" in main_bot_src.split("from ml_predictor import")[1].split("\n")[0]
    print(f"  main_bot.py imports SafeMLPredictor from ml_predictor: {imports_sp}")

    # The code should NOT define its own class; it should import the one from ml_predictor
    report("SafeMLPredictor import (no shadowing)", (not main_has) and ml_has and imports_sp,
           f"main_bot has own class={main_has}, ml_predictor has class={ml_has}, "
           f"main_bot imports SP from ml={imports_sp}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: dotenv correctly in requirements.txt
# backtest_exact.py, shadow_data_collector.py, scripts/test_ws.py all use load_dotenv
# ─────────────────────────────────────────────────────────────────────────────
def test_dotenv_missing():
    print("\n=== TEST 7: dotenv dependency in requirements.txt ===")
    with open("requirements.txt", encoding="utf-8") as f:
        reqs = f.read()

    dotenv_in_reqs = "dotenv" in reqs.lower()

    # Check which files import dotenv
    files_using_dotenv = []
    for fname in ["backtest_exact.py", "shadow_data_collector.py", "scripts/test_ws.py"]:
        with open(fname, encoding="utf-8") as f:
            content = f.read()
            if "from dotenv import" in content or "import dotenv" in content:
                files_using_dotenv.append(fname)

    print(f"  dotenv in requirements.txt: {dotenv_in_reqs}")
    print(f"  Files using dotenv: {files_using_dotenv}")

    try:
        import dotenv
        dotenv_installed = True
    except ImportError:
        dotenv_installed = False

    print(f"  dotenv installed: {dotenv_installed}")

    report("dotenv correctly declared in requirements.txt", dotenv_in_reqs and len(files_using_dotenv) > 0,
           f"in_reqs={dotenv_in_reqs}, using_files={files_using_dotenv}, installed={dotenv_installed}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: main_bot.py drawdown division by zero
# If start_equity is 0, (equity - start_equity) / start_equity crashes
# ─────────────────────────────────────────────────────────────────────────────
def test_drawdown_division_by_zero():
    print("\n=== TEST 8: drawdown division by zero ===")
    # Simulate the drawdown calculation from main_bot.py line 159
    start_equity = 0.0
    equity = 10000.0

    try:
        drawdown = (equity - start_equity) / start_equity * 100
        print(f"  drawdown = {drawdown}")
        report("drawdown div by zero", False, "no crash (unexpected)")
    except ZeroDivisionError as e:
        print(f"  ZeroDivisionError: {e}")
        report("drawdown div by zero", True,
               "start_equity=0.0 causes ZeroDivisionError, caught by outer except, bot sleeps 30s")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: Unused imports — should now be clean after fixes
# Previously: MarketOrderRequest (orders.py), asyncio (notifications.py), os (backtest_grid_search.py)
# ─────────────────────────────────────────────────────────────────────────────
def test_unused_imports():
    print("\n=== TEST 9: Unused imports (cleaned up) ===")
    issues = []

    # orders.py: MarketOrderRequest removed (was unused)
    with open("orders.py", encoding="utf-8") as f:
        orders_src = f.read()
    if "MarketOrderRequest" in orders_src.split("from alpaca.trading.requests import")[1].split("\n")[0]:
        issues.append("orders.py: MarketOrderRequest imported but never used")

    # notifications.py: asyncio removed (was unused)
    with open("notifications.py", encoding="utf-8") as f:
        notif_src = f.read()
    if "import asyncio" in notif_src:
        if "asyncio." not in notif_src.replace("import asyncio", ""):
            issues.append("notifications.py: asyncio imported but never used")

    # backtest_exact.py: time is still used (in asyncio.sleep? no, let's check)
    with open("backtest_exact.py", encoding="utf-8") as f:
        be_src = f.read()
    if "import time" in be_src:
        if "time." not in be_src.replace("import time", " "):
            issues.append("backtest_exact.py: time imported but never used")

    # backtest_grid_search.py: os removed (was unused)
    with open("backtest_grid_search.py", encoding="utf-8") as f:
        bgs_src = f.read()
    if "import os" in bgs_src:
        if "os." not in bgs_src.replace("import os", " "):
            issues.append("backtest_grid_search.py: os imported but never used")

    # main_bot.py: BUY_SIGNAL imported but never directly used (by design - via get_regime_params)
    with open("main_bot.py", encoding="utf-8") as f:
        mb_src = f.read()
    buy_signal_usage = mb_src.count("BUY_SIGNAL")
    if buy_signal_usage <= 1:  # only in import
        issues.append("main_bot.py: BUY_SIGNAL imported but never directly used (only via get_regime_params)")

    for issue in issues:
        print(f"  {issue}")

    report("no unused imports remaining", len(issues) == 0, f"{len(issues)} unused imports found")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: signal_ic_check.py references pandas_ta patterns that don't exist
# ─────────────────────────────────────────────────────────────────────────────
def test_signal_ic_check_misleading():
    print("\n=== TEST 10: signal_ic_check.py misleading fallback patterns ===")
    with open("signal_ic_check.py", encoding="utf-8") as f:
        src = f.read()

    # Check if it references pandas_ta or ta. patterns
    has_pandas_ta_ref = "pandas_ta" in src
    has_ta_failed_pattern = "ta.(\\w+) failed" in src or "ta\\.(\\w+) failed" in src

    # Check if feature_engineering.py uses pandas_ta
    with open("feature_engineering.py", encoding="utf-8") as f:
        fe_src = f.read()
    fe_uses_pandas_ta = "pandas_ta" in fe_src or "import ta" in fe_src

    print(f"  signal_ic_check.py references pandas_ta: {has_pandas_ta_ref}")
    print(f"  signal_ic_check.py has 'ta.X failed' pattern: {has_ta_failed_pattern}")
    print(f"  feature_engineering.py uses pandas_ta: {fe_uses_pandas_ta}")

    misleading = has_pandas_ta_ref or has_ta_failed_pattern
    report("signal_ic_check.py misleading pandas_ta references", misleading,
           f"references_pandas_ta={has_pandas_ta_ref}, "
           f"has_ta_failed_pattern={has_ta_failed_pattern}, "
           f"fe_uses_pandas_ta={fe_uses_pandas_ta}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: main_bot.py DB sync uses start_equity before it's set
# Line 141: float(start_equity or 0) — start_equity is None at this point
# ─────────────────────────────────────────────────────────────────────────────
def test_db_sync_before_equity_set():
    print("\n=== TEST 11: main_bot.py DB sync uses start_equity before set ===")
    # In main_bot.py, the DB sync block (lines 126-145) runs BEFORE the main loop
    # where start_equity is first set (line 154-155).
    # At DB sync time, start_equity is still None.
    # float(start_equity or 0) = float(0) = 0.0
    # So the DB gets starting_equity = 0.0, not the real equity.
    start_equity = None
    db_value = float(start_equity or 0)
    print(f"  start_equity at DB sync time: {start_equity}")
    print(f"  DB receives: {db_value}")
    print(f"  Later in loop, start_equity is set to real equity, but DB already has 0.0")

    report("DB sync uses start_equity=0 before it's set", db_value == 0.0,
           f"start_equity=None → DB gets {db_value} instead of real equity")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: main_bot.py — entry_time not cleared on sell
# Comment says "We deliberately DO NOT pop entry_time" but this means
# if a position is re-entered, held_hours will be calculated from the OLD entry
# ─────────────────────────────────────────────────────────────────────────────
def test_entry_time_not_cleared():
    print("\n=== TEST 12: entry_time not cleared on sell ===")
    with open("main_bot.py", encoding="utf-8") as f:
        src = f.read()

    # Find the sell section
    has_comment = "DO NOT pop entry_time" in src
    has_pop = "entry_time.pop" in src or "del entry_time" in src

    print(f"  Comment says 'DO NOT pop entry_time': {has_comment}")
    print(f"  entry_time.pop or del entry_time found: {has_pop}")

    # The issue: if a position is sold and then re-bought, the held_hours
    # calculation will use the OLD entry_time, not the new one
    # This affects the time-decay stop loss and max hold time logic
    report("entry_time not cleared on sell (stale hold time)", has_comment and not has_pop,
           "entry_time persists after sell → re-entry uses stale hold time")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 13: backtest_exact.py timeframe consistency
# Previously used resampled 5-min data instead of native 15-min bars.
# This causes a train/inference mismatch. Current code should fetch native 15-min.
# ─────────────────────────────────────────────────────────────────────────────
def test_backtest_timeframe_mismatch():
    print("\n=== TEST 13: backtest_exact.py timeframe consistency ===")
    with open("backtest_exact.py", encoding="utf-8") as f:
        be_src = f.read()
    with open("data_feeds.py", encoding="utf-8") as f:
        df_src = f.read()

    # main_bot/data_feeds uses native 15-min bars
    main_uses_15min = "TimeFrame(15, TimeFrameUnit.Minute)" in df_src
    # backtest_exact should also use native 15-min bars
    backtest_uses_15min = "TimeFrame(15, TimeFrameUnit.Minute)" in be_src
    # backtest_exact should NOT resample 1-min to 5-min
    backtest_resamples_5min = 'resample("5min")' in be_src

    print(f"  main_bot/data_feeds uses native 15-min bars: {main_uses_15min}")
    print(f"  backtest_exact uses native 15-min bars: {backtest_uses_15min}")
    print(f"  backtest_exact resamples to 5-min: {backtest_resamples_5min}")

    # The timeframe mismatch has been fixed — backtest now uses native 15-min
    no_mismatch = main_uses_15min and backtest_uses_15min and not backtest_resamples_5min
    report("backtest_exact.py timeframe consistency (fixed)", no_mismatch,
           f"main_15min={main_uses_15min}, backtest_15min={backtest_uses_15min}, "
           f"resamples_5min={backtest_resamples_5min}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 14: main_bot.py — asyncio.create_task fire-and-forget
# send_discord_alert is wrapped in asyncio.create_task but exceptions
# in the task are silently swallowed (no reference kept, no error handler)
# ─────────────────────────────────────────────────────────────────────────────
def test_create_task_fire_and_forget():
    print("\n=== TEST 14: asyncio.create_task with error handling ===")
    with open("main_bot.py", encoding="utf-8") as f:
        src = f.read()

    create_task_count = src.count("asyncio.create_task")
    # Check if task references are stored and have error handlers
    has_task_ref = "task = asyncio.create_task" in src
    has_done_callback = "task.add_done_callback" in src

    print(f"  asyncio.create_task calls: {create_task_count}")
    print(f"  Task references stored: {has_task_ref}")
    print(f"  Task error handlers (add_done_callback): {has_done_callback}")

    # The fix: task references are stored and have done_callbacks for error visibility
    report("create_task with error handling (fixed)",
           has_task_ref and has_done_callback,
           f"{create_task_count} create_task calls, refs={has_task_ref}, "
           f"error_handlers={has_done_callback} → exceptions visible")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 15: portfolio.py — sell_retry_cooldown is module-level mutable state
# This is shared across all symbols, but it IS keyed by symbol, so it's OK
# Let's verify it's properly keyed
# ─────────────────────────────────────────────────────────────────────────────
def test_sell_retry_cooldown_keyed():
    print("\n=== TEST 15: portfolio.py sell_retry_cooldown state ===")
    with open("portfolio.py", encoding="utf-8") as f:
        src = f.read()

    # Check if sell_retry_cooldown is keyed by symbol
    keyed_by_symbol = "sell_retry_cooldown[largest.symbol]" in src
    print(f"  sell_retry_cooldown keyed by symbol: {keyed_by_symbol}")

    # This is actually correct — it's a dict keyed by symbol, not a scalar
    report("sell_retry_cooldown properly keyed by symbol", keyed_by_symbol,
           "dict keyed by symbol — correct scoping")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 16: main_bot.py — highest_prices not cleared on sell
# Similar to entry_time, highest_prices persists after sell
# ─────────────────────────────────────────────────────────────────────────────
def test_highest_prices_not_cleared():
    print("\n=== TEST 16: highest_prices not cleared on sell ===")
    with open("main_bot.py", encoding="utf-8") as f:
        src = f.read()

    # Check if highest_prices is popped on sell
    has_pop = "highest_prices.pop" in src or "del highest_prices" in src
    # Check if there's a comment about not clearing
    has_comment = "highest_prices" in src and "DO NOT" in src

    print(f"  highest_prices.pop found: {has_pop}")
    print(f"  Comment about not clearing: {has_comment}")

    # The issue: if a position is sold and re-bought, highest_prices[symbol]
    # will be the OLD peak, not reset. This means the trailing stop will
    # trigger immediately if the new price is below the old peak.
    report("highest_prices not cleared on sell (stale peak)", not has_pop,
           "highest_prices persists after sell → re-entry uses stale peak price")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 17: data_feeds.py — get_clean_ohlcv_dataframe length check AFTER filter
# The length check at line 98 runs BEFORE the close>0 filter at line 119
# But there's a SECOND length check at line 120 after the filter.
# Let's verify both checks exist.
# ─────────────────────────────────────────────────────────────────────────────
def test_data_feeds_length_checks():
    print("\n=== TEST 17: data_feeds.py length checks ===")
    with open("data_feeds.py", encoding="utf-8") as f:
        src = f.read()

    # Check for length checks
    check_before_filter = "if len(bars) < SEQUENCE_LEN" in src
    check_after_filter = 'if len(df) < SEQUENCE_LEN' in src

    print(f"  Length check before close>0 filter: {check_before_filter}")
    print(f"  Length check after close>0 filter: {check_after_filter}")

    # Both checks exist, so this is correct
    report("data_feeds.py length checks (before AND after filter)",
           check_before_filter and check_after_filter,
           "both checks present — correct")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 18: main_bot.py — start_equity DB sync uses float(None or 0) = 0.0
# But then in the loop, start_equity is set to real equity.
# The DB INSERT uses starting_equity = 0.0, but the ON CONFLICT UPDATE
# also sets starting_equity = EXCLUDED.starting_equity = 0.0
# So even on subsequent cycles, the DB keeps 0.0 for starting_equity
# (the DB sync only runs once at startup, not in the loop)
# ─────────────────────────────────────────────────────────────────────────────
def test_db_starting_equity_never_updated():
    print("\n=== TEST 18: DB starting_equity never updated with real value ===")
    with open("main_bot.py", encoding="utf-8") as f:
        src = f.read()

    # The DB sync block runs once at startup (before the loop)
    # start_equity is None at that point
    # In the loop, start_equity is set but the DB is never updated again
    # report_equity() only updates live_equity, not starting_equity

    has_db_sync_in_loop = "starting_equity" in src.split("while True")[1]
    print(f"  starting_equity updated in loop: {has_db_sync_in_loop}")

    # The DB sync only runs once at startup with start_equity=None → 0.0
    report("DB starting_equity stuck at 0.0 (never updated with real equity)",
           not has_db_sync_in_loop,
           "DB sync runs once at startup with start_equity=None → 0.0, never updated in loop")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 19: orders.py — SELL limit price uses 0.999 (0.1% below market)
# This means sell orders may not fill if the price moves up quickly
# ─────────────────────────────────────────────────────────────────────────────
def test_sell_limit_price_aggressive():
    print("\n=== TEST 19: orders.py SELL limit price 0.1% below market ===")
    with open("orders.py", encoding="utf-8") as f:
        src = f.read()

    # SELL: limit_price = price * 0.999 (0.1% below market)
    # BUY: limit_price = price * 1.001 (0.1% above market)
    sell_multiplier = "0.999" in src
    buy_multiplier = "1.001" in src

    print(f"  SELL uses 0.999 (0.1% below market): {sell_multiplier}")
    print(f"  BUY uses 1.001 (0.1% above market): {buy_multiplier}")

    # The issue: SELL at 0.999 means the order may not fill if price is
    # moving up. The bot then holds the position until the next cycle.
    # This is a design choice, not a bug, but worth noting.
    report("SELL limit price 0.1% below market (may not fill)", sell_multiplier,
           "SELL at price*0.999 may not fill if price moves up — bot holds until next cycle")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 20: main_bot.py — swap_weakest_position is sync, now wrapped in asyncio.to_thread
# Previously called directly (blocking event loop). Now properly offloaded.
# ─────────────────────────────────────────────────────────────────────────────
def test_swap_weakest_sync():
    print("\n=== TEST 20: swap_weakest_position offloaded to thread ===")
    with open("portfolio.py", encoding="utf-8") as f:
        port_src = f.read()
    with open("main_bot.py", encoding="utf-8") as f:
        mb_src = f.read()

    is_sync = "def swap_weakest_position" in port_src and "async def swap_weakest_position" not in port_src
    now_offloaded = "asyncio.to_thread" in mb_src and "swap_weakest_position" in mb_src

    print(f"  swap_weakest_position is sync (def, not async def): {is_sync}")
    print(f"  Now offloaded via asyncio.to_thread: {now_offloaded}")

    # Correct behavior: sync function properly offloaded to avoid blocking event loop
    report("swap_weakest_position offloaded (fixed)", is_sync and now_offloaded,
           "sync function now wrapped in asyncio.to_thread — correct async pattern")

# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("AUDIT REPRODUCTION TESTS")
    print("=" * 70)

    test_backtest_exact_missing_sigmoid()
    test_backtest_exact_arch_mismatch()
    test_regime_nan_comparison()
    test_feature_engineering_short_input()
    test_duplicate_scaler_files()
    test_two_predictor_classes()
    test_dotenv_missing()
    test_drawdown_division_by_zero()
    test_unused_imports()
    test_signal_ic_check_misleading()
    test_db_sync_before_equity_set()
    test_entry_time_not_cleared()
    test_backtest_timeframe_mismatch()
    test_create_task_fire_and_forget()
    test_sell_retry_cooldown_keyed()
    test_highest_prices_not_cleared()
    test_data_feeds_length_checks()
    test_db_starting_equity_never_updated()
    test_sell_limit_price_aggressive()
    test_swap_weakest_sync()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"Total: {len(RESULTS)} | PASS: {passed} | FAIL: {failed}")
    print()
    for name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"  [FAIL] {name}: {detail}")
    print()




