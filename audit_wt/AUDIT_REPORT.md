# Correctness Audit — Grok_alpaca_Apex_v8

**Repo:** https://github.com/brianfreshour944-gif/Grok_alpaca_Apex_v8.git
**Commits audited:** `b25cf68` (initial audit), then re-verified after syncing to
`d0463c3` (origin/main; adds dynamic universe + claimed fixes for F1/F2/F5/F9/F10).
**Method:** full read of every `.py` file + AST-based mechanical scans (await-target
resolution, name-resolution) + 12 executable reproduction tests exercising real repo
code with external APIs stubbed + the repo's own pytest suite (95/95 pass).

Reproduction harness: `_audit/repro_tests.py` (run: `python _audit/repro_tests.py`
from the repo root). Static scans: `_audit/static_audit.py`.

---

## Part 1 — Areas verified CLEAN (checked exhaustively, nothing found)

### 1a. Every `await` traced to its real definition — **0 bugs**
All **73 await sites** in the codebase were mechanically resolved to their target
definitions (`_audit/static_audit.py`, PASS 1):

- `portfolio.cancel_stale_orders_async`, `get_all_positions_async`,
  `get_buying_power_async`, `sell_largest_position`, `swap_weakest_position` → all `async def` ✔
- `orders.place_order` → `async def` ✔
- `data_feeds.get_orderbook_with_retry` → `async def` ✔
- `api_utils.call_with_rate_limit_handling(_async)` → `async def` ✔
- every `asyncio.sleep/to_thread/gather` → library primitives ✔
- the two sites my scanner couldn't statically resolve — `perf_loop_test.py:85
  (await task)` and `tests/test_api_utils.py:48 (await hb)` — were manually
  confirmed to await `asyncio.ensure_future(...)` Task objects, which is valid.

**The specific bug class you asked about (awaiting a sync def, silently swallowed
by broad excepts) does not exist in this codebase.**

### 1b. Import-vs-usage cross-check — production code clean
AST scan of every used name vs. imports/defs per file found **no missing imports in
any runtime module** (main_bot, portfolio, orders, data_feeds, api_utils,
ml_predictor, feature_engineering, regime, exit_logic, config, db_utils).
One latent issue in a dev script: see F12.

### 1c. Shared/global mutable state — correctly scoped
All mutable module-level state is a dict keyed by symbol, not a shared scalar:
`cooldown_until`, `entry_time`, `latest_signals`, `highest_prices` (main_bot.py),
`sell_retry_cooldown` (portfolio.py), `_MODEL_CACHE` (ml_predictor.py, keyed by
checkpoint path). No counter/cache is shared across independent entities. No finding.

### 1d. Boundary / short-data / NaN handling — solid (one positive control run)
- `data_feeds.get_clean_ohlcv_dataframe` re-checks `len(df) >= SEQUENCE_LEN`
  **after** the `close > 0` filter (data_feeds.py:88-90) — verified end-to-end by
  execution (R12 below): 40 raw bars shrinking to 15 valid rows returns `None`
  instead of feeding a short window to the model. ✔
- `regime.py` converts NaN indicator outputs to safe defaults ("normal"/0.0);
  covered by `test_insufficient_bars_defaults_safe_instead_of_nan`. ✔
- `add_features` preserves row count (no dropna) — covered by tests. ✔
- `SafeMLPredictor` raises RuntimeError on feature-count mismatch rather than
  predicting garbage. ✔

### 1e. Duplicate files / entry-point consistency — clean
Single entry point: `Dockerfile` CMD → `python main_bot.py`; `main_bot.py` has an
`if __name__ == "__main__":` guard. No `(1)`-style duplicate copies anywhere;
`feature_scaler.pkl` exists once. The vendored `freqtrade/` directory is a separate
unrelated project and is excluded from the image via `.dockerignore`.

---

## Part 2 — FINDINGS (each verified by executing real repo code unless noted)

Severity scale: crash / silent-failure / correctness / performance / style.

---

### F1 · sell_largest_position submits the wrong symbol format to Alpaca
- **Severity:** silent-failure (order rejected by exchange every time the path runs)
- **Verified:** ✅ EXECUTED (R1) — real `portfolio.sell_largest_position()` with a
  stubbed client captured the outgoing `LimitOrderRequest`.
- **Where:** `portfolio.py:117` (`place_order(candidate["symbol"], ...)`) where
  `candidate["symbol"] = largest.symbol` (~line 110). Alpaca position objects use
  `"BTCUSD"`; every other order path in the bot maps back to slash format
  (`"BTC/USD"` via `SYMBOLS`/`normalize_symbol`) — e.g. `swap_weakest_position`
  (portfolio.py:239) does the mapping, this one doesn't. Same defect at
  `main_bot.py:84`: `close_position(p.symbol)` in startup cleanup also passes the
  raw no-slash symbol.
- **Observed output:** `submit_order received symbol = 'BTCUSD'`
- **Impact:** whenever exposure cap forces a de-risk sell, or a stale position is
  cleaned at startup, the API rejects the order; the retry cooldown then masks it
  for 5 minutes at a time, forever.
- **Fix (proposed, not applied):** in `sell_largest_position`, map
  `largest.symbol` through the same lookup `_select_weakest_position_to_swap`
  uses (`next((s for s in SYMBOLS if normalize_symbol(s) == alpaca_sym), None)`),
  and do the same for `p.symbol` in main_bot's cleanup loop.

### F2 · Duplicate SELL orders while the first limit-sell is unfilled
- **Severity:** correctness (order churn; repeated rejections; fills can double-exit)
- **Verified:** ✅ EXECUTED (R2, decision level) — real `evaluate_exit()` returns an
  exit on three consecutive identical cycles; static check confirms the main loop
  contains no open-order guard before SELL.
- **Where:** `main_bot.py` exit block (~lines 340–370). `cooldown_until` gates
  ENTRIES only; nothing checks for an existing open SELL order for the symbol.
  A limit sell can sit unfilled up to 3 min (`cancel_stale_orders` horizon) while
  the cycle period is ~46 s → up to ~3 extra full-size SELLs per exit event.
- **Observed output:** cycle 1/2/3 exit_reason all fire identically;
  `open-order guard inside main loop before SELL: False`.
- **Fix (proposed):** before placing an exit SELL, query open orders for the symbol
  (or track the pending order client-side) and skip if one exists.

### F3 · backtest_exact.py compounds the whole account on a partial-notional trade
- **Severity:** correctness (reported performance inflated ~3–70× depending on Kelly)
- **Verified:** ✅ EXECUTED (R3) using the file's own `calculate_kelly_multiplier`.
- **Where:** `backtest_exact.py:191` and `:195` do `equity *= (1 + pnl_pct)`
  while lines 200–201 size the trade at `trade_value = equity * risk_pct`
  (risk_pct ≈ 0.3–1.8% of equity). Only the notional earns pnl_pct, but the whole
  equity is credited it.
- **Observed output:** 50 wins at +2% → tool math $26,916 vs economically
  consistent $10,144 (**2.7× overstatement**; grows without bound as trades
  accumulate). Losses are equally exaggerated (full −3% on a −3% stop).
- **Fix (proposed):** `equity += trade_value * pnl_pct` on both exit branches.

### F4 · strict=False checkpoint loading silently produces a half-random model
- **Severity:** silent-failure (research conclusions from grid_search/live_exits invalid)
- **Verified:** ✅ EXECUTED (R4) with the real `GrokGQA_Transformer`.
- **Where:** `backtest_grid_search.py` and `live_exits_check.py` build the model
  with hardcoded arch params and call `load_state_dict(state, strict=False)`.
- **Observed output:** mismatched arch → `missing_keys=64`, **52 of 135 tensors
  loaded, 83 silently skipped, no exception**, script proceeds normally.
- **Fix (proposed):** after loading, `assert not res.missing_keys and not
  res.unexpected_keys` (or log CRITICAL and abort).

### F5 · call_with_rate_limit_handling(max_retries=0) raises `TypeError` without calling fn
- **Severity:** crash (of the wrapped call site; caught by callers' broad excepts → silent skip)
- **Verified:** ✅ EXECUTED (R5).
- **Where:** `api_utils.py` — `last_exception = None` initialized before the loop;
  when `max_retries=0` the loop body never runs and control falls to
  `raise last_exception` → `raise None` → `TypeError: exceptions must derive from
  BaseException`. The intended callable is never invoked.
- **Observed output:** `TypeError: exceptions must derive from BaseException (fn never called: calls=0)`
- **Fix (proposed):** clamp `max_retries = max(1, max_retries)` at function top.

### F6 · perf_detail.py CPU-overhead percentage formula inverted
- **Severity:** style (diagnostic output meaningless)
- **Verified:** ✅ EXECUTED (R6).
- **Where:** `perf_detail.py:164` — `(total_blocking/total_blocking + 5000)*100`.
- **Observed output:** prints `500,100.0%` where the intended value is `4.76%`.
- **Fix (proposed):** `total_blocking / (total_blocking + 5000) * 100`.

### F7 · train_transformer.py crashes on degenerate splits (ZeroDivisionError)
- **Severity:** crash (dev/training script only)
- **Verified:** ✅ EXECUTED (R7) using the real `chrono_split`.
- **Where:** `train_transformer.py` — `chrono_split` (lines ~279–280) has no empty-split
  guard; epoch loop divides by `len(y_tr)`/`len(y_va)` (~lines 148–152). Reachable
  with tiny datasets, and via `balance_dataset` undersampling a single-class train
  split to 0 rows.
- **Observed output:** `ZeroDivisionError: float division by zero` with `len(y_tr)=0`.
- **Fix (proposed):** after split/balance, `if len(y_tr)==0 or len(y_va)==0: raise SystemExit("not enough data")`.

### F8 · signal_ic_check.py POOLED IC shifts forward returns ACROSS symbols
- **Severity:** correctness (the tool's headline "POOLED ACROSS ALL SYMBOLS" number is fabricated)
- **Verified:** ✅ EXECUTED (R8) — exact replication of lines 189–199.
- **Where:** `signal_ic_check.py:191-194` — `pd.concat(pooled_close,
  ignore_index=True)` then `all_close.shift(-h)/all_close - 1.0`. The shift crosses
  symbol boundaries: the last h bars of each symbol get "forward returns" measured
  into the NEXT symbol's price series (e.g., $100 coin → $50k coin ≈ +45,000%
  outliers fed into an outlier-sensitive Pearson r).
- **Observed output:** per-symbol ICs −0.035/+0.027; tool's pooled IC **−0.0398
  (p=0.17)** vs correctly computed pooled IC **−0.0041 (p=0.89)** — sign and
  significance both corrupted by symbol ordering alone.
- **Fix (proposed):** compute per-symbol fwd returns first, then concat the return
  series (as done correctly in the per-file section).

### F9/F10 · Documented environment variables are silently ignored
- **Severity:** silent-failure (operator configuration has no effect)
- **Verified:** ✅ EXECUTED (R9/R10) — set env, reload config, observe values unchanged.
- **Where:** `.env.example` documents `LOG_LEVEL`, `BUY_SIGNAL`, `SELL_SIGNAL`;
  `config.py` hardcodes `logging.basicConfig(level=logging.INFO, ...)` and
  `BUY_SIGNAL = 0.51` / `SELL_SIGNAL = 0.45` and never reads those names.
- **Observed output:** `LOG_LEVEL=DEBUG → logger stays INFO`;
  `BUY_SIGNAL=0.90 → config.BUY_SIGNAL = 0.51`.
- **Fix (proposed):** either wire them up (`os.getenv(...)`) or delete them from
  `.env.example` so operators don't believe they're tuning the bot.

### F11 · scipy undeclared dependency (ImportError inside the shipped image)
- **Severity:** silent-failure (tool crashes on first use in Docker)
- **Verified:** ✅ deterministic file checks executed (R11) + reading.
- **Where:** `signal_ic_check.py` imports `scipy.stats`; `requirements.txt` lacks
  scipy; `.dockerignore` does NOT exclude the file, so it ships in an image that
  cannot run it.
- **Fix (proposed):** add `scipy` to `requirements.txt`.

### F12 · performance_audit.py: `pd` used without import if module is imported
- **Severity:** latent crash (works only when run as `__main__`)
- **Verified:** ⚠️ READ + AST scan only (not executed — script-only usage works today).
- **Where:** `performance_audit.py:45` uses `pd` inside
  `test_model_inference_timing`, but `import pandas` exists only at line 153
  (different function) and line 480 (inside `if __name__ == "__main__"`).
  Importing the module and calling the test directly → NameError.
- **Fix (proposed):** move `import pandas as pd` to the top of the file.

### F13 · Repo-local audits/docs contradict the actual code (stale "verification")
- **Severity:** style/process (false confidence; broken self-checks)
- **Verified:** ✅ EXECUTED (R13) + reading.
- **Items:**
  - `verify_sell_concurrency.py` asserts the OLD implementation
    (`'await asyncio.to_thread(sell_largest_position)' in src`) — running it today
    FAILS its own test 1 even though the current async-def implementation is fine.
  - `KNOWN_ISSUES.md` says "Exit logic has no test coverage" — `tests/test_exit_logic.py`
    exists and passes.
  - `financial_audit.py` prints "[WARN] Trailing stop FIXED at 1%" — trailing stop
    is now dynamic (ATR-scaled, clamped) in `exit_logic.py`; its summary also still
    claims "[FAIL] Realized PnL not tracked separately" though `db_utils` tracks it.
  - `audit_tests.py` TEST 2 constructs its own arch-mismatched model and then
    "passes" — vacuous against the current code; TEST 11/18 assert DB-write
    ordering that no longer exists.
- **Fix (proposed):** update or delete these scripts; they actively mislead.

---

## Part 3 — How to reproduce

```
cd <repo root>
python _audit/static_audit.py     # await-target + name-resolution scans
python -m pytest tests -q         # 95 passed (baseline)
python _audit/repro_tests.py      # R1–R13 evidence above
```

## Post-sync status (re-verified at `d0463c3`, all by execution)

After syncing local ↔ origin/main (`d0463c3`: dynamic-universe feature + fix
commit d0463c3 "Fix duplicate exit-sell orders, symbol format mismatch,
retry-count crash"), every claim was re-tested against the new code:

| Finding | Status at d0463c3 | Evidence |
|---|---|---|
| F1 symbol format | **PARTIALLY FIXED** | R1 executed: force-sell now submits `'BTC/USD'` ✔. **Residual (R1b):** startup cleanup still calls `close_position(p.symbol)` with raw no-slash symbol — untouched by the fix commit |
| F2 duplicate sells | **FIXED** | R2 executed against real `has_pending_exit`/`mark_pending_exit`: cycle 1 submits, cycles 2–3 suppressed; guard wired into main loop AND both forced-sell paths |
| F5 retry crash | **FIXED** | R5 executed: clean `ValueError: max_retries must be >= 1, got 0`, fn never called |
| F9/F10 env vars | **FIXED** | R9/R10 executed incl. fresh-process check (deployment path): `level=DEBUG BUY_SIGNAL=0.9 SELL_SIGNAL=0.1`. Minor caveat: LOG_LEVEL can't take effect via module reload in a long-lived interpreter (basicConfig is a no-op after first import) — irrelevant for docker |
| F3 backtest math | still open | R3 re-executed: identical 2.7× overstatement |
| F4 strict=False | still open | R4 re-executed: identical 52/135 silent partial load |
| F6 perf formula | still open | R6 re-executed: identical 500,100% output |
| F7 empty-split crash | still open | R7 re-executed: identical ZeroDivisionError |
| F8 pooled IC shift | still open | R8 re-executed: identical cross-symbol corruption |
| F11 scipy dep | still open | requirements.txt unchanged in delta |
| F12 pd import | still open | performance_audit.py unchanged in delta |
| F13 stale audits | still open | verify_sell_concurrency.py / KNOWN_ISSUES.md / financial_audit.py unchanged |

Delta code review (new since b25cf68): `scan_stable_assets(candidates=...)`,
`denormalize_symbol`, pending-exit guard, union-of-keys state persistence,
config env wiring — all reviewed; static scans on the new code show **75 await
sites, zero await-on-sync-def**, no new import gaps. Repo's own suite:
**120/120 passed** (up from 95).

## Bottom line

The async/await hygiene and import hygiene you specifically asked about are clean —
verified mechanically, not by impression. The real defects are concentrated in:
(1) one wrong symbol format on the forced-de-risk path (F1), (2) no open-order
guard on exits (F2), (3) two research tools whose numbers are mathematically wrong
in ways that flatter the strategy (F3, F8), and (4) a family of stale self-audit
scripts that claim to verify things they no longer test (F13).