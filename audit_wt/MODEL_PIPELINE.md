# Model Improvement Pipeline — Operations Guide

This directory contains the four-step model-improvement loop plus two
supporting mechanisms. Every piece is wired into the live bot and covered by
`tests/test_model_pipeline.py` (14 tests) alongside the pre-existing suite
(134 tests total, all passing).

## The steps

### Step 0 — Experience capture (`experience_capture.py`)
Every live BUY decision appends an `entry` event (decision-time context +
the EXACT raw feature vector from `predictor.last_features`) to
`live_experiences.jsonl`. Every SELL appends a paired `exit` outcome.
Training later joins them by symbol/time. Shadow inferences are logged as
`shadow` events. Append-only JSONL with size-based rotation (`.1`, `.2`, ...);
malformed lines are skipped on read, never fatal to the trading loop.

### Step 1 — Signal IC check (`signal_ic_check.py`)
Walk-forward Information Coefficient of the champion's raw output vs future
returns, per symbol and pooled. Run BEFORE any tuning:

    python signal_ic_check.py --csv historical_data/BTCUSD.csv
    python signal_ic_check.py --csv-dir historical_data/

Reading the output is documented at the bottom of the script. If |IC| < ~0.02
at every horizon, no execution-side tuning can help — fix features/training.

**Audit note:** the pooled section previously computed forward returns AFTER
concatenating raw close series across symbols, so each symbol's last `h`
bars got "forward returns" measured into the NEXT symbol's prices (a $100
coin "returning" +46,000% into a $50k coin). Fixed: per-symbol returns are
computed first, then stacked. Regression-tested in
`test_pooled_ic_fix_matches_per_symbol_pooling`.

### Step 2 — GBT baseline (`train_gbt_baseline.py`)
Trains a HistGradientBoostingClassifier on IDENTICAL features, labels,
horizon (6 bars / 90 min), and chronological split as the transformer.
Fairness contract is documented in its module docstring; notably
`early_stopping=False` because HGB's internal early stopping would carve a
RANDOM validation split out of overlapping-window rows (leakage).

    python train_gbt_baseline.py            # writes gbt_challenger.joblib
                                            #   + .metrics.json

### Step 3 — Shadow challenger (`shadow_model.py`)
If `gbt_challenger.joblib` exists, the live bot scores every decision-time
feature row through it and logs both probabilities — without ever letting it
influence a trade. Hot-swappable: replace the artifact file and the next
cycle uses the new model (mtime-checked). Failures are non-fatal by design.

### Step 4 — Promotion gate (`promotion_gate.py`)
The ONLY path by which a challenger may become champion. Walk-forward over
contiguous time folds: for fold k the GBT is retrained on everything before
k, then BOTH models are scored on fold k only (champion zero-shot).
Promotion requires winning >= ceil(0.6 * scored_folds) folds AND higher mean
Spearman IC. Report-only by default; `--apply` stages
`champion_promoted.joblib` and prints the exact MODEL_PATH change.

    python promotion_gate.py --folds 5          # report only
    python promotion_gate.py --folds 5 --apply  # stage if gate passes

Caveats printed with every run: champion zero-shot scores are optimistic on
folds overlapping its original training window (judge late folds hardest);
overlapping-window forward returns inflate significance (ICs rank folds,
they are not p-values).

### Step 5 — Champion hot-reload (`ml_predictor.SafeMLPredictor`)
Every `predict_batch()` stats the model/scaler files; changed mtime triggers
an in-place reload — no restart needed to ship a new champion. A failed
reload NEVER takes down trading or swaps in random weights: artifacts load
into locals and commit atomically; on failure the previous weights keep
serving and the signature resyncs so a half-written file doesn't retry-storm.
Also supports sklearn champions (`*.joblib`) so a promoted GBT can serve
through the same interface.

## Verified vs not verified

Verified by automated tests on this machine (no network needed):
- capture round-trip/malformed-line tolerance/rotation
- hot-reload picks up new weights; failed reload keeps old weights serving
- raw feature snapshots match `add_features` exactly
- sklearn-champion prediction path end-to-end
- shadow loader lifecycle incl. artifact hot-swap
- gate decision math (majority+mean rule, None-fold handling)
- fold boundary construction (dtype, span, contiguity)
- NaN-safe fold metrics (constant-input IC -> None, not NaN)
- pooled-IC cross-symbol corruption regression (F8)

NOT yet run against real market data on this machine — blocked, honestly:
- Step 1 IC numbers (needs Alpaca keys or CSVs under `historical_data/`;
  neither present here — `APCA_API_KEY_ID` unset, no CSVs found)
- Step 2 training run (same blocker)
- Step 4 walk-forward run (same blocker)

The scripts are ready; run them where credentials exist. Do NOT tune any
threshold based on intuition before Step 1 produces real IC numbers.

## Config additions (`config.py`)
- `GBT_CHALLENGER_PATH = "gbt_challenger.joblib"` — shadow/gate artifact path
- `EXPERIENCE_LOG_PATH = "live_experiences.jsonl"` — step 0 log location