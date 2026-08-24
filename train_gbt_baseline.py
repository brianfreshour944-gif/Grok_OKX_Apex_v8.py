#!/usr/bin/env python3
"""
train_gbt_baseline.py — Step 2 of the model-improvement pipeline.

Trains a gradient-boosted-tree BASELINE on exactly the same features and the
same labels the transformer champion was trained on, so any later comparison
is an architecture comparison, not a confounded feature-engineering one.

Fairness contract (do not break these without retraining BOTH models):
  * Features : feature_engineering.add_features() / FEATURE_COLS — untouched.
  * Labels   : y = 1 if close[t + TARGET_HORIZON] > close[t], computed from
               raw closes — identical to train_transformer.py's BUG-2 fix.
  * Bars     : native 15-min Alpaca bars via train_transformer.fetch_bars.
  * Horizon  : TARGET_HORIZON = 6 bars (90 min) — same as the transformer.
  * Split    : chronological (first TRAIN_FRAC by timestamp → train, rest →
               validation). NO random splits anywhere: overlapping windows
               share bars, so random splits leak (train_transformer BUG 3).

Deliberate differences from train_transformer.py (and why they're OK):
  * Trees consume a SINGLE feature row, not a 32-bar window — that is the
    architecture difference under test, not a data leak.
  * early_stopping=False: HGB's internal early stopping would carve a RANDOM
    validation split out of the training rows — reintroducing BUG-3 leakage.
    We use fixed iterations and judge honestly on the chronological holdout.

Usage:
    python train_gbt_baseline.py                 # fetches fresh bars
    python train_gbt_baseline.py --days 120      # shorter history
    python train_gbt_baseline.py --out my.joblib # custom artifact path

Output:
    gbt_challenger.joblib   (or --out)      — the model artifact
    <artifact>.metrics.json              — honest val metrics + dataset stats

Backends (--model):
    hgb      scikit-learn HistGradientBoostingClassifier (default)
    lgbm     LightGBM LGBMClassifier        (pip install lightgbm)
    catboost CatBoostClassifier             (pip install catboost)

All three consume IDENTICAL features/labels/split — only the learner
changes. Every backend exposes the sklearn predict_proba interface, so the
shadow loader (shadow_model.py) and the champion serving path
(SafeMLPredictor *.joblib) accept any of them with zero code changes.
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from feature_engineering import add_features, FEATURE_COLS
from train_transformer import (
    _get_data_client, fetch_bars, TRAIN_SYMBOLS, TARGET_HORIZON,
)

# A sample at bar t needs enough preceding bars for rolling features to be
# meaningful (z-scores use 20-bar windows; roll_autocorr uses 10+1).
WARMUP_BARS = 32          # matches SEQUENCE_LEN: same effective history floor
TRAIN_FRAC  = 0.80        # chronological split, same as train_transformer


def build_dataset(client, symbols, days) -> pd.DataFrame:
    """One row per (symbol, bar): FEATURES + label + fwd_ret + ts."""
    frames = []
    for sym in symbols:
        df = fetch_bars(client, sym, days)
        if df is None or len(df) < WARMUP_BARS + TARGET_HORIZON + 50:
            print(f"  {sym}: insufficient bars ({0 if df is None else len(df)}), skipped")
            continue
        feats = add_features(df)[FEATURE_COLS].reset_index(drop=True)
        close = df["close"].reset_index(drop=True)
        n = len(feats) - TARGET_HORIZON
        idx = np.arange(WARMUP_BARS, n)
        cur = close.values[idx]
        fut = close.values[idx + TARGET_HORIZON]
        part = pd.DataFrame(feats.values[idx], columns=FEATURE_COLS)
        part["ts"] = df.index[idx]
        part["symbol"] = sym
        part["fwd_ret"] = fut / cur - 1.0
        part["label"] = (fut > cur).astype(int)
        frames.append(part)
        print(f"  {sym}: {len(part)} samples "
              f"({part['label'].mean():.1%} up) from {len(df)} bars")
    if not frames:
        raise SystemExit("No usable data fetched — check credentials/network.")
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values("ts").reset_index(drop=True)


def make_classifier(kind: str):
    """Factory for the selectable GBT backends. All are sklearn-API
    compatible (fit / predict_proba / classes_) so downstream tooling is
    backend-agnostic."""
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False,       # no random internal split (leakage)
            random_state=42,
        )
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise SystemExit("lightgbm not installed — pip install lightgbm") from e
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=40, reg_lambda=1.0,
            random_state=42, verbosity=-1,
        )
    if kind == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise SystemExit("catboost not installed — pip install catboost") from e
        return CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6,
            l2_leaf_reg=1.0, loss_function="Logloss",
            random_seed=42, verbose=False, allow_writing_files=False,
        )
    raise ValueError(f"unknown backend: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180,
                    help="Days of 15-min history per symbol (default 180)")
    ap.add_argument("--model", default="hgb", choices=["hgb", "lgbm", "catboost"],
                    help="GBT backend (default hgb)")
    ap.add_argument("--out", default=None,
                    help="Artifact path (default gbt_challenger_<model>.joblib)")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"gbt_challenger_{args.model}.joblib"

    print(f"── GBT baseline trainer ── horizon={TARGET_HORIZON} bars "
          f"(90 min), warmup={WARMUP_BARS}, split={TRAIN_FRAC:.0%}/{1 - TRAIN_FRAC:.0%}")
    client = _get_data_client()
    data = build_dataset(client, TRAIN_SYMBOLS, args.days)

    X = data[FEATURE_COLS].values.astype(np.float64)
    y = data["label"].values
    cut = int(len(data) * TRAIN_FRAC)
    X_tr, y_tr = X[:cut], y[:cut]
    X_va, y_va = X[cut:], y[cut:]
    fwd_va = data["fwd_ret"].values[cut:]

    print(f"Dataset: {len(data)} samples | train={len(y_tr)} val={len(y_va)} | "
          f"up-rate train={y_tr.mean():.3f} val={y_va.mean():.3f}")
    print(f"Time span: {data['ts'].iloc[0]} → {data['ts'].iloc[-1]}")

    clf = make_classifier(args.model)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_va)[:, list(clf.classes_).index(1)]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon_bars": TARGET_HORIZON,
        "warmup_bars": WARMUP_BARS,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
        "up_rate_train": float(y_tr.mean()), "up_rate_val": float(y_va.mean()),
        "val_accuracy": float((pred == y_va).mean()),
        "symbols": sorted(data["symbol"].unique().tolist()),
        "time_span": [str(data["ts"].iloc[0]), str(data["ts"].iloc[-1])],
    }
    try:
        metrics["val_auc"] = float(roc_auc_score(y_va, proba))
    except ValueError:
        metrics["val_auc"] = None  # single-class val fold
    try:
        from scipy.stats import spearmanr
        ic_r, ic_p = spearmanr(proba, fwd_va)
        metrics["val_spearman_ic"] = float(ic_r)
        metrics["val_spearman_p"] = float(ic_p)
    except Exception:
        metrics["val_spearman_ic"] = None

    joblib.dump(clf, args.out)
    mpath = args.out + ".metrics.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    auc = metrics.get("val_auc")
    ic = metrics.get("val_spearman_ic")
    print(f"Saved artifact : {args.out}")
    print(f"Saved metrics  : {mpath}")
    print(f"Honest val     : acc={metrics['val_accuracy']:.4f} "
          f"AUC={auc if auc is not None else 'n/a'} "
          f"SpearmanIC={ic if ic is not None else 'n/a'} (n={len(y_va)})")
    print("NOTE: this is a BASELINE, not a promotion. Run promotion_gate.py to")
    print("compare it against the transformer champion walk-forward before any")
    print("champion change is even considered.")


if __name__ == "__main__":
    main()
