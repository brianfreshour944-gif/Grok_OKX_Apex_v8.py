#!/usr/bin/env python3
"""
signal_ic_check.py — Tests whether the ML model's raw prediction actually
correlates with future price movement, BEFORE any further tuning of
thresholds/exits/execution logic.

This is the check that was missing from the whole tuning session: none of
the ~10 config-adjustment iterations verified the model's output contains
real predictive information at all. If the Information Coefficient (IC)
here is near zero, no amount of threshold/exit/position-sizing tuning can
create edge that isn't there -- the fix has to be upstream (features,
model, training), not execution.

Uses a TRUE walk-forward prediction: at every bar t, the model only sees
data up to and including bar t (via a rolling SEQUENCE_LEN window) -- it
never sees the future, exactly matching how it's actually used live.

Usage:
    python3 signal_ic_check.py --csv historical_data/BTCUSD.csv
    python3 signal_ic_check.py --csv-dir historical_data/ --horizons 1,3,6,12,24
"""
import argparse
import contextlib
import glob
import io
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

from ml_predictor import SafeMLPredictor

SEQUENCE_LEN = 32
# Matches config.py's FEATURE_WARMUP_BARS fix: indicators need more history
# than the model's actual input window to compute correctly (confirmed:
# pandas_ta_classic's macd() fails 100% of the time at exactly 32 bars).
# predict() computes features on this full window then slices to the last
# SEQUENCE_LEN rows, so this only affects feature quality, not model input shape.
FEATURE_WARMUP_BARS = 100


def compute_walkforward_predictions(df: pd.DataFrame, predictor: SafeMLPredictor) -> tuple:
    """
    Computes the model's prediction at every bar, using ONLY data available
    up to and including that bar -- a proper walk-forward. Returns
    (predictions, fallback_counts) where fallback_counts tallies how often
    each indicator's pandas_ta call hit its internal short-window bug and
    fell back to a neutral default (see feature_engineering.py) -- useful
    to know since it means that many predictions ran on partially-neutral
    inputs, not a full real feature set.
    """
    preds = pd.Series(index=df.index, dtype=float)
    fallback_counts = {}
    for i in range(FEATURE_WARMUP_BARS - 1, len(df)):
        window = df.iloc[i - FEATURE_WARMUP_BARS + 1: i + 1]
        # Capture feature_engineering.py's print() warnings instead of letting
        # them flood stdout on every single bar -- tally them instead.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preds.iloc[i] = predictor.predict(window)
        for line in buf.getvalue().splitlines():
            m = re.search(r'ta\.(\w+) failed', line)
            if m:
                fallback_counts[m.group(1)] = fallback_counts.get(m.group(1), 0) + 1
        if i % 500 == 0 and i > 0:
            print(f"    ...{i}/{len(df)} bars predicted", file=sys.stderr)
    return preds, fallback_counts


def compute_ic(preds: pd.Series, close: pd.Series, horizons: list) -> dict:
    """Pearson and Spearman IC between prediction and forward return at each horizon."""
    results = {}
    for h in horizons:
        fwd_return = close.shift(-h) / close - 1.0
        valid = preds.notna() & fwd_return.notna()
        if valid.sum() < 30:
            results[h] = {'pearson': None, 'spearman': None, 'n': int(valid.sum())}
            continue
        pearson_r, pearson_p = stats.pearsonr(preds[valid], fwd_return[valid])
        spearman_r, spearman_p = stats.spearmanr(preds[valid], fwd_return[valid])
        results[h] = {
            'pearson': pearson_r, 'pearson_p': pearson_p,
            'spearman': spearman_r, 'spearman_p': spearman_p,
            'n': int(valid.sum()),
        }
    return results


def bucket_analysis(preds: pd.Series, close: pd.Series, horizon: int, n_buckets: int = 5) -> pd.DataFrame:
    """
    Splits predictions into quantile buckets and reports mean forward return
    per bucket. Real predictive skill should show mean_fwd_return increasing
    roughly monotonically from the lowest to highest bucket. Flat or
    non-monotonic = the model's confidence level doesn't track outcomes,
    even if some overall IC number looks nonzero.
    """
    fwd_return = close.shift(-horizon) / close - 1.0
    valid = preds.notna() & fwd_return.notna()
    if valid.sum() < n_buckets * 10:
        return pd.DataFrame()

    d = pd.DataFrame({'pred': preds[valid].values, 'fwd_return': fwd_return[valid].values})
    try:
        d['bucket'] = pd.qcut(d['pred'], n_buckets, labels=False, duplicates='drop')
    except ValueError:
        return pd.DataFrame()
    summary = d.groupby('bucket').agg(
        mean_pred=('pred', 'mean'),
        mean_fwd_return=('fwd_return', 'mean'),
        win_rate=('fwd_return', lambda x: (x > 0).mean()),
        n=('fwd_return', 'count'),
    )
    return summary


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='Single CSV file (needs open,high,low,close,volume columns)')
    parser.add_argument('--csv-dir', help='Directory of CSVs, one per symbol')
    parser.add_argument('--model', default='grok_gqa_v9_best.pth')
    parser.add_argument('--horizons', default='1,3,6,12,24',
                         help='Forward-return horizons in bars, comma-separated (default: 1,3,6,12,24 bars)')
    args = parser.parse_args()

    if not args.csv and not args.csv_dir:
        print("Provide --csv <file> or --csv-dir <directory>")
        sys.exit(1)

    horizons = [int(h) for h in args.horizons.split(',')]
    predictor = SafeMLPredictor(args.model)

    files = [args.csv] if args.csv else sorted(glob.glob(os.path.join(args.csv_dir, '*.csv')))
    if not files:
        print("No CSV files found.")
        sys.exit(1)

    pooled_preds, pooled_close = [], []

    for f in files:
        symbol = os.path.splitext(os.path.basename(f))[0]
        print(f"\n{'=' * 70}\n{symbol}\n{'=' * 70}")
        df = load_csv(f)

        if len(df) < FEATURE_WARMUP_BARS + max(horizons) + 30:
            print(f"  Not enough data ({len(df)} rows), skipping.")
            continue

        print(f"  Computing walk-forward predictions over {len(df)} bars...")
        preds, fallback_counts = compute_walkforward_predictions(df, predictor)
        n_predicted = len(df) - FEATURE_WARMUP_BARS + 1
        if fallback_counts:
            summary = ", ".join(f"{k}: {v}/{n_predicted} ({v/n_predicted*100:.0f}%)"
                                 for k, v in fallback_counts.items())
            print(f"  Indicator fallbacks triggered (pandas_ta's short-window bug): {summary}")
        results = compute_ic(preds, df['close'], horizons)
        pooled_preds.append(preds)
        pooled_close.append(df['close'])

        print(f"\n  {'Horizon (bars)':<16}{'Pearson IC':>12}{'p-value':>10}{'Spearman IC':>14}{'p-value':>10}{'N':>8}")
        for h in horizons:
            r = results[h]
            if r['pearson'] is None:
                print(f"  {h:<16}{'insufficient data':>44}")
                continue
            print(f"  {h:<16}{r['pearson']:>12.4f}{r['pearson_p']:>10.4f}{r['spearman']:>14.4f}{r['spearman_p']:>10.4f}{r['n']:>8}")

        mid_horizon = horizons[len(horizons) // 2]
        print(f"\n  Quintile analysis at horizon={mid_horizon} bars (does higher confidence -> better outcome?):")
        bucket_df = bucket_analysis(preds, df['close'], mid_horizon)
        if not bucket_df.empty:
            print(bucket_df.to_string(float_format=lambda x: f'{x:.4f}'))
        else:
            print("  Not enough data for bucket analysis.")

    if len(pooled_preds) > 1:
        print(f"\n{'=' * 70}\nPOOLED ACROSS ALL SYMBOLS\n{'=' * 70}")
        all_preds = pd.concat(pooled_preds, ignore_index=True)
        # FIX (audit F8): forward returns MUST be computed per symbol BEFORE
        # concatenating. The old code concatenated raw close series across
        # symbols and then shifted, so the last h bars of each symbol got
        # "forward returns" measured into the NEXT symbol's price series
        # (e.g., a $100 coin -> a $50k coin ≈ +49,900% outliers fed into an
        # outlier-sensitive Pearson r). Compute per-symbol returns first,
        # then stack — same method the per-file section already uses.
        for h in horizons:
            fwd_parts = [close.shift(-h) / close - 1.0 for close in pooled_close]
            fwd_return = pd.concat(fwd_parts, ignore_index=True)
            valid = all_preds.notna() & fwd_return.notna()
            if valid.sum() < 30:
                continue
            pearson_r, pearson_p = stats.pearsonr(all_preds[valid], fwd_return[valid])
            spearman_r, spearman_p = stats.spearmanr(all_preds[valid], fwd_return[valid])
            print(f"  Horizon {h:>3} bars | Pearson: {pearson_r:+.4f} (p={pearson_p:.4f}) | "
                  f"Spearman: {spearman_r:+.4f} (p={spearman_p:.4f}) | N={valid.sum()}")

    print(f"""
{'=' * 70}
HOW TO READ THIS
{'=' * 70}
- |IC| < ~0.02-0.03 at every horizon = the model's raw score carries
  little to no monotonic relationship with what actually happens next.
  No amount of threshold/exit tuning can create edge that isn't here --
  the fix has to be upstream (features, model, training), not execution.
- |IC| > ~0.05, especially with p < 0.05, at some horizon = there IS
  real signal. That's when tuning thresholds/exits to monetize it is
  worth doing.
- Check the quintile table: real skill should show mean_fwd_return
  increasing roughly monotonically from the lowest to highest bucket.
  A flat or non-monotonic staircase (even with a nonzero pooled IC) is
  a warning sign of noise, not skill.
- p-values here are a rough guide only -- they assume independent
  observations, which overlapping-window forward returns are not. Don't
  treat p<0.05 as a rigorous significance test, just a loose filter.
""")


if __name__ == '__main__':
    main()
