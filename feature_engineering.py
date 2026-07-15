# feature_engineering.py — Microstructure-based features.
#
# Replaces lagging indicators (RSI, MACD, ATR, BB) with zero-lag features
# derived directly from raw bar data (OHLCV + VWAP + trade_count).
#
# Why microstructure over lagging indicators?
# ─────────────────────────────────────────────
# RSI(14) and MACD(12,26,9) require 26-35+ bars to stabilize and describe
# where price WAS, not where it's going. At a 5-minute bar cadence inside
# a 32-bar sequence (2.7 hours), 26 of those bars are just "warming up" MACD.
# Microstructure features encode what the market IS doing right now:
#   - Who controls the bar (buyers vs sellers): close_position
#   - Where price is relative to fair value: vwap_deviation
#   - Whether activity is accelerating: vol_acceleration, trade_intensity
#   - Intra-bar spread as a volatility proxy: bar_spread
#   - Directional momentum without lag: returns, price_vs_open

import pandas as pd
import numpy as np

# ── Feature columns ─────────────────────────────────────────────────────────────
# NOTE: If you retrain the model, update FEATURE_COLS here and in ml_predictor.py
# (input_dim=len(FEATURE_COLS)). The existing grok_gqa_v9_best.pth was trained
# on 11 features — this new set is also 11 features, so input_dim is unchanged.
FEATURE_COLS = [
    "close",           # raw price level (scaled by scaler)
    "returns",         # bar-over-bar log return  — directional momentum, 0-lag
    "close_position",  # (close-low)/(high-low) — 0=bear bar, 1=bull bar
    "vwap_deviation",  # (close-vwap)/vwap       — above/below fair value
    "bar_spread",      # (high-low)/close         — intra-bar volatility proxy
    "price_vs_open",   # (close-open)/open        — did buyers win the bar?
    "vol_acceleration",# volume / rolling_mean(volume,5) - 1  — surge/dry-up
    "trade_intensity", # trade_count / rolling_mean(trade_count,5) - 1
    "vol_14",          # 14-bar rolling std of returns  — medium-term vol regime
    "volume",          # raw volume (scaled) — absolute activity level
    "trade_count",     # raw trade count (scaled) — activity level cross-check
]

FEATURE_DEFAULTS = {
    "close":           0.0,
    "returns":         0.0,
    "close_position":  0.5,   # neutral: close in middle of bar
    "vwap_deviation":  0.0,   # neutral: at fair value
    "bar_spread":      0.0,
    "price_vs_open":   0.0,
    "vol_acceleration":0.0,   # neutral: average volume
    "trade_intensity": 0.0,   # neutral: average trade count
    "vol_14":          0.0,
    "volume":          0.0,
    "trade_count":     0.0,
}


def _sanitize(series: pd.Series, fill: float = 0.0) -> pd.Series:
    """
    Force a Series to float64 with no None/NaN/inf.
      pd.to_numeric  — non-numeric strings → NaN
      .astype(float) — Python None in object-dtype → NaN  (critical step)
      .replace(inf)  — inf/-inf → fill
      .fillna(fill)  — remaining NaN → fill
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], fill)
        .fillna(fill)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute microstructure features from OHLCV + VWAP + trade_count bars.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, vwap, trade_count
            (vwap and trade_count fall back gracefully to neutral values if missing)

    Returns:
        DataFrame with exactly FEATURE_COLS columns, all float64, no NaN/inf.
        Always has the same number of rows as the input.
    """
    if df.empty:
        return pd.DataFrame(
            FEATURE_DEFAULTS, index=df.index, columns=FEATURE_COLS
        ).astype(float)

    d = df.copy()

    # ── Step 1: Sanitize all raw input columns ────────────────────────────────
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = _sanitize(d[col] if col in d.columns else pd.Series(0.0, index=d.index))

    # vwap falls back to close if not supplied (e.g. during unit tests)
    d["vwap"] = _sanitize(
        d["vwap"] if "vwap" in d.columns else d["close"],
        fill=0.0
    )
    # Where vwap is still 0 (bar had no volume), substitute close
    d["vwap"] = d["vwap"].where(d["vwap"] > 0, d["close"])

    d["trade_count"] = _sanitize(
        d["trade_count"] if "trade_count" in d.columns
        else pd.Series(0.0, index=d.index)
    )

    close  = d["close"]
    high   = d["high"]
    low    = d["low"]
    open_  = d["open"]
    volume = d["volume"]
    vwap   = d["vwap"]
    tc     = d["trade_count"]

    # ── Step 2: Compute features ──────────────────────────────────────────────

    # Bar-over-bar log return (pct_change approximated with log for additivity)
    d["returns"] = _sanitize(close.pct_change(), fill=0.0)

    # 14-bar rolling volatility of returns (medium-term regime)
    d["vol_14"] = _sanitize(d["returns"].rolling(14, min_periods=1).std(), fill=0.0)

    # Where did close land within the bar's range? [0=low, 1=high]
    bar_range = (high - low).replace(0, np.nan)
    d["close_position"] = _sanitize((close - low) / bar_range, fill=0.5)

    # Price vs VWAP — positive = above fair value, negative = below
    safe_vwap = vwap.replace(0, np.nan)
    d["vwap_deviation"] = _sanitize((close - safe_vwap) / safe_vwap, fill=0.0)

    # Intra-bar spread as a normalised volatility proxy
    safe_close = close.replace(0, np.nan)
    d["bar_spread"] = _sanitize((high - low) / safe_close, fill=0.0)

    # Did buyers win the bar? +ve = close above open
    safe_open = open_.replace(0, np.nan)
    d["price_vs_open"] = _sanitize((close - safe_open) / safe_open, fill=0.0)

    # Volume acceleration: current bar vs 5-bar rolling mean (surge = +ve)
    vol_ma5 = volume.rolling(5, min_periods=1).mean().replace(0, np.nan)
    d["vol_acceleration"] = _sanitize(volume / vol_ma5 - 1, fill=0.0)

    # Trade intensity: same idea for trade count
    tc_ma5 = tc.rolling(5, min_periods=1).mean().replace(0, np.nan)
    d["trade_intensity"] = _sanitize(tc / tc_ma5 - 1, fill=0.0)

    # ── Step 3: Final guard — all FEATURE_COLS present, correct dtype ─────────
    for col in FEATURE_COLS:
        if col not in d.columns:
            d[col] = FEATURE_DEFAULTS.get(col, 0.0)
        d[col] = _sanitize(d[col], fill=FEATURE_DEFAULTS.get(col, 0.0))

    return d[FEATURE_COLS]
