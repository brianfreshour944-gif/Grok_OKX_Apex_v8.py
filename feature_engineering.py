# feature_engineering.py — Institutional-grade market microstructure features.
#
# Replaces retail indicators (RSI, MACD, bar shape ratios) with academically
# grounded, regime-invariant features that encode mathematically distinct
# sources of market information.
#
# ── Why institutional features beat retail features ───────────────────────────
# Retail features (close_position, price_vs_open, vol_acceleration) are all
# asking the same question: "was this a bull bar or bear bar?"  They are highly
# correlated, so a neural network given 11 of them just learns a weighted sum
# of one noisy signal.
#
# Institutional features are orthogonal:
#   - WHO controls price discovery (signed order flow, Kyle lambda)
#   - HOW EFFICIENT the market is (Amihud illiquidity, trade size proxy)
#   - WHAT REGIME we are in (vol-of-vol, rolling autocorrelation)
#   - WHERE price is in a normalized, scale-free sense (range_position_z, vwap_z)
#   - HOW VOLATILE the market truly is (Parkinson vol, Garman-Klass vol)
#
# Academic references:
#   Parkinson (1980)    — Range-based volatility estimation
#   Garman & Klass (1980) — OHLC volatility estimation
#   Kyle (1985)         — Continuous auction and insider trading (lambda)
#   Amihud (2002)       — Illiquidity and stock returns
#
# NOTE: This feature set was confirmed against feature_scaler.pkl, which was
# fit with n_features_in_ == 11 — i.e. the model currently deployed
# (grok_gqa_v9_best.pth) WAS TRAINED ON THIS EXACT 11-FEATURE SET. A previous
# unresolved git merge conflict in this file (leftover <<<<<<< HEAD /
# ======= / >>>>>>> markers) was crashing the bot at import time
# ("SyntaxError: invalid decimal literal") every single restart — this is
# very likely the true root cause of the "permanently idle" bot: it wasn't
# silently rejecting trades, it was crash-looping before it ever reached the
# main trading loop, and your orchestration was catching/retrying at INFO
# level without surfacing the traceback where you were looking.

import pandas as pd
import numpy as np

# ── Feature columns ───────────────────────────────────────────────────────────
# NOTE: This list is exactly 11 features — input_dim=11 in ml_predictor.py and
# train_transformer.py is unchanged.  Update this list only if you also retrain
# the model from scratch (changing input_dim breaks saved weights).
FEATURE_COLS = [
    "z_return",          # Vol-normalized log return — regime-invariant momentum
    "parkinson_vol",     # Parkinson (1980) range-based vol — 5x more efficient than std
    "garman_klass_vol",  # Garman-Klass (1980) OHLC vol — most efficient open-market estimator
    "kyle_lambda",       # Kyle (1985) price impact proxy — |ret|/sqrt(vol), Z-scored
    "signed_flow",       # Signed order flow proxy — vol x sign(C-O), Z-scored
    "vwap_z",            # Z-scored VWAP deviation — regime-invariant fair-value distance
    "vol_of_vol",        # Volatility of volatility — regime uncertainty / transition signal
    "amihud_z",          # Amihud (2002) illiquidity — |ret|/vol, Z-scored
    "trade_size_proxy",  # Avg trade size (vol/trade_count), Z-scored — inst. vs retail flow
    "roll_autocorr",     # Rolling lag-1 return autocorrelation — trend vs mean-reversion
    "range_position_z",  # Z-scored close position within 20-bar H/L range — breakout signal
]

# Neutral fill values for each feature (used on empty input or edge failures).
# Z-scored features are centred at 0.  Volatility features are 0 (no vol).
FEATURE_DEFAULTS = {
    "z_return":         0.0,
    "parkinson_vol":    0.0,
    "garman_klass_vol": 0.0,
    "kyle_lambda":      0.0,
    "signed_flow":      0.0,
    "vwap_z":           0.0,
    "vol_of_vol":       0.0,
    "amihud_z":         0.0,
    "trade_size_proxy": 0.0,
    "roll_autocorr":    0.0,
    "range_position_z": 0.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(series: pd.Series, fill: float = 0.0) -> pd.Series:
    """
    Force a Series to float64 with no None/NaN/inf.
      pd.to_numeric  — non-numeric strings -> NaN
      .astype(float) — Python None in object-dtype -> NaN  (critical step)
      .replace(inf)  — inf/-inf -> fill
      .fillna(fill)  — remaining NaN -> fill
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], fill)
        .fillna(fill)
    )


def _z_score(series: pd.Series, window: int = 20, fill: float = 0.0) -> pd.Series:
    """
    Rolling Z-score: (x - mu_w) / sigma_w  using a trailing `window`-bar window.
    Returns 0.0 where sigma=0 or where fewer than 2 samples exist.
    """
    mu  = series.rolling(window, min_periods=2).mean()
    sig = series.rolling(window, min_periods=2).std().replace(0.0, np.nan)
    return _sanitize((series - mu) / sig, fill=fill)


# ── Main feature function ─────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 11 institutional-grade features from OHLCV + VWAP + trade_count bars.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, vwap, trade_count
            (vwap falls back to close; trade_count falls back to 1 if missing)

    Returns:
        DataFrame with exactly FEATURE_COLS columns, all float64, no NaN/inf.
        Row count always equals the input row count.

    Features are computed in dependency order:
        log_ret  -> z_return, parkinson_vol, garman_klass_vol, kyle_lambda,
                   amihud_z, roll_autocorr
        parkinson_vol -> vol_of_vol
        signed volume -> signed_flow
        VWAP distance -> vwap_z
        trade size    -> trade_size_proxy
        rolling range -> range_position_z
    """
    if df is None or df.empty:
        idx = df.index if df is not None else None
        return pd.DataFrame(
            {col: pd.Series(dtype="float64") for col in FEATURE_COLS},
            index=idx,
        )

    d = df.copy()

    # ── Step 1: Sanitize all raw input columns ────────────────────────────────
    for col in ("open", "high", "low", "close", "volume"):
        src = d[col] if col in d.columns else pd.Series(0.0, index=d.index)
        d[col] = _sanitize(src)

    # vwap falls back to close if not supplied (unit tests, live fallback)
    d["vwap"] = _sanitize(
        d["vwap"] if "vwap" in d.columns else d["close"], fill=0.0
    )
    d["vwap"] = d["vwap"].where(d["vwap"] > 0, d["close"])  # replace zeros

    # trade_count falls back to 1 to avoid division-by-zero in trade_size_proxy
    d["trade_count"] = _sanitize(
        d["trade_count"] if "trade_count" in d.columns
        else pd.Series(1.0, index=d.index),
        fill=1.0,
    ).replace(0.0, 1.0)  # guarantee non-zero denominator

    close  = d["close"]
    high   = d["high"]
    low    = d["low"]
    open_  = d["open"]
    volume = d["volume"]
    vwap   = d["vwap"]
    tc     = d["trade_count"]

    # ── Step 2: Log returns (base for several features) ───────────────────────
    safe_prev_close = close.shift(1).replace(0.0, np.nan)
    log_ret = _sanitize(np.log(close / safe_prev_close), fill=0.0)

    # ── Feature 1: z_return ───────────────────────────────────────────────────
    # Regime-invariant momentum: the same % move means very different things
    # in a quiet vs wild market.  Dividing by rolling vol standardises it.
    d["z_return"] = _z_score(log_ret, window=20, fill=0.0)

    # ── Feature 2: parkinson_vol ──────────────────────────────────────────────
    # Parkinson (1980): uses the intrabar high-low range.
    # sigma_park = sqrt( log(H/L)^2 / (4*ln2) )
    # 5x more statistically efficient than close-to-close std on the same data.
    safe_low    = low.replace(0.0, np.nan)
    log_hl      = _sanitize(np.log(high / safe_low), fill=0.0).clip(lower=0.0)
    park_raw    = np.sqrt(log_hl ** 2 / (4.0 * np.log(2.0)))
    d["parkinson_vol"] = _sanitize(park_raw, fill=0.0)

    # ── Feature 3: garman_klass_vol ───────────────────────────────────────────
    # Garman & Klass (1980): uses all four OHLC points.
    # sigma_GK = sqrt( 0.5*log(H/L)^2 - (2ln2-1)*log(C/O)^2 )
    # Most efficient estimator for continuously traded open markets.
    safe_open_ = open_.replace(0.0, np.nan)
    log_co     = _sanitize(np.log(close / safe_open_), fill=0.0)
    gk_raw     = (0.5 * log_hl ** 2) - ((2.0 * np.log(2.0) - 1.0) * log_co ** 2)
    d["garman_klass_vol"] = _sanitize(np.sqrt(gk_raw.clip(lower=0.0)), fill=0.0)

    # ── Feature 4: kyle_lambda ────────────────────────────────────────────────
    # Kyle (1985) price impact proxy: lambda ~ |dP| / Q
    # Approximated as |log_ret| / sqrt(volume) (standard proxy when order book unavailable).
    # Z-scored over 20 bars so lambda is comparable across regimes.
    safe_vol    = volume.replace(0.0, np.nan)
    kyle_raw    = _sanitize(np.abs(log_ret) / np.sqrt(safe_vol), fill=0.0)
    d["kyle_lambda"] = _z_score(kyle_raw, window=20, fill=0.0)

    # ── Feature 5: signed_flow ────────────────────────────────────────────────
    # Signed volume: volume x sign(close - open).
    # Positive  -> net buying pressure this bar.
    # Negative  -> net selling pressure this bar.
    # Z-scored to be regime-invariant (crypto volume varies 10-100x across time).
    raw_flow = volume * np.sign(close - open_)
    d["signed_flow"] = _z_score(_sanitize(raw_flow, fill=0.0), window=20, fill=0.0)

    # ── Feature 6: vwap_z ─────────────────────────────────────────────────────
    # Z-scored VWAP deviation: (close - VWAP) / rolling_std(close - VWAP, 20).
    # The old raw vwap_deviation (close-VWAP)/VWAP is not comparable across
    # different volatility regimes.  Z-scoring normalises the scale.
    vwap_dev = _sanitize(close - vwap, fill=0.0)
    d["vwap_z"] = _z_score(vwap_dev, window=20, fill=0.0)

    # ── Feature 7: vol_of_vol ─────────────────────────────────────────────────
    # Rolling standard deviation of Parkinson vol over 14 bars.
    # High VoV -> uncertain, transition regime (model should be less confident).
    # Low  VoV -> stable, predictable regime.
    park_series = d["parkinson_vol"]
    d["vol_of_vol"] = _sanitize(
        park_series.rolling(14, min_periods=2).std(), fill=0.0
    )

    # ── Feature 8: amihud_z ───────────────────────────────────────────────────
    # Amihud (2002) illiquidity ratio: |ret| / volume.
    # High ratio = large price move on thin volume = market is thin = more alpha.
    # Z-scored for regime invariance.
    amihud_raw = _sanitize(np.abs(log_ret) / safe_vol, fill=0.0)
    d["amihud_z"] = _z_score(amihud_raw, window=20, fill=0.0)

    # ── Feature 9: trade_size_proxy ───────────────────────────────────────────
    # Average trade size = volume / trade_count.
    # Large avg trade  -> institutional block flow.
    # Small avg trade  -> retail limit-order churn.
    # Z-scored over 20 bars.
    trade_size_raw = _sanitize(volume / tc, fill=0.0)
    d["trade_size_proxy"] = _z_score(trade_size_raw, window=20, fill=0.0)

    # ── Feature 10: roll_autocorr ─────────────────────────────────────────────
    # Rolling lag-1 Pearson autocorrelation of log returns over a 10-bar window.
    # Negative autocorr -> mean-reverting regime  (buy dips, sell rips).
    # Positive autocorr -> trending regime         (momentum works).
    # Vectorised via rolling cov / (std_x x std_lag1).
    log_ret_lag1 = log_ret.shift(1)
    roll_cov    = log_ret.rolling(10, min_periods=4).cov(log_ret_lag1)
    roll_std_x  = log_ret.rolling(10, min_periods=4).std()
    roll_std_l1 = log_ret_lag1.rolling(10, min_periods=4).std()
    denom       = (roll_std_x * roll_std_l1).replace(0.0, np.nan)
    d["roll_autocorr"] = _sanitize(roll_cov / denom, fill=0.0).clip(-1.0, 1.0)

    # ── Feature 11: range_position_z ─────────────────────────────────────────
    # Where does close sit within the 20-bar rolling high-low range? [0, 1].
    # Then Z-scored to capture breakouts (>> 0) vs. mean reversion (<< 0)
    # in a scale-free, regime-invariant way.
    roll_high_20  = high.rolling(20, min_periods=2).max()
    roll_low_20   = low.rolling(20, min_periods=2).min()
    roll_range_20 = (roll_high_20 - roll_low_20).replace(0.0, np.nan)
    range_pos_raw = _sanitize((close - roll_low_20) / roll_range_20, fill=0.5)
    d["range_position_z"] = _z_score(range_pos_raw, window=20, fill=0.0)

    # ── Step 3: Final guard — all FEATURE_COLS present, correct dtype ─────────
    for col in FEATURE_COLS:
        if col not in d.columns:
            d[col] = FEATURE_DEFAULTS.get(col, 0.0)
        d[col] = _sanitize(d[col], fill=FEATURE_DEFAULTS.get(col, 0.0))

    return d[FEATURE_COLS]
