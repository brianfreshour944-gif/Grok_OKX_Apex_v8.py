# feature_engineering.py - Robust, pandas_ta-free implementation
#
# FIX: Removed pandas_ta_classic dependency entirely. All indicators are now
# computed inline with pure pandas/numpy, matching deployed_bot.py's approach.
# This eliminates the root cause of:
#   TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
# which was triggered inside pandas_ta_classic's internal ATR arithmetic when
# columns contained Python None objects that survived object-dtype coercion.

import pandas as pd
import numpy as np

# Define the feature columns that will be used by the model
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'returns', 'vol_14', 'rsi', 'macd', 'atr', 'bb_width'
]

# Define default values for features, used if calculation fails or data is insufficient
FEATURE_DEFAULTS = {
    'returns': 0.0,
    'vol_14':  0.0,
    'rsi':     50.0,   # Neutral RSI
    'macd':    0.0,    # Neutral MACD
    'atr':     0.0,
    'bb_width': 0.0,
}

def _sanitize_col(series: pd.Series) -> pd.Series:
    """
    Force a Series to float64 with no None/NaN/inf values.
    - pd.to_numeric(errors='coerce') converts non-numeric to NaN
    - .astype(float) evicts Python None from object-dtype columns;
      without it None can survive and later cause:
        TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
    - .replace([inf, -inf], 0.0) removes inf that breaks scalers
    - .fillna(0.0) removes any remaining NaN
    """
    return (
        pd.to_numeric(series, errors='coerce')
        .astype(float)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds technical indicator features to the DataFrame.
    All indicators computed with pure pandas/numpy — no external TA library.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.

    Returns:
        DataFrame with exactly FEATURE_COLS columns, all float64, no None/NaN/inf.
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=FEATURE_COLS).fillna(FEATURE_DEFAULTS)

    df_copy = df.copy()

    # ── Step 1: Sanitize all OHLCV inputs ──────────────────────────────────────
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_copy.columns:
            df_copy[col] = 0.0
        df_copy[col] = _sanitize_col(df_copy[col])

    close  = df_copy['close']
    high   = df_copy['high']
    low    = df_copy['low']

    # ── Step 2: Returns & volatility ───────────────────────────────────────────
    df_copy['returns'] = close.pct_change().fillna(0.0)
    df_copy['vol_14']  = df_copy['returns'].rolling(14).std().fillna(0.0)

    # ── Step 3: RSI (Wilder / rolling-mean method) ─────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df_copy['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0).clip(0, 100)

    # ── Step 4: MACD (12/26/9 EMA) ─────────────────────────────────────────────
    exp1        = close.ewm(span=12).mean()
    exp2        = close.ewm(span=26).mean()
    macd_line   = exp1 - exp2
    signal_line = macd_line.ewm(span=9).mean()
    df_copy['macd'] = (macd_line - signal_line).fillna(0.0)

    # ── Step 5: ATR (14-period) ─────────────────────────────────────────────────
    # Use sanitized local Series so .shift() NaN does not interact with None
    prev_close = close.shift(1).fillna(close)   # fill first row with itself
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df_copy['atr'] = tr.rolling(14).mean().fillna(0.0)

    # ── Step 6: Bollinger Band Width (20-period, 2σ) ────────────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # Width = (upper - lower) / mid  (same as BBB in pandas_ta)
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    df_copy['bb_width'] = bb_width.fillna(0.0)

    # ── Step 7: Final sanitization pass on every output column ─────────────────
    for col in FEATURE_COLS:
        if col not in df_copy.columns:
            df_copy[col] = FEATURE_DEFAULTS.get(col, 0.0)
        df_copy[col] = _sanitize_col(df_copy[col])

    return df_copy[FEATURE_COLS]
