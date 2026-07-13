# feature_engineering.py - pandas_ta-based indicators, with crash-safe sanitization.
#
# Uses pandas_ta_classic for RSI/MACD/ATR/Bollinger Bands so live predictions
# match what the model was actually trained on (trading_env.py's training
# pipeline imported add_features() from this module while it used
# pandas_ta_classic -- a hand-rolled rolling-mean RSI, for example, differs
# from pandas_ta's Wilder-smoothed RSI by an average of ~5 points and up to
# ~20 points on the same data, which is a real train/serve mismatch, not a
# cosmetic difference).
#
# The actual bug that prompted removing pandas_ta_classic entirely
# (TypeError: unsupported operand type(s) for -: 'float' and 'NoneType')
# is fixed here differently: by sanitizing every column to float64 -- via
# .astype(float), which evicts stray Python None from object-dtype columns --
# both BEFORE handing data to pandas_ta and AFTER getting results back, so a
# None can't survive to reach pandas_ta's internal arithmetic OR leak out of
# it into the final feature matrix.

import pandas as pd
import numpy as np
import pandas_ta_classic as ta

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'returns', 'vol_14', 'rsi', 'macd', 'atr', 'bb_width'
]

FEATURE_DEFAULTS = {
    'returns':  0.0,
    'vol_14':   0.0,
    'rsi':      50.0,   # Neutral RSI
    'macd':     0.0,    # Neutral MACD
    'atr':      0.0,
    'bb_width': 0.0,
}


def _sanitize_col(series: pd.Series) -> pd.Series:
    """
    Force a Series to float64 with no None/NaN/inf values.
    - pd.to_numeric(errors='coerce') converts non-numeric to NaN
    - .astype(float) evicts Python None from object-dtype columns; without
      it, None can survive pd.to_numeric under some pandas versions and
      later cause: TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
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
    Calculates and adds technical indicator features to the DataFrame using
    pandas_ta_classic (matching the pipeline trading_env.py used for
    training), with float64 sanitization on both sides of every pandas_ta
    call so a stray None can never reach its internal arithmetic or escape
    into the model's input matrix.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.

    Returns:
        DataFrame with exactly FEATURE_COLS columns, all float64, no None/NaN/inf.
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=FEATURE_COLS).fillna(FEATURE_DEFAULTS)

    df_copy = df.copy()

    # ── Step 1: Sanitize all OHLCV inputs BEFORE they reach pandas_ta ──────────
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_copy.columns:
            df_copy[col] = 0.0
        df_copy[col] = _sanitize_col(df_copy[col])

    close = df_copy['close']

    # ── Step 2: Returns & volatility ────────────────────────────────────────────
    df_copy['returns'] = close.pct_change().fillna(0.0)
    df_copy['vol_14']  = df_copy['returns'].rolling(window=14).std()

    # ── Step 3: RSI (pandas_ta, Wilder-smoothed -- matches training) ───────────
    # Wrapped in try/except: pandas_ta_classic==0.3.14b1 has an internal bug
    # where short windows (e.g. the model's 32-bar SEQUENCE_LEN, well under
    # what a 26+9-period MACD needs to fully stabilize) can produce a raw
    # Python None in an internal intermediate Series, which then crashes on
    # subtraction with TypeError: unsupported operand type(s) for -: 'float'
    # and 'NoneType' -- reproduced directly against this pinned version. This
    # isn't fixable by sanitizing our input (it already is clean float64);
    # it's the library's own arithmetic that produces the None internally.
    try:
        rsi_result = ta.rsi(close, length=14)
        df_copy['rsi'] = rsi_result if rsi_result is not None else np.nan
    except Exception as e:
        print(f"Warning: ta.rsi failed ({e}), using NaN (will fall back to neutral default)")
        df_copy['rsi'] = np.nan

    # ── Step 4: MACD (pandas_ta) ────────────────────────────────────────────────
    try:
        macd_result = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.empty and 'MACD_12_26_9' in macd_result.columns:
            df_copy['macd'] = macd_result['MACD_12_26_9']
        else:
            df_copy['macd'] = np.nan
    except Exception as e:
        print(f"Warning: ta.macd failed ({e}), using NaN (will fall back to neutral default)")
        df_copy['macd'] = np.nan

    # ── Step 5: ATR (pandas_ta) ─────────────────────────────────────────────────
    try:
        atr_result = ta.atr(df_copy['high'], df_copy['low'], close, length=14)
        df_copy['atr'] = atr_result if atr_result is not None else np.nan
    except Exception as e:
        print(f"Warning: ta.atr failed ({e}), using NaN (will fall back to neutral default)")
        df_copy['atr'] = np.nan

    # ── Step 6: Bollinger Band Width (pandas_ta) ────────────────────────────────
    try:
        bbands_result = ta.bbands(close, length=20, std=2.0)
        if bbands_result is not None and not bbands_result.empty and 'BBB_20_2.0' in bbands_result.columns:
            df_copy['bb_width'] = bbands_result['BBB_20_2.0']
        else:
            df_copy['bb_width'] = np.nan
    except Exception as e:
        print(f"Warning: ta.bbands failed ({e}), using NaN (will fall back to neutral default)")
        df_copy['bb_width'] = np.nan

    # ── Step 7: Fill rolling-window NaNs at the start of the series ────────────
    df_copy = df_copy.ffill().bfill()

    # ── Step 8: Final sanitization pass -- catches anything pandas_ta itself
    # left as None/NaN/inf, so it can never reach the scaler/model ────────────
    for col in FEATURE_COLS:
        if col not in df_copy.columns:
            df_copy[col] = FEATURE_DEFAULTS.get(col, 0.0)
        df_copy[col] = _sanitize_col(df_copy[col])

    return df_copy[FEATURE_COLS]
