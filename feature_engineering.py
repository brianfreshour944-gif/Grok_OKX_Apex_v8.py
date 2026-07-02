# feature_engineering.py - Refined for robustness

import pandas as pd
import pandas_ta_classic as ta
import numpy as np

# Define the feature columns that will be used by the model
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'returns', 'vol_14', 'rsi', 'macd', 'atr', 'bb_width'
]

# Define default values for features, used if calculation fails or data is insufficient
FEATURE_DEFAULTS = {
    'returns': 0.0,
    'vol_14': 0.0,
    'rsi': 50.0,    # Neutral RSI
    'macd': 0.0,    # Neutral MACD
    'atr': 0.0,
    'bb_width': 0.0
}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds technical indicator features to the DataFrame.
    Handles cases where df might be too short for some indicators.

    Args:
        df (pd.DataFrame): Input DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.

    Returns:
        pd.DataFrame: DataFrame with added features, structured exactly for the GQA Transformer.
    """
    if df.empty:
        # Return an empty DataFrame with FEATURE_COLS as columns, filled with defaults
        return pd.DataFrame(index=df.index, columns=FEATURE_COLS).fillna(FEATURE_DEFAULTS)

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Ensure required columns are numeric and present
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_copy.columns:
            print(f"Warning: Missing critical column '{col}' in DataFrame for feature engineering.")
            # Create a DataFrame with default values for missing critical columns
            temp_df = pd.DataFrame(index=df.index, columns=FEATURE_COLS)
            for feature in FEATURE_COLS:
                temp_df[feature] = FEATURE_DEFAULTS.get(feature, 0.0) 
            return temp_df
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Calculate 'returns'
    df_copy['returns'] = df_copy['close'].pct_change().fillna(0)

    # Calculate 'vol_14' (simple rolling standard deviation of returns)
    df_copy['vol_14'] = df_copy['returns'].rolling(window=14).std()

    # Apply pandas_ta indicators safely
    # RSI
    df_copy['rsi'] = ta.rsi(df_copy['close'], length=14)

    # MACD
    macd_result = ta.macd(df_copy['close'], fast=12, slow=26, signal=9)
    if macd_result is not None and not macd_result.empty and 'MACD_12_26_9' in macd_result.columns:
        df_copy['macd'] = macd_result['MACD_12_26_9']
    else:
        df_copy['macd'] = np.nan

    # ATR (Average True Range)
    df_copy['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=14)

    # Bollinger Bands Width
    bbands_result = ta.bbands(df_copy['close'], length=20, std=2.0)
    if bbands_result is not None and not bbands_result.empty and 'BBB_20_2.0' in bbands_result.columns:
        df_copy['bb_width'] = bbands_result['BBB_20_2.0']
    else:
        df_copy['bb_width'] = np.nan

    # Fill rolling windows NaNs at the very beginning of the dataset series
    # Using forward fill first to preserve valid recent observations, then backfill
    df_copy = df_copy.ffill().bfill()

    # Hard safeguard: Fill any remaining NaNs with hardcoded neutral FEATURE_DEFAULTS values
    for col, default_val in FEATURE_DEFAULTS.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(default_val)
        else:
            # Ensure all FEATURE_COLS are present, even if data was missing to calculate them
            df_copy[col] = default_val

    # Final check to guarantee column sorting order matches training matrices perfectly
    for col in FEATURE_COLS:
        if col not in df_copy.columns:
            df_copy[col] = FEATURE_DEFAULTS.get(col, 0.0)
            
    return df_copy[FEATURE_COLS]
