# regime.py — Market regime classification and volatility-adjusted risk sizing.

import pandas as pd
from config import logger, BASE_RISK_PERCENT, MAX_SINGLE_TRADE_USD


def compute_regime_and_trend(df: pd.DataFrame):
    """
    Classifies the current market regime (wild / normal / quiet) by ATR%,
    and the trend direction (up / down) relative to EMA-50.
    Returns (regime, trend, atr_pct).

    All column arithmetic uses sanitized float64 local Series to prevent:
        TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
    """
    try:
        high  = pd.to_numeric(df["high"],  errors="coerce").astype(float).fillna(0.0)
        low   = pd.to_numeric(df["low"],   errors="coerce").astype(float).fillna(0.0)
        close = pd.to_numeric(df["close"], errors="coerce").astype(float).fillna(0.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr     = tr.rolling(14).mean().iloc[-1]
        price   = close.iloc[-1]
        atr_pct = (atr / price) * 100 if price > 0 else 0.0
        ema50   = close.ewm(span=50).mean().iloc[-1]

        trend  = "up"    if price > ema50  else "down"
        regime = "wild"  if atr_pct > 4.0 else "normal" if atr_pct > 2.0 else "quiet"
        return regime, trend, round(atr_pct, 2)

    except Exception:
        return "normal", "neutral", 2.0


def calculate_adjusted_risk(equity: float, atr_pct: float) -> float:
    """
    Scales the per-trade risk dollar amount down when ATR% exceeds a baseline.
    Returns a dollar amount capped at MAX_SINGLE_TRADE_USD.

    atr_pct is already expressed as a percentage number (e.g. 9.0 = 9%).
    """
    baseline_vol = 1.5   # % — matches atr_pct scale
    if atr_pct > baseline_vol and atr_pct > 0:
        vol_scaler = baseline_vol / atr_pct
        adjusted   = equity * BASE_RISK_PERCENT * vol_scaler
        logger.info(
            f"⚠️ High volatility (ATR%={atr_pct:.2f}%) — "
            f"scaling risk by {vol_scaler:.2f}x"
        )
    else:
        adjusted = equity * BASE_RISK_PERCENT

    return min(adjusted, MAX_SINGLE_TRADE_USD)
