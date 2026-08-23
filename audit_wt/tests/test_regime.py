# tests/test_regime.py — market regime classification and risk scaling.

import numpy as np
import pandas as pd
import pytest

from regime import compute_regime_and_trend, calculate_adjusted_risk
from config import MAX_SINGLE_TRADE_USD


def _make_df(n=40, base=100.0, daily_vol_pct=0.0, seed=0):
    """Synthetic OHLC bars with controllable bar-to-bar range as a % of price."""
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, base * daily_vol_pct / 100.0, n))
    closes = np.clip(closes, 1.0, None)
    highs = closes * (1 + daily_vol_pct / 100.0 / 2)
    lows = closes * (1 - daily_vol_pct / 100.0 / 2)
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes})


def test_quiet_regime_for_low_volatility():
    df = _make_df(n=40, daily_vol_pct=0.3, seed=1)
    regime, trend, atr_pct = compute_regime_and_trend(df)
    assert regime == "quiet"
    assert atr_pct < 2.0


def test_wild_regime_for_high_volatility():
    df = _make_df(n=40, daily_vol_pct=8.0, seed=2)
    regime, trend, atr_pct = compute_regime_and_trend(df)
    assert regime == "wild"
    assert atr_pct > 4.0


def test_insufficient_bars_defaults_safe_instead_of_nan():
    """
    Fewer than 14 bars means the ATR rolling window can't fill -> atr_pct
    would be NaN. NaN compared against any threshold silently evaluates to
    False in Python, which would fall through to "quiet" (least risk-averse)
    if unguarded. Must explicitly default to normal/neutral instead.
    """
    df = _make_df(n=5, daily_vol_pct=1.0, seed=3)
    regime, trend, atr_pct = compute_regime_and_trend(df)
    assert regime == "normal"
    assert trend == "neutral"
    assert not np.isnan(atr_pct)


def test_malformed_input_fails_safe():
    """
    All-None input sanitizes to price=0, which takes the explicit
    `atr_pct = 0.0 if price <= 0` branch (not the NaN-guard branch) --
    this is intentional, deterministic degenerate-data handling, not a
    crash or a NaN slipping through a comparison.
    """
    df = pd.DataFrame({"open": [None] * 20, "high": [None] * 20, "low": [None] * 20, "close": [None] * 20})
    regime, trend, atr_pct = compute_regime_and_trend(df)
    assert atr_pct == 0.0
    assert regime == "quiet"


def test_adjusted_risk_scales_down_in_high_volatility():
    equity = 10000.0
    low_vol_risk = calculate_adjusted_risk(equity, atr_pct=1.0)
    high_vol_risk = calculate_adjusted_risk(equity, atr_pct=9.0)
    assert high_vol_risk < low_vol_risk


def test_adjusted_risk_capped_at_max_single_trade():
    assert calculate_adjusted_risk(equity=10_000_000.0, atr_pct=1.0) <= MAX_SINGLE_TRADE_USD


def test_adjusted_risk_zero_atr_uses_full_base_risk():
    from config import BASE_RISK_PERCENT
    equity = 10000.0
    assert calculate_adjusted_risk(equity, atr_pct=0.0) == pytest.approx(equity * BASE_RISK_PERCENT)
