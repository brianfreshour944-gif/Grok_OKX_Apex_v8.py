# tests/test_feature_engineering.py — add_features() row-preservation and
# NaN/inf-safety guarantees. These matter because ml_predictor.py slices
# `.tail(seq_len)` off the OUTPUT of add_features() and only checks the
# resulting length -- if add_features() ever dropped rows (a dropna(), a
# boolean-mask filter, etc.) that length check would be silently checking
# the wrong thing after a sequence had already been shortened upstream.

import numpy as np
import pandas as pd
import pytest

from feature_engineering import add_features, FEATURE_COLS


def _make_ohlcv(n=32, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.clip(close, 1.0, None)
    high = close * 1.01
    low = close * 0.99
    open_ = close
    volume = rng.uniform(100, 1000, n)
    vwap = close
    trade_count = rng.integers(1, 100, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "vwap": vwap, "trade_count": trade_count},
        index=idx,
    )


def test_row_count_is_preserved():
    df = _make_ohlcv(32)
    out = add_features(df)
    assert len(out) == len(df)


def test_output_has_exactly_feature_cols_no_more_no_less():
    df = _make_ohlcv(32)
    out = add_features(df)
    assert list(out.columns) == FEATURE_COLS


def test_no_nan_or_inf_in_output():
    df = _make_ohlcv(32)
    out = add_features(df)
    assert not out.isna().any().any()
    assert np.isfinite(out.to_numpy()).all()


def test_empty_dataframe_returns_empty_with_correct_columns():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"])
    out = add_features(empty)
    assert len(out) == 0
    assert list(out.columns) == FEATURE_COLS


def test_none_dataframe_returns_empty():
    out = add_features(None)
    assert len(out) == 0
    assert list(out.columns) == FEATURE_COLS


def test_missing_vwap_and_trade_count_columns_fall_back_safely():
    df = _make_ohlcv(32).drop(columns=["vwap", "trade_count"])
    out = add_features(df)
    assert len(out) == len(df)
    assert not out.isna().any().any()


def test_zero_high_low_range_does_not_produce_nan():
    """A completely flat bar (high == low == close, zero range) must not
    produce NaN/inf via log(H/L) or division by a zero range."""
    df = _make_ohlcv(32)
    df["high"] = df["close"]
    df["low"] = df["close"]
    out = add_features(df)
    assert not out.isna().any().any()
    assert np.isfinite(out.to_numpy()).all()


def test_single_row_input_does_not_crash():
    df = _make_ohlcv(1)
    out = add_features(df)
    assert len(out) == 1
    assert not out.isna().any().any()
