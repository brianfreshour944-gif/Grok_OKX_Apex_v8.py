# tests/test_data_feeds.py — scan_stable_assets() and get_clean_ohlcv_dataframe(),
# with data_client mocked out (no real Alpaca API calls, no network).

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from data_feeds import scan_stable_assets, get_clean_ohlcv_dataframe
from feature_engineering import add_features, FEATURE_COLS
from config import SEQUENCE_LEN
from conftest import run_async


def _fake_bar(volume, close):
    return MagicMock(volume=volume, close=close)


def _mock_data_client(volumes: dict, raise_for: set = frozenset()):
    """volumes: {symbol: (volume, close)}. raise_for: symbols whose fetch should fail."""
    def fake_get_crypto_bars(req):
        symbol = req.symbol_or_symbols
        if symbol in raise_for:
            raise RuntimeError(f"no data for {symbol}")
        result = MagicMock()
        vol, close = volumes[symbol]
        result.data = {symbol: [_fake_bar(vol, close)]}
        return result
    return MagicMock(get_crypto_bars=fake_get_crypto_bars)


def test_ranks_by_dollar_volume_descending(monkeypatch):
    import data_feeds
    volumes = {
        "BTC/USD": (100, 50000),   # $5,000,000
        "ETH/USD": (1000, 2000),  # $2,000,000
        "SOL/USD": (10000, 100),  # $1,000,000
    }
    monkeypatch.setattr(data_feeds, "data_client", _mock_data_client(volumes))

    result = run_async(scan_stable_assets(limit_scope=10, candidates=list(volumes.keys())))

    assert result == ["BTC/USD", "ETH/USD", "SOL/USD"]


def test_respects_limit_scope():
    import data_feeds as df_module
    volumes = {"BTC/USD": (100, 50000), "ETH/USD": (1000, 2000), "SOL/USD": (10000, 100)}
    import unittest.mock as um
    with um.patch.object(df_module, "data_client", _mock_data_client(volumes)):
        result = run_async(scan_stable_assets(limit_scope=2, candidates=list(volumes.keys())))
    assert result == ["BTC/USD", "ETH/USD"]


def test_restricts_to_given_candidates_only(monkeypatch):
    """This is the whole point of the `candidates` param: the live bot passes
    DYNAMIC_UNIVERSE_CANDIDATES (10 symbols the model was trained on) instead
    of the full 24-symbol DEFAULT_SCAN_CANDIDATES."""
    import data_feeds
    seen = []

    def fake_get_crypto_bars(req):
        symbol = req.symbol_or_symbols
        seen.append(symbol)
        result = MagicMock()
        result.data = {symbol: [_fake_bar(100, 100)]}
        return result

    monkeypatch.setattr(data_feeds, "data_client", MagicMock(get_crypto_bars=fake_get_crypto_bars))

    run_async(scan_stable_assets(limit_scope=10, candidates=["BTC/USD", "ETH/USD"]))

    assert set(seen) == {"BTC/USD", "ETH/USD"}


def test_falls_back_to_safe_default_when_everything_fails(monkeypatch):
    import data_feeds
    monkeypatch.setattr(
        data_feeds, "data_client",
        MagicMock(get_crypto_bars=MagicMock(side_effect=RuntimeError("API down"))),
    )

    result = run_async(scan_stable_assets(limit_scope=5, candidates=["BTC/USD", "ETH/USD"]))

    assert result == ["BTC/USD", "ETH/USD", "SOL/USD"]


def test_one_bad_symbol_does_not_abort_the_whole_scan(monkeypatch):
    import data_feeds
    volumes = {"BTC/USD": (100, 50000), "SOL/USD": (100, 100)}
    monkeypatch.setattr(data_feeds, "data_client", _mock_data_client(volumes, raise_for={"ETH/USD"}))

    result = run_async(scan_stable_assets(limit_scope=5, candidates=["BTC/USD", "ETH/USD", "SOL/USD"]))

    assert set(result) == {"BTC/USD", "SOL/USD"}


def test_default_candidates_used_when_none_given(monkeypatch):
    import data_feeds
    seen = []

    def fake_get_crypto_bars(req):
        seen.append(req.symbol_or_symbols)
        raise RuntimeError("just counting calls")

    monkeypatch.setattr(data_feeds, "data_client", MagicMock(get_crypto_bars=fake_get_crypto_bars))

    run_async(scan_stable_assets(limit_scope=5))  # no candidates passed

    assert set(seen) == set(data_feeds.DEFAULT_SCAN_CANDIDATES)


# ── get_clean_ohlcv_dataframe ────────────────────────────────────────────────
#
# Regression coverage for a real train/serve skew: this function used to
# truncate to exactly SEQUENCE_LEN raw bars BEFORE any features were
# computed. ml_predictor.predict_batch() then ran add_features() on that
# already-truncated frame, so rolling-window features (20-bar Z-scores,
# 14-bar vol-of-vol, etc.) for roughly the first half of the model's input
# window were computed from a partial window instead of real prior history.
# train_transformer.py computes add_features() on the FULL raw series first,
# then slices SEQUENCE_LEN-bar windows out of the feature matrix -- so the
# live bot's inputs diverged from what the model was trained on. Verified
# against the real model checkpoint: up to ~0.08 signal difference and
# BUY/no-BUY flips on ~17% of trials at the live BUY_SIGNAL threshold.
# The fix: this function no longer truncates before returning; predict_batch
# computes add_features() on the full frame it receives, matching training.

def _fake_15m_bar(ts, o, h, l, c, v, vwap, trade_count):
    return MagicMock(
        timestamp=ts, open=o, high=h, low=l, close=c,
        volume=v, vwap=vwap, trade_count=trade_count,
    )


def _make_bars(n, seed=0):
    rng = np.random.RandomState(seed)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Oldest bar first, matching Alpaca's returned order; last bar closes
    # well in the past so it isn't filtered as "still forming".
    start = now - timedelta(minutes=15 * (n + 2))
    dates = [start + timedelta(minutes=15 * i) for i in range(n)]
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n) * 0.3)
    low = close - np.abs(rng.randn(n) * 0.3)
    open_ = close + rng.randn(n) * 0.1
    volume = np.abs(rng.randn(n) * 1000 + 5000)
    vwap = close + rng.randn(n) * 0.05
    trade_count = np.abs(rng.randn(n) * 50 + 200)
    return [
        _fake_15m_bar(dates[i], open_[i], high[i], low[i], close[i], volume[i], vwap[i], trade_count[i])
        for i in range(n)
    ], dates, close, high, low, open_, volume, vwap, trade_count


def _mock_bars_client(bars, symbol="BTC/USD"):
    result = MagicMock()
    result.data = {symbol: bars}
    return MagicMock(get_crypto_bars=MagicMock(return_value=result))


def test_returns_more_than_sequence_len_when_available(monkeypatch):
    """The whole point of the fix: don't throw away the extra fetched bars
    before features are computed."""
    import data_feeds
    bars, *_ = _make_bars(64)
    monkeypatch.setattr(data_feeds, "data_client", _mock_bars_client(bars))

    df = run_async(get_clean_ohlcv_dataframe("BTC/USD"))

    assert df is not None
    assert len(df) == 64
    assert len(df) > SEQUENCE_LEN


def test_returns_none_when_fewer_than_sequence_len_bars(monkeypatch):
    import data_feeds
    bars, *_ = _make_bars(SEQUENCE_LEN - 1)
    monkeypatch.setattr(data_feeds, "data_client", _mock_bars_client(bars))

    df = run_async(get_clean_ohlcv_dataframe("BTC/USD"))

    assert df is None


def test_returns_none_when_close_filter_drops_below_sequence_len(monkeypatch):
    """Length check must run AFTER the close>0 filter, not just before it."""
    import data_feeds
    bars, dates, close, high, low, open_, volume, vwap, trade_count = _make_bars(SEQUENCE_LEN + 5)
    # Zero out enough closes that fewer than SEQUENCE_LEN valid bars remain
    # post-filter, even though the pre-filter bar count clears the bar.
    for i in range(6):
        bars[i] = _fake_15m_bar(dates[i], open_[i], high[i], low[i], 0.0, volume[i], vwap[i], trade_count[i])
    monkeypatch.setattr(data_feeds, "data_client", _mock_bars_client(bars))

    df = run_async(get_clean_ohlcv_dataframe("BTC/USD"))

    assert df is None


def test_features_on_returned_frame_match_train_style_computation(monkeypatch):
    """The actual regression test for the train/serve skew bug: features
    computed the way predict_batch does it (add_features on the full frame
    this function returns, then tail(seq_len)) must exactly match features
    computed the way train_transformer.py does it (add_features on the full
    raw series, then tail(seq_len)) -- for the SAME underlying price data."""
    import data_feeds
    n = 64
    bars, dates, close, high, low, open_, volume, vwap, trade_count = _make_bars(n, seed=42)
    monkeypatch.setattr(data_feeds, "data_client", _mock_bars_client(bars))

    df_live = run_async(get_clean_ohlcv_dataframe("BTC/USD"))
    assert df_live is not None

    # predict_batch's order of operations: add_features on the full frame,
    # then take the last seq_len rows of FEATURES.
    live_features = add_features(df_live.copy())[FEATURE_COLS].tail(SEQUENCE_LEN).reset_index(drop=True)

    # train_transformer.py's order of operations: add_features on the full
    # raw series, then slice a seq_len-bar window out of the feature matrix.
    df_full = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "vwap": vwap, "trade_count": trade_count,
    }, index=pd.DatetimeIndex(dates))
    train_features = add_features(df_full)[FEATURE_COLS].tail(SEQUENCE_LEN).reset_index(drop=True)

    pd.testing.assert_frame_equal(live_features, train_features, check_exact=False, atol=1e-9)
