# tests/test_ml_predictor.py — SafeMLPredictor's hot-reload, decision-time
# feature capture, and sklearn/GBT dual-format serving (steps 3-5 of the
# model-improvement pipeline).
#
# Before this, main_bot.py instantiated SafeMLPredictor exactly once at
# startup and never checked again -- a promoted model would sit on disk
# unused until the process was restarted. These tests pin down that a
# reload actually happens on an mtime change, that a corrupt artifact never
# takes down the live predictor, and that the raw feature vector consumed
# by experience_capture.py (step 0) is captured correctly.

import os

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from feature_engineering import add_features, FEATURE_COLS


def make_ohlcv(n=40, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(1e3, 1e5, n),
        "vwap": (high + low + close) / 3,
        "trade_count": rng.integers(50, 500, n).astype(float),
    })


def tiny_torch_state(seed):
    from ml_predictor import GrokGQA_Transformer
    torch.manual_seed(seed)
    m = GrokGQA_Transformer(input_dim=11, seq_len=32, embed_dim=16,
                            num_layers=1, num_q_heads=4, num_kv_heads=2,
                            dropout=0.0)
    return m.state_dict()


@pytest.fixture()
def torch_champion(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    path = d / "champion.pth"
    torch.save(tiny_torch_state(1), path)
    # scaler present so the torch path loads one
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.random.default_rng(0).normal(size=(200, 11)))
    joblib.dump(sc, d / "feature_scaler.pkl")
    return str(path)


def _tiny_predictor(model_path):
    from ml_predictor import SafeMLPredictor
    return SafeMLPredictor(model_path=model_path, input_dim=11, seq_len=32,
                           embed_dim=16, num_layers=1, num_q_heads=4,
                           num_kv_heads=2, dropout=0.0)


def test_hot_reload_picks_up_new_weights(torch_champion):
    p = _tiny_predictor(torch_champion)
    df = make_ohlcv(40)
    out1 = p.predict_batch({"BTC/USD": df.copy()})
    assert p.scaler is not None

    torch.save(tiny_torch_state(99), torch_champion)   # new weights
    os.utime(torch_champion, (os.path.getmtime(torch_champion) + 10,) * 2)
    reloaded = p.reload_if_changed()
    assert reloaded is True
    out2 = p.predict_batch({"BTC/USD": df.copy()})
    assert out1["BTC/USD"] != out2["BTC/USD"], "weights did not change output"


def test_hot_reload_happens_automatically_inside_predict_batch(torch_champion):
    """reload_if_changed() is called FROM predict_batch() -- the live loop
    never calls it directly, so this must work without an explicit call."""
    p = _tiny_predictor(torch_champion)
    df = make_ohlcv(40)
    out1 = p.predict_batch({"BTC/USD": df.copy()})

    torch.save(tiny_torch_state(42), torch_champion)
    os.utime(torch_champion, (os.path.getmtime(torch_champion) + 10,) * 2)
    out2 = p.predict_batch({"BTC/USD": df.copy()})  # no reload_if_changed() call
    assert out1["BTC/USD"] != out2["BTC/USD"], "predict_batch did not auto-reload"


def test_hot_reload_failure_keeps_old_weights(torch_champion, capsys):
    p = _tiny_predictor(torch_champion)
    old_model = p.model
    df = make_ohlcv(40)
    before = p.predict_batch({"BTC/USD": df.copy()})

    with open(torch_champion, "wb") as f:
        f.write(b"this is not a torch checkpoint")
    os.utime(torch_champion, (os.path.getmtime(torch_champion) + 10,) * 2)

    out = p.predict_batch({"BTC/USD": df.copy()})   # must NOT raise
    assert p.model is old_model                     # previous weights still serving
    assert out["BTC/USD"] == before["BTC/USD"]
    assert "Hot-reload failed" in capsys.readouterr().out


def test_no_reload_when_artifact_unchanged(torch_champion):
    p = _tiny_predictor(torch_champion)
    p.predict_batch({"BTC/USD": make_ohlcv(40)})
    assert p.reload_if_changed() is False


def test_last_features_captured_raw(torch_champion):
    p = _tiny_predictor(torch_champion)
    df = make_ohlcv(40)
    p.predict_batch({"ETH/USD": df})
    snap = p.last_features.get("ETH/USD")
    assert snap is not None and set(snap.keys()) == set(FEATURE_COLS)
    expected = add_features(df)[FEATURE_COLS].iloc[-1]
    for c in FEATURE_COLS:
        assert snap[c] == pytest.approx(float(expected[c]), abs=1e-9)


def test_last_features_absent_when_window_too_short(torch_champion):
    p = _tiny_predictor(torch_champion)
    short_df = make_ohlcv(5)  # well under seq_len=32
    out = p.predict_batch({"BTC/USD": short_df})
    assert out["BTC/USD"] == 0.5
    assert "BTC/USD" not in p.last_features


# ── sklearn-champion path (a GBT that won the promotion gate) ────────────────
def test_sklearn_champion_predict(tmp_path):
    from ml_predictor import SafeMLPredictor
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 11))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=20).fit(X, y)
    path = str(tmp_path / "gbt.joblib")
    joblib.dump(clf, path)

    p = SafeMLPredictor(model_path=path, seq_len=32)
    assert p.scaler is None  # sklearn path needs no scaler
    df = make_ohlcv(40)
    out = p.predict_batch({"SOL/USD": df})
    assert 0.0 <= out["SOL/USD"] <= 1.0
    assert set(p.last_features["SOL/USD"].keys()) == set(FEATURE_COLS)


def test_missing_model_file_raises_file_not_found(tmp_path):
    from ml_predictor import SafeMLPredictor
    with pytest.raises(FileNotFoundError):
        SafeMLPredictor(model_path=str(tmp_path / "does_not_exist.pth"), seq_len=32)
