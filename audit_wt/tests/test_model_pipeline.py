"""
Tests for the model-improvement pipeline (steps 0-5):
experience capture, champion hot-reload (+failure resilience),
sklearn-champion path, shadow GBT loader, promotion-gate logic,
and the pooled-IC cross-symbol fix in signal_ic_check.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import joblib
import torch

from feature_engineering import add_features, FEATURE_COLS


# ── helpers ────────────────────────────────────────────────────────────────────
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
def exp_log(tmp_path, monkeypatch):
    from config import logger  # noqa: F401  (ensures config imported first)
    import experience_capture
    path = str(tmp_path / "live_experiences.jsonl")
    monkeypatch.setattr(experience_capture, "EXPERIENCE_LOG_PATH", path)
    yield path


# ── Step 0: experience capture ────────────────────────────────────────────────
def test_entry_exit_shadow_events_roundtrip(exp_log):
    from experience_capture import (
        log_entry_experience, log_exit_outcome, log_shadow_prediction,
        load_experiences,
    )
    feats = {c: float(i) for i, c in enumerate(FEATURE_COLS)}
    assert log_entry_experience("BTC/USD", signal=0.62, regime="normal",
                                trend="up", atr_pct=1.5, price=50000.0,
                                qty=0.001, trade_value=50.0, features=feats)
    assert log_exit_outcome("BTC/USD", avg_entry=50000.0, price=51000.0,
                            qty=0.001, exit_reason="PROFIT_TARGET",
                            regime="normal", held_hours=1.2, pnl_pct=0.02)
    assert log_shadow_prediction("BTC/USD", gbt_prob=0.57,
                                 transformer_signal=0.62, regime="normal",
                                 trend="up", atr_pct=1.5, price=50000.0)

    events = load_experiences(exp_log)
    assert [e["type"] for e in events] == ["entry", "exit", "shadow"]
    assert events[0]["features"] == feats
    assert events[1]["pnl_pct"] == 0.02
    assert events[2]["gbt_prob"] == 0.57


def test_load_experiences_skips_malformed_lines(exp_log):
    from experience_capture import load_experiences
    with open(exp_log, "w", encoding="utf-8") as f:
        f.write('{"type":"entry","symbol":"BTC/USD"}\n')
        f.write("GARBAGE LINE\n")
        f.write("\n")
    events = load_experiences(exp_log)
    assert len(events) == 1 and events[0]["symbol"] == "BTC/USD"


def test_rotation_creates_backup(exp_log, monkeypatch):
    import experience_capture
    monkeypatch.setattr(experience_capture, "_MAX_BYTES", 300)
    for i in range(6):  # each entry line > 300 bytes total eventually
        experience_capture.log_entry_experience(
            "BTC/USD", signal=0.6, regime="normal", trend="up", atr_pct=1.0,
            price=100.0 + i, qty=0.1, trade_value=10.0,
            features={c: 0.0 for c in FEATURE_COLS})
    assert os.path.exists(exp_log + ".1"), "rotation backup missing"


# ── Step 5: champion hot-reload ───────────────────────────────────────────────
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


def test_hot_reload_picks_up_new_weights(torch_champion):
    from ml_predictor import SafeMLPredictor
    p = SafeMLPredictor(model_path=torch_champion, input_dim=11, seq_len=32,
                        embed_dim=16, num_layers=1, num_q_heads=4,
                        num_kv_heads=2, dropout=0.0)
    df = make_ohlcv(40)
    out1 = p.predict_batch({"BTC/USD": df.copy()})
    assert p.scaler is not None

    torch.save(tiny_torch_state(99), torch_champion)   # new weights
    os.utime(torch_champion, (os.path.getmtime(torch_champion) + 10,) * 2)
    reloaded = p.reload_if_changed()
    assert reloaded is True
    out2 = p.predict_batch({"BTC/USD": df.copy()})
    assert out1["BTC/USD"] != out2["BTC/USD"], "weights did not change output"


def test_hot_reload_failure_keeps_old_weights(torch_champion, capsys):
    from ml_predictor import SafeMLPredictor
    p = SafeMLPredictor(model_path=torch_champion, input_dim=11, seq_len=32,
                        embed_dim=16, num_layers=1, num_q_heads=4,
                        num_kv_heads=2, dropout=0.0)
    old_model = p.model
    df = make_ohlcv(40)
    before = p.predict_batch({"BTC/USD": df.copy()})

    with open(torch_champion, "wb") as f:
        f.write(b"this is not a torch checkpoint")
    os.utime(torch_champion, (os.path.getmtime(torch_champion) + 10,) * 2)

    out = p.predict_batch({"BTC/USD": df.copy()})   # must NOT raise
    assert p.model is old_model                     # previous weights serving
    assert out["BTC/USD"] == before["BTC/USD"]
    assert "Hot-reload failed" in capsys.readouterr().out


def test_last_features_captured_raw(torch_champion):
    from ml_predictor import SafeMLPredictor
    p = SafeMLPredictor(model_path=torch_champion, input_dim=11, seq_len=32,
                        embed_dim=16, num_layers=1, num_q_heads=4,
                        num_kv_heads=2, dropout=0.0)
    df = make_ohlcv(40)
    p.predict_batch({"ETH/USD": df})
    snap = p.last_features.get("ETH/USD")
    assert snap is not None and set(snap.keys()) == set(FEATURE_COLS)
    expected = add_features(df)[FEATURE_COLS].iloc[-1]
    for c in FEATURE_COLS:
        assert snap[c] == pytest.approx(float(expected[c]), abs=1e-9)


# ── sklearn-champion path (step 4 promotion target) ──────────────────────────
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
    df = make_ohlcv(40)
    out = p.predict_batch({"SOL/USD": df})
    assert 0.0 <= out["SOL/USD"] <= 1.0
    assert set(p.last_features["SOL/USD"].keys()) == set(FEATURE_COLS)


# ── Step 3: shadow GBT loader ─────────────────────────────────────────────────
def _train_tiny_gbt(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=15,
                                          random_state=0).fit(X, y)


def test_shadow_gbt_lifecycle(tmp_path, monkeypatch):
    import shadow_model
    path = str(tmp_path / "gbt_challenger.joblib")
    monkeypatch.setattr(shadow_model, "GBT_CHALLENGER_PATH", path)

    s = shadow_model.ShadowGBT(path=path)
    assert s.available() is False
    assert s.predict_row({c: 0.0 for c in FEATURE_COLS}) is None

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 11))
    y = (X[:, 2] > 0).astype(int)
    joblib.dump(_train_tiny_gbt(X, y), path)
    row = {c: float(v) for c, v in zip(FEATURE_COLS, X[0])}
    p1 = s.predict_row(row)
    assert p1 is not None and 0.0 <= p1 <= 1.0

    # hot-swap artifact -> new predictions without recreating the object
    X2 = rng.normal(size=(300, 11)) + 5.0
    y2 = (X2[:, 2] > 5.0).astype(int)
    joblib.dump(_train_tiny_gbt(X2, y2), path)
    os.utime(path, (os.path.getmtime(path) + 10,) * 2)
    p2 = s.predict_row(row)
    assert p2 != p1


# ── Step 4: promotion-gate pure logic ─────────────────────────────────────────
def test_decide_promotion_majority_and_mean():
    from promotion_gate import decide_promotion
    ok, why = decide_promotion([0.05, 0.04, 0.06, 0.03], [-0.01, 0.00, 0.01, -0.02])
    assert ok is True and "won 4/4" in why
    ok, _ = decide_promotion([-0.05, -0.04, -0.06], [0.01, 0.00, 0.02])
    assert ok is False
    # exactly 2 wins of 4 < need=ceil(0.6*4)=3 -> not promoted
    ok, why = decide_promotion([0.05, 0.06, -0.10, -0.20],
                               [0.00, 0.01, 0.00, -0.03])
    assert ok is False and "won 2/4" in why


def test_decide_promotion_drops_none_folds():
    from promotion_gate import decide_promotion
    ok, why = decide_promotion([None, 0.05, 0.04], [None, 0.00, 0.01])
    assert ok is True and "won 2/2" in why
    ok, why = decide_promotion([None, None], [None, 0.01])
    assert ok is False and why == "no scoreable folds"


def test_fold_boundaries_contiguous_and_complete():
    from promotion_gate import fold_boundaries
    ts = pd.Series(pd.date_range("2026-01-01", periods=100, freq="h"))
    folds = fold_boundaries(ts, 5)
    assert len(folds) == 5
    # edges keep the series' native dtype and span the full timeline
    assert all(isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp)
               for a, b in folds)
    assert folds[0][0] == ts.min() and folds[-1][1] == ts.max()
    starts = [a for a, _ in folds]
    assert starts == sorted(starts)
    for (a1, b1), (a2, b2) in zip(folds, folds[1:]):
        assert a2 > b1  # contiguous, non-overlapping


def test_evaluate_fold_metrics_nan_safe():
    from promotion_gate import evaluate_fold_metrics
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(0.3, 0.7, 500)
    fwd = rng.normal(0, 0.01, 500)
    m = evaluate_fold_metrics(y, p, fwd)
    assert m is not None and m["auc"] is not None and m["ic"] is not None
    # constant probabilities -> undefined IC must be None, not NaN
    m2 = evaluate_fold_metrics(y, np.full(500, 0.5), fwd)
    assert m2["ic"] is None
    assert evaluate_fold_metrics(y[:50], p[:50], fwd[:50]) is None  # too small


# ── F8 regression: pooled IC must not shift across symbol boundaries ─────────
def test_pooled_ic_fix_matches_per_symbol_pooling():
    """The fixed pooled computation (per-symbol shift BEFORE concat) must
    equal the correctly-pooled series, and differ from the buggy concat-shift
    when price scales differ across symbols."""
    rng = np.random.default_rng(11)
    n, h = 600, 6
    # Oracle-signal construction: preds track each symbol's ACTUAL h-bar
    # forward return, so per-symbol/pooled-fixed IC is strongly positive.
    # This is the scenario where the buggy method's fabricated cross-symbol
    # rows have maximum leverage on Pearson r.
    def make_symbol(level):
        w = rng.normal(0, .01, n)
        close = pd.Series(level * np.exp(np.cumsum(w)))
        fwd = close.shift(-h) / close - 1.0
        preds = 0.5 + 8 * fwd.fillna(0.0) + pd.Series(rng.normal(0, .02, n))
        return close, preds

    close_A, preds_A = make_symbol(100.0)
    close_B, preds_B = make_symbol(50000.0)
    preds = pd.concat([preds_A, preds_B], ignore_index=True)

    # FIXED method (what signal_ic_check.py now does)
    fwd_fixed = pd.concat([close_A.shift(-h) / close_A - 1.0,
                           close_B.shift(-h) / close_B - 1.0],
                          ignore_index=True)
    # OLD buggy method (concat closes THEN shift)
    cc = pd.concat([close_A, close_B], ignore_index=True)
    fwd_buggy = cc.shift(-h) / cc - 1.0

    v = preds.notna() & fwd_fixed.notna()
    vb = preds.notna() & fwd_buggy.notna()
    from scipy.stats import pearsonr
    r_fixed = pearsonr(preds[v], fwd_fixed[v])[0]
    r_buggy = pearsonr(preds[vb], fwd_buggy[vb])[0]

    # Structural difference at the boundary: the fixed method has NO data
    # for each symbol's last h bars (no future within the symbol -> NaN,
    # correctly masked out); the buggy method fabricates cross-symbol
    # "returns" there (e.g. $100 coin -> $50k coin).
    assert fwd_fixed.iloc[n - h:n].isna().all()
    assert fwd_buggy.iloc[n - h:n].abs().max() > 1.0   # ~+46,000% junk

    # With real signal present, the fabricated rows must materially corrupt
    # the headline pooled Pearson IC.
    assert r_fixed > 0.5, f"sanity: oracle pooled IC should be high, got {r_fixed:.4f}"
    assert abs(r_fixed - r_buggy) > 0.3, (
        f"expected buggy pooling to distort IC; got fixed={r_fixed:.4f} "
        f"buggy={r_buggy:.4f}")


# ── LightGBM / CatBoost backends (optional deps; skipped if not installed) ───
def _tiny_boost_artifact(tmp_path, name, kind):
    """Train a tiny boosted classifier of the given backend and dump it."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 11))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    if kind == "lgbm":
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=15, verbosity=-1,
                             random_state=0).fit(X, y)
    else:
        pytest.importorskip("catboost")
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(iterations=15, verbose=False,
                                 allow_writing_files=False,
                                 random_seed=0).fit(X, y)
    path = str(tmp_path / name)
    joblib.dump(clf, path)
    return path


@pytest.mark.parametrize("kind", ["lgbm", "catboost"])
def test_boost_backends_serve_as_champion(tmp_path, kind):
    """A promoted LightGBM/CatBoost artifact must serve through the SAME
    SafeMLPredictor *.joblib path with zero code changes."""
    from ml_predictor import SafeMLPredictor
    path = _tiny_boost_artifact(tmp_path, f"champ_{kind}.joblib", kind)
    p = SafeMLPredictor(model_path=path, seq_len=32)
    out = p.predict_batch({"BTC/USD": make_ohlcv(40)})
    assert 0.0 <= out["BTC/USD"] <= 1.0
    assert set(p.last_features["BTC/USD"].keys()) == set(FEATURE_COLS)


@pytest.mark.parametrize("kind", ["lgbm", "catboost"])
def test_boost_backends_work_as_shadow(tmp_path, monkeypatch, kind):
    """LightGBM/CatBoost artifacts must load through the ShadowGBT loader."""
    import shadow_model
    path = _tiny_boost_artifact(tmp_path, f"shadow_{kind}.joblib", kind)
    monkeypatch.setattr(shadow_model, "GBT_CHALLENGER_PATH", path)
    s = shadow_model.ShadowGBT(path=path)
    assert s.available() is True
    row = {c: float(i % 3 - 1) for i, c in enumerate(FEATURE_COLS)}
    prob = s.predict_row(row)
    assert prob is not None and 0.0 <= prob <= 1.0


def test_trainer_backend_factory():
    from train_gbt_baseline import make_classifier
    hgb = make_classifier("hgb")
    assert hasattr(hgb, "predict_proba")
    try:
        import lightgbm  # noqa: F401
        assert hasattr(make_classifier("lgbm"), "predict_proba")
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401
        assert hasattr(make_classifier("catboost"), "predict_proba")
    except ImportError:
        pass
    with pytest.raises(ValueError):
        make_classifier("xgboost")


# ── main_bot wiring sanity (source-level; full loop needs live API) ──────────
def test_main_bot_wired_for_steps_0_and_3():
    src = open(os.path.join(os.path.dirname(__file__), "..", "main_bot.py"),
               encoding="utf-8").read()
    assert "log_entry_experience(" in src
    assert "log_exit_outcome(" in src
    assert "log_shadow_prediction(" in src
    assert "get_shadow_gbt()" in src
    assert "predictor.last_features.get(symbol)" in src