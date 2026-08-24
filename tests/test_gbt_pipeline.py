# tests/test_gbt_pipeline.py — Steps 2 and 4: GBT baseline trainer
# (train_gbt_baseline.py) and the walk-forward champion/challenger
# promotion gate (promotion_gate.py). No network/live data needed: the
# backend factory and promotion-decision logic are pure, and the serving
# path is exercised against tiny artifacts trained on synthetic data.

import numpy as np
import pandas as pd
import pytest
import joblib

from feature_engineering import FEATURE_COLS


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


# ── train_gbt_baseline.make_classifier: backend factory ──────────────────────
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


# ── LightGBM / CatBoost backends serve through the real live paths ───────────
# (optional deps; skipped if not installed)
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
def test_boost_backends_work_as_shadow(tmp_path, kind):
    """LightGBM/CatBoost artifacts must load through the ShadowGBT loader."""
    import shadow_model
    path = _tiny_boost_artifact(tmp_path, f"shadow_{kind}.joblib", kind)
    s = shadow_model.ShadowGBT(path=path)
    assert s.available() is True
    row = {c: float(i % 3 - 1) for i, c in enumerate(FEATURE_COLS)}
    prob = s.predict_row(row)
    assert prob is not None and 0.0 <= prob <= 1.0


# ── promotion_gate.py: pure decision logic (no data/network needed) ──────────
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
