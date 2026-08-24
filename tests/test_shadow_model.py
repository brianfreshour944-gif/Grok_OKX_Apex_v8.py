# tests/test_shadow_model.py — Step 3: non-trading GBT challenger inference.
#
# ShadowGBT never influences a trade; main_bot.py calls predict_row() once
# per symbol per cycle purely to log (challenger probability, champion
# signal, later outcome) rows for the step-4 bake-off. Covers: absence
# handled gracefully, first load, and mtime-based hot-swap without
# recreating the object (same reload contract as SafeMLPredictor).

import os

import joblib
import numpy as np

from feature_engineering import FEATURE_COLS


def _train_tiny_gbt(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=15, random_state=0).fit(X, y)


def test_available_false_and_predict_none_when_no_artifact(tmp_path):
    import shadow_model
    path = str(tmp_path / "gbt_challenger.joblib")
    s = shadow_model.ShadowGBT(path=path)
    assert s.available() is False
    assert s.predict_row({c: 0.0 for c in FEATURE_COLS}) is None


def test_predict_row_none_when_features_missing_or_empty(tmp_path):
    import shadow_model
    path = str(tmp_path / "gbt_challenger.joblib")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 11))
    y = (X[:, 0] > 0).astype(int)
    joblib.dump(_train_tiny_gbt(X, y), path)
    s = shadow_model.ShadowGBT(path=path)
    assert s.predict_row(None) is None
    assert s.predict_row({}) is None


def test_shadow_gbt_lifecycle(tmp_path):
    import shadow_model
    path = str(tmp_path / "gbt_challenger.joblib")

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


def test_load_failure_keeps_unavailable_and_does_not_raise(tmp_path):
    import shadow_model
    path = str(tmp_path / "gbt_challenger.joblib")
    with open(path, "wb") as f:
        f.write(b"not a real joblib artifact")
    s = shadow_model.ShadowGBT(path=path)
    assert s.available() is True  # file exists...
    assert s.predict_row({c: 0.0 for c in FEATURE_COLS}) is None  # ...but load failed, never raises


def test_get_shadow_gbt_returns_same_singleton():
    from shadow_model import get_shadow_gbt
    assert get_shadow_gbt() is get_shadow_gbt()
