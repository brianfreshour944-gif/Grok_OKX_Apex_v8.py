"""
shadow_model.py — Step 3 support: non-trading challenger inference.

Loads the GBT challenger artifact (gbt_challenger.joblib, produced by
train_gbt_baseline.py) lazily and hot-reloads it on mtime change — same
pattern as SafeMLPredictor's champion reload, so a promotion-gate retrain is
picked up by the live loop without a restart.

This model NEVER trades. main_bot.py calls predict_row() once per symbol per
cycle purely to log (challenger probability, champion signal, regime, later
outcome) rows into shadow_predictions.jsonl for the step-4 bake-off.
"""

import os

import joblib

from config import logger, GBT_CHALLENGER_PATH
from feature_engineering import FEATURE_COLS


class ShadowGBT:
    def __init__(self, path: str = GBT_CHALLENGER_PATH):
        self.path = path
        self._clf = None
        self._mtime = None

    def available(self) -> bool:
        return os.path.exists(self.path)

    def _ensure_loaded(self) -> None:
        if not self.available():
            self._clf = None
            self._mtime = None
            return
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if self._clf is not None and mtime == self._mtime:
            return
        try:
            clf = joblib.load(self.path)
            self._clf = clf
            self._mtime = mtime
            logger.info(f"🧠 Shadow GBT loaded from {self.path}")
        except Exception as e:
            if self._clf is None:
                # First load failed: stay unavailable, retry on next mtime change.
                logger.warning(f"⚠️ Shadow GBT load failed ({e}); shadow mode inactive")
                self._mtime = None
            else:
                # Keep serving previous weights; resync mtime to avoid retry storm.
                logger.warning(f"⚠️ Shadow GBT reload failed, keeping previous: {e}")
                self._mtime = mtime

    def predict_row(self, features: dict | None) -> float | None:
        """P(up) for one decision-time feature row. None when no model."""
        self._ensure_loaded()
        if self._clf is None or not features:
            return None
        try:
            row = [[float(features.get(c, 0.0)) for c in FEATURE_COLS]]
            proba = self._clf.predict_proba(row)[0]
            classes = list(getattr(self._clf, "classes_", [0, 1]))
            return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
        except Exception as e:
            logger.warning(f"⚠️ Shadow GBT predict failed: {e}")
            return None


_singleton: ShadowGBT | None = None


def get_shadow_gbt() -> ShadowGBT:
    global _singleton
    if _singleton is None:
        _singleton = ShadowGBT()
    return _singleton