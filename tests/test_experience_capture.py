# tests/test_experience_capture.py — Step 0 of the model-improvement
# pipeline: decision-time feature vectors + outcomes, logged as JSONL so
# later training/gating tools have something to actually train/evaluate on.
# database.record_trade() only ever logged execution facts (symbol, side,
# price, qty, fee, fill_price) -- never the feature vector the model saw --
# so nothing downstream (GBT training, promotion-gate evaluation, a real
# signal IC on live data) was buildable without this.

import os

import pytest

from feature_engineering import FEATURE_COLS


@pytest.fixture()
def exp_log(tmp_path, monkeypatch):
    import experience_capture
    path = str(tmp_path / "live_experiences.jsonl")
    monkeypatch.setattr(experience_capture, "EXPERIENCE_LOG_PATH", path)
    yield path


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


def test_load_experiences_returns_empty_list_when_file_missing(tmp_path):
    from experience_capture import load_experiences
    missing = str(tmp_path / "does_not_exist.jsonl")
    assert load_experiences(missing) == []


def test_rotation_creates_backup(exp_log, monkeypatch):
    import experience_capture
    monkeypatch.setattr(experience_capture, "_MAX_BYTES", 300)
    for i in range(6):  # each entry line pushes well past 300 bytes total
        experience_capture.log_entry_experience(
            "BTC/USD", signal=0.6, regime="normal", trend="up", atr_pct=1.0,
            price=100.0 + i, qty=0.1, trade_value=10.0,
            features={c: 0.0 for c in FEATURE_COLS})
    assert os.path.exists(exp_log + ".1"), "rotation backup missing"


def test_log_functions_never_raise_on_write_failure(exp_log, monkeypatch):
    """Logging must never be able to take down the trading loop."""
    import experience_capture

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(experience_capture, "_rotate_if_needed", _boom)
    assert experience_capture.log_entry_experience(
        "BTC/USD", signal=0.6, regime="normal", trend="up", atr_pct=1.0,
        price=100.0, qty=0.1, trade_value=10.0, features={}) is False
