"""
experience_capture.py — Step 0 of the model-improvement pipeline.

Captures the thing every retraining/evaluation plan needs and nothing in this
repo previously recorded: the FEATURE VECTOR the champion model saw at
decision time, paired with what actually happened afterwards.

Why JSONL: append-only, crash-safe (one line per event), zero DB migration,
trivially replayable by training scripts. Trades are rare (~a few per day),
so a single flat file with size-based rotation is more than sufficient.

Event schema (schema=1):
  entry : one per successful BUY fill submission
      {schema, type, ts, symbol, signal, regime, trend, atr_pct, price,
       qty, trade_value, order_id, features:{FEATURE_COLS...}}
  exit  : one per successful SELL fill submission
      {schema, type, ts, symbol, avg_entry, price, qty, exit_reason,
       regime, held_hours, pnl_pct, order_id}
  shadow: one per symbol per cycle while a challenger GBT exists (step 3)
      {schema, type, ts, symbol, gbt_prob, transformer_signal, regime,
       trend, atr_pct, price}

Training joins entries→exits by symbol over time (an exit closes the most
recent still-open entry for that symbol); shadow rows are used directly as
(probability, forward outcome) pairs once joined to later prices.
"""

import json
import os
import threading
from datetime import datetime, timezone

from config import logger, EXPERIENCE_LOG_PATH

_WRITE_LOCK = threading.Lock()
_MAX_BYTES = 50 * 1024 * 1024  # rotate at 50 MB, keep one .1 backup


def _now_iso(ts=None) -> str:
    if ts is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.isoformat()


def _rotate_if_needed(path: str) -> None:
    try:
        if os.path.exists(path) and os.path.getsize(path) > _MAX_BYTES:
            backup = path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(path, backup)
    except OSError:
        pass  # rotation is best-effort; never block trading on it


def _append_event(event: dict) -> bool:
    """Append one JSON line. Returns False (never raises) on any failure so
    logging problems can never take down the trading loop."""
    try:
        line = json.dumps(event, separators=(",", ":"), default=str)
        with _WRITE_LOCK:
            _rotate_if_needed(EXPERIENCE_LOG_PATH)
            with open(EXPERIENCE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return True
    except Exception as e:
        logger.warning(f"⚠️ experience log write failed ({type(e).__name__}: {e})")
        return False


def log_entry_experience(symbol: str, *, signal: float, regime: str, trend: str,
                         atr_pct: float, price: float, qty: float,
                         trade_value: float, features: dict | None,
                         order_id: str | None = None, ts: float | None = None) -> bool:
    """Record the decision-time context + feature vector for a BUY."""
    ev = {
        "schema": 1,
        "type": "entry",
        "ts": _now_iso(ts),
        "symbol": symbol,
        "signal": round(float(signal), 6),
        "regime": regime,
        "trend": trend,
        "atr_pct": round(float(atr_pct), 6),
        "price": float(price),
        "qty": float(qty),
        "trade_value": round(float(trade_value), 2),
        "order_id": order_id,
        "features": {k: float(v) for k, v in (features or {}).items()},
    }
    ok = _append_event(ev)
    if ok:
        n_feat = len(ev["features"])
        logger.info(f"🧾 experience logged: ENTRY {symbol} @ {price:.6g} "
                    f"(signal={signal:.3f}, features={n_feat})")
    return ok


def log_exit_outcome(symbol: str, *, avg_entry: float, price: float, qty: float,
                     exit_reason: str, regime: str, held_hours: float,
                     pnl_pct: float, order_id: str | None = None,
                     ts: float | None = None) -> bool:
    """Record the realized outcome for a SELL (joins back to its entry row)."""
    ev = {
        "schema": 1,
        "type": "exit",
        "ts": _now_iso(ts),
        "symbol": symbol,
        "avg_entry": float(avg_entry),
        "price": float(price),
        "qty": float(qty),
        "exit_reason": str(exit_reason),
        "regime": regime,
        "held_hours": round(float(held_hours), 4),
        "pnl_pct": round(float(pnl_pct), 6),
        "order_id": order_id,
    }
    ok = _append_event(ev)
    if ok:
        logger.info(f"🧾 experience logged: EXIT {symbol} @ {price:.6g} "
                    f"(pnl={pnl_pct * 100:+.2f}%)")
    return ok


def log_shadow_prediction(symbol: str, *, gbt_prob: float | None,
                          transformer_signal: float, regime: str, trend: str,
                          atr_pct: float, price: float,
                          ts: float | None = None) -> bool:
    """Record one non-trading challenger prediction alongside the champion's."""
    ev = {
        "schema": 1,
        "type": "shadow",
        "ts": _now_iso(ts),
        "symbol": symbol,
        "gbt_prob": None if gbt_prob is None else round(float(gbt_prob), 6),
        "transformer_signal": round(float(transformer_signal), 6),
        "regime": regime,
        "trend": trend,
        "atr_pct": round(float(atr_pct), 6),
        "price": float(price),
    }
    return _append_event(ev)


def load_experiences(path: str | None = None) -> list[dict]:
    """Read all events (skipping malformed lines) — used by training/gate tools."""
    path = path or EXPERIENCE_LOG_PATH
    events = []
    if not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events