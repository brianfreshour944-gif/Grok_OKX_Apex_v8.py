"""Verify every runtime dependency of the bot imports cleanly."""
import importlib
import os
import sys

# Optional GBT backends are checked separately (warn, don't fail).
CORE_MODS = [
    "numpy", "pandas", "scipy", "torch", "sklearn", "joblib",
    "alpaca", "alpaca.trading", "alpaca.data", "aiohttp", "psycopg2",
    "dotenv", "pytz",
]
OPTIONAL_MODS = ["lightgbm", "catboost"]
failed = []
missing_optional = []
for m in CORE_MODS:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"OK   {m:<18} {ver}")
    except Exception as e:
        failed.append(m)
        print(f"FAIL {m:<18} {type(e).__name__}: {e}")

for m in OPTIONAL_MODS:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"OK   {m:<18} {ver} (optional GBT backend)")
    except Exception as e:
        missing_optional.append(m)
        print(f"MISS {m:<18} optional — pip install {m} ({type(e).__name__})")

# Also import every bot module end-to-end (catches missing names/typos).
# Resolve bot modules relative to THIS script's location (audit_wt/_audit/
# -> audit_wt), mirroring how the deployed container runs from /app.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("APCA_API_KEY_ID", "test")
os.environ.setdefault("APCA_API_SECRET_KEY", "test")
for m in ["config", "data_feeds", "feature_engineering", "ml_predictor",
          "regime", "portfolio", "orders", "exit_logic", "database",
          "experience_capture", "shadow_model", "notifications", "money"]:
    try:
        importlib.import_module(m)
        print(f"OK   module {m}")
    except Exception as e:
        failed.append(m)
        print(f"FAIL module {m}: {type(e).__name__}: {e}")

if failed:
    print("RESULT:", f"FAILED: {failed}")
else:
    note = f" (optional backends not installed: {missing_optional})" if missing_optional else ""
    print("RESULT: ALL_OK" + note)
