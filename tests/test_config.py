# tests/test_config.py — config.py env-var overrides (LOG_LEVEL, BUY_SIGNAL,
# SELL_SIGNAL). .env.example has always documented these as configurable;
# config.py used to hardcode them regardless of what was set in the
# environment, silently ignoring the documented override.
#
# These run in a FRESH subprocess rather than importlib.reload()-ing config
# in-process: config.py calls logging.basicConfig() at import time, which
# is a documented no-op if the root logger already has handlers configured
# (true after the first `import config` anywhere earlier in this test
# session) -- reloading in-process would silently fail to re-apply
# LOG_LEVEL, giving an unreliable test. A subprocess also matches how
# config.py is actually loaded in production: exactly once, in a fresh
# process.

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _base_env(**overrides):
    env = dict(os.environ)
    env["APCA_API_KEY_ID"] = "test"
    env["APCA_API_SECRET_KEY"] = "test"
    for key in ("BUY_SIGNAL", "SELL_SIGNAL", "LOG_LEVEL"):
        env.pop(key, None)
    env.update(overrides)
    return env


def _run(code: str, env: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    return result.stdout.strip().splitlines()[-1]


def test_buy_sell_signal_overridable_via_env():
    out = _run(
        "from config import BUY_SIGNAL, SELL_SIGNAL; print(f'{BUY_SIGNAL},{SELL_SIGNAL}')",
        _base_env(BUY_SIGNAL="0.60", SELL_SIGNAL="0.40"),
    )
    assert out == "0.6,0.4"


def test_buy_sell_signal_default_when_env_unset():
    out = _run(
        "from config import BUY_SIGNAL, SELL_SIGNAL; print(f'{BUY_SIGNAL},{SELL_SIGNAL}')",
        _base_env(),
    )
    assert out == "0.51,0.45"


def test_log_level_overridable_via_env():
    out = _run(
        "from config import logger; print(logger.getEffectiveLevel())",
        _base_env(LOG_LEVEL="DEBUG"),
    )
    assert out == "10"  # logging.DEBUG


def test_log_level_defaults_to_info_when_env_unset():
    out = _run(
        "from config import logger; print(logger.getEffectiveLevel())",
        _base_env(),
    )
    assert out == "20"  # logging.INFO


def test_log_level_invalid_value_falls_back_to_info():
    """getattr(logging, 'NOT_A_LEVEL', logging.INFO) -- an unrecognized
    LOG_LEVEL must fail safe to INFO rather than crashing at import time."""
    out = _run(
        "from config import logger; print(logger.getEffectiveLevel())",
        _base_env(LOG_LEVEL="NOT_A_REAL_LEVEL"),
    )
    assert out == "20"
