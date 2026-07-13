#!/usr/bin/env python3
"""
regime_windows_backtest.py

Run the FROZEN Grok_alpaca_Apex_v8 strategy + frozen config.backtest.json
against 4 disjoint windows spanning DIFFERENT visible market regimes
(bull blow-off, bear crash, sideways chop, ETF bull).

No tuning, no config/strategy changes -- pure robustness test.

For each window it:
  1. downloads the required 5m + 1h OHLCV from OKX (with left padding for
     the strategy's startup candle count),
  2. runs the frozen freqtrade backtest,
  3. parses the Total Profit %.

Finally it prints the cross-window return distribution:
  mean, std dev, worst (min) case, best (max) case.
"""
import subprocess
import re
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(REPO, "freqtrade", "config.backtest.json")
STRAT = "Grok_alpaca_Apex_v8"
STRAT_PATH = os.path.join(REPO, "user_data", "strategies")
OUT_DIR = os.path.join(REPO, "user_data", "backtest_results")

# label -> freqtrade timerange (disjoint, one calendar month each)
# spanning genuinely different historical BTC regimes.
WINDOWS = [
    ("W1_2021_bull_blowoff", "20211001-20211101"),  # Oct 2021  +~40% blow-off top
    ("W2_2022_bear_luna",    "20220501-20220601"),  # May 2022  LUNA/UST collapse, -15% bear
    ("W3_2023_chop",         "20230801-20230901"),  # Aug 2023  sideways chop / low vol
    ("W4_2024_bull_etf",     "20240201-20240301"),  # Feb 2024  spot-ETF approval rally
]

PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["5m", "1h"]
# left-pad each window by 15 days so startup candles (200 x 5m + 200 x 1h)
# are available before the window starts.
PAD_DAYS = 15


def _pad_timerange(tr):
    start = tr.split("-")[0]
    y, m, d = int(start[:4]), int(start[4:6]), int(start[6:8])
    # subtract PAD_DAYS
    import datetime
    dt = datetime.date(y, m, d) - datetime.timedelta(days=PAD_DAYS)
    padded = dt.strftime("%Y%m%d")
    return f"{padded}-{tr.split('-', 1)[1]}"


def download_data(timerange):
    cmd = [
        sys.executable, "-m", "freqtrade", "download-data",
        "--config", CONFIG,
        "--pairs", *PAIRS,
        "--timeframes", *TIMEFRAMES,
        "--timerange", timerange,
    ]
    print(f"  [download] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO)
    return proc.returncode == 0


def run_window(label, timerange):
    log_path = os.path.join(OUT_DIR, f"win_{label}.log")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ensure data exists for this window
    dl_tr = _pad_timerange(timerange)
    download_data(dl_tr)

    cmd = [
        sys.executable, "-m", "freqtrade", "backtesting",
        "--config", CONFIG,
        "--strategy", STRAT,
        "--strategy-path", STRAT_PATH,
        "--timerange", timerange,
        "--export", "none",
    ]
    print(f"\n=== {label}  ({timerange}) ===")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  [WARN] freqtrade exited {proc.returncode}; see {log_path}")
    # parse results from the log
    with open(log_path) as f:
        text = f.read()
    total_pct = None
    trades = 0
    # SUMMARY METRICS line: "| Total profit %  | -28.18%  |"
    m = re.search(r"Total profit %\s*\|?\s*([-\d.]+)", text, re.IGNORECASE)
    if m:
        total_pct = float(m.group(1))
    # STRATEGY SUMMARY line: "| Grok_alpaca_Apex_v8 |  316 | ..."
    m = re.search(r"Grok_alpaca_Apex_v8\s*\|\s*(\d+)", text)
    if not m:
        # fallback: TOTAL row in BACKTESTING REPORT
        m = re.search(r"TOTAL\s*\|\s*(\d+)", text)
    if m:
        trades = int(m.group(1))
    if total_pct is None:
        # no trades / no summary -> treat as 0.0 return
        total_pct = 0.0
    print(f"  Total Profit %: {total_pct:+.2f}   Trades: {trades}")
    return label, timerange, total_pct, trades


def main():
    results = []
    for label, tr in WINDOWS:
        results.append(run_window(label, tr))

    returns = [r[2] for r in results]
    n = len(returns)
    mean = sum(returns) / n
    var = sum((x - mean) ** 2 for x in returns) / n
    std = var ** 0.5
    worst = min(returns)
    best = max(returns)

    print("\n" + "=" * 60)
    print("PER-WINDOW RETURNS (Total Profit %)")
    print("=" * 60)
    for label, tr, pct, t in results:
        print(f"  {label:22s} {tr:20s}  {pct:+8.2f}%   trades={t}")
    print("-" * 60)
    print(f"  mean   : {mean:+8.2f}%")
    print(f"  std dev: {std:8.2f}%")
    print(f"  worst  : {worst:+8.2f}%")
    print(f"  best   : {best:+8.2f}%")
    print("=" * 60)

    # honest verdict hint
    if worst <= 0 and mean <= 1.0:
        print("\nInterpretation: flat-to-negative across all windows incl. trending")
        print("month (W4_2024_bull_etf). Consistent with NO edge yet under frozen config.")
    else:
        print("\nInterpretation: see per-window numbers above.")


if __name__ == "__main__":
    main()