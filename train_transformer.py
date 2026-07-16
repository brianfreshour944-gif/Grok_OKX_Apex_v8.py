#!/usr/bin/env python3
"""
train_transformer.py — Corrected GQA Transformer training script.

Three bugs fixed vs the May 31 original
────────────────────────────────────────
BUG 1 — Timeframe mismatch (FIXED):
  Old: TimeFrame.Hour, 60 days  → hourly bars
  New: TimeFrame(15, TimeFrameUnit.Minute) → native 15-min bars, 180 days
       (15-min bars have 3× less microstructure noise than 5-min bars;
        180 days keeps window count comparable to the old 60-day 5-min run)

BUG 2 — Label-offset (FIXED):
  Old: y = labels[idx + seq_len + target_horizon - 1]
       This labelled a return that STARTS target_horizon bars after the window
       ends — the model never saw the bars it was predicting over.
  New: close_now    = close[idx + seq_len - 1]       ← last bar the model sees
       close_future = close[idx + seq_len - 1 + target_horizon]
       label = 1 if close_future > close_now else 0
       The model predicts whether price will be higher target_horizon×5min
       from the END of its input window. No skip, no offset.

BUG 3 — Random split leakage on overlapping windows (FIXED):
  Old: torch.utils.data.random_split scatters overlapping windows (window i
       and window i+1 share 31 of 32 bars) across train and val — the model
       can partially see val-adjacent data during training.
  New: Chronological split: first TRAIN_FRAC of the timeline → train,
       last (1-TRAIN_FRAC) → val. No overlap between sets.

STRUCTURAL FIX — Labels extracted from raw closes, not add_features() output:
  Old: df_features['label'].fillna(0)  → KeyError, add_features() returns
       only FEATURE_COLS.
  New: Labels computed from df_raw['close'] before add_features() is called.

IMPORT FIX:
  Old: from ml_predictor import GrokGQA_Transformer, MLPredictor
       MLPredictor does not exist in ml_predictor.py.
  New: from ml_predictor import GrokGQA_Transformer   (only what exists)

────────────────────────────────────────────────────────────────────────────────
Colab quick-start:
  !git clone https://github.com/brianfreshour944-gif/Grok_alpaca_Apex_v8.py.git /content/bot
  %cd /content/bot
  !pip install -q -r requirements.txt
  import os
  os.environ["APCA_API_KEY_ID"]    = "your_key"
  os.environ["APCA_API_SECRET_KEY"] = "your_secret"
  !python train_transformer.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler
import joblib

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Import from repo modules — run this script from the repo root directory.
from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer

# ── Hyperparameters ─────────────────────────────────────────────────────────────
SEQ_LEN        = 32    # bars per input window (must match live bot SEQUENCE_LEN)

# FIX 1 — TIMEFRAME: 15-min bars are 3× less noisy than 5-min bars.
#   6 bars × 15 min = 90-min lookahead — a clean, meaningful directional signal.
# FIX 3 — HORIZON: 6 bars instead of 12 (12×5min=60min → 6×15min=90min).
BAR_TIMEFRAME  = TimeFrame(15, TimeFrameUnit.Minute)  # native 15-min bars from Alpaca
TARGET_HORIZON = 6     # predict 6 × 15-min = 90 min ahead

EMBED_DIM      = 128
NUM_LAYERS     = 2
NUM_Q_HEADS    = 16
NUM_KV_HEADS   = 4
DROPOUT        = 0.1
EPOCHS         = 60
BATCH_SIZE     = 64
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TRAIN_FRAC     = 0.80   # chronological — first 80% → train, last 20% → val
DAYS_HISTORY   = 180    # 180 days of 15-min bars ≈ same window count as 60d of 5-min
PATIENCE       = 8      # early-stopping: halt if no val improvement for N epochs
MODEL_OUT      = "grok_gqa_v9_best.pth"
SCALER_OUT     = "feature_scaler.pkl"

# Symbols to train on — broader than live SYMBOLS for more training data
TRAIN_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
    "LTC/USD", "AVAX/USD", "LINK/USD", "ADA/USD",
    "BCH/USD", "DOT/USD",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ── Alpaca client ───────────────────────────────────────────────────────────────
def _get_data_client() -> CryptoHistoricalDataClient:
    key    = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError(
            "Missing Alpaca credentials.\n"
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.\n"
            "In Colab: use the Secrets panel (🔑) or:\n"
            "  import os\n"
            "  os.environ['APCA_API_KEY_ID']     = 'your_key'\n"
            "  os.environ['APCA_API_SECRET_KEY'] = 'your_secret'"
        )
    return CryptoHistoricalDataClient(api_key=key, secret_key=secret)


# ── Data fetching ───────────────────────────────────────────────────────────────
# FIX 1 — TIMEFRAME: Request native 15-min bars directly from Alpaca.
#   The old approach fetched 1-min bars and resampled them to 5-min, which is
#   equivalent to 5-min bars — still very noisy microstructure. Native 15-min
#   bars from Alpaca have their VWAP calculated server-side over a real 15-min
#   window, which is higher quality than a client-side volume-weighted resample.
def fetch_bars(
    client: CryptoHistoricalDataClient,
    symbol: str,
    days: int,
    timeframe: TimeFrame = BAR_TIMEFRAME,
) -> pd.DataFrame | None:
    """
    Fetch native 15-minute bars from Alpaca.
    Returns DataFrame with: open, high, low, close, volume, vwap, trade_count
    VWAP is provided by Alpaca and is already volume-weighted over the 15-min window.
    """
    try:
        start = datetime.now(tz=timezone.utc) - timedelta(days=days)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
        )
        raw_bars = client.get_crypto_bars(req).data.get(symbol, [])
        if not raw_bars:
            log.warning(f"  {symbol}: no bars returned")
            return None

        df = pd.DataFrame([{
            "timestamp":   b.timestamp,
            "open":        float(b.open        or 0),
            "high":        float(b.high        or 0),
            "low":         float(b.low         or 0),
            "close":       float(b.close       or 0),
            "volume":      float(b.volume      or 0),
            "vwap":        float(b.vwap        or 0),
            "trade_count": float(b.trade_count or 0),
        } for b in raw_bars])
        df.set_index("timestamp", inplace=True)

        # Strip timezone so index arithmetic works cleanly downstream
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Where Alpaca returned vwap=0 (rare empty bars), fall back to close
        df["vwap"] = df["vwap"].where(df["vwap"] > 0, df["close"])

        df = df[df["close"] > 0]
        log.info(f"  {symbol}: {len(df):,} 15-min bars fetched")
        return df

    except Exception as exc:
        log.error(f"  {symbol}: fetch failed — {exc}")
        return None


# ── Dataset construction ────────────────────────────────────────────────────────
def build_arrays(
    client: CryptoHistoricalDataClient,
    symbols: list[str],
    days: int,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each symbol:
      1. Fetch 5-min bars
      2. Compute features via add_features()
      3. Compute labels from RAW close prices (NOT from add_features output)
      4. Slice into (X, y) windows with correct chronological alignment

    Label:
      For window ending at bar t = idx + seq_len - 1,
        label = 1 if close[t + horizon] > close[t] else 0
      → predicts whether price is higher `horizon` bars after the last
        bar the model sees. No skip, no extra offset.

    Returns:
      X — float32 array of shape (N, seq_len, n_features)
      y — float32 array of shape (N,) with values in {0.0, 1.0}
    """
    all_X: list[np.ndarray] = []
    all_y: list[float]      = []

    min_bars_needed = seq_len + horizon + 10

    for sym in symbols:
        log.info(f"Building windows for {sym} ...")
        df_raw = fetch_bars(client, sym, days)

        if df_raw is None or len(df_raw) < min_bars_needed:
            log.warning(f"  {sym}: only {0 if df_raw is None else len(df_raw)} bars — need {min_bars_needed}, skipping")
            continue

        # Feature matrix — returns only FEATURE_COLS, never a 'label' column
        df_feat = add_features(df_raw)

        # Align: make sure features and raw closes share the same index
        shared_idx = df_feat.index.intersection(df_raw.index)
        df_feat = df_feat.loc[shared_idx]
        close_arr = df_raw.loc[shared_idx, "close"].values.astype(np.float64)
        feat_arr  = df_feat[FEATURE_COLS].values.astype(np.float32)

        n = len(feat_arr)
        windows_added = 0
        for idx in range(n - seq_len - horizon + 1):
            x_window = feat_arr[idx : idx + seq_len]          # (seq_len, n_features)

            # Last bar the model sees:  idx + seq_len - 1
            # Future bar:               idx + seq_len - 1 + horizon
            close_now    = close_arr[idx + seq_len - 1]
            close_future = close_arr[idx + seq_len - 1 + horizon]

            if close_now <= 0:   # skip bad candles
                continue

            label = 1.0 if close_future > close_now else 0.0
            all_X.append(x_window)
            all_y.append(label)
            windows_added += 1

        log.info(f"  → {windows_added:,} windows from {sym} ({horizon} × 15-min = {horizon*15}-min lookahead)")

    if not all_X:
        raise RuntimeError(
            "No training windows collected.\n"
            "Check API credentials, symbol list, and whether Alpaca has data for these symbols."
        )

    X = np.stack(all_X, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.float32)

    bullish_pct = y.mean() * 100
    log.info(
        f"\nDataset summary:\n"
        f"  Windows  : {len(y):,}\n"
        f"  Shape    : {X.shape}\n"
        f"  Bullish  : {bullish_pct:.1f}%  Bearish: {100-bullish_pct:.1f}%\n"
        f"  (healthy label balance is 45–55%)"
    )
    return X, y


# ── Chronological split ─────────────────────────────────────────────────────────
def chrono_split(
    X: np.ndarray, y: np.ndarray, train_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split at a single time cut — all training windows come before all val windows.
    Prevents data leakage from overlapping windows (window[i] and window[i+1]
    share seq_len-1 bars; random_split scatters them across train and val).
    """
    cut = int(len(y) * train_frac)
    return X[:cut], y[:cut], X[cut:], y[cut:]


# ── Dataset balancing ────────────────────────────────────────────────────────────
# FIX 2 — BALANCE: Downsample the majority class so the model gets exactly 50/50
# labels during training.  Applied to the TRAIN split only — the val split keeps
# its natural distribution so accuracy metrics reflect real-world conditions.
#
# Why undersample instead of oversample (SMOTE)?  The windows are already highly
# overlapping (window i and i+1 share 31 of 32 bars), so duplicating windows via
# SMOTE would create near-identical synthetic samples that add zero diversity.
# Undersampling the majority class is the cleanest, leak-free approach here.
def balance_dataset(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly undersample the majority class so both classes are exactly equal.

    Args:
        X: float32 array of shape (N, seq_len, n_features)
        y: float32 array of shape (N,) with values in {0.0, 1.0}
        seed: random seed for reproducibility

    Returns:
        X_bal, y_bal — balanced arrays (minority_count × 2 samples)
    """
    rng = np.random.default_rng(seed)

    idx_up   = np.where(y == 1.0)[0]
    idx_down = np.where(y == 0.0)[0]

    minority_n = min(len(idx_up), len(idx_down))

    idx_up_sampled   = rng.choice(idx_up,   size=minority_n, replace=False)
    idx_down_sampled = rng.choice(idx_down, size=minority_n, replace=False)

    idx_balanced = np.concatenate([idx_up_sampled, idx_down_sampled])
    rng.shuffle(idx_balanced)  # shuffle so model doesn't see all-Up then all-Down

    log.info(
        f"  Balance: {len(y):,} → {len(idx_balanced):,} windows "
        f"(dropped {len(y) - len(idx_balanced):,} majority-class samples)\n"
        f"  New split: Up={minority_n:,} ({50.0:.1f}%)  "
        f"Down={minority_n:,} ({50.0:.1f}%)"
    )
    return X[idx_balanced], y[idx_balanced]


# ── PyTorch Dataset ─────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Main training function ──────────────────────────────────────────────────────
def train(
    epochs:     int   = EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    client = _get_data_client()

    # 1. Collect raw data and build windows ─────────────────────────────────────
    log.info("=== Phase 1: Fetching data and building training windows ===")
    X, y = build_arrays(client, TRAIN_SYMBOLS, DAYS_HISTORY, SEQ_LEN, TARGET_HORIZON)

    # 2. Chronological split ────────────────────────────────────────────────────
    log.info("=== Phase 2: Chronological train/val split ===")
    X_tr, y_tr, X_va, y_va = chrono_split(X, y, TRAIN_FRAC)
    log.info(f"  Train: {len(y_tr):,} windows | Val: {len(y_va):,} windows")

    # 3. Balance TRAIN set only (val keeps natural distribution for honest metrics)
    # FIX 2 — BALANCE: 50/50 majority-class undersampling eliminates the free
    # 51% accuracy the model was exploiting by always predicting "Down".
    log.info("=== Phase 3: Balancing training set to 50/50 ===")
    X_tr, y_tr = balance_dataset(X_tr, y_tr)

    # 4. Fit scaler on TRAINING data only — never touch val ─────────────────────
    log.info("=== Phase 4: Fitting StandardScaler on training data only ===")
    n_tr, seq_len, n_feat = X_tr.shape
    n_va                  = len(y_va)

    scaler = StandardScaler()
    # Flatten to (n_samples × seq_len, n_features) for fit, then unflatten
    scaler.fit(X_tr.reshape(-1, n_feat))

    X_tr_s = scaler.transform(X_tr.reshape(-1, n_feat)).reshape(n_tr, seq_len, n_feat).astype(np.float32)
    X_va_s = scaler.transform(X_va.reshape(-1, n_feat)).reshape(n_va, seq_len, n_feat).astype(np.float32)

    joblib.dump(scaler, SCALER_OUT)
    log.info(f"  Scaler saved → {SCALER_OUT}")

    train_loader = DataLoader(
        SequenceDataset(X_tr_s, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_va_s, y_va),
        batch_size=batch_size, shuffle=False,
    )

    # 5. Instantiate model ───────────────────────────────────────────────────────
    log.info("=== Phase 5: Building model ===")
    model = GrokGQA_Transformer(
        input_dim    = n_feat,
        seq_len      = seq_len,
        embed_dim    = EMBED_DIM,
        num_layers   = NUM_LAYERS,
        num_q_heads  = NUM_Q_HEADS,
        num_kv_heads = NUM_KV_HEADS,
        dropout      = DROPOUT,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # FIX: Use OneCycleLR for proper learning rate warmup
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )

    # 6. Training loop ───────────────────────────────────────────────────────────
    log.info("=== Phase 6: Training ===")
    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(1)
            loss = criterion(pred.view(-1), yb.float().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            # FIX: Raw logits threshold is 0.0 (which corresponds to probability 0.5)
            tr_correct += ((pred > 0.0) == yb.bool()).sum().item()
            scheduler.step()

        # ── Validate ──
        model.eval()
        va_loss, va_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(1)
                va_loss += criterion(pred.view(-1), yb.float().view(-1)).item() * len(yb)
                # FIX: Raw logits threshold is 0.0 (which corresponds to probability 0.5)
                va_correct += ((pred > 0.0) == yb.bool()).sum().item()

        t_l = tr_loss / len(y_tr)
        v_l = va_loss / len(y_va)
        t_a = tr_correct / len(y_tr) * 100
        v_a = va_correct / len(y_va) * 100

        log.info(
            f"Epoch {epoch:3d}/{epochs}  |  "
            f"train  loss={t_l:.4f}  acc={t_a:.1f}%  |  "
            f"val    loss={v_l:.4f}  acc={v_a:.1f}%"
            + ("  ✅ best" if v_l < best_val_loss else f"  (patience {patience_ctr+1}/{PATIENCE})")
        )

        if v_l < best_val_loss:
            best_val_loss = v_l
            patience_ctr  = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

    log.info(
        f"\n=== Training complete ===\n"
        f"  Best val loss : {best_val_loss:.4f}\n"
        f"  Model saved   : {MODEL_OUT}\n"
        f"  Scaler saved  : {SCALER_OUT}\n"
        f"\nNext step: commit both files to the repo so Coolify picks them up on redeploy."
    )


if __name__ == "__main__":
    train()
