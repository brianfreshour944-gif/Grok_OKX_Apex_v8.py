import os
import time
import pandas as pd
import asyncio
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import torch
import numpy as np

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from ml_predictor import GrokGQA_Transformer, FEATURE_COLS
import joblib

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

SEQUENCE_LEN = 32

def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    from feature_engineering import add_features as _add_features
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    for col in ['vwap', 'trade_count']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return _add_features(df)

class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS),
            seq_len=seq_len,
            embed_dim=128,
            num_layers=4,
            num_q_heads=8,
            num_kv_heads=2,
            dropout=0.1
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        scaler_path = os.path.join(os.path.dirname(model_path), "feature_scaler.pkl")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    def predict(self, df):
        try:
            df_feat = safe_add_features(df.copy())
            data    = df_feat.tail(self.seq_len).values.astype(np.float32)
            if len(data) < self.seq_len: return 0.5
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = np.clip(data, -1e6, 1e6)
            if self.scaler:
                data = self.scaler.transform(data).astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.tensor(data).unsqueeze(0).to(self.device)
            with torch.no_grad(): return float(torch.sigmoid(self.model(x)).item())
        except Exception:
            return 0.5

def emulate_clean_ohlcv(bars_slice):
    """
    Builds a feature-ready dataframe directly from NATIVE 15-minute bars
    (no resampling). This MUST match data_feeds.py's get_clean_ohlcv_dataframe()
    and train_transformer.py's BAR_TIMEFRAME (TimeFrame(15, TimeFrameUnit.Minute)) —
    feeding the model 5-min resampled bars instead of native 15-min bars puts
    every feature out-of-distribution relative to training.
    """
    if len(bars_slice) == 0: return None
    df = pd.DataFrame([{
        "timestamp":   b.timestamp,
        "open":        float(b.open or 0),
        "high":        float(b.high or 0),
        "low":         float(b.low or 0),
        "close":       float(b.close or 0),
        "volume":      float(b.volume or 0),
        "vwap":        float(b.vwap or 0),
        "trade_count": float(b.trade_count or 0),
    } for b in bars_slice[-SEQUENCE_LEN * 4:]])
    df.set_index("timestamp", inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Where Alpaca returned vwap=0 (rare empty bars), fall back to close
    df["vwap"] = df["vwap"].where(df["vwap"] > 0, df["close"])
    df = df[df["close"] > 0]
    if len(df) < SEQUENCE_LEN: return None
    return df.tail(SEQUENCE_LEN)

def compute_regime_and_trend(df):
    try:
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low']  - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        price = df['close'].iloc[-1]
        atr_pct = (atr / price) * 100 if price > 0 else 0.0
        
        # VAMA Trend Filter
        base_span = 50
        baseline_vol = 1.5
        if atr_pct > 0:
            vol_ratio = max(0.5, min(atr_pct / baseline_vol, 5.0))
            dynamic_span = max(10, int(base_span / vol_ratio))
        else:
            dynamic_span = base_span
            
        adaptive_ma = df['close'].ewm(span=dynamic_span).mean().iloc[-1]
        trend = "up" if price > adaptive_ma else "down"
        return trend, atr_pct
    except:
        return "neutral", 2.0

def calculate_kelly_multiplier(signal_prob, profit_target_pct, stop_loss_pct):
    w = signal_prob
    r = profit_target_pct / stop_loss_pct if stop_loss_pct > 0 else 1.0
    kelly_fraction = w - ((1.0 - w) / r)
    if kelly_fraction <= 0: return 0.5
    multiplier = 0.5 + (kelly_fraction * 15.0)
    return max(0.5, min(multiplier, 3.0))

async def main():
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    target_date = datetime(2026, 7, 18, tzinfo=timezone.utc)
    start_time = target_date.replace(hour=0, minute=0, second=0)
    end_time = datetime(2026, 7, 19, 2, 0, tzinfo=timezone.utc)
    
    print("Fetching historical data...")
    all_data = {}
    for sym in symbols:
        req = CryptoBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start_time, end=end_time,
        )
        bars = data_client.get_crypto_bars(req).data.get(sym, [])
        all_data[sym] = bars
        print(f"{sym}: {len(bars)} native 15-minute bars fetched.")

    predictor = SafeMLPredictor("grok_gqa_v9_best.pth", 32)
    BUY_SIGNAL = 0.51
    PROFIT_TARGET_PCT = 0.02
    STOP_LOSS_PCT = 0.03
    
    print("\nStarting Advanced Backtest with VAMA, Kelly, and Trailing Stops...")
    current_time = target_date.replace(hour=12, minute=0, second=0)
    
    positions = {} # sym -> {entry_price, highest_seen}
    equity = 10000.0
    
    while current_time <= end_time:
        for sym in symbols:
            bars_slice = [b for b in all_data[sym] if b.timestamp <= current_time]
            if len(bars_slice) > 0:
                df = emulate_clean_ohlcv(bars_slice)
                if df is not None:
                    signal = predictor.predict(df)
                    trend, atr_pct = compute_regime_and_trend(df)
                    price = df['close'].iloc[-1]
                    
                    if sym in positions:
                        pos = positions[sym]
                        pnl_pct = (price - pos['entry_price']) / pos['entry_price']
                        
                        if price > pos['highest_seen']:
                            pos['highest_seen'] = price
                            
                        # Trailing Stop
                        if pos['highest_seen'] > pos['entry_price'] * (1 + PROFIT_TARGET_PCT):
                            trailing_stop = pos['highest_seen'] * 0.99
                            if price <= trailing_stop:
                                print(f"[{current_time.strftime('%H:%M')}] SELL {sym}: Trailing Stop Hit. PnL: {pnl_pct*100:.2f}%")
                                equity *= (1 + pnl_pct)
                                del positions[sym]
                        elif pnl_pct <= -STOP_LOSS_PCT:
                            print(f"[{current_time.strftime('%H:%M')}] SELL {sym}: Stop Loss Hit. PnL: {pnl_pct*100:.2f}%")
                            equity *= (1 + pnl_pct)
                            del positions[sym]
                            
                    elif signal > BUY_SIGNAL and trend == "up":
                        kelly_mult = calculate_kelly_multiplier(signal, PROFIT_TARGET_PCT, STOP_LOSS_PCT)
                        risk_pct = 0.006 * kelly_mult
                        trade_value = equity * risk_pct
                        print(f"[{current_time.strftime('%H:%M')}] BUY  {sym}: Signal={signal:.4f}, VAMA={trend}, KellyMult={kelly_mult:.2f}, Size=${trade_value:.2f}")
                        positions[sym] = {'entry_price': price, 'highest_seen': price}
                        
        current_time += timedelta(minutes=15)
        
    print(f"\nFinal Simulated Equity: ${equity:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
