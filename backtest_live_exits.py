import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import data_client, SEQUENCE_LEN
import os
import torch
import joblib
import numpy as np

# Live bot config values
PROFIT_TARGET_PCT = 0.02
STOP_LOSS_PCT = 0.03
MAX_HOLD_HOURS = 4.0
MIN_HOLD_HOURS_BEFORE_SIGNAL = 0.5
BUY_SIGNAL = 0.51
SELL_SIGNAL = 0.45

def get_full_day_5m(bars_slice):
    if len(bars_slice) == 0:
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
    } for b in bars_slice])
    df.set_index("timestamp", inplace=True)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df["vwap_x_vol"] = df["vwap"] * df["volume"]
    df_5m = df.resample("5min").agg({
        "open":        "first",
        "high":        "max",
        "low":         "min",
        "close":       "last",
        "volume":      "sum",
        "vwap_x_vol":  "sum",
        "trade_count": "sum",
    }).fillna(0)

    df_5m["vwap"] = (
        df_5m["vwap_x_vol"] / df_5m["volume"].replace(0, float("nan"))
    ).fillna(df_5m["close"])
    df_5m.drop(columns=["vwap_x_vol"], inplace=True)

    df_5m = df_5m[df_5m["close"] > 0]
    return df_5m

async def main():
    symbols = ["BTC/USD", "ETH/USD", "AAVE/USD", "LTC/USD", "GRT/USD", "BAT/USD"]
    
    target_date = datetime(2026, 7, 18, tzinfo=timezone.utc)
    start_time = target_date.replace(hour=0, minute=0, second=0) - timedelta(days=2)
    end_time = datetime(2026, 7, 19, 2, 0, tzinfo=timezone.utc)
    
    print("Fetching data...")
    all_data_5m = {}
    for sym in symbols:
        req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=start_time, end=end_time)
        bars = data_client.get_crypto_bars(req).data.get(sym, [])
        df_5m = get_full_day_5m(bars)
        if df_5m is not None:
            all_data_5m[sym] = df_5m

    from feature_engineering import add_features, FEATURE_COLS
    from ml_predictor import GrokGQA_Transformer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrokGQA_Transformer(
        input_dim=len(FEATURE_COLS), seq_len=SEQUENCE_LEN,
        embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1,
    ).to(device)
    state = torch.load("grok_gqa_v9_best.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    scaler = joblib.load("feature_scaler.pkl")
    
    print("Precomputing signals...")
    precomputed_signals = {}
    eval_start = datetime(2026, 7, 18, 13, 50, 0)
    
    for sym in symbols:
        df_5m = all_data_5m.get(sym)
        if df_5m is None: continue
        
        signals = []
        prices = []
        timestamps = []
        
        for i in range(len(df_5m)):
            current_time = df_5m.index[i]
            if current_time < eval_start:
                continue
                
            sub_df = df_5m.iloc[:i+1]
            if len(sub_df) < SEQUENCE_LEN:
                continue
                
            try:
                df_feat = add_features(sub_df.copy())
                data = df_feat.tail(SEQUENCE_LEN).values.astype(np.float32)
                if len(data) < SEQUENCE_LEN: continue
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data = np.clip(data, -1e6, 1e6)
                data = scaler.transform(data).astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                x = torch.tensor(data).unsqueeze(0).to(device)
                with torch.no_grad():
                    sig = float(torch.sigmoid(model(x)).item())
                    
                signals.append(sig)
                prices.append(sub_df["close"].iloc[-1])
                timestamps.append(current_time)
            except Exception as e:
                pass
                
        precomputed_signals[sym] = pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "signal": signals
        })

    print(f"\nRunning backtest with LIVE EXIT LOGIC:")
    print(f"BUY_SIGNAL = {BUY_SIGNAL}, SELL_SIGNAL = {SELL_SIGNAL}")
    print(f"PROFIT_TARGET_PCT = {PROFIT_TARGET_PCT*100:.1f}%, STOP_LOSS_PCT = {STOP_LOSS_PCT*100:.1f}%")
    print(f"MAX_HOLD_HOURS = {MAX_HOLD_HOURS}h, MIN_HOLD = {MIN_HOLD_HOURS_BEFORE_SIGNAL}h\n")
    
    total_pnl = 0
    total_trades = 0
    trade_logs = []
    
    for sym in symbols:
        sig_df = precomputed_signals.get(sym)
        if sig_df is None or len(sig_df) == 0:
            continue
            
        position = 0
        entry_price = 0
        entry_time = None
        
        for _, row in sig_df.iterrows():
            now = row["timestamp"]
            price = row["price"]
            signal = row["signal"]
            
            if position == 0 and signal > BUY_SIGNAL:
                position = 1
                entry_price = price
                entry_time = now
                trade_logs.append(f"[{now.strftime('%H:%M:%S')}] BUY  {sym} @ {price:.2f} (sig: {signal:.3f})")
            
            elif position == 1:
                pnl_pct = (price - entry_price) / entry_price
                held_hours = (now - entry_time).total_seconds() / 3600.0
                
                exit_reason = None
                if pnl_pct >= PROFIT_TARGET_PCT:
                    exit_reason = "Profit Target"
                elif pnl_pct <= -STOP_LOSS_PCT:
                    exit_reason = "Stop Loss"
                elif held_hours >= MAX_HOLD_HOURS:
                    exit_reason = "Max Hold Time"
                elif held_hours >= MIN_HOLD_HOURS_BEFORE_SIGNAL and signal < SELL_SIGNAL:
                    exit_reason = "Weak Signal"
                    
                if exit_reason:
                    total_pnl += pnl_pct
                    total_trades += 1
                    position = 0
                    trade_logs.append(f"[{now.strftime('%H:%M:%S')}] SELL {sym} @ {price:.2f} ({exit_reason}, PnL: {pnl_pct*100:+.2f}%)")
                    
        # Force close at end of day
        if position == 1:
            final_price = sig_df.iloc[-1]["price"]
            pnl_pct = (final_price - entry_price) / entry_price
            total_pnl += pnl_pct
            total_trades += 1
            trade_logs.append(f"[{sig_df.iloc[-1]['timestamp'].strftime('%H:%M:%S')}] SELL {sym} @ {final_price:.2f} (End of Day, PnL: {pnl_pct*100:+.2f}%)")

    for log_msg in trade_logs:
        print(log_msg)
        
    print("\n--- FINAL RESULTS ---")
    print(f"Total Return: {total_pnl * 100:+.2f}%")
    print(f"Total Trades: {total_trades}")

if __name__ == "__main__":
    asyncio.run(main())
