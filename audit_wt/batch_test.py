#!/usr/bin/env python3
import time
import torch
import numpy as np
import pandas as pd
from ml_predictor import GrokGQA_Transformer, FEATURE_COLS
from feature_engineering import add_features

model = GrokGQA_Transformer(input_dim=len(FEATURE_COLS), seq_len=32, embed_dim=128,
                            num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1)

# Load actual trained weights
try:
    state = torch.load("grok_gqa_v9_best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    print("Model weights loaded successfully")
except Exception as e:
    print("Could not load model weights: {}".format(e))
    print("Using default weights for testing")

model.to('cpu').eval()

np.random.seed(42)
dfs = {}
for sym in ['BTC/USD', 'ETH/USD', 'SOL/USD']:
    df = pd.DataFrame({
        'open': np.random.uniform(99, 101, 64),
        'high': np.random.uniform(100, 102, 64),
        'low': np.random.uniform(98, 100, 64),
        'close': np.random.uniform(99, 101, 64),
        'volume': np.random.uniform(1000, 5000, 64),
        'vwap': np.random.uniform(99, 101, 64),
        'trade_count': np.random.uniform(100, 300, 64),
    })
    dfs[sym] = df

def batched_predict(df_dict):
    tensors = []
    symbols_in_order = []
    
    for symbol, df in df_dict.items():
        df = df.copy()
        df_features = add_features(df)
        data = df_features[FEATURE_COLS].tail(32).values.astype(np.float32)
        x = torch.tensor(data).unsqueeze(0)
        tensors.append(x)
        symbols_in_order.append(symbol)
    
    batch = torch.cat(tensors, dim=0)
    with torch.no_grad():
        raw_logits = model(batch)
        preds = torch.sigmoid(raw_logits).squeeze(-1)
    
    return {sym: float(preds[i].item()) for i, sym in enumerate(symbols_in_order)}

# Warmup
_ = batched_predict(dfs)

# Measure batched inference
times = []
for _ in range(20):
    start = time.perf_counter()
    signals = batched_predict(dfs)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

print("BATCH inference (3 symbols):")
print("  Average: {:.2f}ms".format(sum(times)/len(times)))
print("  Max: {:.2f}ms".format(max(times)))
print("  Min: {:.2f}ms".format(min(times)))
print("  Signals: {}".format(signals))

# Compare with sequential
times_seq = []
for _ in range(20):
    start = time.perf_counter()
    for sym in dfs:
        df = dfs[sym].copy()
        df_feat = add_features(df)
        data = df_feat[FEATURE_COLS].tail(32).values.astype(np.float32)
        x = torch.tensor(data).unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(model(x)).item()
    elapsed = (time.perf_counter() - start) * 1000
    times_seq.append(elapsed)

print("\nSEQUENTIAL inference (3 symbols):")
print("  Average: {:.2f}ms".format(sum(times_seq)/len(times_seq)))
print("  Max: {:.2f}ms".format(max(times_seq)))
print("  Min: {:.2f}ms".format(min(times_seq)))

speedup = sum(times_seq)/len(times_seq) / (sum(times)/len(times))
print("\nSpeedup from batching: {:.2f}x".format(speedup))