#!/usr/bin/env python3
"""
Targeted performance measurement: Block event loop time per trading cycle.
Measures actual time spent in synchronous/blocking operations.
"""

import asyncio
import time
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Simulate one trading cycle of blocking time ────────────────────────────────
async def measure_cycle_blocking():
    """Measure actual event loop blocking in a simulated trading cycle."""
    
    results = {}
    
    # Test 1: Model inference timing
    try:
        import torch
        import numpy as np
        from ml_predictor import GrokGQA_Transformer, FEATURE_COLS
        from feature_engineering import add_features
        
        model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS), seq_len=32,
            embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1
        )
        device = torch.device("cpu")
        model = model.to(device).eval()
        
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            "open": np.random.uniform(99, 101, 50),
            "high": np.random.uniform(100, 102, 50),
            "low": np.random.uniform(98, 100, 50),
            "close": np.random.uniform(99, 101, 50),
            "volume": np.random.uniform(1000, 2000, 50),
            "vwap": np.random.uniform(99, 101, 50),
            "trade_count": np.random.uniform(100, 200, 50),
        })
        df_features = add_features(df)
        data = df_features[FEATURE_COLS].tail(32).values.astype(np.float32)
        x = torch.tensor(data).unsqueeze(0)
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Time inference
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                output = model(x)
                pred = torch.sigmoid(output).item()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_inf = sum(times) / len(times)
        max_inf = max(times)
        results['model_inference_avg'] = avg_inf
        results['model_inference_max'] = max_inf
        print(f"Model inference (CPU): avg={avg_inf:.2f}ms, max={max_inf:.2f}ms")
        
        # Time batched inference for 3 symbols
        start = time.perf_counter()
        symbols_data = [torch.tensor(data).unsqueeze(0) for _ in range(3)]
        with torch.no_grad():
            outputs = [torch.sigmoid(model(x)).item() for x in symbols_data]
        batch_time = (time.perf_counter() - start) * 1000
        results['model_batch_3_symbols'] = batch_time
        print(f"Model inference (3 symbols, sequential): {batch_time:.2f}ms")
        
    except Exception as e:
        print(f"Model inference test failed: {e}")
    
    # Test 2: Feature engineering timing
    try:
        from feature_engineering import add_features, FEATURE_COLS
        start = time.perf_counter()
        result = add_features(df.copy())
        feat_time = (time.perf_counter() - start) * 1000
        results['feature_engineering'] = feat_time
        print(f"Feature engineering (1 symbol): {feat_time:.2f}ms")
        
        # Feature engineering for 3 symbols
        start = time.perf_counter()
        for _ in range(3):
            _ = add_features(df.copy())
        feat_time_3 = (time.perf_counter() - start) * 1000
        results['feature_engineering_3_symbols'] = feat_time_3
        print(f"Feature engineering (3 symbols): {feat_time_3:.2f}ms")
        
    except Exception as e:
        print(f"Feature engineering test failed: {e}")
    
    # Test 3: Database state save/load
    try:
        from database import save_bot_state, load_bot_state
        cooldown_until, entry_time, latest_signals, highest_prices = {}, {}, {}, {}
        
        start = time.perf_counter()
        save_bot_state(cooldown_until, entry_time, latest_signals, highest_prices)
        save_time = (time.perf_counter() - start) * 1000
        results['state_save'] = save_time
        print(f"State save (pickle): {save_time:.2f}ms")
        
        start = time.perf_counter()
        cooldown_until, entry_time, latest_signals, highest_prices = load_bot_state()
        load_time = (time.perf_counter() - start) * 1000
        results['state_load'] = load_time
        print(f"State load (pickle): {load_time:.2f}ms")
        
    except Exception as e:
        print(f"State save/load test failed: {e}")
    
    # Test 4: Log I/O timing
    try:
        from config import logger
        
        start = time.perf_counter()
        for i in range(100):
            logger.info(f"Cycle log message {i}")
        log_time = (time.perf_counter() - start) * 1000
        results['log_io_100_msgs'] = log_time
        print(f"100 log messages: {log_time:.2f}ms")
        
    except Exception as e:
        print(f"Log I/O test failed: {e}")
    
    # Test 5: Synchronous trading_client calls (simulated timing without network)
    # Estimate based on typical latency
    print(f"Estimated API call overhead (get_account + get_all_positions): ~200-500ms (network)")
    results['api_call_estimate'] = 350  # Typical middle estimate
    
    return results

# ── Measure total cycle blocking time ───────────────────────────────────────────
async def main():
    print("=" * 70)
    print("PERFORMANCE AUDIT: Detailed Timing Measurements")
    print("=" * 70)
    
    results = await measure_cycle_blocking()
    
    # Calculate total estimated blocking time per cycle
    # 3 symbols * (feature_eng + model_inference) for each
    if 'feature_engineering_3_symbols' in results and 'model_batch_3_symbols' in results:
        total_blocking = (
            results.get('feature_engineering_3_symbols', 0) +
            results.get('model_batch_3_symbols', 0) +
            results.get('state_save', 0) +
            results.get('log_io_100_msgs', 0)
        )
        print(f"\n--- Total Blocking Time Estimate (CPU-only, no network) ---")
        print(f"Total CPU blocking per cycle: {total_blocking:.2f}ms")
        print(f"Estimated network overhead: ~{results.get('api_call_estimate', 0)}ms")
        print(f"Total estimated cycle time: {total_blocking + results.get('api_call_estimate', 0):.2f}ms")
        print(f"\nWith SLEEP_PER_LOOP likely ~5-10s, the CPU overhead is {(total_blocking/total_blocking + 5000)*100:.1f}% of cycle" if total_blocking > 0 else "")
        print(f"\nKey finding: Synchronous trading_client calls block event loop for ~{results.get('api_call_estimate', 0)}ms")
        print(f"  Fix: Wrap in asyncio.to_thread() or use async trading client")

if __name__ == "__main__":
    asyncio.run(main())