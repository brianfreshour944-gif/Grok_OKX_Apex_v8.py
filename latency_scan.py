#!/usr/bin/env python3
"""
Bot latency scan - measures component latencies without network dependencies.
"""
import asyncio
import time
import sys
import os
import tracemalloc

sys.path.insert(0, '.')

def test_model_inference():
    """Measure model inference latency."""
    print("=== Model Inference Latency ===")
    try:
        import torch
        import numpy as np
        import pandas as pd
        from ml_predictor import GrokGQA_Transformer, FEATURE_COLS
        
        model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS), seq_len=32,
            embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1
        ).to("cpu").eval()
        
        np.random.seed(42)
        df = pd.DataFrame({
            "open": np.random.uniform(99, 101, 64),
            "high": np.random.uniform(100, 102, 64),
            "low": np.random.uniform(98, 100, 64),
            "close": np.random.uniform(99, 101, 64),
            "volume": np.random.uniform(1000, 5000, 64),
            "vwap": np.random.uniform(99, 101, 64),
            "trade_count": np.random.uniform(100, 300, 64),
        })
        
        from feature_engineering import add_features
        df_feat = add_features(df).tail(32)
        data = df_feat[FEATURE_COLS].values.astype(np.float32)
        x = torch.tensor(data).unsqueeze(0)
        
        # Warmup
        with torch.no_grad():
            _ = torch.sigmoid(model(x)).item()
        
        # Measure
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                pred = torch.sigmoid(model(x)).item()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg = sum(times) / len(times)
        mx = max(times)
        mn = min(times)
        
        print(f"  Average: {avg:.2f}ms")
        print(f"  Max:     {mx:.2f}ms")
        print(f"  Min:     {mn:.2f}ms")
        print(f"  CPU Threads: {torch.get_num_threads()}")
        
    except Exception as e:
        print(f"  Error: {e}")

def test_feature_engineering():
    """Measure feature engineering latency."""
    print("\n=== Feature Engineering Latency ===")
    try:
        import numpy as np
        import pandas as pd
        from feature_engineering import add_features
        
        np.random.seed(42)
        df = pd.DataFrame({
            "open": np.random.uniform(99, 101, 100),
            "high": np.random.uniform(100, 102, 100),
            "low": np.random.uniform(98, 100, 100),
            "close": np.random.uniform(99, 101, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "vwap": np.random.uniform(99, 101, 100),
            "trade_count": np.random.uniform(100, 300, 100),
        })
        
        # Warmup
        _ = add_features(df.copy())
        
        # Measure
        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = add_features(df.copy())
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg = sum(times) / len(times)
        mx = max(times)
        mn = min(times)
        
        print(f"  Average: {avg:.2f}ms")
        print(f"  Max:     {mx:.2f}ms")
        print(f"  Min:     {mn:.2f}ms")
        
    except Exception as e:
        print(f"  Error: {e}")

def test_state_io():
    """Measure state save/load latency."""
    print("\n=== State I/O Latency ===")
    try:
        from database import save_bot_state, load_bot_state
        
        cooldown_until = {}
        entry_time = {}
        latest_signals = {}
        highest_prices = {}
        
        # Measure save
        times = []
        for _ in range(100):
            start = time.perf_counter()
            save_bot_state(cooldown_until, entry_time, latest_signals, highest_prices)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg = sum(times) / len(times)
        print(f"  Save (pickle): avg={avg:.3f}ms, max={max(times):.3f}ms")
        
        # Measure load
        times = []
        for _ in range(100):
            start = time.perf_counter()
            load_bot_state()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg = sum(times) / len(times)
        print(f"  Load (pickle): avg={avg:.3f}ms, max={max(times):.3f}ms")
        
    except Exception as e:
        print(f"  Error: {e}")

def test_event_loop_responsiveness():
    """Measure event loop blocking using a heartbeat."""
    print("\n=== Event Loop Responsiveness ===")
    
    async def run_test():
        blocker_times = []
        
        async def heartbeat():
            last = time.perf_counter()
            for _ in range(50):
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gap = (now - last) * 1000
                blocker_times.append(gap)
                last = now
        
        # Test 1: Pure async (should be very fast)
        await heartbeat()
        max_pure_async = max(blocker_times)
        avg_pure_async = sum(blocker_times) / len(blocker_times)
        print(f"  Pure async heartbeat: max={max_pure_async:.2f}ms, avg={avg_pure_async:.2f}ms")
        
        # Test 2: Simulated blocking call
        blocker_times.clear()
        last = time.perf_counter()
        for _ in range(20):
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gap = (now - last) * 1000
            blocker_times.append(gap)
            last = now
            
            # Simulate a 50ms blocking call mid-cycle
            if _ == 10:
                time.sleep(0.05)  # Blocking call
        
        blocked_events = [t for t in blocker_times if t > 25]
        if blocked_events:
            print(f"  With blocking call: max={max(blocked_events):.2f}ms, blocked events={len(blocked_events)}")
            print(f"  -> This would delay other async tasks by ~50ms")
        else:
            print(f"  No blocking detected")
        
        return blocker_times
    
    asyncio.run(run_test())

def test_memory_usage():
    """Measure memory usage patterns."""
    print("\n=== Memory Usage ===")
    
    try:
        # Check file size growth instead of importing main_bot (avoids emoji chars)
        import os
        
        for f in ['main_bot.py', 'portfolio.py', 'data_feeds.py']:
            size = os.path.getsize(f)
            print(f"  {f}: {size} bytes")
        
        # Check if stale cleanup is present
        with open('main_bot.py', encoding='utf-8') as f:
            src = f.read()
        
        has_cleanup = 'cutoff' in src and 'cooldown_until.pop' in src
        print(f"  Stale cleanup logic: {'Yes' if has_cleanup else 'No'}")
        
        with open('portfolio.py', encoding='utf-8') as f:
            port_src = f.read()
        
        has_sync_cleanup = 'stale' in port_src.lower() or 'pop' in port_src
        print(f"  Portfolio cleanup logic: {'Yes' if has_sync_cleanup else 'No'}")
        
    except Exception as e:
        print(f"  Error: {e}")

def test_api_call_analysis():
    """Analyze API call patterns in the code."""
    print("\n=== API Call Analysis ===")
    
    try:
        with open('main_bot.py', encoding='utf-8') as f:
            src = f.read()
        
        # Check for synchronous trading_client calls
        sync_calls = []
        async_wrappers = []
        
        # Count synchronous calls
        import re
        # Find trading_client calls not wrapped in to_thread
        sync_pattern = r'(?<!asyncio\.to_thread\(trading_client\.)trading_client\.(\w+)\('
        matches = re.findall(sync_pattern, src)
        sync_calls.extend(matches)
        
        # Count async wrappers
        wrapper_pattern = r'asyncio\.to_thread\((?:trading_client\.)?(\w+)\('
        wrappers = re.findall(wrapper_pattern, src)
        async_wrappers.extend(wrappers)
        
        # Also check for async wrapper functions
        if 'get_all_positions_async' in src:
            async_wrappers.append('get_all_positions_async')
        if 'get_buying_power_async' in src:
            async_wrappers.append('get_buying_power_async')
        if 'cancel_stale_orders_async' in src:
            async_wrappers.append('cancel_stale_orders_async')
        
        print(f"  Synchronous trading_client calls: {len(sync_calls)}")
        if sync_calls:
            for call in set(sync_calls):
                print(f"    - {call}")
        
        print(f"  Async/offloaded calls: {len(async_wrappers)}")
        if async_wrappers:
            for call in set(async_wrappers):
                print(f"    - {call}")
        
        # Check data_feeds.py
        with open('data_feeds.py', encoding='utf-8') as f:
            df_src = f.read()
        
        df_async = df_src.count('asyncio.to_thread')
        print(f"  data_feeds.py uses asyncio.to_thread: {df_async} times")
        
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("BOT LATENCY SCAN")
    print("=" * 60)
    
    test_model_inference()
    test_feature_engineering()
    test_state_io()
    test_event_loop_responsiveness()
    test_memory_usage()
    test_api_call_analysis()
    
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)