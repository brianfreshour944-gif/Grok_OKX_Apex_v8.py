#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Verification test for crash-recovery fixes.

Verifies:
 1. cancel_stale_orders_async() is called on startup in main_bot.py
 2. sync_existing_positions() seeds entry_time and highest_prices for
    active positions that lack them after a restart
 3. highest_prices is seeded from max(avg_entry, current_price) on exchange
"""

import sys
import os
import time

sys.path.insert(0, '.')

# == Mock trading client for testing ==
class MockOrder:
    def __init__(self, order_id, symbol):
        self.id = order_id
        self.symbol = symbol

class MockTradingClient:
    def __init__(self):
        self.positions = {}
        self.cancelled_orders = []
        self.open_orders = []
        self.get_orders_called = False
    
    def get_all_positions(self):
        class FakePosition:
            def __init__(self, data):
                self.symbol = data["symbol"]
                self.qty = data["qty"]
                self.avg_entry_price = data["avg_entry"]
                self.market_value = data["mv"]
                self.current_price = data["cp"]
        return [FakePosition(d) for d in self.positions.values()]
    
    def get_orders(self, request):
        self.get_orders_called = True
        return self.open_orders
    
    def cancel_order_by_id(self, order_id):
        self.cancelled_orders.append(order_id)

class MockOrderRequest:
    def __init__(self, status):
        self.status = status

# == Test 1: cancel_stale_orders_async called on startup ==
def test_stale_order_cancellation_on_startup():
    print("\n=== Test 1: Stale Order Cancellation on Startup ===")
    
    with open('main_bot.py', encoding='utf-8') as f:
        src = f.read()
    
    # Check that cancel_stale_orders_async is called before the main loop
    startup_section = src.split('while True')[0]
    
    has_startup_cancel = 'cancel_stale_orders_async' in startup_section
    has_startup_sync = 'sync_existing_positions' in startup_section
    
    print(f"  cancel_stale_orders_async in startup: {has_startup_cancel}")
    print(f"  sync_existing_positions in startup: {has_startup_sync}")
    
    if has_startup_cancel and has_startup_sync:
        order_cancel_pos = startup_section.find('cancel_stale_orders_async')
        sync_pos = startup_section.find('sync_existing_positions')
        
        if order_cancel_pos > sync_pos:
            print("  [PASS] cancel_stale_orders_async called AFTER sync_existing_positions")
            print("  [PASS] Stale orders from previous session will be cancelled on startup")
            return True
        else:
            print("  [WARN] cancel_stale_orders_async called BEFORE sync_existing_positions")
            return True  # Still functionally correct
    else:
        print("  [FAIL] cancel_stale_orders_async NOT called during startup")
        return False

# == Test 2: sync_existing_positions seeds missing state ==
def test_state_seeding():
    print("\n=== Test 2: State Seeding After Restart ===")
    
    from portfolio import sync_existing_positions, normalize_symbol
    import portfolio
    
    # Mock the trading client
    mock_client = MockTradingClient()
    mock_client.positions = {
        "BTC/USD": {
            "symbol": "BTCUSD",
            "qty": "0.5",
            "avg_entry": 50000.0,
            "mv": "25000.0",
            "cp": 51000.0,
        }
    }
    
    original_client = portfolio.trading_client
    portfolio.trading_client = mock_client
    
    # Start with EMPTY local state (simulating fresh start after crash)
    local_entry_time = {}
    local_highest_prices = {}
    
    from config import SYMBOLS
    portfolio.SYMBOLS = SYMBOLS  # Ensure SYMBOLS is available
    
    try:
        sync_existing_positions(local_entry_time, local_highest_prices)
        
        # Verify BTC/USD got seeded
        has_entry_time = "BTC/USD" in local_entry_time
        has_highest_price = "BTC/USD" in local_highest_prices
        
        print(f"  entry_time seeded for BTC/USD: {has_entry_time}")
        print(f"  highest_prices seeded for BTC/USD: {has_highest_price}")
        
        if has_entry_time:
            entry_time_val = local_entry_time["BTC/USD"]
            print(f"  entry_time value: {entry_time_val} (should be recent timestamp)")
            # Verify it's a reasonable timestamp (within last 60 seconds)
            now = time.time()
            if abs(now - entry_time_val) < 60:
                print("  [PASS] entry_time seeded with current timestamp")
            else:
                print("  [FAIL] entry_time seeded with unreasonable value")
                return False
        else:
            print("  [FAIL] entry_time NOT seeded for active position")
            return False
        
        if has_highest_price:
            peak = local_highest_prices["BTC/USD"]
            expected = max(50000.0, 51000.0)  # max(avg_entry, current_price)
            print(f"  highest_price value: {peak} (expected {expected})")
            if peak == expected:
                print("  [PASS] highest_prices seeded from max(avg_entry, current_price)")
            else:
                print(f"  [FAIL] highest_prices = {peak}, expected {expected}")
                return False
        else:
            print("  [FAIL] highest_prices NOT seeded for active position")
            return False
        
        print("  [PASS] Active position state correctly seeded after restart")
        return True
        
    finally:
        portfolio.trading_client = original_client

# == Test 3: sync_existing_positions preserves existing state ==
def test_state_preservation():
    print("\n=== Test 3: Existing State Preservation ===")
    
    from portfolio import sync_existing_positions
    import portfolio
    
    mock_client = MockTradingClient()
    mock_client.positions = {
        "BTC/USD": {
            "symbol": "BTCUSD",
            "qty": "0.5",
            "avg_entry": 50000.0,
            "mv": "25000.0",
            "cp": 51000.0,
        }
    }
    
    original_client = portfolio.trading_client
    portfolio.trading_client = mock_client
    
    # Start with EXISTING local state (should be preserved)
    original_entry_val = time.time() - 3600  # 1 hour ago
    original_peak_val = 53000  # higher than current
    existing_entry_time = {"BTC/USD": original_entry_val}
    existing_highest_prices = {"BTC/USD": original_peak_val}
    
    try:
        sync_existing_positions(existing_entry_time, existing_highest_prices)
        
        entry_preserved = existing_entry_time.get("BTC/USD") == original_entry_val
        peak_preserved = existing_highest_prices.get("BTC/USD") == original_peak_val
        
        print(f"  entry_time preserved: {entry_preserved}")
        print(f"  highest_prices preserved: {peak_preserved}")
        
        if entry_preserved and peak_preserved:
            print("  [PASS] Existing state correctly preserved (not overwritten)")
            return True
        else:
            print("  [FAIL] Existing state was overwritten on restart")
            return False
        
    finally:
        portfolio.trading_client = original_client

# == Test 4: Stale state cleanup ==
def test_stale_state_cleanup():
    print("\n=== Test 4: Stale State Cleanup ===")
    
    from portfolio import sync_existing_positions
    import portfolio
    
    mock_client = MockTradingClient()
    # Exchange has NO positions (all sold)
    mock_client.positions = {}
    
    original_client = portfolio.trading_client
    portfolio.trading_client = mock_client
    
    # Local state has stale entries
    local_entry_time = {"BTC/USD": time.time(), "ETH/USD": time.time()}
    local_highest_prices = {"BTC/USD": 50000, "ETH/USD": 3000}
    
    try:
        sync_existing_positions(local_entry_time, local_highest_prices)
        
        has_btc = "BTC/USD" in local_entry_time
        has_eth = "ETH/USD" in local_entry_time
        has_btc_peak = "BTC/USD" in local_highest_prices
        has_eth_peak = "ETH/USD" in local_highest_prices
        
        print(f"  BTC/USD cleaned up: {not has_btc}")
        print(f"  ETH/USD cleaned up: {not has_eth}")
        print(f"  BTC/USD peak cleaned: {not has_btc_peak}")
        print(f"  ETH/USD peak cleaned: {not has_eth_peak}")
        
        if not has_btc and not has_eth and not has_btc_peak and not has_eth_peak:
            print("  [PASS] Stale state correctly cleaned up for non-held symbols")
            return True
        else:
            print("  [FAIL] Stale state not fully cleaned up")
            return False
        
    finally:
        portfolio.trading_client = original_client

# == Test 5: Cancel stale orders actually cancels open orders ==
def test_cancel_stale_orders_function():
    print("\n=== Test 5: cancel_stale_orders Functionality ===")
    
    from portfolio import cancel_stale_orders
    import portfolio
    
    mock_client = MockTradingClient()
    # Simulate an open order that's been sitting for >3 minutes
    from datetime import datetime, timezone, timedelta
    old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    
    class MockOpenOrder:
        def __init__(self, order_id, symbol, created_at):
            self.id = order_id
            self.symbol = symbol
            self.created_at = created_at
    
    mock_client.open_orders = [
        MockOpenOrder("order_123", "BTCUSD", old_time),
    ]
    
    original_client = portfolio.trading_client
    portfolio.trading_client = mock_client
    
    try:
        cancel_stale_orders(timeout_minutes=3)
        
        cancelled_count = len(mock_client.cancelled_orders)
        print(f"  Open orders: {len(mock_client.open_orders)}")
        print(f"  Orders cancelled: {cancelled_count}")
        print(f"  get_orders called: {mock_client.get_orders_called}")
        
        if cancelled_count == 1 and mock_client.cancelled_orders == ["order_123"]:
            print("  [PASS] Stale order correctly cancelled")
            return True
        else:
            print(f"  [FAIL] Expected 1 cancellation, got {cancelled_count}")
            return False
        
    finally:
        portfolio.trading_client = original_client

# == Main ==
if __name__ == "__main__":
    print("=" * 70)
    print("CRASH RECOVERY FIX VERIFICATION")
    print("=" * 70)
    
    results = []
    results.append(("Stale Order Cancellation on Startup", test_stale_order_cancellation_on_startup()))
    results.append(("State Seeding After Restart", test_state_seeding()))
    results.append(("Existing State Preservation", test_state_preservation()))
    results.append(("Stale State Cleanup", test_stale_state_cleanup()))
    results.append(("cancel_stale_orders Functionality", test_cancel_stale_orders_function()))
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print()
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    if passed == total:
        print("\nAll crash recovery fixes verified successfully!")
    else:
        print(f"\n{total - passed} test(s) failed - fixes need adjustment")