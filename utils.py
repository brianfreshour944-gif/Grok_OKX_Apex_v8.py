import requests
import os

# Define the reporting logic here once
def log_trade(bot_name, symbol, side, qty, entry_price, pnl=None, ...):
    # Validation logic to keep your database clean
    if entry_price is None or float(entry_price) <= 0:
        return 
    
    payload = {
        "bot_name": bot_name,
        "symbol": symbol,
        "side": side,
        # ... your payload ...
    }
    # API POST logic here
