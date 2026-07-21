import os
import time
import pandas as pd
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestOrderbookRequest

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

def fetch_orderbook_imbalance():
    req = CryptoLatestOrderbookRequest(symbol_or_symbols=SYMBOLS)
    try:
        books = data_client.get_crypto_latest_orderbook(req)
        records = []
        now = datetime.now()
        for sym, book in books.items():
            # Alpaca orderbook returns lists of OrderbookQuote objects with p (price) and s (size)
            total_bid_size = sum(b.s for b in book.bids)
            total_ask_size = sum(a.s for a in book.asks)
            imbalance = total_bid_size / total_ask_size if total_ask_size > 0 else 1.0
            
            records.append({
                "timestamp": now.isoformat(),
                "symbol": sym,
                "bid_ask_imbalance": imbalance,
                "total_bid_size": total_bid_size,
                "total_ask_size": total_ask_size
            })
            
        df = pd.DataFrame(records)
        csv_file = "shadow_features.csv"
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
        print(f"[{now.strftime('%H:%M:%S')}] ✅ Saved orderbook imbalance for {len(records)} symbols.")
    except Exception as e:
        print(f"❌ Error fetching orderbook: {e}")

async def run_collector():
    print("🚀 Starting Shadow Data Collector (v10 Prep)...")
    print("This script passively collects Level 2 orderbook data without executing trades.")
    while True:
        fetch_orderbook_imbalance()
        await asyncio.sleep(300)  # Collect every 5 minutes to avoid rate limits while building dataset

if __name__ == "__main__":
    asyncio.run(run_collector())
