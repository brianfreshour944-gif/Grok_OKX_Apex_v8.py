import asyncio
from datetime import datetime, timedelta, timezone
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import data_client

async def test():
    symbol = "BTC/USD"
    # Assuming today is the date
    end = datetime.now(timezone.utc)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end)
    bars = data_client.get_crypto_bars(req).data.get(symbol, [])
    print(f"Fetched {len(bars)} 1-min bars for {symbol} from {start} to {end}")
    if bars:
        print(f"First: {bars[0].timestamp}, Last: {bars[-1].timestamp}")

if __name__ == "__main__":
    asyncio.run(test())
