import asyncio
import os
from dotenv import load_dotenv
from alpaca.data.live import CryptoDataStream

load_dotenv()

async def trade_handler(trade):
    print("Trade:", trade)

async def main():
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    
    stream = CryptoDataStream(api_key, api_secret)
    stream.subscribe_trades(trade_handler, "BTC/USD")
    
    # How to run this alongside a while loop?
    print("Starting stream...")
    # asyncio.create_task(stream._run_forever()) # Try internal
    asyncio.get_event_loop().create_task(stream._run_forever())
    
    for i in range(5):
        print("Main loop running:", i)
        await asyncio.sleep(2)
        
    print("Done")

if __name__ == "__main__":
    asyncio.run(main())
