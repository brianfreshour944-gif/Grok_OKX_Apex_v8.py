import asyncio
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
import logging
import json
import os
import csv
from datetime import datetime
from brokers import GrokOKXBroker
from ml_predictor import MLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class GrokApexIroncladBot:
    def __init__(self, paper_mode: bool = True):
        self.broker = GrokOKXBroker(paper_mode=paper_mode)
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        # Set to look for your newly verified v9 weights file
        self.ml = MLPredictor(model_path="grok_gqa_v9_best.pth", seq_len=512)
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.running = True
        self.start_balance = None
        self.load_state()
        self.init_trade_log()

    def init_trade_log(self):
        """Create trade_log.csv with headers if it doesn't exist"""
        if not os.path.exists("trade_log.csv"):
            with open("trade_log.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "action", "price", 
                    "size", "score", "pnl_usdt", "balance_usdt"
                ])
            logger.info("📁 Created new trade_log.csv file")

    def log_trade(self, symbol, action, price, size, score, pnl=0.0):
        """Write each trade to CSV"""
        balance = self.broker.get_balance('USDT')
        with open("trade_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                action,
                price,
                size,
                f"{score:.3f}",
                f"{pnl:.2f}",
                f"{balance:.2f}"
            ])
        logger.info(f"📝 Logged {action} {size} {symbol} @ ${price} | PnL: ${pnl:.2f}")

    def load_state(self):
        if os.path.exists("grok_apex_state.json"):
            with open("grok_apex_state.json") as f:
                data = json.load(f)
                self.positions = data.get("positions", {})
                self.trades = data.get("trades", [])

    def save_state(self):
        with open("grok_apex_state.json", "w") as f:
            json.dump({"positions": self.positions, "trades": self.trades[-100:]}, f)

    async def run(self):
        exchange = ccxtpro.okx({
            'apiKey': self.broker.api_key,
            'secret': self.broker.secret,
            'password': self.broker.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        if self.broker.paper_mode:
            exchange.set_sandbox_mode(True)
            logger.info("🔁 Running in PAPER TRADING mode (sandbox)")

        # Get starting balance
        self.start_balance = self.broker.get_balance('USDT')
        logger.info(f"💰 Starting balance: ${self.start_balance:.2f} USDT")

        while self.running:
            try:
                balance = self.broker.get_balance('USDT')
                
                for symbol in self.symbols:
                    try:
                        ticker = await exchange.watch_ticker(symbol)
                        
                        # 📈 FIXED: Shifted from '15m' to '5m' timeframe and set sequence limit to 512
                        ohlcv = await exchange.fetch_ohlcv(symbol, '5m', limit=512)
                        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # Basic feature engineering for the predictor
                        df['returns'] = df['close'].pct_change()
                        df['vol_14'] = df['returns'].rolling(14).std()
                        
                        # Technical indicators with fallback (no pandas-ta required)
                        try:
                            # Simple RSI calculation
                            delta = df['close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            df['rsi'] = 100 - (100 / (1 + rs))
                            
                            # Simple MACD
                            exp1 = df['close'].ewm(span=12, adjust=False).mean()
                            exp2 = df['close'].ewm(span=26, adjust=False).mean()
                            df['macd'] = exp1 - exp2
                            
                            # Simple ATR
                            high_low = df['high'] - df['low']
                            high_close = (df['high'] - df['close'].shift()).abs()
                            low_close = (df['low'] - df['close'].shift()).abs()
                            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                            df['atr'] = tr.rolling(window=14).mean()
                            
                            # Bollinger Bands width
                            sma = df['close'].rolling(window=20).mean()
                            std = df['close'].rolling(window=20).std()
                            upper_band = sma + (std * 2)
                            lower_band = sma - (std * 2)
                            df['bb_width'] = (upper_band - lower_band) / sma
                            
                        except Exception as e:
                            logger.warning(f"Indicator calculation failed, using defaults: {e}")
                            df['rsi'] = 50
                            df['macd'] = 0
                            df['atr'] = df['close'].rolling(14).std()
                            df['bb_width'] = 0.05
                        
                        # Fill any NaN values
                        df = df.fillna(0)
                        
                        score = self.ml.predict(df)
                        price = ticker['last']
                        logger.info(f"{symbol} | Price: ${price:.2f} | Score: {score:.3f} | Balance: ${balance:.2f}")
                        
                        # Trading logic with logging
                        if score > 0.67 and symbol not in self.positions:
                            size = 0.01  # Fixed size for now
                            order = await exchange.create_order(symbol, 'market', 'buy', size)
                            self.positions[symbol] = {'price': price, 'size': size, 'entry_score': score}
                            self.log_trade(symbol, "BUY", price, size, score)
                            
                        elif score < 0.36 and symbol in self.positions:
                            size = self.positions[symbol]['size']
                            entry_price = self.positions[symbol]['price']
                            order = await exchange.create_order(symbol, 'market', 'sell', size)
                            
                            # Calculate P&L
                            pnl = (price - entry_price) * size
                            self.log_trade(symbol, "SELL", price, size, score, pnl)
                            del self.positions[symbol]
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                
            # Sleep interval for real-time websocket check heartbeat
            await asyncio.sleep(15)
        
        await exchange.close()

    def stop(self):
        self.running = False
        self.save_state()
        logger.info("🛑 Bot stopped. State saved.")


if __name__ == "__main__":
    # Read paper_mode from environment variable (default to True/paper trading)
    paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
    
    bot = GrokApexIroncladBot(paper_mode=paper_mode)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Bot shutdown complete")
