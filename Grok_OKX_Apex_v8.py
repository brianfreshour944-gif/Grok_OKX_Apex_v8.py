import asyncio
import ccxt.pro as ccxtpro
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("OKX_Bot")

# =================================================================
# PART 1: THE BRAIN (GQA Transformer Architecture)
# =================================================================

class GQA_TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_q_heads=16, num_kv_heads=4, dropout=0.1):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        
        self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        
        x = self.norm1(residual + self.dropout(attn))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class GrokGQA_Transformer(nn.Module):
    def __init__(self, input_features=12, embed_dim=128, num_layers=6, seq_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Linear(input_features, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList([GQA_TransformerBlock(embed_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_enc[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))

class MLPredictor:
    def __init__(self, model_path="grok_gqa_v8.pth", seq_len=512):
        self.seq_len = seq_len
        self.device = torch.device("cpu")
        self.model = GrokGQA_Transformer(seq_len=seq_len).to(self.device)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                logger.info(f"✅ Loaded trained GQA Transformer Weights")
            except Exception as e:
                logger.warning(f"Weights load failed: {e}. Using fresh model.")
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 100: return 0.5
            feat = df[['open', 'high', 'low', 'close', 'volume']].tail(self.seq_len).copy()
            for col in feat.columns:
                mean, std = feat[col].mean(), feat[col].std()
                feat[col] = (feat[col] - mean) / (std + 1e-8)
            
            vals = feat.values
            if vals.shape[1] < 12:
                padding = np.zeros((vals.shape[0], 12 - vals.shape[1]))
                vals = np.hstack([vals, padding])
            
            X = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = self.model(X).item()
            return float(np.clip(prob + np.random.normal(0, 0.01), 0.1, 0.9))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5

# =================================================================
# PART 2: THE BODY (OKX Integration)
# =================================================================

class Grok_OKX_Apex_v8:
    def __init__(self, paper_mode=True):
        self.ml = MLPredictor()
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        self.positions = {}
        self.paper_mode = paper_mode
        
        # Pulling keys from Environment Variables
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')

    async def run(self):
        while True:
            # Injecting Secure Keys
            exchange = ccxtpro.okx({
                'apiKey': self.api_key,
                'secret': self.secret,
                'password': self.passphrase,
                'enableRateLimit': True, 
                'options': {'defaultType': 'swap'}
            })
            
            if self.paper_mode:
                exchange.set_sandbox_mode(True)
            
            logger.info("🌌 GROK OKX APEX v8.1 | ONLINE | ORACLE CLOUD")
            
            try:
                while True:
                    for symbol in self.symbols:
                        try:
                            ticker = await exchange.watch_ticker(symbol)
                            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=600)
                            df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                            
                            score = self.ml.predict(df)
                            price = ticker['last']
                            
                            logger.info(f"OKX-{symbol} | ${price:,.2f} | Score: {score:.3f}")

                            if score > 0.68 and symbol not in self.positions:
                                qty = 0.01 if 'BTC' in symbol else 0.5
                                await exchange.create_order(symbol, 'market', 'buy', qty)
                                self.positions[symbol] = qty
                                logger.info(f"🚀 OKX-BUY {symbol} @ {price}")

                            elif score < 0.35 and symbol in self.positions:
                                await exchange.create_order(symbol, 'market', 'sell', 
                                                          self.positions[symbol], 
                                                          params={'reduceOnly': True})
                                del self.positions[symbol]
                                logger.info(f"🔻 OKX-CLOSE {symbol}")

                        except Exception as e:
                            logger.debug(f"Scan delay: {e}")
                    
                    await asyncio.sleep(15)

            except Exception as e:
                logger.error(f"🚨 Connection lost: {e}. Rewaking in 10s...")
                await asyncio.sleep(10)
            finally:
                await exchange.close()

if __name__ == "__main__":
    bot = Grok_OKX_Apex_v8(paper_mode=True)
    asyncio.run(bot.run())
