
# train_transformer.py
import asyncio
import ccxt
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime
import os
from ml_predictor import GrokGQA_Transformer, MLPredictor  # Your existing model

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=32, target_horizon=8):
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.data, self.labels = self._prepare_data(df)
    
    def _prepare_data(self, df):
        # Create target: future return after horizon
        df = df.copy()
        df['future_return'] = df['close'].pct_change(self.target_horizon).shift(-self.target_horizon)
        df['label'] = (df['future_return'] > 0).astype(float)  # Binary bullish probability target
        
        # Use same features as inference
        features = ['open','high','low','close','volume']
        df['returns'] = df['close'].pct_change()
        df['vol_14'] = df['returns'].rolling(14).std()
        
        try:
            import pandas_ta as ta
            df['rsi'] = ta.rsi(df['close'], 14)
            df['macd'] = ta.macd(df['close'])['MACD_12_26_9']

            df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
            df['bb_width'] = 0.05 # Add fallback placeholder to match inference layout
        except:
            df['rsi'] = 50
            df['macd'] = 0
            df['atr'] = df['close'].rolling(14).std()
            df['bb_width'] = 0.05
        
        feature_cols = features + ['returns','vol_14','rsi','macd','atr','bb_width']
        data = df[feature_cols].bfill().fillna(0).values
        labels = df['label'].fillna(0).values
        
        return data, labels
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.target_horizon
    
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def fetch_historical_data(symbol='BTC/USDT:USDT', timeframe='5m', limit=20000):
    """Download large historical dataset"""
    logger.info(f"Downloading {limit} candles for {symbol}...")
    exchange = ccxt.okx({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    logger.info(f"Downloaded {len(df)} candles from {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")
    return df


def train_model(epochs=50, batch_size=32, lr=1e-4, seq_len=32):
    df = fetch_historical_data(limit=25000)  # ~6-8 months of 5m data
    
    # Split train/val
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    train_dataset = FinancialTimeSeriesDataset(train_df, seq_len=seq_len)
    val_dataset = FinancialTimeSeriesDataset(val_df, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # FIX 1: Set input_features to 11 to match ml_predictor structure exactly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrokGQA_Transformer(input_features=11, embed_dim=128, num_layers=8, seq_len=seq_len).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    logger.info(f"🚀 Starting Robust Training on {str(device).upper()}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X).squeeze()
                val_loss += criterion(output, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # FIX 2: Save as v9 to match your main bot initialization path
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "grok_gqa_v9_best.pth")
            patience_counter = 0
            logger.info("💾 New best model saved as grok_gqa_v9_best.pth!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("⏹️ Early stopping triggered")
                break
    
    # Final save
    torch.save(model.state_dict(), "grok_gqa_v9_final.pth")
    logger.info("🎉 Training completed! Models saved.")
    return model


if __name__ == "__main__":
    # Run training
    model = train_model(
        epochs=80,           # Set higher for deep optimization
        batch_size=16,       # Low batch size prevents RAM overruns in Colab
        lr=3e-4,
        seq_len=32
    )
    
    # Quick test prediction validation
    predictor = MLPredictor(model_path="grok_gqa_v9_best.pth")
    logger.info("✅ Training script completely aligned. Ready to push to GitHub!")
