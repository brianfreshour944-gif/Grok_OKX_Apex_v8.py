import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
import pandas as pd

from feature_engineering import add_features, FEATURE_COLS, FEATURE_DEFAULTS

# This module's own print() calls below use emoji (checkmarks/warnings).
# main_bot.py reconfigures stdout/stderr to UTF-8 before importing this
# module, but every diagnostic/backtest script that imports SafeMLPredictor
# directly (backtest_exact.py, backtest_grid_search.py, backtest_live_exits.py,
# batch_test.py, latency_scan.py, perf_detail.py, signal_ic_check.py,
# train_transformer.py) does NOT -- on Windows (or any console using a
# non-UTF-8 default codepage, e.g. cp1252), that crashes with
# UnicodeEncodeError at model-load time, before any real work happens.
# Reconfiguring here, once, at the source of the emoji output, protects every
# caller regardless of whether it remembered to do this itself.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # stream doesn't support reconfigure (e.g. some capture shims) -- non-fatal


# ==============================================================================
# MODEL ARCHITECTURE  (Matches train_transformer.py exactly)
# ==============================================================================

class GQA_TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_q_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = embed_dim // num_q_heads
        self.q_proj   = nn.Linear(embed_dim, num_q_heads  * self.head_dim)
        self.k_proj   = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj   = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim)
        self.norm1    = nn.LayerNorm(embed_dim)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.ffn      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN: Normalize BEFORE attention, residual bypasses normalization
        residual = x
        norm_x = self.norm1(x)
        batch, seq, _ = norm_x.shape
        
        q = self.q_proj(norm_x).view(batch, seq, self.num_q_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.num_q_heads * self.head_dim)
        x = residual + self.dropout(self.out_proj(attn))
        
        # Pre-LN: Normalize BEFORE feed-forward network
        residual = x
        norm_x = self.norm2(x)
        x = residual + self.ffn(norm_x)
        return x


class GrokGQA_Transformer(nn.Module):
    def __init__(
        self, input_dim, seq_len=32,
        embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder      = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.dropout          = nn.Dropout(dropout)
        self.layers           = nn.ModuleList([
            GQA_TransformerBlock(embed_dim, num_q_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm             = nn.LayerNorm(embed_dim)
        self.output_head      = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoder 
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_head(x[:, -1, :])
        return x   # <--- FIX: raw logits, no sigmoid


# ==============================================================================
# MLPredictor (for inference)
# ==============================================================================

class SafeMLPredictor:
    def __init__(
        self, model_path="grok_gqa_v9_best.pth", input_dim=11, seq_len=32,
        embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2,
        dropout=0.0
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = GrokGQA_Transformer(
            input_dim=input_dim, seq_len=seq_len,
            embed_dim=embed_dim, num_layers=num_layers,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            dropout=dropout
        ).to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
            print(f"✅ Model weights loaded from {model_path}")
        except Exception as e:
            # Fail loudly rather than silently continuing with a partially-
            # initialized (effectively random-weight) model.
            print(f"❌ Model state dict load failed: {e}")
            raise

        self.seq_len = seq_len

        scaler_path = os.path.join(os.path.dirname(model_path), 'feature_scaler.pkl')
        self.scaler = None
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print(
                f"⚠️  Scaler file '{scaler_path}' not found.\n"
                f"   Predictions will be unreliable without normalisation.\n"
                f"   Re-run train_transformer.py to generate feature_scaler.pkl."
            )

    def predict(self, df: pd.DataFrame) -> float:
        """
        Returns a probability in [0, 1].
        >0.51 -> bullish, <0.49 -> bearish, ~0.5 -> no signal.
        """
        result = self.predict_batch({"single": df})
        return result.get("single", 0.5)

    def predict_batch(self, df_dict: dict) -> dict:
        """
        Batch inference for multiple symbols. Takes a dict of {symbol: df}
        and returns {symbol: signal} with a single model forward pass.

        df values may contain MORE than seq_len raw bars (data_feeds.py no
        longer pre-truncates). add_features() is deliberately called on the
        full frame BEFORE the tail(seq_len) slice below, matching
        train_transformer.py's order of operations, so rolling-window
        features (20-bar Z-scores etc.) for the window's early bars are
        computed with real prior history instead of a truncated, partial
        window -- see the note in data_feeds.get_clean_ohlcv_dataframe.
        """
        try:
            processed = {}
            tensors = []
            symbols_in_order = []

            for symbol, df in df_dict.items():
                df = df.copy()
                df_features = add_features(df)
                data = df_features[FEATURE_COLS].tail(self.seq_len).values.astype(np.float32)
                
                if len(data) < self.seq_len:
                    processed[symbol] = 0.5
                    continue
                
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data = np.clip(data, -1e6, 1e6)
                
                if self.scaler is not None:
                    data = self.scaler.transform(data).astype(np.float32)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                x = torch.tensor(data).unsqueeze(0)
                tensors.append(x)
                symbols_in_order.append(symbol)
            
            if not tensors:
                return {s: 0.5 for s in df_dict.keys()}
            
            # Stack all tensors for batched inference: (batch_size, seq_len, n_features)
            batch = torch.cat(tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                raw_logits = self.model(batch)  # (batch_size, 1)
                preds = torch.sigmoid(raw_logits).squeeze(-1)  # (batch_size,)
            
            result = {}
            for symbol in df_dict.keys():
                if symbol in symbols_in_order:
                    idx = symbols_in_order.index(symbol)
                    result[symbol] = float(preds[idx].item())
                else:
                    result[symbol] = processed.get(symbol, 0.5)
            
            return result
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return {symbol: 0.5 for symbol in df_dict.keys()}
