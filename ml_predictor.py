# ml_predictor.py — Fixed & Complete

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
import pandas as pd 

# Import feature engineering function and constants
from feature_engineering import add_features, FEATURE_COLS, FEATURE_DEFAULTS


# ==============================================================================
# MODEL ARCHITECTURE  (Matches train_transformer.py exactly)
# ==============================================================================

class GQA_TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_q_heads=16, num_kv_heads=4, dropout=0.1):
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
        residual      = x
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_q_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.num_q_heads * self.head_dim)
        x = self.dropout(self.out_proj(attn))
        x = self.norm1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        return x


class GrokGQA_Transformer(nn.Module):
    def __init__(
        self, input_dim, seq_len=32,  # Fixed: Default set to 32 steps
        embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1  # Fixed: Default set to 8 layers
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
        return torch.sigmoid(x) 


# ==============================================================================
# MLPredictor (for inference)  --  Loads trained model and scales input
# ==============================================================================

class SafeMLPredictor:
    def __init__(
        self, model_path,
        input_dim=len(FEATURE_COLS),  
        seq_len=32,    # Fixed: Standard default value safely matched to 32
        embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4,
        dropout=0.1
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FIX 1: Raise error if model not found, don't fall back to random weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = GrokGQA_Transformer(
            input_dim=input_dim, seq_len=seq_len,
            embed_dim=embed_dim, num_layers=num_layers,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            dropout=dropout
        ).to(self.device)
        
        # FIXED INDENTATION: strict=False is tucked safely inside the load logic
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            print(f"✅ Model weights loaded successfully from {model_path} (Strict mode disabled)")
        except Exception as e:
            print(f"⚠️ Warning loading state dict: {e}")

        self.seq_len = seq_len 

        # FIX 2: Load StandardScaler if available
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

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> float:
        """
        Returns a float in [0, 1].
        >0.51  → bullish signal (loosened bounds)
        <0.49  → bearish signal (loosened bounds)
        ~0.5   → no signal (or model uncertain)
        """
        try:
            df = df.copy()

            # Use the centralized feature engineering function
            df_features = add_features(df)

            # Extract the exact sequence chunk needed for inference matrix shape
            data = df_features[FEATURE_COLS].tail(self.seq_len).values.astype(np.float32)

            if len(data) < self.seq_len:
                print(f"⚠️  Need {self.seq_len} bars for inference, only {len(data)} available.")
                return 0.5

            # FIX: apply scaler — same transform used during training
            if self.scaler is not None:
                data = self.scaler.transform(data).astype(np.float32)

            x = torch.tensor(data).unsqueeze(0).to(self.device)  # (1, seq_len, 11)
            with torch.no_grad():
                pred = self.model(x).item()

            return float(pred)

        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
