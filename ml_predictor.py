import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        x = self.norm1(residual + self.dropout(attn))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class GrokGQA_Transformer(nn.Module):
    def __init__(self, input_features=11, embed_dim=128, num_layers=8, seq_len=512):
        super().__init__()
        self.embed = nn.Linear(input_features, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList([GQA_TransformerBlock(embed_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
    
    def forward(self, x):
        x = self.embed(x) + self.pos_enc[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))

class MLPredictor:
    def __init__(self, model_path="grok_gqa_v9_best.pth", seq_len=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GrokGQA_Transformer(seq_len=seq_len).to(self.device)
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Loaded model from {self.model_path}")
        else:
            print(f"⚠️ Model file {self.model_path} not found. Using default predictions.")

    def predict(self, df):
        try:
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'vol_14', 'rsi', 'macd', 'atr', 'bb_width']
            
            # Check if all columns exist, if not create defaults
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.5 if col in ['rsi', 'macd', 'atr', 'bb_width'] else 0.0
            
            x = torch.tensor(df[feature_cols].tail(512).values, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model(x).item()
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
