import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
import pandas as pd

from feature_engineering import add_features, FEATURE_COLS, FEATURE_DEFAULTS


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
    """
    Live inference wrapper around the champion model.

    Supports two champion formats, selected purely by file extension:
      *.pth          -> the GQA transformer (sequence model, needs scaler)
      *.joblib/.pkl  -> a sklearn classifier (e.g. the GBT baseline) applied
                        to the LAST feature row; no scaler required.

    Hot-reload (step 5): every predict_batch() call stats the model (and
    scaler) files; if either mtime changed since load, the artifact is
    reloaded in place. A failed reload NEVER takes down the caller -- the
    previously-loaded weights keep serving and the error is printed once;
    the next file change retries.
    """

    def __init__(
        self, model_path="grok_gqa_v9_best.pth", input_dim=11, seq_len=32,
        embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2,
        dropout=0.0
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.seq_len = seq_len
        self._torch_kwargs = dict(
            input_dim=input_dim, seq_len=seq_len, embed_dim=embed_dim,
            num_layers=num_layers, num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads, dropout=dropout,
        )
        # Decision-time feature snapshots, populated by predict_batch().
        # {symbol: {feature_col: value}} — the RAW (unscaled) last-row vector
        # the model actually saw, for experience capture (step 0).
        self.last_features: dict = {}

        self._kind = None      # "torch" | "sklearn"
        self.model = None      # torch module (kind == "torch")
        self.clf = None        # sklearn estimator (kind == "sklearn")
        self.scaler = None
        self._sig = None       # (path, mtime) signature watched for reload

        self._load()

    # ── loading / hot-reload ────────────────────────────────────────────────
    def _scaler_path(self) -> str:
        return os.path.join(os.path.dirname(self.model_path) or ".", "feature_scaler.pkl")

    def _current_sig(self):
        sig = [self.model_path, os.path.getmtime(self.model_path)]
        sp = self._scaler_path()
        if self._kind == "torch" and os.path.exists(sp):
            sig.append(sp)
            sig.append(os.path.getmtime(sp))
        return tuple(sig)

    def _load(self):
        """Load artifacts into LOCALS first and commit to self only after
        every step succeeded — a failed hot-reload must leave the previously
        loaded weights serving, never a half-built random-weight model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        ext = os.path.splitext(self.model_path)[1].lower()
        if ext in (".joblib", ".pkl"):
            clf = joblib.load(self.model_path)   # may raise -> nothing mutated
            kind, model, scaler = "sklearn", None, None
            print(f"✅ Sklearn champion loaded from {self.model_path}")
        else:
            model = GrokGQA_Transformer(**self._torch_kwargs).to(self.device)
            try:
                model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device),
                    strict=True,
                )
                print(f"✅ Model weights loaded from {self.model_path}")
            except Exception as e:
                # Fail loudly rather than silently continuing with a partially-
                # initialized (effectively random-weight) model.
                print(f"❌ Model state dict load failed: {e}")
                raise

            scaler = None
            scaler_path = self._scaler_path()
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded from {scaler_path}")
            else:
                print(
                    f"⚠️  Scaler file '{scaler_path}' not found.\n"
                    f"   Predictions will be unreliable without normalisation.\n"
                    f"   Re-run train_transformer.py to generate feature_scaler.pkl."
                )
            kind = "torch"

        # ── commit point: everything above succeeded ──
        self._kind = kind
        self.model = model
        self.clf = clf if kind == "sklearn" else None
        self.scaler = scaler
        self._sig = self._current_sig()

    def reload_if_changed(self) -> bool:
        """Reload model+scaler if the artifact files changed on disk.

        Returns True if a reload happened. Never raises: on failure the
        previously loaded weights keep serving and we resync the signature so
        a half-written file doesn't cause a retry storm (the next completed
        write changes mtime again and retries naturally).
        """
        try:
            sig = self._current_sig()
        except OSError:
            return False
        if sig == self._sig:
            return False
        print(f"🔁 Model artifact changed on disk — reloading {self.model_path}")
        try:
            self._load()
            return True
        except Exception as e:
            print(f"❌ Hot-reload failed, keeping previous weights: {e}")
            self._sig = sig
            return False

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

        Side effects (intentional, step 0/5 plumbing):
          * checks artifact mtimes and hot-reloads if they changed
          * populates self.last_features[symbol] with the RAW last-row
            feature vector used for that symbol's prediction
        """
        self.reload_if_changed()
        if self._kind == "sklearn":
            return self._predict_batch_sklearn(df_dict)
        try:
            processed = {}
            tensors = []
            symbols_in_order = []
            
            for symbol, df in df_dict.items():
                df = df.copy()
                df_features = add_features(df)
                window = df_features[FEATURE_COLS].tail(self.seq_len)

                if len(window) < self.seq_len:
                    processed[symbol] = 0.5
                    continue

                # Step 0: snapshot the RAW decision-time feature row (the last
                # bar of the window the model is about to see) BEFORE any
                # clipping/scaling, so captured vectors match training-space
                # features exactly.
                self.last_features[symbol] = {
                    col: float(window.iloc[-1][col]) for col in FEATURE_COLS
                }
                data = window.values.astype(np.float32)

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

    def _predict_batch_sklearn(self, df_dict: dict) -> dict:
        """Champion path for sklearn artifacts (e.g. promoted GBT baseline).

        Applies the SAME feature engineering; uses only the last feature row
        (gradient-boosted trees are not sequence models). Populates
        self.last_features exactly like the torch path.
        """
        result = {}
        for symbol, df in df_dict.items():
            try:
                feats = add_features(df.copy())
                if len(feats) == 0:
                    result[symbol] = 0.5
                    continue
                last = feats[FEATURE_COLS].iloc[-1]
                self.last_features[symbol] = {c: float(last[c]) for c in FEATURE_COLS}
                row = feats[FEATURE_COLS].iloc[-1:].values.astype(np.float64)
                proba = self.clf.predict_proba(row)[0]
                classes = list(getattr(self.clf, "classes_", [0, 1]))
                result[symbol] = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
            except Exception as e:
                print(f"Sklearn champion predict error for {symbol}: {e}")
                result[symbol] = 0.5
        return result
