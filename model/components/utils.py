import torch
import torch.nn as nn
import math

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(query, context, context)
        x = self.norm1(query + attn_out)
        return self.norm2(x + self.ff(x))
    
class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, 1, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        query = self.query.expand(x.size(0), -1, -1)
        pooled, weights = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1), weights.squeeze(1) # [B, d_model], [B, n_windows]
    
class SequenceAggregator(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, max_len: int = 2048):
        super().__init__()
        self.register_buffer("pos_encoding", self._build_sinusoidal(max_len, d_model))
        layer        = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.pooling = AttentionPooling(d_model)

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, window_embeddings: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        Args: 
            window_embeddings: [B, n_windows, d_model]
        """
        n_windows = window_embeddings.size(1)
        x = self.encoder(window_embeddings + self.pos_encoding[:n_windows], src_key_padding_mask=key_padding_mask)
        return self.pooling(x, key_padding_mask=key_padding_mask)  # [B, d_model], [B, n_windows]
