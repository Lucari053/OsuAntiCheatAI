import torch
import torch.nn as nn
from model.components.utils import CrossAttentionBlock

class WindowEncoder(nn.Module):
    def __init__(
        self,
        n_beatmap_features: int = 4,
        n_replay_features: int = 13,
        n_object: int = 10,
        n_frame: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_self_layers: int = 2
    ):
        super().__init__()
        self.beatmap_proj = nn.Linear(n_beatmap_features, d_model)
        self.replay_proj  = nn.Linear(n_replay_features, d_model)

        self.beatmap_pos = nn.Parameter(torch.randn(1, n_object, d_model) * 0.02)
        self.replay_pos  = nn.Parameter(torch.randn(1, n_frame,  d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True)
        self.beatmap_encoder = nn.TransformerEncoder(layer, n_self_layers)
        self.replay_encoder  = nn.TransformerEncoder(layer, n_self_layers)

        self.cross_attn = CrossAttentionBlock(d_model, n_heads)

    def forward(self, beatmap_ctx: torch.Tensor, replay_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
           beatmap_ctx: [B, n_object, n_beatmap_features]
           replay_ctx:  [B, n_frame, n_replay_features]
        """
        b = self.beatmap_encoder(self.beatmap_proj(beatmap_ctx) + self.beatmap_pos)
        r = self.replay_encoder(self.replay_proj(replay_ctx) + self.replay_pos)

        fused = self.cross_attn(query=b, context=r) # [B, n_object, d_model]
        return fused.mean(dim=1) # [B, d_model]