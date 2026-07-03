import torch
import torch.nn as nn
from model.components.window_encoder import WindowEncoder
from model.components.utils import SequenceAggregator



class CheatDetector(nn.Module):
    def __init__(self, d_model: int = 128, **window_kwargs):
        super().__init__()
        self.window_encoder = WindowEncoder(d_model=d_model, **window_kwargs)
        self.sequence_aggregator = SequenceAggregator(d_model=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self, 
        beatmap_windows: torch.Tensor, 
        replay_windows: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            beatmap_windows: [B, n_windows, n_object, n_beatmap_features]
            replay_windows : [B, n_windows, n_frame, n_replay_features]
        """
        B, N = beatmap_windows.shape[:2]

        b_flat = beatmap_windows.flatten(0, 1)
        r_flat = replay_windows.flatten(0, 1)

        window_emb = self.window_encoder(b_flat, r_flat).view(B, N, -1)

        pooled, attn_weights = self.sequence_aggregator(window_emb, key_padding_mask=key_padding_mask)
        logit = self.head(pooled).squeeze(-1) # [B,]
        return logit, attn_weights