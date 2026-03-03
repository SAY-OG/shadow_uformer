from .attention import SwinWindowAttention
from .leff import LeFF
import torch.nn as nn


class LocalEnhancedBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, shift=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SwinWindowAttention(
            dim,
            window_size,
            num_heads,
            shift=shift
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = LeFF(dim)

    def forward(self, x):

        B, C, H, W = x.shape

        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)

        x = x + self.attn(x_norm)

        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_perm).permute(0, 3, 1, 2)

        x = x + self.ffn(x_norm)

        return x
