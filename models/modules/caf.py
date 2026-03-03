import torch
import torch.nn as nn
from einops import rearrange


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()

        self.ws = window_size
        self.heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, skip):

        B, C, H, W = x.shape
        ws = self.ws

        x = rearrange(
            x,
            "b c (h ws1) (w ws2) -> (b h w) (ws1 ws2) c",
            ws1=ws,
            ws2=ws
        )

        skip = rearrange(
            skip,
            "b c (h ws1) (w ws2) -> (b h w) (ws1 ws2) c",
            ws1=ws,
            ws2=ws
        )

        q = self.q(x)
        k, v = self.kv(skip).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.proj(out)

        out = rearrange(
            out,
            "(b h w) (ws1 ws2) c -> b c (h ws1) (w ws2)",
            b=B,
            h=H // ws,
            w=W // ws,
            ws1=ws,
            ws2=ws
        )

        return out
