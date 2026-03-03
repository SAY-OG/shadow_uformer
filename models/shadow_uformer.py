import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()

        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws

        assert H % ws == 0 and W % ws == 0

        x = rearrange(
            x,
            "b c (h ws1) (w ws2) -> (b h w) (ws1 ws2) c",
            ws1=ws, ws2=ws
        )

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
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

class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ShadowUformer(nn.Module):
    def __init__(
        self,
        base_dim=48,
        window_size=8,
        num_heads=8
    ):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, base_dim, 3, padding=1)
        self.block1 = TransformerBlock(base_dim, window_size, num_heads)

        self.down1 = Downsample(base_dim, base_dim * 2)
        self.block2 = TransformerBlock(base_dim * 2, window_size, num_heads)

        self.down2 = Downsample(base_dim * 2, base_dim * 4)
        self.block3 = TransformerBlock(base_dim * 4, window_size, num_heads)

        # Bottleneck
        self.bottleneck = TransformerBlock(base_dim * 4, window_size, num_heads)

        # Decoder
        self.up1 = Upsample(base_dim * 4, base_dim * 2)
        self.dec1 = TransformerBlock(base_dim * 2, window_size, num_heads)

        self.up2 = Upsample(base_dim * 2, base_dim)
        self.dec2 = TransformerBlock(base_dim, window_size, num_heads)

        # Output
        self.out_conv = nn.Conv2d(base_dim, 3, 3, padding=1)

    def forward(self, x):

        # Encoder
        e1 = self.block1(self.enc1(x))
        e2 = self.block2(self.down1(e1))
        e3 = self.block3(self.down2(e2))

        b = self.bottleneck(e3)

        # Decoder
        d1 = self.up1(b) + e2
        d1 = self.dec1(d1)

        d2 = self.up2(d1) + e1
        d2 = self.dec2(d2)

        out = self.out_conv(d2)

        return torch.clamp(out, 0, 1)
