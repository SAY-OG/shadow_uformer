import torch
import torch.nn as nn
from .modules.blocks import LocalEnhancedBlock

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Replacing ConvTranspose2d  with Bilinear + Conv to stop aliasing
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ShadowUformer(nn.Module):
    def __init__(self, base_dim=48, window_size=8, num_heads=8):
        super().__init__()

        # Encoder: Alternating shifts to fix shadow outlines [cite: 373]
        self.enc1 = nn.Conv2d(3, base_dim, 3, padding=1)
        self.block1 = LocalEnhancedBlock(base_dim, window_size, num_heads, shift=False)

        self.down1 = Downsample(base_dim, base_dim * 2)
        self.block2 = LocalEnhancedBlock(base_dim * 2, window_size, num_heads, shift=True)

        self.down2 = Downsample(base_dim * 2, base_dim * 4)
        self.block3 = LocalEnhancedBlock(base_dim * 4, window_size, num_heads, shift=False)

        # Bottleneck
        self.bottleneck = LocalEnhancedBlock(base_dim * 4, window_size, num_heads, shift=True)

        # Decoder
        self.up1 = Upsample(base_dim * 4, base_dim * 2)
        self.dec1 = LocalEnhancedBlock(base_dim * 2, window_size, num_heads, shift=False)

        self.up2 = Upsample(base_dim * 2, base_dim)
        self.dec2 = LocalEnhancedBlock(base_dim, window_size, num_heads, shift=True)

        self.out_conv = nn.Conv2d(base_dim, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.block1(self.enc1(x))
        e2 = self.block2(self.down1(e1))
        e3 = self.block3(self.down2(e2))
        b = self.bottleneck(e3)
        d1 = self.up1(b) + e2
        d1 = self.dec1(d1)
        d2 = self.up2(d1) + e1
        d2 = self.dec2(d2)
        return torch.clamp(self.out_conv(d2), 0, 1)
