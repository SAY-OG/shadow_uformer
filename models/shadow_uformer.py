import torch
import torch.nn as nn
from .modules.blocks import LocalEnhancedBlock
from .modules.caf import WindowCrossAttention
from .modules.modulator import MultiScaleModulator

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ShadowUformer(nn.Module):
    def __init__(self, base_dim=48, window_size=8, num_heads=8):
        super().__init__()

        self.enc1 = nn.Conv2d(3, base_dim, 3, padding=1)
        self.block1 = LocalEnhancedBlock(base_dim, window_size, num_heads, shift=False)

        self.down1 = Downsample(base_dim, base_dim * 2)
        self.block2 = LocalEnhancedBlock(base_dim * 2, window_size, num_heads, shift=True)

        self.down2 = Downsample(base_dim * 2, base_dim * 4)
        self.block3 = LocalEnhancedBlock(base_dim * 4, window_size, num_heads, shift=False)

        self.bottleneck = LocalEnhancedBlock(base_dim * 4, window_size, num_heads, shift=True)

        self.up1 = Upsample(base_dim * 4, base_dim * 2)
        self.mod1 = MultiScaleModulator(base_dim * 2)
        self.caf1 = WindowCrossAttention(base_dim * 2, window_size, num_heads)
        self.dec1 = LocalEnhancedBlock(base_dim * 2, window_size, num_heads, shift=False)

        self.up2 = Upsample(base_dim * 2, base_dim)
        self.mod2 = MultiScaleModulator(base_dim)
        self.caf2 = WindowCrossAttention(base_dim, window_size, num_heads)
        self.dec2 = LocalEnhancedBlock(base_dim, window_size, num_heads, shift=True)

        self.out_conv = nn.Conv2d(base_dim, 3, 3, padding=1)

    def forward(self, x):
        input_img = x

        e1 = self.block1(self.enc1(x))
        e2 = self.block2(self.down1(e1))
        e3 = self.block3(self.down2(e2))
        
        b = self.bottleneck(e3)
        
        d1 = self.up1(b)
        e2_mod = self.mod1(e2)
        d1 = self.caf1(d1, e2_mod)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        e1_mod = self.mod2(e1)
        d2 = self.caf2(d2, e1_mod)
        d2 = self.dec2(d2)

        out = self.out_conv(d2) + input_img
        
        return torch.clamp(out, 0, 1)
