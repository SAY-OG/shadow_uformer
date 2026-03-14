import torch
import torch.nn as nn
from .modules.blocks import LocalEnhancedBlock
from .modules.caf import WindowCrossAttention
from .modules.modulator import MultiScaleModulator

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Fixes Aliasing: Bilinear upsampling is smoother than ConvTranspose2d [cite: 134]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Downsampling using strided convolution [cite: 135]
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ShadowUformer(nn.Module):
    """
    ShadowUformer: Transformer-based Image Shadow Removal
    Integrates CAF for feature fusion and Multi-Scale Modulators for illumination calibration.
     [cite: 954, 135-139]
    """
    def __init__(self, base_dim=48, window_size=8, num_heads=8):
        super().__init__()

        # --- Encoder Path ---
        # Level 1: [B, 3, H, W] -> [B, 48, H, W]
        self.enc1 = nn.Conv2d(3, base_dim, 3, padding=1)
        self.block1 = LocalEnhancedBlock(base_dim, window_size, num_heads, shift=False)

        # Level 2: [B, 48, H, W] -> [B, 96, H/2, W/2]
        self.down1 = Downsample(base_dim, base_dim * 2)
        self.block2 = LocalEnhancedBlock(base_dim * 2, window_size, num_heads, shift=True)

        # Level 3: [B, 96, H/2, W/2] -> [B, 192, H/4, W/4]
        self.down2 = Downsample(base_dim * 2, base_dim * 4)
        self.block3 = LocalEnhancedBlock(base_dim * 4, window_size, num_heads, shift=False)

        # --- Bottleneck ---
        # Processing at lowest resolution [B, 192, H/4, W/4]
        self.bottleneck = LocalEnhancedBlock(base_dim * 4, window_size, num_heads, shift=True)

        # --- Decoder Path ---
        # Level 2 Decoder: Feature Fusion using CAF and Modulator
        self.up1 = Upsample(base_dim * 4, base_dim * 2)
        self.mod1 = MultiScaleModulator(base_dim * 2) # Calibrates encoder skip features [cite: 151]
        self.caf1 = WindowCrossAttention(base_dim * 2, window_size, num_heads) # Advanced fusion [cite: 181]
        self.dec1 = LocalEnhancedBlock(base_dim * 2, window_size, num_heads, shift=False)

        # Level 1 Decoder: Final restoration stage
        self.up2 = Upsample(base_dim * 2, base_dim)
        self.mod2 = MultiScaleModulator(base_dim)
        self.caf2 = WindowCrossAttention(base_dim, window_size, num_heads)
        self.dec2 = LocalEnhancedBlock(base_dim, window_size, num_heads, shift=True)

        # Final Convolution: Reconstructing RGB image
        self.out_conv = nn.Conv2d(base_dim, 3, 3, padding=1)

    def forward(self, x):
        # 1. Encoder Flow
        e1 = self.block1(self.enc1(x))
        e2 = self.block2(self.down1(e1))
        e3 = self.block3(self.down2(e2))
        
        # 2. Bottleneck
        b = self.bottleneck(e3)
        
        # 3. Decoder Stage 1 (CAF + Modulator)
        # Instead of simple addition, we use Cross-Attention to fuse upsampled features with modulated skip features
        d1 = self.up1(b)
        e2_mod = self.mod1(e2) # Apply learnable gamma/beta calibration [cite: 152]
        d1 = self.caf1(d1, e2_mod) # Query = Up-feature, Key/Value = Modulated Encoder feature [cite: 185]
        d1 = self.dec1(d1)
        
        # 4. Decoder Stage 2 (Final Fusion)
        d2 = self.up2(d1)
        e1_mod = self.mod2(e1)
        d2 = self.caf2(d2, e1_mod)
        d2 = self.dec2(d2)
        
        # Output Clamp ensures pixel values remain within [0, 1] [cite: 139]
        return torch.clamp(self.out_conv(d2), 0, 1)
