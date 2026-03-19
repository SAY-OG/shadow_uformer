import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class JointLoss(nn.Module):
    def __init__(self, device):
        super(JointLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.charbonnier = nn.L1Loss()

    def ssim_loss(self, pred, target):
        return 1 - F.cosine_similarity(pred, target, dim=1).mean()

    def forward(self, pred, target, mask):
        loss_pixel = self.charbonnier(pred, target)
        loss_shadow = self.charbonnier(pred * mask, target * mask)
        loss_perceptual = F.mse_loss(self.vgg(pred), self.vgg(target))
        total_loss = loss_pixel + (2.0 * loss_shadow) + (0.05 * loss_perceptual)
        return total_loss
