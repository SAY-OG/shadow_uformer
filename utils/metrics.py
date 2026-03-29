import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def calculate_psnr(pred, target, data_range=1.0):
    mse = torch.nn.functional.mse_loss(pred, target)
    if mse == 0: return torch.tensor(100.0).to(pred.device)
    return 10 * torch.log10(data_range**2 / mse)

def calculate_ssim(pred, target):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return ssim_metric(pred, target)
