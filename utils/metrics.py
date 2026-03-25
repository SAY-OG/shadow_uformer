import torch

def calculate_psnr(pred, target, data_range=1.0):
    mse = torch.nn.functional.mse_loss(pred, target)
    if mse == 0: return torch.tensor(100.0).to(pred.device)
    return 10 * torch.log10(data_range**2 / mse)

def calculate_ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)

    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    return ssim
