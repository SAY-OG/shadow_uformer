import torch
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from models.shadow_uformer import ShadowUformer
from datasets.istd import ISTDDataset
from datasets.transforms import ValTransform
from utils.metrics import calculate_psnr, calculate_ssim

# Reuse the boundary-aware tiled inference logic
def tapered_tiled_inference(model, img_tensor, patch_size=256, stride=128, device='cuda'):
    b, c, h, w = img_tensor.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    img_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, ph, pw = img_padded.shape

    output = torch.zeros_like(img_padded)
    weight_mask = torch.zeros_like(img_padded)
    
    # Create blending window
    vec = torch.linspace(0, 1, steps=patch_size//4, device=device)
    middle = torch.ones(patch_size - patch_size//2, device=device)
    ramp = torch.cat([vec, middle, vec.flip(0)])
    window = (ramp.unsqueeze(0) * ramp.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    def get_coords(full_size, patch_s, strd):
        coords = list(range(0, full_size - patch_s + 1, strd))
        if coords[-1] != full_size - patch_s:
            coords.append(full_size - patch_s)
        return coords

    y_coords = get_coords(ph, patch_size, stride)
    x_coords = get_coords(pw, patch_size, stride)

    for y in y_coords:
        for x in x_coords:
            patch = img_padded[:, :, y:y+patch_size, x:x+patch_size]
            with torch.inference_mode():
                res = model(patch)
            output[:, :, y:y+patch_size, x:x+patch_size] += res * window
            weight_mask[:, :, y:y+patch_size, x:x+patch_size] += window

    output = output / (weight_mask + 1e-8)
    return output[:, :, :h, :w]

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = ShadowUformer().to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Setup Dataset (Using ValTransform for consistent testing)
    test_dataset = ISTDDataset("data/ISTD", split="test", transform=ValTransform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 3. Output Directory
    res_dir = "results/test_outputs"
    os.makedirs(res_dir, exist_ok=True)

    avg_psnr = 0.0
    avg_ssim = 0.0
    
    print(f"Starting testing on {len(test_dataset)} images...")

    # 4. Evaluation Loop
    for i, (img, target) in enumerate(tqdm(test_loader)):
        img, target = img.to(device), target.to(device)
        
        # Perform Tiled Inference
        with torch.inference_mode():
            pred = tapered_tiled_inference(model, img, patch_size=256, stride=128, device=device)
            pred = torch.clamp(pred, 0, 1)

        # Calculate Metrics
        psnr = calculate_psnr(pred, target).item()
        ssim = calculate_ssim(pred, target).item()
        avg_psnr += psnr
        avg_ssim += ssim

        # Save some visual results (every 10th image)
        if i % 10 == 0:
            pred_img = TF.to_pil_image(pred.squeeze(0).cpu())
            pred_img.save(os.path.join(res_dir, f"test_{i}_psnr_{psnr:.2f}.png"))

    # 5. Final Summary
    avg_psnr /= len(test_loader)
    avg_ssim /= len(test_loader)
    
    print("\n" + "="*30)
    print(f"Test Results:")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*30)

if __name__ == "__main__":
    test()
