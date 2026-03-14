import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from models.shadow_uformer import ShadowUformer

def load_model(checkpoint_path, device):
    model = ShadowUformer().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_tapered_window(patch_size, device):
    """Creates a 2D linear ramp window to blend patches smoothly."""
    vec = torch.linspace(0, 1, steps=patch_size//4, device=device)
    middle = torch.ones(patch_size - patch_size//2, device=device)
    ramp = torch.cat([vec, middle, vec.flip(0)])
    window_2d = ramp.unsqueeze(0) * ramp.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

def tiled_inference(model, img_tensor, patch_size=256, stride=128, device='cuda'):
    b, c, h, w = img_tensor.shape
    
    # 1. Pad image so it's at least one patch size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, ph, pw = img_padded.shape

    output = torch.zeros_like(img_padded)
    weight_mask = torch.zeros_like(img_padded)
    window = get_tapered_window(patch_size, device)

    # 2. Generate coordinates that GUARANTEE boundary coverage
    def get_coords(full_size, patch_s, strd):
        coords = list(range(0, full_size - patch_s + 1, strd))
        if coords[-1] != full_size - patch_s:
            coords.append(full_size - patch_s)
        return coords

    y_coords = get_coords(ph, patch_size, stride)
    x_coords = get_coords(pw, patch_size, stride)

    # 3. Process Patches
    for y in y_coords:
        for x in x_coords:
            patch = img_padded[:, :, y:y+patch_size, x:x+patch_size]
            
            with torch.inference_mode():
                res = model(patch)
            
            output[:, :, y:y+patch_size, x:x+patch_size] += res * window
            weight_mask[:, :, y:y+patch_size, x:x+patch_size] += window

    # 4. Final Blend and Crop
    # Add a tiny epsilon (1e-8) to avoid division by zero
    output = output / (weight_mask + 1e-8)
    return output[:, :, :h, :w]

def run_inference(model, image_path, output_path, device):
    img = Image.open(image_path).convert('RGB')
    input_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    # Reduced stride (128) provides more overlap for better shadow blending
    output_tensor = tiled_inference(model, input_tensor, patch_size=256, stride=128, device=device)
    
    # Clamp to 0-1 range to ensure valid pixel values
    output_tensor = torch.clamp(output_tensor, 0, 1)
    
    output_img = TF.to_pil_image(output_tensor.squeeze(0).cpu())
    output_img.save(output_path)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "checkpoints/best_model.pth"
    INPUT_PATH = "source.jpg"
    OUTPUT_PATH = "result.png"

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, DEVICE)
        run_inference(model, INPUT_PATH, OUTPUT_PATH, DEVICE)
    else:
        print(f"Error: Checkpoint not found at {MODEL_PATH}")
