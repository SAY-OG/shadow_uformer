import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
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

def tiled_inference(model, img_tensor, patch_size=256, stride=192, device='cuda'):
    b, c, h, w = img_tensor.shape
    
    # 1. Pad image so patches fit perfectly
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, ph, pw = img_padded.shape

    # 2. Create empty canvases for output and weight mask (for blending)
    output = torch.zeros_like(img_padded)
    weight_mask = torch.zeros_like(img_padded)
    
    # 3. Create a linear blending window to smooth seams
    window = torch.ones((1, 1, patch_size, patch_size)).to(device)
    # Optional: could add a gaussian taper here for even smoother seams

    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            # Extract patch
            patch = img_padded[:, :, y:y+patch_size, x:x+patch_size]
            
            with torch.inference_mode(): # Faster/Leaner than no_grad
                # Process patch
                res = model(patch)
            
            # Add result to output and increment weight mask
            output[:, :, y:y+patch_size, x:x+patch_size] += res * window
            weight_mask[:, :, y:y+patch_size, x:x+patch_size] += window
            
            # Clear cache if VRAM is extremely tight
            # torch.cuda.empty_cache() 

    # 4. Average the overlapping areas and crop back to original size
    output /= weight_mask
    return output[:, :, :h, :w]

def run_inference(model, image_path, output_path, device):
    img = Image.open(image_path).convert('RGB')
    input_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    # Use Tiled Inference instead of one single forward pass
    output_tensor = tiled_inference(model, input_tensor, patch_size=256, stride=192, device=device)
    
    output_img = TF.to_pil_image(output_tensor.squeeze(0).clamp(0, 1).cpu())
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
        print(f"Checkpoint not found.")
