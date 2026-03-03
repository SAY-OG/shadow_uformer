import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models.shadow_uformer import ShadowUformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- SETTINGS --------
model_path = "checkpoints/best_model.pth"
input_dir = "data/ISTD/test/test_A"
output_dir = "results"
image_size = 256  # must match training
# --------------------------

os.makedirs(output_dir, exist_ok=True)

# Load model
model = ShadowUformer(
    base_dim=48,
    window_size=8,
    num_heads=8
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

with torch.no_grad():
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(device)

        pred = model(img_tensor)
        pred = torch.clamp(pred, 0, 1)

        save_image(
            pred,
            os.path.join(output_dir, filename)
        )

print("Inference complete.")
