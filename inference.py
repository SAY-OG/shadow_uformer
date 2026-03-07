import torch
import torchvision.transforms as transforms
from PIL import Image

from models.shadow_uformer import ShadowUformer


MODEL_PATH = "checkpoints/best_model.pth"
INPUT_IMAGE = "input.jpg"
OUTPUT_IMAGE = "output.png"
IMAGE_SIZE = 256


def load_model(device):

    model = ShadowUformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model


def preprocess():

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(INPUT_IMAGE).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    return tensor, img.size


def postprocess(tensor, original_size):

    tensor = tensor.squeeze().cpu().clamp(0,1)

    img = transforms.ToPILImage()(tensor)
    img = img.resize(original_size)

    img.save(OUTPUT_IMAGE)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)

    img_tensor, original_size = preprocess()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    postprocess(output, original_size)

    print("Saved result to", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
