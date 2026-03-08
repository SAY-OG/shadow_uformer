import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class ISTDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__()
        assert split in ["train", "test"]

        self.root = root
        self.split = split
        self.dir_A = os.path.join(root, split, f"{split}_A")
        self.dir_C = os.path.join(root, split, f"{split}_C")
        
        # Ensure we only have string filenames in this list
        self.images = sorted([f for f in os.listdir(self.dir_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Force idx to be an integer to avoid returning a list of names
        if isinstance(idx, list):
            idx = idx[0]
            
        name = self.images[idx]
        path_A = os.path.join(self.dir_A, name)
        path_C = os.path.join(self.dir_C, name)

        # Load as numpy for your custom TrainTransform [cite: 224]
        img = np.array(Image.open(path_A).convert("RGB"))
        target = np.array(Image.open(path_C).convert("RGB"))

        if self.transform:
            img, target = self.transform(img, target)

        # Convert to tensors manually for better control 
        img = F.to_tensor(img)
        target = F.to_tensor(target)

        return img, target
