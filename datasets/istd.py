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
        self.dir_B = os.path.join(root, split, f"{split}_B")
        self.dir_C = os.path.join(root, split, f"{split}_C")
        
        self.images = sorted([f for f in os.listdir(self.dir_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = np.array(Image.open(os.path.join(self.dir_A, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.dir_B, name)).convert("L"))
        target = np.array(Image.open(os.path.join(self.dir_C, name)).convert("RGB"))

        if self.transform:
            return self.transform(img, mask, target)

        return F.to_tensor(img), F.to_tensor(mask), F.to_tensor(target)


