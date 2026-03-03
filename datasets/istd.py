import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ISTDDataset(Dataset):

    def __init__(self, root, split="train", image_size=256):
        super().__init__()

        assert split in ["train", "test"]

        self.root = root
        self.split = split
        self.image_size = image_size

        self.dir_A = os.path.join(root, split, f"{split}_A")
        self.dir_C = os.path.join(root, split, f"{split}_C")

        self.images = sorted(os.listdir(self.dir_A))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        path_A = os.path.join(self.dir_A, name)
        path_C = os.path.join(self.dir_C, name)

        img = Image.open(path_A).convert("RGB")
        target = Image.open(path_C).convert("RGB")

        img = self.transform(img)
        target = self.transform(target)

        return img, target
