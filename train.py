import torch
from torch.utils.data import DataLoader

from datasets.istd import ISTDDataset
from datasets.transforms import TrainTransform, ValTransform # Added imports
from models.shadow_uformer import ShadowUformer
from engine.trainer import Trainer

DATA_ROOT = "data/ISTD"
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 200
LR = 2e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass the Transform objects instead of the IMAGE_SIZE integer
    train_transform = TrainTransform(crop_size=IMAGE_SIZE)
    val_transform = ValTransform()

    train_dataset = ISTDDataset(DATA_ROOT, "train", transform=train_transform)
    val_dataset = ISTDDataset(DATA_ROOT, "test", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True # Recommended for faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2
    )

    model = ShadowUformer().to(device)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        EPOCHS,
        LR,
        device
    )

    trainer.train()

if __name__ == "__main__":
    main()
