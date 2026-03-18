import torch
import argparse
import os
from torch.utils.data import DataLoader
from datasets.istd import ISTDDataset
from datasets.transforms import TrainTransform, ValTransform
from models.shadow_uformer import ShadowUformer
from engine.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="ShadowUformer Training Script")
    
    # --- 1. Dataset & Path Arguments ---
    parser.add_argument('--data_root', type=str, required=True, help='Path to ISTD dataset root')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Where to save models')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # --- 2. Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()

    # Create the save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. Initialize Data Loaders ---
    train_dataset = ISTDDataset(
        args.data_root, "train", 
        transform=TrainTransform(crop_size=args.img_size)
    )
    val_dataset = ISTDDataset(
        args.data_root, "test", 
        transform=ValTransform(crop_size=args.img_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, 
        shuffle=False, num_workers=args.num_workers
    )

    # --- 4. Initialize Model ---
    model = ShadowUformer().to(device)

    # --- 5. Initialize Trainer ---
    # Note: We pass args.save_dir here so the trainer knows where to save pth files
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        resume_path=args.resume,
        save_dir=args.save_dir 
    )

    trainer.train()

if __name__ == "__main__":
    main()
