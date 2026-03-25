import torch
import argparse
import os
from torch.utils.data import DataLoader
from datasets.istd import ISTDDataset
from datasets.transforms import TrainTransform, ValTransform
from models.shadow_uformer import ShadowUformer
from engine.trainer import Trainer

def get_warmup_lr_scheduler(optimizer, warmup_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    parser = argparse.ArgumentParser(description="ShadowUformer Training")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Stabilize attention at start')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=256)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ISTDDataset(args.data_root, "train", transform=TrainTransform(crop_size=args.img_size))
    val_dataset = ISTDDataset(args.data_root, "test", transform=ValTransform(crop_size=args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = ShadowUformer().to(device)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        resume_path=args.resume
    )
    if args.warmup_epochs > 0 and args.resume is None:
        print(f"Starting Warmup for {args.warmup_epochs} epochs...")
        warmup_scheduler = get_warmup_lr_scheduler(trainer.optimizer, args.warmup_epochs, args.lr)
        for epoch in range(args.warmup_epochs):
            trainer._train_one_epoch(epoch)
            warmup_scheduler.step()
        print("Warmup complete. Entering main training phase.")

    trainer.train()

if __name__ == "__main__":
    main()
