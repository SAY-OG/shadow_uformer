import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import calculate_psnr, calculate_ssim
from utils.checkpoint import save_checkpoint, load_checkpoint
from losses.charbonnier import CharbonnierLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, lr, device, resume_path=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.best_psnr = 0.0
        self.start_epoch = 0

        # Charbonnier Loss handles edges better than L1
        self.criterion = CharbonnierLoss(eps=1e-3)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Fixed: Attribute 'scaler' initialized here
        self.scaler = torch.amp.GradScaler("cuda")
        self.writer = SummaryWriter("runs")

        if resume_path:
            self.start_epoch = load_checkpoint(
                resume_path, self.model, self.optimizer, self.scheduler, self.scaler
            )
            print(f"Resuming from epoch {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_one_epoch(epoch)
            val_psnr, val_ssim = self._validate(epoch)

            self.scheduler.step()

            print(f"Epoch {epoch+1}: Loss={train_loss:.6f} PSNR={val_psnr:.4f}")

            # State dictionary for resuming
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'best_psnr': self.best_psnr
            }
            
            save_checkpoint(state, "/content/drive/MyDrive/models/latest_model.pth")
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                save_checkpoint(state, "/content/drive/MyDrive/models/best_model.pth")

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        bar = tqdm(self.train_loader)
        for img, target in bar:
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = self.model(img)
                loss = self.criterion(pred, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        psnr, ssim = 0, 0
        with torch.no_grad():
            for img, target in self.val_loader:
                img, target = img.to(self.device), target.to(self.device)
                pred = self.model(img)
                psnr += calculate_psnr(pred, target).item()
                ssim += calculate_ssim(pred, target).item()
        return psnr / len(self.val_loader), ssim / len(self.val_loader)
