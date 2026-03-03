import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import calculate_psnr, calculate_ssim
from utils.checkpoint import save_checkpoint


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs,
        lr,
        device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )

        self.scaler = torch.amp.GradScaler("cuda")
        self.writer = SummaryWriter("runs")

        self.best_psnr = 0.0

    def train(self):

        for epoch in range(self.epochs):

            train_loss = self._train_one_epoch(epoch)
            val_psnr, val_ssim = self._validate(epoch)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1}: "
                f"TrainLoss={train_loss:.6f} "
                f"PSNR={val_psnr:.4f} "
                f"SSIM={val_ssim:.4f}"
            )

            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                save_checkpoint(
                    self.model,
                    "checkpoints/best_model.pth"
                )

    def _train_one_epoch(self, epoch):

        self.model.train()
        total_loss = 0

        bar = tqdm(self.train_loader)

        for img, target in bar:

            img = img.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = self.model(img)
                loss = self.criterion(pred, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Train/Loss", avg_loss, epoch)

        return avg_loss

    def _validate(self, epoch):

        self.model.eval()

        total_psnr = 0
        total_ssim = 0

        with torch.no_grad():
            for img, target in self.val_loader:

                img = img.to(self.device)
                target = target.to(self.device)

                pred = self.model(img)

                total_psnr += calculate_psnr(pred, target).item()
                total_ssim += calculate_ssim(pred, target).item()

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        self.writer.add_scalar("Val/PSNR", avg_psnr, epoch)
        self.writer.add_scalar("Val/SSIM", avg_ssim, epoch)

        return avg_psnr, avg_ssim
