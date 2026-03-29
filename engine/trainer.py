import torch
import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from utils.metrics import calculate_psnr, calculate_ssim
from utils.checkpoint import save_checkpoint, load_checkpoint
from losses.joint_loss import JointLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, warmup_epochs, lr, device, save_dir, resume_path=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = save_dir
        self.best_psnr = 0.0
        self.start_epoch = 0

        self.criterion = JointLoss(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        
        self.scaler = torch.amp.GradScaler("cuda")

        if resume_path:
            self.start_epoch = load_checkpoint(resume_path, self.model, self.optimizer, self.scheduler, self.scaler)
            print(f"Resuming from epoch {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_one_epoch(epoch)
            val_psnr, val_ssim = self._validate()

            self.scheduler.step()

            print(f"Epoch {epoch}: Loss={train_loss:.6f} PSNR={val_psnr:.4f} SSIM={val_ssim:.4f}")

            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_psnr': self.best_psnr
                }
                save_checkpoint(state, os.path.join(self.save_dir, "best_model.pth"))

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for img, mask, target in bar:
            img, mask, target = img.to(self.device), mask.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                pred = self.model(img)
                loss = self.criterion(pred, target, mask)

            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        psnr_val, ssim_val = 0.0, 0.0
        with torch.no_grad():
            for img, _, target in self.val_loader:
                img, target = img.to(self.device), target.to(self.device)
                pred = self.model(img)
                
                psnr_val += calculate_psnr(pred, target).item()
                ssim_val += calculate_ssim(pred, target).item()

        return psnr_val / len(self.val_loader), ssim_val / len(self.val_loader)
