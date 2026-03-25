import torch
import torchvision.transforms.functional as F
import random
import numpy as np

class TrainTransform:
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, img, mask, target):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            target = np.fliplr(target).copy()

        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
            target = np.flipud(target).copy()

        h, w, _ = img.shape
        new_h, new_w = self.crop_size, self.crop_size
        
        if h < new_h or w < new_w:
            img = F.to_pil_image(img).resize((new_w, new_h))
            mask = F.to_pil_image(mask).resize((new_w, new_h))
            target = F.to_pil_image(target).resize((new_w, new_h))
            img, mask, target = np.array(img), np.array(mask), np.array(target)
            top, left = 0, 0
        else:
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]
        target = target[top: top + new_h, left: left + new_w]

        img_t = F.to_tensor(img)       
        target_t = F.to_tensor(target)
        
        mask_t = F.to_tensor(mask)
        mask_t = (mask_t > 0.5).float() 

        return img_t, mask_t, target_t

class ValTransform:
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, img, mask, target):
        h, w, _ = img.shape
        new_h, new_w = self.crop_size, self.crop_size
        
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        img = img[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]
        target = target[top: top + new_h, left: left + new_w]

        img_t = F.to_tensor(img)
        target_t = F.to_tensor(target)
        mask_t = (F.to_tensor(mask) > 0.5).float()

        return img_t, mask_t, target_t
