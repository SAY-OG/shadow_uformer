import random
import numpy as np


class TrainTransform:
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, shadow, gt):
        H, W, _ = shadow.shape

        # Random crop
        if H > self.crop_size and W > self.crop_size:
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)

            shadow = shadow[top:top+self.crop_size, left:left+self.crop_size]
            gt = gt[top:top+self.crop_size, left:left+self.crop_size]

        # Random flip
        if random.random() > 0.5:
            shadow = np.fliplr(shadow)
            gt = np.fliplr(gt)

        if random.random() > 0.5:
            shadow = np.flipud(shadow)
            gt = np.flipud(gt)

        return shadow.copy(), gt.copy()


class ValTransform:
    def __call__(self, shadow, gt):
        return shadow, gt
