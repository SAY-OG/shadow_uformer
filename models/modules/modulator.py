import torch
import torch.nn as nn


class MultiScaleModulator(nn.Module):
    """
    Lightweight learnable feature modulator.
    """

    def __init__(self, dim):
        super(MultiScaleModulator, self).__init__()

        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta
