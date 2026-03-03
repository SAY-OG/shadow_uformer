import torch
import torch.nn as nn


class LeFF(nn.Module):
    """
    Local Enhanced Feed Forward Network
    Linear → Depthwise Conv → GELU → Linear
    """

    def __init__(self, dim, hidden_dim=None):
        super(LeFF, self).__init__()

        hidden_dim = hidden_dim or dim * 4

        self.linear1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim
        )
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
