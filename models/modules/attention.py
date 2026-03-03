import torch
import torch.nn as nn
from einops import rearrange


def window_partition(x, window_size):
    return rearrange(
        x,
        "b c (h ws1) (w ws2) -> (b h w) ws1 ws2 c",
        ws1=window_size,
        ws2=window_size
    )


def window_reverse(windows, window_size, H, W, B):
    return rearrange(
        windows,
        "(b h w) ws1 ws2 c -> b c (h ws1) (w ws2)",
        b=B,
        h=H // window_size,
        w=W // window_size,
        ws1=window_size,
        ws2=window_size
    )


class SwinWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, shift=False):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_bias = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1),
                num_heads
            )
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij"
        ))

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        self.register_buffer(
            "relative_index",
            relative_coords.sum(-1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        if self.shift:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(2, 3))

        x = x.permute(0, 2, 3, 1)
        windows = window_partition(x, ws)
        windows = windows.view(-1, ws * ws, C)

        qkv = self.qkv(windows).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(
                t.shape[0],
                t.shape[1],
                self.num_heads,
                self.head_dim
            ).transpose(1, 2),
            qkv
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_bias = self.relative_bias[
            self.relative_index.view(-1)
        ].view(ws * ws, ws * ws, -1).permute(2, 0, 1)

        attn = attn + relative_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(
            windows.shape[0],
            ws * ws,
            C
        )

        out = self.proj(out)

        out = out.view(-1, ws, ws, C)
        x = window_reverse(out, ws, H, W, B)

        if self.shift:
            x = torch.roll(x, shifts=(ws // 2, ws // 2), dims=(2, 3))

        return x
