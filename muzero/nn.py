# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/09_nn.ipynb.

# %% auto 0
__all__ = ['ShortcutProjection', 'ResidualBlock', 'BottleneckResidualBlock', 'RepresentationNetwork', 'PredictionNetwork',
           'DynamicsNetwork']

# %% ../nbs/09_nn.ipynb 3
from typing import List, Optional
import torch
from torch import nn

# %% ../nbs/09_nn.ipynb 4
class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

# %% ../nbs/09_nn.ipynb 5
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride
            )
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.act2(out + residual)


# %% ../nbs/09_nn.ipynb 6
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=bottleneck_channels,
            kernel_size=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_channels, out_channels=bottleneck_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_channels, out_channels=out_channels,
            kernel_size=1, stride=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride
            )
        else:
            self.shortcut = nn.Identity()

        self.act3 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return self.act3(out + residual)

# %% ../nbs/09_nn.ipynb 8
class RepresentationNetwork(nn.Module):
    def __init__(
        self, img_channels: int, n_blocks: List[int], n_channels: List[int],
        first_kernel_size: int, bottlenecks: Optional[List[int]] = None
    ):
        super().__init__()
        assert len(n_blocks) == len(n_channels)

        self.conv = nn.Conv2d(
            in_channels=img_channels, out_channels=n_channels[0],
            kernel_size=first_kernel_size, stride=1, padding=first_kernel_size//2
        )

        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []
        prev_channels = n_channels[0]

        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1

            if bottlenecks is None:
                blocks.append(ResidualBlock(
                    in_channels=prev_channels, out_channels=channels,
                    stride=stride
                ))
            else:
                blocks.append(BottleneckResidualBlock(
                    in_channels=prev_channels, bottleneck_channels=bottlenecks[i],
                    out_channels=channels, stride=stride
                ))

            prev_channels = channels

            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    
                    blocks.append(ResidualBlock(
                        in_channels=channels, out_channels=channels,
                        stride=1
                    ))
                else:
                    blocks.append(BottleneckResidualBlock(
                        in_channels=channels, bottleneck_channels=bottlenecks[i],
                        out_channels=channels, stride=1
                    ))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        x = self.blocks(x)
        x = x.view(x.shape[0], x.shape[1], -1)

        return x.mean(dim=-1)

# %% ../nbs/09_nn.ipynb 10
class PredictionNetwork(nn.Module):
    pass

# %% ../nbs/09_nn.ipynb 12
class DynamicsNetwork(nn.Module):
    pass
