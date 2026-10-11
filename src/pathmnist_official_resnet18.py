"""Official-style 28x28 ResNet-18 used by the PathMNIST victim pilot.

Architecture source:
https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py

The network uses the small-image/CIFAR stem required by the frozen protocol:
a 3x3 stride-one convolution, no max-pool, BasicBlock depths [2, 2, 2, 2],
64/128/256/512 channels, adaptive average pooling, and a nine-class head.
Input normalization is part of the model and is fixed to mean/std 0.5 per
channel, matching ``Normalize((.5, .5, .5), (.5, .5, .5))``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


N_CLASSES = 9
HIDDEN_DIMENSION = 512
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn1(self.conv1(values)), inplace=False)
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + self.shortcut(values), inplace=False)


class PathMNISTResNet18(nn.Module):
    """CIFAR-style ResNet-18 with an inspectable 512-D penultimate layer."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be at least two")
        self.register_buffer(
            "channel_mean",
            torch.tensor(NORMALIZATION_MEAN).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "channel_std",
            torch.tensor(NORMALIZATION_STD).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Preserve the upstream module name so official MedMNIST checkpoints
        # load with ``strict=True`` and require no key remapping.
        self.linear = nn.Linear(HIDDEN_DIMENSION, n_classes)

    def _make_layer(self, channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        strides = (stride,) + (1,) * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_channels, channels, block_stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def hidden(self, values: torch.Tensor) -> torch.Tensor:
        values = (values - self.channel_mean) / self.channel_std
        values = F.relu(self.bn1(self.conv1(values)), inplace=False)
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        values = self.layer4(values)
        return self.pool(values).flatten(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(self.hidden(values))


def make_model(seed: int, n_classes: int = N_CLASSES) -> PathMNISTResNet18:
    torch.manual_seed(int(seed))
    return PathMNISTResNet18(n_classes=n_classes)
