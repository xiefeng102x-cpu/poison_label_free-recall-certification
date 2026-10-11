"""Official-style 28x28 ResNet-50 for the PathMNIST blind extension.

This matches the small-image ResNet implementation used by the official
MedMNIST v2 benchmark: a 3x3 stride-one stem, no max pool, bottleneck depths
[3, 4, 6, 3], adaptive average pooling, and a nine-class ``linear`` head.
The module keeps the upstream names so the official ``resnet50_28_*.pth``
checkpoints load strictly without key remapping.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


N_CLASSES = 9
HIDDEN_DIMENSION = 2048
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(
            channels, self.expansion * channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * channels)
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn1(self.conv1(values)), inplace=False)
        hidden = F.relu(self.bn2(self.conv2(hidden)), inplace=False)
        hidden = self.bn3(self.conv3(hidden))
        return F.relu(hidden + self.shortcut(values), inplace=False)


class PathMNISTResNet50(nn.Module):
    """CIFAR-style ResNet-50 with an inspectable 2048-D representation."""

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
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(HIDDEN_DIMENSION, n_classes)

    def _make_layer(self, channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        strides = (stride,) + (1,) * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(Bottleneck(self.in_channels, channels, block_stride))
            self.in_channels = channels * Bottleneck.expansion
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


# The audited training engine constructs this historical symbol.  Keeping the
# alias lets the extension reuse its checkpoint/resume contract unchanged.
PathMNISTResNet18 = PathMNISTResNet50


def make_model(seed: int, n_classes: int = N_CLASSES) -> PathMNISTResNet50:
    torch.manual_seed(int(seed))
    return PathMNISTResNet50(n_classes=n_classes)
