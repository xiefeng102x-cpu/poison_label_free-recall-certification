#!/usr/bin/env python3
"""CIFAR-native VGG16-BN used by the prospective blind certificate block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_CFG_D: tuple[int | str, ...] = (
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    512, 512, 512, "M", 512, 512, 512, "M",
)


class CIFAR10VGG16BN(nn.Module):
    """VGG16-BN with a CIFAR-size stem and a 512-D penultimate feature."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for item in _CFG_D:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            width = int(item)
            layers.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)
        self.register_buffer(
            "input_mean",
            torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_with_features(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = (values - self.input_mean) / self.input_std
        features = self.features(values)
        penultimate = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(penultimate), penultimate

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(values)[0]


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in CIFAR10VGG16BN().parameters())

