from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agpd_feature_adapter import (  # noqa: E402
    collect_activation_gradients,
    cosine_similarity_rows,
    one_sided_mad,
    progressive_filter,
    upstream_rho,
)


class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(3, 2, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv(values))
        return self.head(self.pool(hidden).flatten(1))


def test_cosine_similarity_and_rho_are_finite() -> None:
    values = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    similarity = cosine_similarity_rows(values, np.asarray([1.0, 0.0]))
    assert similarity[0] == pytest.approx(1.0)
    assert similarity[-1] == pytest.approx(-1.0)
    assert upstream_rho(values, np.asarray([1.0, 0.0]), [2]) > 0.0


def test_one_sided_mad_only_marks_upper_outlier() -> None:
    scores = one_sided_mad([0.1, 0.11, 0.09, 0.1, 1.0])
    assert np.all(scores[:4] < scores[4])
    assert scores[4] > np.exp(2.0)


def test_progressive_filter_never_drops_initial_outlier() -> None:
    clean = np.asarray([1.0, 0.0])
    values = np.vstack(
        [np.tile(clean, (20, 1)), np.asarray([[-1.0, 0.0], [-0.9, 0.1]])]
    )
    flagged, iterations, repairs = progressive_filter(values, clean, tau=0.5)
    assert 20 in flagged
    assert iterations >= 1
    assert isinstance(repairs, tuple)


def test_collect_activation_gradients_matches_batch_length() -> None:
    torch.manual_seed(4)
    model = TinyConv()
    images = torch.rand(7, 1, 4, 4)
    labels = torch.as_tensor([0, 1, 0, 1, 0, 1, 0])
    gradients = collect_activation_gradients(
        model, images, labels, "conv", batch_size=3, device="cpu"
    )
    assert gradients.shape == (7, 3)
    assert np.isfinite(gradients).all()


def test_collect_activation_gradients_rejects_unknown_layer() -> None:
    model = TinyConv()
    with pytest.raises(KeyError):
        collect_activation_gradients(
            model,
            torch.rand(2, 1, 4, 4),
            torch.zeros(2, dtype=torch.long),
            "missing",
        )
