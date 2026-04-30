from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """target: float in {0,1}, same spatial size as logits."""
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + (1.0 - self.bce_weight) * dice_loss_with_logits(
            logits, target
        )
