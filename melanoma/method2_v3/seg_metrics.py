from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def mean_dice_soft(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Soft Dice (same as training log): sigmoid probs vs soft GT."""
    model.eval()
    parts: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = torch.sigmoid(model(x))
        inter = (p * y).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-6
        dice = (2 * inter / union).mean().item()
        parts.append(dice)
    return float(sum(parts) / max(1, len(parts)))


@torch.no_grad()
def mean_iou_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresh: float = 0.5,
) -> float:
    """IoU with binarized prediction (prob >= thresh) vs binary GT mask."""
    model.eval()
    ious: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = (torch.sigmoid(model(x)) >= thresh).float()
        for b in range(x.size(0)):
            pb = p[b, 0]
            yb = y[b, 0]
            inter = (pb * yb).sum()
            union = pb.sum() + yb.sum() - inter
            ious.append(float((inter / (union + 1e-6)).item()))
    return float(sum(ious) / max(1, len(ious)))
