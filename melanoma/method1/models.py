from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_classifier(backbone: str, pretrained: bool = True) -> nn.Module:
"""
Binary melanoma probability at logits (use BCEWithLogitsLoss).
backbone: efficientnet_b0 | vit_b_16
"""
    b = backbone.lower().strip()
    weights = "DEFAULT" if pretrained else None

    if b == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, 1)
        return m

    if b == "vit_b_16":
        m = models.vit_b_16(weights=weights)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, 1)
        return m

    raise ValueError(f"Unknown backbone '{backbone}'. Use efficientnet_b0 or vit_b_16.")
