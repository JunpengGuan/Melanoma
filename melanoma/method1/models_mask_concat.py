import torch
import torch.nn as nn
from torchvision import models


def build_classifier_mask_concat(backbone, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    backbone = backbone.lower().strip()

    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)

        old_conv = model.features[0][0]

        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        model.features[0][0] = new_conv

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

        return model

    raise ValueError("Only efficientnet_b0 is supported for image + mask right now.")