from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import functional as TF

from melanoma.config import SEG_IMG_SIZE


def load_rgb_hwc_uint8(image_path: Path, size: int = SEG_IMG_SIZE) -> np.ndarray:
    img = read_image(str(image_path))
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img[:3, :, :]
    img = TF.resize(img, [size, size], antialias=True)
    return img.permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def image_to_model_input(path: Path, size: int = SEG_IMG_SIZE) -> torch.Tensor:
    """Normalized CHW float tensor, batch dim included."""
    img = read_image(str(path))
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img[:3, :, :].float() / 255.0
    img = TF.resize(img, [size, size], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0)


@torch.no_grad()
def predict_mask_bool(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    size: int = SEG_IMG_SIZE,
    thresh: float = 0.5,
) -> np.ndarray:
    model.eval()
    x = image_to_model_input(image_path, size).to(device)
    logit = model(x)
    prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    return prob >= thresh
