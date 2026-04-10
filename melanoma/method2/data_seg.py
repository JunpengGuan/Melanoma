from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as TF

from melanoma.config import SEG_IMG_SIZE


def mask_path_for_id(mask_dir: Path, image_id: str) -> Path:
    return Path(mask_dir) / f"{image_id}_Segmentation.png"


def load_binary_mask(path: Path, out_hw: tuple[int, int]) -> torch.Tensor:
    m = read_image(str(path)).float()
    m = m.max(dim=0, keepdim=True).values  # 1,H,W
    m = m / 255.0
    m = (m > 0.5).float()
    m = TF.resize(m, list(out_hw), interpolation=TF.InterpolationMode.NEAREST)
    return m


class LesionSegDataset(Dataset):
    """RGB image + binary lesion mask (0/1 float). rows: list of (image_id, _) — label ignored."""

    def __init__(
        self,
        rows: list[tuple[str, int]],
        image_dir: Path,
        mask_dir: Path,
        image_size: int = SEG_IMG_SIZE,
    ) -> None:
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self._hw = (image_size, image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, _ = self.rows[i]
        img = read_image(str(self.image_dir / f"{image_id}.jpg"))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = img[:3, :, :]
        img = TF.resize(img, list(self._hw), antialias=True).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        mpath = mask_path_for_id(self.mask_dir, image_id)
        mask = load_binary_mask(mpath, self._hw)
        return img, mask


def filter_rows_with_masks(
    rows: list[tuple[str, int]],
    image_dir: Path,
    mask_dir: Path,
) -> list[tuple[str, int]]:
    image_dir = Path(image_dir)
    out: list[tuple[str, int]] = []
    for image_id, y in rows:
        if not (image_dir / f"{image_id}.jpg").is_file():
            continue
        if not mask_path_for_id(mask_dir, image_id).is_file():
            continue
        out.append((image_id, y))
    return out
