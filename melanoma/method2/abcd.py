from __future__ import annotations

import numpy as np
from skimage import measure
from skimage.color import rgb2hsv

FEATURE_NAMES: list[str] = [
    "area_norm",
    "circularity",
    "border_irregularity",
    "solidity",
    "eccentricity",
    "axis_ratio",
    "asym_h",
    "asym_v",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "std_h",
    "std_s",
    "std_v",
]


def _mirror_asymmetry(patch: np.ndarray) -> tuple[float, float]:
    """patch: binary float/uint8 H,W. Returns asym_h, asym_v in [0,1] (higher = more asymmetric)."""
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0, 0.0
    p = (patch > 0).astype(np.float32)
    inter = float(np.logical_and(p, np.fliplr(p)).sum())
    union = float(np.logical_or(p, np.fliplr(p)).sum()) + 1e-6
    asym_h = 1.0 - inter / union
    inter2 = float(np.logical_and(p, np.flipud(p)).sum())
    union2 = float(np.logical_or(p, np.flipud(p)).sum()) + 1e-6
    asym_v = 1.0 - inter2 / union2
    return asym_h, asym_v


def extract_abcd(image_uint8_hwc: np.ndarray, mask_bool_hw: np.ndarray) -> np.ndarray:
    """image: H,W,3 uint8. mask: H,W bool. Returns fixed-length float vector."""
    h, w = mask_bool_hw.shape
    if image_uint8_hwc.shape[:2] != (h, w):
        raise ValueError("image and mask spatial shape must match")
    m = mask_bool_hw.astype(bool)
    area = int(m.sum())
    if area < 10:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    im = image_uint8_hwc.astype(np.float32) / 255.0
    hsv = rgb2hsv(im)

    labeled = measure.label(m.astype(np.uint8))
    props = measure.regionprops(labeled)
    if not props:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    prop = max(props, key=lambda p: p.area)
    area_f = float(prop.area)
    perim = float(prop.perimeter) if prop.perimeter > 0 else 1e-6
    circularity = float(4.0 * np.pi * area_f / (perim**2))
    circularity = float(np.clip(circularity, 0.0, 1.0))
    border_irreg = float(1.0 - circularity)
    solidity = float(prop.solidity) if prop.solidity is not None else 0.0
    ecc = float(prop.eccentricity) if prop.eccentricity is not None else 0.0
    maj = float(prop.major_axis_length) if prop.major_axis_length else 1.0
    min_ax = float(prop.minor_axis_length) if prop.minor_axis_length else 1.0
    axis_ratio = float(min_ax / (maj + 1e-6))

    rmin, cmin, rmax, cmax = prop.bbox
    pad = 2
    r0, r1 = max(0, rmin - pad), min(h, rmax + pad)
    c0, c1 = max(0, cmin - pad), min(w, cmax + pad)
    patch = m[r0:r1, c0:c1]
    asym_h, asym_v = _mirror_asymmetry(patch)

    rr, cc = np.where(m)
    pix = im[rr, cc]
    mean_rgb = pix.mean(axis=0)
    std_rgb = pix.std(axis=0)
    pix_h = hsv[rr, cc]
    mean_hsv = pix_h.mean(axis=0)
    std_hsv = pix_h.std(axis=0)

    area_norm = area_f / float(h * w + 1e-6)

    return np.array(
        [
            area_norm,
            circularity,
            border_irreg,
            solidity,
            ecc,
            axis_ratio,
            asym_h,
            asym_v,
            mean_rgb[0],
            mean_rgb[1],
            mean_rgb[2],
            std_rgb[0],
            std_rgb[1],
            std_rgb[2],
            mean_hsv[0],
            mean_hsv[1],
            mean_hsv[2],
            std_hsv[0],
            std_hsv[1],
            std_hsv[2],
        ],
        dtype=np.float32,
    )
