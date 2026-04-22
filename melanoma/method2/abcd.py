from __future__ import annotations

import numpy as np
from skimage import measure
from skimage.color import rgb2hsv, rgb2lab
from skimage import morphology
from skimage.transform import rotate

FEATURE_NAMES: list[str] = [
    "A_major",
    "A_minor",
    "B_circularity",
    "B_solidity",
    "B_radial_cv",
    "B_perimeter_area_ratio",
    "C_H_std",
    "C_H_circular_std",
    "C_S_std",
    "C_V_std",
    "C_H_entropy",
    "C_S_entropy",
    "C_V_entropy",
    "C_L_std",
    "C_a_std",
    "C_b_std",
    "C_V_contrast_p95_p05",
    "C_L_contrast_p95_p05",
    "D_major_axis",
    "D_minor_axis",
    "D_equivalent_diameter",
    "D_area",
    "D_major_axis_ratio",
    "D_minor_axis_ratio",
    "D_equivalent_diameter_ratio",
    "D_area_ratio",
]

def _zeros():
    return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

def _clean_mask(mask_bool_hw):
    m = mask_bool_hw.astype(bool)
    if not m.any():
        return m

    labeled = measure.label(m.astype(np.uint8), connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return np.zeros_like(m, dtype=bool)
    keep = max(props, key=lambda p: p.area).label
    m = labeled == keep

    area = int(m.sum())
    hole_threshold = max(64, int(area * 0.01))
    min_object = max(32, int(area * 0.02))
    try:
        m = morphology.remove_small_holes(m, max_size=hole_threshold)
    except TypeError:
        m = morphology.remove_small_holes(m, area_threshold=hole_threshold)
    m = morphology.closing(m, morphology.disk(3))
    m = morphology.opening(m, morphology.disk(1))
    try:
        m = morphology.remove_small_objects(m, max_size=min_object)
    except TypeError:
        m = morphology.remove_small_objects(m, min_size=min_object)

    labeled = measure.label(m.astype(np.uint8), connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return np.zeros_like(m, dtype=bool)
    keep = max(props, key=lambda p: p.area).label
    return labeled == keep


def _principal_axis_angle_deg(mask_bool_hw):
    rr, cc = np.where(mask_bool_hw)
    coords = np.column_stack([cc.astype(np.float32), rr.astype(np.float32)])
    centroid = coords.mean(axis=0, keepdims=True)
    centered = coords - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    return float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))


def _crop_centered_square(mask_bool_hw, side):
    rr, cc = np.where(mask_bool_hw)
    center_r = float(rr.mean())
    center_c = float(cc.mean())

    half = side // 2
    r0 = int(round(center_r)) - half
    c0 = int(round(center_c)) - half
    r1 = r0 + side
    c1 = c0 + side

    out = np.zeros((side, side), dtype=bool)
    src_r0 = max(0, r0)
    src_c0 = max(0, c0)
    src_r1 = min(mask_bool_hw.shape[0], r1)
    src_c1 = min(mask_bool_hw.shape[1], c1)
    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = mask_bool_hw[src_r0:src_r1, src_c0:src_c1]
    return out


def _iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 1.0
    return inter / union

def _axis_asymmetry(mask_bool_hw):
    if mask_bool_hw.sum() < 10:
        return 0.0, 0.0

    prop = measure.regionprops(mask_bool_hw.astype(np.uint8))[0]
    height = prop.bbox[2] - prop.bbox[0]
    width = prop.bbox[3] - prop.bbox[1]
    side = int(np.ceil(np.hypot(height, width))) + 12
    side = max(side, 16)

    patch = _crop_centered_square(mask_bool_hw, side=side)
    angle_deg = _principal_axis_angle_deg(mask_bool_hw)
    rotated = rotate(
        patch.astype(np.float32),
        angle=-angle_deg,
        resize=False,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    ) > 0.5

    h, w = rotated.shape
    half_h = min(h // 2, h - h // 2)
    half_w = min(w // 2, w - w // 2)

    top = rotated[h // 2 - half_h:h // 2, :]
    bottom = rotated[h // 2:h // 2 + half_h, :]
    left = rotated[:, w // 2 - half_w:w // 2]
    right = rotated[:, w // 2:w // 2 + half_w]

    major_iou = _iou(top, np.flipud(bottom))
    minor_iou = _iou(left, np.fliplr(right))
    return float(1.0 - major_iou), float(1.0 - minor_iou)


def _entropy(values, bins):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist.astype(np.float64) / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _hue_circular_std(h_vals):
    if h_vals.size == 0:
        return 0.0
    angles = 2.0 * np.pi * h_vals.astype(np.float64)
    sin_mean = float(np.sin(angles).mean())
    cos_mean = float(np.cos(angles).mean())
    resultant = np.sqrt(sin_mean**2 + cos_mean**2)
    resultant = float(np.clip(resultant, 1e-8, 1.0))
    return float(np.sqrt(-2.0 * np.log(resultant)) / (2.0 * np.pi))


def _p95_p05(values):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def _region_axis_length(prop, new_name: str, old_name: str):
    value = getattr(prop, new_name, None)
    if value is None:
        value = getattr(prop, old_name, 0.0)
    return float(value or 0.0)


def extract_abcd(image_uint8_hwc: np.ndarray, mask_bool_hw: np.ndarray) -> np.ndarray:
    if image_uint8_hwc.ndim != 3 or image_uint8_hwc.shape[2] < 3:
        raise ValueError("image must be HxWx3")

    h, w = mask_bool_hw.shape
    if image_uint8_hwc.shape[:2] != (h, w):
        raise ValueError("image and mask spatial shape must match")

    m = _clean_mask(mask_bool_hw)
    area = int(m.sum())
    if area < 10:
        return _zeros()

    rgb = image_uint8_hwc[..., :3].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    im = rgb
    hsv = rgb2hsv(im)
    lab = rgb2lab(im)

    labeled = measure.label(m.astype(np.uint8), connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return _zeros()
    prop = max(props, key=lambda region: region.area)

    area_f = float(prop.area)
    perim = float(measure.perimeter(m, neighborhood=8))
    perim = max(perim, 1e-6)
    circularity = float(np.clip(4.0 * np.pi * area_f / (perim**2), 0.0, 1.0))
    b_circularity = float(1.0 - circularity)
    solidity = float(prop.solidity) if prop.solidity is not None else 0.0
    b_solidity = float(1.0 - np.clip(solidity, 0.0, 1.0))

    contours = measure.find_contours(m.astype(np.float32), level=0.5)
    if contours:
        contour = max(contours, key=len)
        centroid_rc = np.array(prop.centroid, dtype=np.float32)
        radial = np.linalg.norm(contour - centroid_rc[None, :], axis=1)
        radial_mean = float(radial.mean()) if radial.size else 0.0
        radial_std = float(radial.std()) if radial.size else 0.0
        b_radial_cv = radial_std / max(radial_mean, 1e-6)
    else:
        b_radial_cv = 0.0

    a_major, a_minor = _axis_asymmetry(m)
    b_perimeter_area_ratio = float(perim / np.sqrt(area_f + 1e-6))

    rr, cc = np.where(m)
    lesion_hsv = hsv[rr, cc]
    h_vals = lesion_hsv[:, 0]
    s_vals = lesion_hsv[:, 1]
    v_vals = lesion_hsv[:, 2]

    c_h_std = float(h_vals.std())
    c_h_circular_std = _hue_circular_std(h_vals)
    c_s_std = float(s_vals.std())
    c_v_std = float(v_vals.std())
    c_h_entropy = _entropy(h_vals)
    c_s_entropy = _entropy(s_vals)
    c_v_entropy = _entropy(v_vals)
    lesion_lab = lab[rr, cc]
    l_vals = lesion_lab[:, 0]
    a_vals = lesion_lab[:, 1]
    b_vals = lesion_lab[:, 2]
    c_l_std = float(l_vals.std())
    c_a_std = float(a_vals.std())
    c_b_std = float(b_vals.std())
    c_v_contrast = _p95_p05(v_vals)
    c_l_contrast = _p95_p05(l_vals)

    d_major_axis = _region_axis_length(prop, "axis_major_length", "major_axis_length")
    d_minor_axis = _region_axis_length(prop, "axis_minor_length", "minor_axis_length")
    d_equivalent_diameter = float(np.sqrt(4.0 * area_f / np.pi))
    d_area = area_f
    image_area = float(h * w)
    image_span = float(max(h, w))
    d_major_axis_ratio = d_major_axis / max(image_span, 1e-6)
    d_minor_axis_ratio = d_minor_axis / max(image_span, 1e-6)
    d_equivalent_diameter_ratio = d_equivalent_diameter / max(image_span, 1e-6)
    d_area_ratio = d_area / max(image_area, 1e-6)

    return np.array(
        [
            a_major,
            a_minor,
            b_circularity,
            b_solidity,
            float(b_radial_cv),
            b_perimeter_area_ratio,
            c_h_std,
            c_h_circular_std,
            c_s_std,
            c_v_std,
            c_h_entropy,
            c_s_entropy,
            c_v_entropy,
            c_l_std,
            c_a_std,
            c_b_std,
            c_v_contrast,
            c_l_contrast,
            d_major_axis,
            d_minor_axis,
            d_equivalent_diameter,
            d_area,
            d_major_axis_ratio,
            d_minor_axis_ratio,
            d_equivalent_diameter_ratio,
            d_area_ratio,
        ],
        dtype=np.float32,
    )
