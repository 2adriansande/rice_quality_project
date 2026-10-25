"""Grain-mask and chalk-oriented features (used by 01_data_preparation.py and check_grain_mask.py).

These features are measured INSIDE the grain only, so the background cannot dilute them. They aim at the
whole-versus-chalk confusion: chalky grains tend to be opaque white (often in the centre or belly), while
whole grains are more translucent.
IMPORTANT: the thresholds below (0.8 brightness, 0.25 saturation, ...) are starting guesses. Run
check_grain_mask.py on your real images and look at the masks before trusting these features.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure

CHALK_KEYS = ["chalk_share_bright", "chalk_share_pale", "chalk_v_mean", "chalk_v_std", "chalk_v_p90",
              "chalk_sat_mean", "chalk_largest_patch", "chalk_core_minus_edge"]


def grain_mask(gray):
    """Boolean mask of the largest object (the grain), found with Otsu thresholding.
    Assumes the grain is the smaller of the two regions. Returns None if nothing is found."""
    mask = gray > filters.threshold_otsu(gray)
    if mask.mean() > 0.5:
        mask = ~mask
    lab = measure.label(mask)
    if lab.max() == 0:
        return None
    biggest = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    grain = ndi.binary_fill_holes(lab == biggest)
    return grain if grain.sum() >= 30 else None


def chalk_features(hsv, gray) -> dict:
    """Eight numbers describing how chalky the grain itself looks (all 0.0 if no grain is found)."""
    f = dict.fromkeys(CHALK_KEYS, 0.0)
    try:
        grain = grain_mask(gray)
        if grain is None:
            return f
        area = float(grain.sum())
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        bright = (v > 0.8) & (s < 0.25) & grain                 # opaque white pixels inside the grain
        f["chalk_share_bright"] = bright.sum() / area            # share of the grain that is chalky white
        f["chalk_share_pale"] = ((s < 0.15) & grain).sum() / area  # share with almost no colour
        f["chalk_v_mean"] = float(v[grain].mean())
        f["chalk_v_std"] = float(v[grain].std())
        f["chalk_v_p90"] = float(np.percentile(v[grain], 90))
        f["chalk_sat_mean"] = float(s[grain].mean())
        lab = measure.label(bright)                              # biggest single white patch, as share of grain
        if lab.max() > 0:
            f["chalk_largest_patch"] = float(np.bincount(lab.ravel())[1:].max() / area)
        steps = max(1, int(round(np.sqrt(area) * 0.15)))         # core vs edge ring (chalk is often central)
        core = ndi.binary_erosion(grain, iterations=steps)
        edge = grain & ~core
        if core.sum() > 0 and edge.sum() > 0:
            f["chalk_core_minus_edge"] = float(v[core].mean() - v[edge].mean())
    except Exception:
        pass
    return f
