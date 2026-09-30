"""
Sticker / overlay detection - Rule 6(3).

WHAT THE RULE ACTUALLY SAYS
---------------------------
Rule 6(3) prohibits ALTERING a mandatory declaration by affixing a
sticker. But it carries a proviso that is easy to miss and expensive to
get wrong: a sticker bearing a REVISED LOWER retail sale price IS
permitted, provided it does not cover the original MRP printed by the
manufacturer or packer. Rule 6(4) further allows stickers freely for any
NON-mandatory declaration.

So "a sticker exists" is not a violation. A naive detector that flags
every pasted label will fire on a large share of legitimate retail stock
and be switched off by the first inspector who uses it.

The three things that ARE violations:

  1. the sticker CONCEALS the printed MRP;
  2. the sticker price is HIGHER than the printed price (that is not a
     revision downward, it is overcharging);
  3. the sticker covers some other mandatory declaration.

VISUAL SIGNALS
--------------
A pasted sticker differs from printed packaging in ways a camera can
see: its paper stock has a different luminance from the panel it sits
on, it has straight cut edges producing a rectangular luminance
discontinuity, and it usually casts a faint shadow along one edge.

We look for large, rectangular, roughly uniform regions whose background
luminance differs from the panel's dominant background, then reason
about what they overlap. Detection alone is never enough - the verdict
comes from the geometry and from the price comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core.schema import BBox


@dataclass
class Overlay:
    """A candidate pasted region."""
    bbox: BBox
    mean_luminance: float
    surround_luminance: float
    luminance_delta: float
    rectangularity: float
    area_frac: float
    confidence: float


def _dominant_background(gray: np.ndarray) -> float:
    """
    Modal luminance of the panel, ignoring ink.

    Uses the histogram peak rather than the mean: text pulls a mean
    downward, and on a densely printed label that shift is large enough
    to make the paper itself look like an anomaly.
    """
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    return float(np.argmax(hist) * 4 + 2)


def _boundary_step(gray: np.ndarray, x: int, y: int, w: int, h: int,
                   band: int = 6) -> float:
    """
    Mean luminance step across the region's border.

    A sticker's cut edge and the shadow beneath it produce a step from
    just-inside to just-outside the boundary. This survives the common
    case that defeats fill-difference detection: a near-white sticker on
    near-white packaging, where the two faces differ by only a few grey
    levels but the seam between them is sharp.
    """
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 * band or y1 - y0 < 2 * band:
        return 0.0

    inner = gray[y0 + band:y1 - band, x0 + band:x1 - band]
    ox0, oy0 = max(0, x0 - band), max(0, y0 - band)
    ox1, oy1 = min(W, x1 + band), min(H, y1 + band)
    outer = gray[oy0:oy1, ox0:ox1].astype(np.float32).copy()
    outer[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = np.nan
    vals = outer[~np.isnan(outer)]
    if vals.size == 0 or inner.size == 0:
        return 0.0
    return abs(float(np.median(inner)) - float(np.median(vals)))


def detect_overlays(
    image: np.ndarray,
    panel_bbox: Optional[BBox] = None,
    min_area_frac: float = 0.03,
    max_area_frac: float = 0.55,
    min_step: float = 2.5,
    marker_corners: Optional[np.ndarray] = None,
) -> list[Overlay]:
    """
    Find candidate pasted regions inside the panel.

    Edge-first, not fill-first. The initial version thresholded on the
    difference between a region's fill and the panel's dominant
    background; on real stock that difference is tiny (measured 242 vs
    238 - four grey levels) and nothing was ever detected. What actually
    distinguishes a sticker is its geometry: a closed rectangular seam
    with a luminance step across it.

    Deliberately tuned to under-report. A missed sticker is one
    undetected violation; a false sticker on every second package is a
    tool nobody uses.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    if marker_corners is not None:
        pts = marker_corners.astype(np.int32).reshape(-1, 1, 2)
        mx, my, mw, mh = cv2.boundingRect(pts)
        pad = int(0.3 * max(mw, mh))
        gray[max(0, my - pad):my + mh + pad, max(0, mx - pad):mx + mw + pad] = int(
            np.median(gray)
        )

    if panel_bbox is not None:
        x0 = max(0, int(panel_bbox.x))
        y0 = max(0, int(panel_bbox.y))
        x1 = min(gray.shape[1], int(panel_bbox.x2))
        y1 = min(gray.shape[0], int(panel_bbox.y2))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return []
        roi = gray[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        roi = gray

    roi_area = float(roi.shape[0] * roi.shape[1])

    # Denoise hard before looking for the seam. A sticker edge is
    # large-scale structure, so aggressive smoothing helps rather than
    # hurts - whereas sensor noise on a real photograph completely
    # swamps a step of only a few grey levels. Without this the detector
    # works on clean renders and finds nothing at all on noisy ones,
    # which is the worst possible place to discover the limitation.
    # TWO VIEWS, unioned. Neither alone is sufficient, and each fails on
    # exactly the images the other handles:
    #
    #   raw     - keeps the sticker's thin cut outline, which is the
    #             strongest cue on a clean image, but sensor noise buries
    #             the fill step (measured 6.0 grey levels clean, 3.0
    #             noisy).
    #   median  - lifts the noisy step back to 4.0, but a 5px median
    #             erases a 1px outline entirely, so clean images go
    #             undetected.
    #
    # Tuning one path alone flipped the detector between "works clean,
    # blind on noise" and the exact reverse. A bilateral filter on top
    # of the median crushed the step to 1.0 and blinded it on both.
    denoised = cv2.medianBlur(roi, 5)
    views = (roi, denoised)
    k = max(5, int(min(roi.shape[:2]) * 0.012) | 1)
    contours = []
    for view in views:
        e = cv2.Canny(cv2.GaussianBlur(view, (5, 5), 0), 10, 40)
        e = cv2.morphologyEx(
            e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        )
        cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)

    out: list[Overlay] = []
    seen: list[tuple] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area / roi_area < min_area_frac or area / roi_area > max_area_frac:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rect = area / float(w * h) if w * h else 0.0
        if rect < 0.70:
            continue
        if w < 20 or h < 20:
            continue

        step = max(_boundary_step(roi, x, y, w, h),
                   _boundary_step(denoised, x, y, w, h))
        if step < min_step:
            continue

        # Collapse near-duplicate contours from the inner and outer side
        # of the same seam.
        if any(abs(x - sx) < 12 and abs(y - sy) < 12 and abs(w - sw) < 20
               for sx, sy, sw, _ in seen):
            continue
        seen.append((x, y, w, h))

        inner = denoised[y:y + h, x:x + w]

        # A barcode is a rectangle with a luminance step, which is
        # exactly the signature this detector looks for - and every
        # label has one, so without this guard every package reports a
        # pasted sticker. Bar patterns are distinguishable by their
        # density of vertical edges: a sticker face is comparatively
        # uniform, a barcode is nothing but alternating edges.
        vert = cv2.Sobel(inner, cv2.CV_32F, 1, 0, ksize=3)
        vert_density = float((np.abs(vert) > 40).mean())
        if vert_density > 0.18:
            continue

        out.append(
            Overlay(
                bbox=BBox(float(x0 + x), float(y0 + y), float(w), float(h)),
                mean_luminance=float(inner.mean()),
                surround_luminance=float(inner.mean()) - step,
                luminance_delta=step,
                rectangularity=rect,
                area_frac=area / roi_area,
                confidence=min(1.0, (step / 25.0) * 0.6 + rect * 0.4),
            )
        )

    return sorted(out, key=lambda o: -o.confidence)


def assess_price_overlay(
    overlays: list[Overlay],
    mrp_bbox: Optional[BBox],
    printed_price: Optional[float] = None,
    sticker_price: Optional[float] = None,
    conceal_iou: float = 0.25,
    price_value_boxes: Optional[list] = None,
    price_label_boxes: Optional[list] = None,
) -> tuple[bool, str, Optional[Overlay]]:
    """
    Decide whether an overlay near the price is actually unlawful.

    Returns (is_violation, reason, overlay). The proviso is honoured:
    a lower revised price on a sticker that leaves the printed MRP
    visible is compliant and must not be flagged.
    """
    if not overlays:
        return False, "No pasted region detected over the price declaration.", None

    # Concealment is checked FIRST. If the sticker covers the printed
    # MRP then the printed price is, by definition, unreadable - so any
    # "printed price" we think we have came from some other line and
    # comparing against it yields a right answer for a wrong reason.
    if mrp_bbox is not None:
        for ov in overlays:
            if ov.bbox.iou(mrp_bbox) >= conceal_iou or ov.bbox.contains(mrp_bbox, tol=2.0):
                # The price was READ inside the panel: the panel carries the
                # price, it does not hide it. On real packs this is the white
                # inkjet coding box ("Rs. 40.00 / Rs. 1.00/g / batch / dates"
                # on a Troovy pouch), which a rectangle detector cannot tell
                # from a pasted label.
                # ...but only when the MRP LABEL is printed OUTSIDE the panel,
                # pointing into it. A self-contained "MRP Rs. 150" rectangle
                # over the price is what a shop pastes over the printed MRP -
                # the commonest real sticker violation - and is judged as one.
                labels_in = [b for b in (price_label_boxes or [])
                             if ov.bbox.contains(b, tol=4.0) or ov.bbox.iou(b) > 0.3]
                labels_out = [b for b in (price_label_boxes or []) if b not in labels_in]
                if (price_value_boxes and labels_out and not labels_in and any(
                        ov.bbox.contains(b, tol=4.0) for b in price_value_boxes)):
                    return (
                        False,
                        "No sticker hides the printed price: the rectangular panel "
                        "around it carries the price itself (a coding box or price "
                        "label), and the price is readable.",
                        None,
                    )
                return (
                    True,
                    "A pasted label conceals the retail sale price printed by the "
                    "manufacturer or packer.",
                    ov,
                )

    if (
        printed_price is not None
        and sticker_price is not None
        and sticker_price > printed_price
    ):
        return (
            True,
            f"Sticker price Rs. {sticker_price:.2f} exceeds the printed retail "
            f"sale price Rs. {printed_price:.2f}. A sticker may only revise the "
            f"price downward.",
            overlays[0],
        )

    if mrp_bbox is None:
        return False, "No price declaration located; overlay not assessed.", None

    if printed_price is not None and sticker_price is not None:
        return (
            False,
            f"Sticker revises the price downward to Rs. {sticker_price:.2f} "
            f"without concealing the printed price - permitted under the "
            f"proviso to Rule 6(3).",
            overlays[0],
        )

    # A candidate region that conceals nothing is usually printed design -
    # the red price panel on a real Kesar Chandan jar was reported as a
    # "pasted region". Say what was checked, not that a sticker exists.
    return False, "No sticker concealing or raising the printed price was found.", None


def find_top_token(
    spans,
    token: str,
    panel_bbox: Optional[BBox],
    top_fraction: float = 0.25,
) -> tuple[bool, bool]:
    """
    Rule 6(7): genetically modified food must bear 'GM' AT THE TOP of the
    principal display panel.

    Returns (present_anywhere, present_at_top) so the engine can tell the
    two failures apart - a package with no mark at all and one with the
    mark in the wrong place are different findings, and an inspector
    needs to know which.
    """
    # Whole-word match inside the span, not equality with it. PaddleOCR
    # returns LINES, so a "GM" printed close to other text arrives as
    # "GM  NET WT 500 g" and an equality test never sees it. The word
    # boundary keeps "GMP" or "GMT" from counting.
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", re.I)
    hits = [s for s in spans if pat.search(s.text.strip())]
    if not hits:
        return False, False
    if panel_bbox is None:
        return True, False
    cutoff = panel_bbox.y + panel_bbox.h * top_fraction
    return True, any(s.bbox.cy <= cutoff for s in hits)
