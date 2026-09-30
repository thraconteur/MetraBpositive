"""
Principal display panel detection.

WHY THIS MODULE EXISTS
----------------------
Rule 7(4) defines the PDP area as a property of the PHYSICAL PANEL:
for a rectangular package, the height multiplied by the width of the
display face. It is not the region covered by ink.

The first version of this pipeline approximated the PDP as the bounding
box of all detected text. That is wrong, and wrong in a dangerous
direction: printed matter occupies only part of a panel, so the area
comes out too small, which selects a LOWER band in Table-I, which
applies a SMALLER minimum height, which lets genuinely undersized text
pass as compliant. On a 90x130mm panel (117 cm2) the text-extent
estimate returned about 46 cm2 - dropping from the 2.5mm band to the
1.0mm band.

Since the 2017 amendment keyed Table-I to panel area rather than net
quantity, EVERY font-size decision depends on this number. It is the
most load-bearing measurement in the system.

APPROACH
--------
The panel is a bright, roughly rectangular region with a detectable
border against the surrounding surface. We:

  1. mask out the fiducial marker so it can never be mistaken for the
     panel (it is a strong black square and otherwise wins contour
     ranking outright);
  2. find candidate quadrilaterals via edge detection and contour
     approximation;
  3. score them by area, rectangularity and how much of the detected
     text they contain - the true panel should contain nearly all the
     declarations;
  4. fall back to a text-extent estimate ONLY if nothing plausible is
     found, and mark the result low-confidence so the rules engine can
     abstain rather than assert a threshold it cannot justify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core.schema import BBox, Calibration


@dataclass
class PanelDetection:
    bbox: BBox
    quad: Optional[np.ndarray]          # (4,2) corners when available
    method: str                         # "contour" | "text_extent" | "frame"
    confidence: float
    text_coverage: float = 0.0          # fraction of text spans inside

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.5


def _mask_marker(gray: np.ndarray, marker_corners: Optional[np.ndarray]) -> np.ndarray:
    """
    Paint over the fiducial so contour search cannot latch onto it.

    The marker is a high-contrast black square with clean corners, i.e.
    exactly what a quadrilateral detector likes best. Left in place it
    frequently outranks the real panel.
    """
    if marker_corners is None:
        return gray
    out = gray.copy()
    pts = marker_corners.astype(np.int32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    pad = int(0.35 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(out.shape[1], x + w + pad), min(out.shape[0], y + h + pad)
    # Fill with the median of the surrounding region so the patch does
    # not itself create a new strong edge.
    out[y0:y1, x0:x1] = int(np.median(out))
    return out


def _candidate_quads(gray: np.ndarray, min_area_frac: float = 0.06) -> list[np.ndarray]:
    h, w = gray.shape[:2]
    img_area = float(h * w)
    quads: list[np.ndarray] = []

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Two complementary views: intensity thresholding catches a light
    # panel on a darker surround, edges catch a printed border.
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    for binary in (th, edges):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area * min_area_frac:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float64))
            else:
                # Fall back to the minimum-area rectangle: panels with
                # rounded corners or a partially occluded edge will not
                # approximate to exactly four points.
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                if cv2.contourArea(box.astype(np.float32)) >= img_area * min_area_frac:
                    quads.append(box.astype(np.float64))
    return quads


def _quad_bbox(quad: np.ndarray) -> BBox:
    x, y = quad[:, 0].min(), quad[:, 1].min()
    return BBox(float(x), float(y), float(quad[:, 0].max() - x), float(quad[:, 1].max() - y))


def _rectangularity(quad: np.ndarray) -> float:
    """1.0 for a perfect axis-aligned rectangle; lower for skewed shapes."""
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    bbox = _quad_bbox(quad)
    return area / bbox.area if bbox.area > 0 else 0.0


def detect_panel(
    image: np.ndarray,
    text_boxes: Optional[list[BBox]] = None,
    marker_corners: Optional[np.ndarray] = None,
) -> PanelDetection:
    """
    Locate the principal display panel.

    `text_boxes` is used for scoring, not for defining the panel: the
    true panel should CONTAIN the declarations, which is a strong signal,
    but its extent must come from the physical boundary.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]
    img_area = float(h * w)
    text_boxes = text_boxes or []

    masked = _mask_marker(gray, marker_corners)
    quads = _candidate_quads(masked)

    best, best_score, best_cov = None, -1.0, 0.0
    for q in quads:
        bbox = _quad_bbox(q)
        area_frac = bbox.area / img_area
        if area_frac > 0.98:
            continue  # the whole frame, not a panel
        rect = _rectangularity(q)
        if rect < 0.75:
            continue

        if text_boxes:
            inside = sum(1 for t in text_boxes if bbox.contains(t, tol=6.0))
            coverage = inside / len(text_boxes)
        else:
            coverage = 0.0

        # Text coverage dominates: a panel that does not contain the
        # declarations is not the principal display panel, however
        # cleanly rectangular it is. Area breaks ties toward the outer
        # boundary rather than an inner printed frame.
        score = coverage * 2.0 + rect * 0.5 + area_frac * 0.5
        if score > best_score:
            best, best_score, best_cov = q, score, coverage

    if best is not None and best_cov >= 0.6:
        return PanelDetection(
            bbox=_quad_bbox(best),
            quad=best,
            method="contour",
            confidence=min(1.0, 0.5 + best_cov / 2.0),
            text_coverage=best_cov,
        )

    # ---- fallback -------------------------------------------------
    # No trustworthy boundary. Return the text extent but say so, and
    # keep the confidence below the reliability threshold so the rules
    # engine abstains on the height check instead of applying a band
    # derived from an area we do not actually believe.
    if text_boxes:
        x = min(b.x for b in text_boxes)
        y = min(b.y for b in text_boxes)
        x2 = max(b.x2 for b in text_boxes)
        y2 = max(b.y2 for b in text_boxes)
        return PanelDetection(
            bbox=BBox(x, y, x2 - x, y2 - y),
            quad=None,
            method="text_extent",
            confidence=0.3,
            text_coverage=1.0,
        )

    return PanelDetection(
        bbox=BBox(0, 0, float(w), float(h)),
        quad=None,
        method="frame",
        confidence=0.1,
    )


def panel_area_cm2(
    detection: PanelDetection,
    calibration: Calibration,
    geometry: str = "rectangular",
) -> Optional[float]:
    """
    Convert a detected panel to an area in cm^2 per Rule 7(4).

    Rectangular packages use height x width of the display face.
    Cylindrical packages are 40% of height x circumference - and a single
    view cannot recover circumference, so we return None and let the
    caller ask the user for a diameter rather than silently substituting
    the flat-face area, which would understate the panel and (again)
    select too small a threshold.
    """
    if not calibration.available:
        return None

    # The quad's own side lengths, not its upright bounding box: a panel
    # turned 3 degrees read 100.1 cm2 instead of 89.8 and crossed a Table-I
    # band (false font-size violation).
    if detection.quad is not None and len(detection.quad) == 4:
        q = np.asarray(detection.quad, dtype=np.float64)
        d = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
        w_px, h_px = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
    else:
        w_px, h_px = detection.bbox.w, detection.bbox.h
    w_mm = calibration.px_to_mm(w_px)
    h_mm = calibration.px_to_mm(h_px)
    if not w_mm or not h_mm:
        return None

    if geometry == "rectangular":
        return (w_mm / 10.0) * (h_mm / 10.0)
    if geometry == "cylindrical":
        return None  # needs circumference; caller must prompt
    # "Any other shape": 40% of total surface. Without a 3D model we
    # cannot compute total surface, so treat the visible face as a lower
    # bound and flag it rather than guess.
    return None
