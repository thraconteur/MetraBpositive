"""
Pixel-to-millimetre calibration.

THE CORE PROBLEM
----------------
A photograph carries no absolute scale. A 2 mm numeral shot close up and
a 6 mm numeral shot from far away can occupy exactly the same number of
pixels. So Rule 7(2) - the minimum numeral height in millimetres - is
simply not checkable from an uncalibrated photo, no matter how good your
OCR is. Any team claiming otherwise is guessing.

The fix is a reference object of known physical size in the same frame.
We use an ArUco marker printed at a known edge length. Detect it, measure
its side in pixels, and you have px/mm for that specific photograph.

WHY PER-PHOTO AND NOT PER-DEVICE
--------------------------------
Two separate things get confused here:

  * Camera calibration (lens intrinsics, distortion) is a property of
    the device. Done once per phone model, cached, invisible to the user.
  * SCALE is a property of the shot. It changes every time the user
    moves the camera. There is no way around putting a reference in the
    frame, or getting metric depth from AR.

So the UX is: shoot normally for the checks that need no scale, and
prompt for the reference card only when a millimetre measurement is
actually required. See docs in README.

PERSPECTIVE
-----------
If the marker plane is tilted relative to the sensor, the marker appears
foreshortened and px/mm is wrong. We estimate tilt from the marker's
corner geometry and either correct via homography or refuse. Refusing is
better than a silently wrong measurement that ends up on a legal report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core.schema import Calibration, CalibrationMethod

ARUCO_DICTS = {
    "ARUCO_4X4_50": cv2.aruco.DICT_4X4_50,
    "ARUCO_5X5_100": cv2.aruco.DICT_5X5_100,
    "ARUCO_6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray          # (4,2) float32, clockwise from top-left
    side_lengths_px: tuple[float, float, float, float]
    mean_side_px: float
    squareness: float            # 1.0 = perfect square; lower = tilted
    area_px: float


# ---------------------------------------------------------------------
# Marker generation (print this and keep it in the inspector's wallet)
# ---------------------------------------------------------------------

def generate_marker_card(
    out_path: str,
    marker_id: int = 0,
    marker_size_mm: float = 25.0,
    dpi: int = 300,
    dict_name: str = "ARUCO_4X4_50",
    quiet_zone_ratio: float = 0.25,
) -> str:
    """
    Render a printable calibration card.

    Print at 100% scale (no 'fit to page' - that silently rescales and
    destroys the whole point). Verify with a ruler after printing: if the
    marker measures 24 mm instead of 25 mm, every downstream measurement
    inherits a 4% error.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    px_per_mm = dpi / 25.4
    marker_px = int(round(marker_size_mm * px_per_mm))

    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)

    quiet = int(marker_px * quiet_zone_ratio)
    canvas_w = marker_px + 2 * quiet
    canvas_h = marker_px + 2 * quiet + int(px_per_mm * 12)  # label strip
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas[quiet:quiet + marker_px, quiet:quiet + marker_px] = marker

    label = f"LMPC CAL  id={marker_id}  {marker_size_mm:.1f}mm  PRINT AT 100%"
    cv2.putText(
        canvas, label,
        (quiet // 2, canvas_h - int(px_per_mm * 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        px_per_mm * 0.055, 0, max(1, int(px_per_mm * 0.06)), cv2.LINE_AA,
    )
    cv2.imwrite(out_path, canvas)
    return out_path


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

def detect_markers(
    image: np.ndarray,
    dict_name: str = "ARUCO_4X4_50",
) -> list[MarkerDetection]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement matters a lot here: a half-pixel error
    # on a 100 px marker is a 0.5% scale error, which at a 2 mm threshold
    # is 0.01 mm. Cheap insurance.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    # Most photos have no card in them, and a full search of a 4096 px
    # photo costs ~0.7 s (several seconds on a laptop CPU), three times per
    # scan. A search at half size says whether anything marker-like is
    # there at all: a marker usable for calibration (>= 40 px side) is
    # >= 28 px at 0.7x, above ArUco's own minimum for that frame. Only
    # when something is found does the exact full-size search run, so a
    # photo WITH a card gets exactly the result it got before.
    if max(gray.shape[:2]) > 2000:
        # 0.7x, not 0.5x: at half size a usable 40-45 px marker turned 45
        # degrees or printed at low contrast was missed (audit, 30 Sept).
        half = cv2.resize(gray, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        probe = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        if probe.detectMarkers(half)[1] is None:
            return []

    corners, ids, _ = detector.detectMarkers(gray)
    out: list[MarkerDetection] = []
    if ids is None:
        return out

    for c, i in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float64)
        sides = tuple(
            float(np.linalg.norm(pts[k] - pts[(k + 1) % 4])) for k in range(4)
        )
        mean_side = float(np.mean(sides))
        squareness = float(min(sides) / max(sides)) if max(sides) > 0 else 0.0
        area = float(cv2.contourArea(pts.astype(np.float32)))
        # A real calibration card in a usable photo is tens of pixels
        # across and square. A 10 px "marker" is a pattern that happens to
        # decode - the inside of a QR code on a real jam cup decoded as
        # ArUco id 17, the photo was "rectified" to its plane (a 1280x7932
        # smear) and OCR then read nothing at all.
        if mean_side < 30.0 or squareness < 0.75:
            continue
        out.append(
            MarkerDetection(
                marker_id=int(i),
                corners=pts,
                side_lengths_px=sides,
                mean_side_px=mean_side,
                squareness=squareness,
                area_px=area,
            )
        )
    return out


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def calibrate_from_image(
    image: np.ndarray,
    marker_size_mm: float = 25.0,
    dict_name: str = "ARUCO_4X4_50",
    min_squareness: float = 0.80,
    min_marker_px: float = 40.0,
) -> Calibration:
    """
    Derive px/mm from the largest well-conditioned marker in the frame.

    Refuses (returns an unavailable Calibration with a reason) when:
      * no marker is found;
      * the marker is too small in pixels to measure reliably;
      * the marker is too foreshortened, meaning the card is not
        coplanar with the label and the scale would not transfer.
    """
    dets = detect_markers(image, dict_name=dict_name)
    if not dets:
        return Calibration(
            method=CalibrationMethod.NONE,
            notes="No calibration marker detected in frame.",
        )

    det = max(dets, key=lambda d: d.area_px)

    if det.mean_side_px < min_marker_px:
        return Calibration(
            method=CalibrationMethod.NONE,
            notes=(
                f"Marker too small ({det.mean_side_px:.0f} px side). Move "
                f"closer or increase resolution."
            ),
        )

    if det.squareness < min_squareness:
        return Calibration(
            method=CalibrationMethod.NONE,
            notes=(
                f"Marker too foreshortened (squareness {det.squareness:.2f}). "
                f"Shoot square-on, with the card flat against the label face."
            ),
        )

    px_per_mm = det.mean_side_px / marker_size_mm
    # Spread across the four sides is a decent proxy for residual
    # perspective plus corner-localisation noise - plus the card itself:
    # printed with "fit to page" it is 94-97% of 25 mm, and nothing in the
    # photo can tell. 6% of the scale is carried for that (A4 "fit to page"
    # on Letter-size printers shrinks to ~94%).
    side_std = float(np.std(det.side_lengths_px))
    uncertainty = side_std / marker_size_mm + 0.06 * px_per_mm

    return Calibration(
        px_per_mm=px_per_mm,
        method=CalibrationMethod.FIDUCIAL,
        uncertainty_px_per_mm=uncertainty,
        marker_id=det.marker_id,
        marker_size_mm=marker_size_mm,
        squareness=det.squareness,
        notes=(
            f"Calibrated from marker {det.marker_id}: "
            f"{det.mean_side_px:.1f} px / {marker_size_mm} mm, "
            f"squareness {det.squareness:.3f}."
        ),
    )


def calibrate_from_known_dimension(
    measured_px: float,
    known_mm: float,
    assumed_error_mm: float = 1.0,
) -> Calibration:
    """
    Fallback: the user tells us one real dimension (e.g. bottle height).

    Much weaker than a fiducial - the user's own measurement error goes
    straight into the scale - so we inflate the uncertainty accordingly
    and let the abstain band absorb it.
    """
    if known_mm <= 0 or measured_px <= 0:
        return Calibration(method=CalibrationMethod.NONE, notes="Invalid dimension.")
    px_per_mm = measured_px / known_mm
    rel_err = assumed_error_mm / known_mm
    return Calibration(
        px_per_mm=px_per_mm,
        method=CalibrationMethod.USER_DIMENSION,
        uncertainty_px_per_mm=px_per_mm * rel_err,
        notes=(
            f"User-supplied dimension {known_mm} mm (assumed +/-"
            f"{assumed_error_mm} mm). Lower confidence than a fiducial."
        ),
    )


# ---------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------

def rectify_to_marker_plane(
    image: np.ndarray,
    marker_size_mm: float = 25.0,
    target_px_per_mm: float = 10.0,
    dict_name: str = "ARUCO_4X4_50",
) -> tuple[Optional[np.ndarray], Calibration]:
    """
    Warp the image so the marker plane is fronto-parallel and the scale
    is exactly `target_px_per_mm` everywhere.

    After this, measurement is trivial: every pixel is a known fraction
    of a millimetre, uniformly across the image, with no perspective
    gradient. Do this before glyph measurement whenever a marker exists.
    """
    dets = detect_markers(image, dict_name=dict_name)
    if not dets:
        return None, Calibration(
            method=CalibrationMethod.NONE, notes="No marker for rectification."
        )

    det = max(dets, key=lambda d: d.area_px)
    # The same limits as calibrate_from_image: a small or foreshortened
    # marker (card not flat on the label) gave a confident scale here while
    # the calibration refused it - the text came out stretched 1.28x.
    if det.mean_side_px < 40.0 or det.squareness < 0.80:
        return None, Calibration(
            method=CalibrationMethod.NONE,
            notes=(f"Marker not usable for rectification (side {det.mean_side_px:.0f} px, "
                   f"squareness {det.squareness:.2f}); lay the card flat and move closer."))
    side_px = marker_size_mm * target_px_per_mm

    # ArUco corner order is TL, TR, BR, BL.
    dst = np.array(
        [[0, 0], [side_px, 0], [side_px, side_px], [0, side_px]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(det.corners.astype(np.float32), dst)

    h, w = image.shape[:2]
    corners_img = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img, H).reshape(-1, 2)

    xmin, ymin = warped_corners.min(axis=0)
    xmax, ymax = warped_corners.max(axis=0)

    # Keep the output bounded - a grazing view can warp to something huge.
    out_w, out_h = int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))
    # ...and a rectification that stretches the photo far beyond its own
    # size is a bad fit, not a better view.
    if (out_w <= 0 or out_h <= 0 or out_w > 12000 or out_h > 12000
            or out_w * out_h > 9 * w * h):
        return None, Calibration(
            method=CalibrationMethod.NONE,
            notes="Rectified extent out of bounds; view too oblique.",
        )

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    warped = cv2.warpPerspective(image, T @ H, (out_w, out_h), flags=cv2.INTER_CUBIC)

    calib = Calibration(
        px_per_mm=target_px_per_mm,
        method=CalibrationMethod.FIDUCIAL,
        # 1% fit + 6% for a card printed off-size ("fit to page")
        uncertainty_px_per_mm=target_px_per_mm * 0.07,
        marker_id=det.marker_id,
        marker_size_mm=marker_size_mm,
        squareness=det.squareness,
        notes=(
            f"Rectified to marker plane at {target_px_per_mm} px/mm "
            f"(marker {det.marker_id})."
        ),
    )
    return warped, calib
