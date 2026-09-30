"""
Image quality gating and preprocessing.

TWO SEPARATE JOBS
-----------------
1. The QUALITY GATE decides whether the image is good enough to make any
   claim at all. This is not a nicety - the PS asks for readability
   analysis, and "this photo is too blurry to judge" is a legitimate and
   useful output. A system that always produces a verdict is a system
   that sometimes produces a confident wrong one.

2. PREPROCESSING improves what we can. Each stage is optional and
   measurable: run your evaluation harness with stages toggled off to
   produce the "CER before/after each stage" table for your pitch. If a
   stage does not move the number, cut it - it is costing you latency
   and demo risk for nothing.

MODEL-BASED STAGES
------------------
Super-resolution and learned deblurring are left as adapter hooks rather
than baked in. Do not train these. Wire a pretrained checkpoint (e.g.
Real-ESRGAN for SR) behind `SuperResolutionAdapter` when you have one,
and keep the classical fallback so the demo never hard-fails offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Protocol

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------

@dataclass
class QualityReport:
    ok: bool
    blur_score: float             # variance of Laplacian; higher = sharper
    mean_luminance: float
    glare_fraction: float         # fraction of near-saturated pixels
    contrast: float
    est_min_glyph_px: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def specular_glare_fraction(gray: np.ndarray, sat_threshold: int = 245) -> float:
    """
    Fraction of the frame lost to SPECULAR GLARE, as distinct from a
    legitimately light background.

    This distinction matters more than it sounds. A naive
    `(gray > 245).mean()` counts the white paper of a clean label as
    glare and rejects a perfectly readable image - we hit exactly that
    bug on synthetic labels, where the white margin alone was 44% of the
    frame and the gate refused every single one.

    The physical difference: a paper or background region is one large
    connected area that reaches the frame edge, whereas a specular
    highlight is an interior blob sitting on top of the subject. So we
    drop saturated components that touch the border and count only what
    is left.
    """
    sat = (gray > sat_threshold).astype(np.uint8)
    if sat.sum() == 0:
        return 0.0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    h, w = gray.shape[:2]
    interior_area = 0

    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        touches_border = (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)
        if touches_border:
            continue  # background, not glare
        interior_area += int(stats[i, cv2.CC_STAT_AREA])

    return interior_area / float(gray.size)


def assess_quality(
    image: np.ndarray,
    min_blur_var: float = 100.0,
    max_glare_fraction: float = 0.12,
    min_contrast: float = 25.0,
) -> QualityReport:
    """
    Cheap, deterministic gate. Runs in milliseconds, before anything else.

    Thresholds here are defaults, not law. Tune them on YOUR collected
    photos and record the tuned values, because they directly control
    your abstention rate - which you should be reporting alongside
    precision and recall, since a gate this aggressive can flatter every
    other number in the table.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(gray.mean())
    contrast = float(gray.std())
    glare = specular_glare_fraction(gray)

    problems = []
    if blur < min_blur_var:
        problems.append(f"image is soft or out of focus (sharpness {blur:.0f})")
    if glare > max_glare_fraction:
        problems.append(f"specular glare on {glare*100:.0f}% of the subject")
    if contrast < min_contrast:
        problems.append(f"low contrast ({contrast:.0f})")

    # Exposure is judged by DETAIL LOSS, not by brightness. A white label
    # shot well is bright and perfectly readable; the failure case is
    # bright AND flat, where the print has been washed out.
    if mean_lum < 40 and contrast < 40:
        problems.append("underexposed - detail lost in shadow")
    elif mean_lum > 235 and contrast < 30:
        problems.append("overexposed - print washed out")

    return QualityReport(
        ok=not problems,
        blur_score=blur,
        mean_luminance=mean_lum,
        glare_fraction=glare,
        contrast=contrast,
        notes="; ".join(problems),
    )


# ---------------------------------------------------------------------
# Classical preprocessing stages
# ---------------------------------------------------------------------

def remove_specular_highlights(
    image: np.ndarray,
    sat_threshold: int = 250,
    inpaint_radius: int = 3,
    max_glare_fraction: float = 0.25,
    min_blob_fraction: float = 0.0005,
    texture_edge_threshold: float = 0.02,
) -> np.ndarray:
    """
    Remove specular blowout from foil pouches and glossy laminates.

    THE TRAP THIS FUNCTION EXISTS TO AVOID
    --------------------------------------
    The obvious implementation - mask every bright, low-saturation pixel
    and inpaint it - destroys ordinary labels. A white paper carton has
    value ~255 and saturation ~0 across its whole background, so the
    naive mask selects the entire label and inpainting smears the black
    print into the white paper. OCR then returns confident garbage,
    which is the worst possible failure mode because nothing looks wrong
    until you read the output.

    So brightness alone cannot identify glare. Two further signals do:

      1. GLARE IS LOCAL. A blown-out highlight covers part of a frame.
         If the "glare" mask covers most of the image, it is not glare,
         it is a light-coloured background - bail out entirely.

      2. GLARE IS TEXTURELESS. Inside a real specular highlight the
         sensor has clipped, so there are no edges. White paper bearing
         printed text is full of edges. Measuring edge density inside
         each candidate blob separates the two cleanly.

    Only blobs that are bright, bounded in size, and texture-free get
    inpainted. Everything else is left alone, because an honest glare
    patch reported as unreadable beats invented texture where print used
    to be.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    candidate = ((v >= sat_threshold) & (s < 40)).astype(np.uint8)

    total_px = candidate.size
    if candidate.sum() == 0:
        return image

    # Signal 1: if "glare" dominates the frame it is just a bright
    # background. Refusing here is what keeps white cartons readable.
    if candidate.sum() / total_px > max_glare_fraction:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    mask = np.zeros_like(candidate)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area / total_px < min_blob_fraction:
            continue  # speckle, not a highlight

        blob = labels == i
        # Signal 2: texture inside the blob. Printed text -> many edges.
        if edges[blob].mean() > texture_edge_threshold:
            continue

        mask[blob] = 255

    if mask.sum() == 0:
        return image

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def enhance_local_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel. Helps low-contrast print without blowing colour."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray, strength: int = 5) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def unsharp(image: np.ndarray, amount: float = 1.0, sigma: float = 1.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    Classical fallback for super-resolution.

    Note honestly in your write-up that Lanczos does NOT recover
    information - it only helps the OCR engine's own downsampling
    behave. Real gains need a learned SR model behind the adapter below.
    """
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4
    )


# ---------------------------------------------------------------------
# Adapter for learned models
# ---------------------------------------------------------------------

class SuperResolutionAdapter(Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray: ...


class ClassicalSR:
    """Default adapter. Swap for Real-ESRGAN or similar when available."""

    name = "lanczos-2x"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return upscale(image, 2.0)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Every stage toggleable, so the ablation table writes itself."""
    deglare: bool = True
    contrast: bool = True
    denoise: bool = False        # slow; off by default, measure before enabling
    sharpen: bool = True
    superres: bool = False
    sr_adapter: Optional[SuperResolutionAdapter] = None


def preprocess(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
) -> tuple[np.ndarray, list[str]]:
    """Returns the processed image and the list of stages that ran."""
    cfg = config or PreprocessConfig()
    out = image.copy()
    applied: list[str] = []

    if cfg.deglare:
        out = remove_specular_highlights(out)
        applied.append("deglare")
    if cfg.contrast:
        out = enhance_local_contrast(out)
        applied.append("clahe")
    if cfg.denoise:
        out = denoise(out)
        applied.append("denoise")
    if cfg.superres:
        adapter = cfg.sr_adapter or ClassicalSR()
        out = adapter(out)
        applied.append(getattr(adapter, "name", "superres"))
    if cfg.sharpen:
        out = unsharp(out)
        applied.append("unsharp")

    return out, applied


# ---------------------------------------------------------------------
# Document dewarping
# ---------------------------------------------------------------------

def detect_panel_quad(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the dominant quadrilateral - usually the label or the flat face
    of the package. Used to rectify perspective when no fiducial marker
    is present.

    Classical contour approach. It fails on cluttered shelves and on
    curved surfaces; when it returns None the caller should fall back to
    the unrectified image rather than guess.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = image.shape[0] * image.shape[1]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.15:
            break
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order corners TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def dewarp_to_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Four-point homography onto a fronto-parallel rectangle."""
    q = order_quad(quad)
    (tl, tr, br, bl) = q
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w <= 0 or h <= 0:
        return image
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_CUBIC)


# =====================================================================
# Super-resolution adapters
# =====================================================================
# Small print is the main real-world OCR failure: a phone photo of a
# 1.5mm consumer-care line often has too few pixels per glyph to
# recognise. Upscaling before OCR genuinely helps.
#
# Both adapters are OPTIONAL and load LAZILY. An earlier version
# imported torch, basicsr and realesrgan at module top and constructed
# the GAN inside PipelineConfig's default - so a machine without those
# packages could not import the pipeline AT ALL. Not "super-resolution
# unavailable": every scan, every rule, every test dead on an ImportError
# from an optional enhancement. Heavy optional dependencies must never
# sit on the critical import path.


class ClassicalSR:
    """Lanczos upscale. No dependencies, always available."""

    name = "classical_sr"

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(
            image, (int(w * self.scale), int(h * self.scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )


class ESRGANAdapter:
    """
    Real-ESRGAN upscaler. Markedly better than Lanczos on small print,
    at the cost of torch plus a ~64MB weights file.

    Everything is deferred to first use, and any failure - missing
    package, missing weights, no memory - degrades to ClassicalSR rather
    than taking down the scan. Reconstructing text that was never legible
    is a real risk with a generative upscaler, so treat its output as an
    aid to recognition, not as evidence: measurement still runs on the
    original pixels.
    """

    name = "real_esrgan"

    def __init__(self, weights: str = "weights/RealESRGAN_x4plus.pth",
                 outscale: float = 2.0):
        self.weights = weights
        self.outscale = outscale
        self._upsampler = None
        self._failed = False
        self._fallback = ClassicalSR(scale=outscale)

    def available(self) -> bool:
        import importlib.util
        from pathlib import Path

        if self._failed:
            return False
        for mod in ("torch", "basicsr", "realesrgan"):
            if importlib.util.find_spec(mod) is None:
                return False
        return Path(self.weights).exists()

    def _load(self):
        if self._upsampler is not None or self._failed:
            return
        try:
            # basicsr still imports torchvision.transforms.functional_tensor,
            # removed in newer torchvision. Alias it before importing, and
            # only inside this function - patching sys.modules as a side
            # effect of importing our own module would surprise anyone else
            # in the process.
            import sys
            import torchvision.transforms.functional as tv_f

            sys.modules.setdefault(
                "torchvision.transforms.functional_tensor", tv_f
            )
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            self._upsampler = RealESRGANer(
                scale=4, model_path=self.weights, model=model, tile=0
            )
        except Exception:
            logger.warning(
                "Real-ESRGAN unavailable, falling back to Lanczos upscaling",
                exc_info=True,
            )
            self._failed = True

    def __call__(self, image: np.ndarray) -> np.ndarray:
        self._load()
        if self._upsampler is None:
            return self._fallback(image)
        try:
            out, _ = self._upsampler.enhance(image, outscale=self.outscale)
            return out
        except Exception:
            logger.warning("Real-ESRGAN inference failed", exc_info=True)
            self._failed = True
            return self._fallback(image)
