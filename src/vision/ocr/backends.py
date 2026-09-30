"""
OCR backends.

PaddleOCR is the only real engine (see paddle.py). Tesseract has been
removed entirely: it read Indian packaging far worse than PP-OCR, and
keeping it as a silent fallback meant a machine without Paddle produced
reports that LOOKED normal but were read by the weakest engine. There is
now no silent fallback of any kind - if Paddle is not installed,
`build_default_ocr()` raises and says so.

Also here:
  * SyntheticOCR - test-only. Returns the exact text lines the synthetic
    label renderer drew, as line-level spans shaped like Paddle's output.
    Lets the whole pipeline downstream of OCR be tested deterministically
    with no model loaded.
  * ReplayOCR    - replays a saved `.paddle.json` dump (defined in paddle.py).
  * StubOCR      - geometry-only, for developing without any text.
  * VLMFallback  - hook for a vision-language model on hard crops.

Choosing the engine: `SIH_OCR_BACKEND` = paddle (default) | synthetic | stub.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ...core.schema import BBox, TextSpan
from .base import OCRBackend
from .paddle import PaddleConfig, PaddleOCRBackend, ReplayOCR  # noqa: F401

logger = logging.getLogger(__name__)


# =====================================================================
# StubOCR
# =====================================================================

class StubOCR(OCRBackend):
    """
    Geometry without transcription: MSER-style text regions with text
    supplied from a caller-provided ground-truth list, or empty.
    Not an OCR engine and does not pretend to be.
    """

    name = "stub"
    supports_languages = ("en",)

    def __init__(self, ground_truth: Optional[list[tuple[str, BBox]]] = None):
        self.ground_truth = ground_truth or []

    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        if self.ground_truth:
            return [
                TextSpan(text=t, bbox=b, confidence=0.99, source_engine=self.name)
                for t, b in self.ground_truth
            ]
        return [
            TextSpan(text="", bbox=b, confidence=0.5, source_engine=self.name)
            for b in detect_text_regions(image)
        ]


def detect_text_regions(image: np.ndarray, min_area: int = 60) -> list[BBox]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grad = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
    )
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area or h < 6:
            continue
        boxes.append(BBox(float(x), float(y), float(w), float(h)))
    boxes.sort(key=lambda b: (b.y, b.x))
    return boxes


# =====================================================================
# SyntheticOCR - test oracle shaped like PaddleOCR output
# =====================================================================

SYNTH_REGISTRY = Path(
    os.environ.get("SIH_SYNTH_REGISTRY",
                   Path(tempfile.gettempdir()) / "sih_synth_registry")
)


def pixel_key(arr: np.ndarray) -> str:
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


class SyntheticOCR(OCRBackend):
    """
    Perfect line-level OCR for images made by the synthetic renderer.

    The renderer records every line of text it draws. This backend finds
    that record (a `.lines.json` sidecar next to the image, or a registry
    entry keyed by the image's pixels - which is how an API upload of a
    rendered label is still recognised), maps the boxes into whatever
    frame the pipeline hands it via the ArUco marker, and returns one
    span per printed line - the same granularity PaddleOCR returns.

    It measures everything DOWNSTREAM of OCR. It says nothing about how
    well PaddleOCR reads real packaging; only real photos can.
    """

    name = "synthetic"
    supports_languages = ("en",)

    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        manifest = self._find_manifest(source)
        if manifest is None:
            return []
        H = self._homography(manifest, image)
        spans = []
        for ln in manifest["lines"]:
            pts = np.float32(ln["poly"]).reshape(-1, 1, 2)
            if H is not None:
                pts = cv2.perspectiveTransform(pts, H)
            pts = pts.reshape(-1, 2)
            x, y = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            spans.append(TextSpan(
                text=ln["text"], bbox=BBox(x, y, x2 - x, y2 - y),
                confidence=0.97, source_engine=self.name,
            ))
        return spans

    @staticmethod
    def _find_manifest(source: Optional[str]) -> Optional[dict]:
        if not source:
            return None
        side = Path(f"{source}.lines.json")
        if side.exists():
            return json.loads(side.read_text(encoding="utf-8"))
        img = cv2.imread(str(source))
        if img is None:
            return None
        reg = SYNTH_REGISTRY / f"{pixel_key(img)}.json"
        if reg.exists():
            return json.loads(reg.read_text(encoding="utf-8"))
        return None

    @staticmethod
    def _homography(manifest: dict, image: np.ndarray) -> Optional[np.ndarray]:
        src = manifest.get("marker_corners")
        if src:
            from ..calibration import detect_markers
            try:
                dets = detect_markers(image)
            except Exception:
                dets = []
            if dets:
                dst = max(dets, key=lambda d: d.area_px).corners
                H, _ = cv2.findHomography(np.float32(src), np.float32(dst))
                if H is not None:
                    return H
        sh = manifest.get("shape", [])[:2]
        if sh and tuple(sh) != tuple(image.shape[:2]):
            sy, sx = image.shape[0] / sh[0], image.shape[1] / sh[1]
            return np.float64([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return None


# =====================================================================
# VLM fallback (hook)
# =====================================================================

class VLMFallback(OCRBackend):
    """
    Vision-language model for crops PaddleOCR finds hard. Scope to
    LOW-CONFIDENCE CROPS ONLY, and keep the strict-JSON prompt: a VLM
    that paraphrases will invent plausible prices.
    """

    name = "vlm"
    supports_languages = ("en", "hi")

    PROMPT = (
        "Transcribe every piece of text visible in this image of a product "
        "label. Return ONLY a JSON array, no prose and no markdown fences. "
        'Each element: {"text": "<exact text>", "confidence": <0.0-1.0>}. '
        "Transcribe exactly what is printed, including punctuation and "
        "currency symbols. If a character is illegible use '?'. Never guess "
        "a digit you cannot see - an invented price is worse than a gap."
    )

    def __init__(self, client=None, model: str = ""):
        self.client = client
        self.model = model

    def available(self) -> bool:
        return self.client is not None

    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        if not self.available():
            return []
        h, w = image.shape[:2]
        try:
            items = self._call_model(image)
        except Exception as exc:
            logger.warning("VLM call failed: %s", exc)
            return []
        return [
            TextSpan(
                text=it.get("text", ""), bbox=BBox(0.0, 0.0, float(w), float(h)),
                confidence=float(it.get("confidence", 0.7)), source_engine=self.name,
            )
            for it in items if it.get("text")
        ]

    def _call_model(self, image: np.ndarray) -> list[dict]:
        raise NotImplementedError("Wire your VLM provider here.")


# =====================================================================
# Factory
# =====================================================================

class OCRUnavailable(RuntimeError):
    pass


def build_default_ocr(vlm_client=None, backend: Optional[str] = None, **paddle_kwargs):
    """
    The engine the pipeline uses when none is passed explicitly.

    paddle (default) - PaddleOCR. Raises OCRUnavailable if it is not
                       installed, rather than quietly reading with
                       something weaker.
    synthetic        - test oracle for rendered labels.
    stub             - geometry only, no text.
    """
    choice = (backend or os.environ.get("SIH_OCR_BACKEND") or "paddle").lower()
    if choice == "synthetic":
        engine: OCRBackend = SyntheticOCR()
    elif choice == "stub":
        engine = StubOCR()
    elif choice == "paddle":
        engine = PaddleOCRBackend(**paddle_kwargs)
        if not engine.available():
            raise OCRUnavailable(
                "PaddleOCR is not installed. Run:\n"
                "    pip install paddlepaddle paddleocr\n"
                "The first scan downloads the PP-OCR models (~200 MB) and caches "
                "them under ~/.paddlex."
            )
    else:
        raise ValueError(f"Unknown SIH_OCR_BACKEND={choice!r}")

    if vlm_client is not None:
        from .base import EnsembleOCR
        return EnsembleOCR([engine], fallback=VLMFallback(client=vlm_client))
    return engine
