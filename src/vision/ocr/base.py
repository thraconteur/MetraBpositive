"""
OCR backend abstraction.

WHY AN ABSTRACTION AND NOT JUST "import paddleocr"
--------------------------------------------------
Three reasons, all of which bite during a hackathon:

  1. PaddleOCR is the engine, but tests must not need its models: the
     pipeline downstream of OCR is tested with SyntheticOCR, which
     returns Paddle-shaped line spans for rendered labels.

  2. A saved Paddle dump can be replayed (ReplayOCR), so extraction and
     rules are debugged against real Paddle output without re-running it.

  3. The optional VLM fallback wraps the engine for hard crops only.

Implement `recognise` and you are done. Everything downstream consumes
TextSpan objects and does not care where they came from.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ...core.schema import BBox, TextSpan

logger = logging.getLogger(__name__)


class OCRBackend(ABC):
    """Base class for all OCR engines."""

    name: str = "base"
    supports_languages: tuple[str, ...] = ("en",)

    @abstractmethod
    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        """
        Return every text span found in the image.

        `source` is the path of the photo being scanned, when there is
        one. Backends use it to name OCR dumps and (in tests) to find a
        rendered label's line manifest. Pixels always come from `image`.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """Whether this backend can actually run right now."""
        return True

    def upright_rotation(self, image: np.ndarray, source: Optional[str] = None) -> tuple[int, float]:
        """
        How to turn this photo so its text reads upright: (quarter turns
        clockwise in degrees - 0/90/180/270, then a small tilt correction in
        degrees, counter-clockwise positive as in cv2.getRotationMatrix2D).
        Backends that cannot tell return (0, 0.0).
        """
        return 0, 0.0

    def note_orientation(self, turn: int, tilt: float) -> None:
        """Told that the pipeline turned the photo (turn, tilt) itself."""
        return None

    def __repr__(self) -> str:
        return f"<OCRBackend {self.name}>"


# ---------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------

class EnsembleOCR(OCRBackend):
    """
    Run several backends and merge their outputs.

    Merge policy: spans that overlap heavily (IoU above `iou_threshold`)
    are treated as the same physical text, and we keep the one with the
    highest confidence. Non-overlapping spans are all kept, because a
    miss by one engine is exactly what the ensemble exists to cover.

    A refinement worth trying once the basics work: instead of taking
    the highest-confidence transcription wholesale, do character-level
    voting across engines for the overlapping spans. Helps most on
    digits, which is where you care most.
    """

    name = "ensemble"

    def __init__(
        self,
        backends: list[OCRBackend],
        iou_threshold: float = 0.5,
        fallback: Optional[OCRBackend] = None,
        low_confidence: float = 0.55,
    ):
        self.backends = [b for b in backends if b.available()]
        self.iou_threshold = iou_threshold
        self.fallback = fallback
        self.low_confidence = low_confidence
        if not self.backends:
            raise RuntimeError(
                "No OCR backend is available. At minimum StubOCR should be."
            )
        self.supports_languages = tuple(
            sorted({l for b in self.backends for l in b.supports_languages})
        )

    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        all_spans: list[TextSpan] = []
        for backend in self.backends:
            try:
                all_spans.extend(backend.recognise(image, source=source))
            except Exception:  # one bad engine must not kill the scan
                logger.warning("OCR backend %s failed", backend.name, exc_info=True)

        merged = self._merge(all_spans)

        # Escalate only the hard crops to the expensive fallback (a VLM),
        # never the whole image - that is slow and usually unnecessary.
        if self.fallback is not None:
            for i, span in enumerate(merged):
                if span.confidence < self.low_confidence:
                    try:
                        crop = _crop(image, span.bbox)
                        better = self.fallback.recognise(crop)
                        if better:
                            best = max(better, key=lambda s: s.confidence)
                            if best.confidence > span.confidence:
                                merged[i] = TextSpan(
                                    text=best.text,
                                    bbox=span.bbox,
                                    confidence=best.confidence,
                                    source_engine=f"{self.fallback.name}(fallback)",
                                    language=best.language,
                                )
                    except Exception:
                        logger.warning("Fallback failed on span %d", i, exc_info=True)

        return merged

    def _merge(self, spans: list[TextSpan]) -> list[TextSpan]:
        kept: list[TextSpan] = []
        for span in sorted(spans, key=lambda s: -s.confidence):
            if any(span.bbox.iou(k.bbox) >= self.iou_threshold for k in kept):
                continue
            kept.append(span)
        return sort_reading_order(kept)


def sort_reading_order(spans: list[TextSpan]) -> list[TextSpan]:
    """
    Sort spans the way a human reads: top line first, then left to right.

    A naive sort on (bbox.y, bbox.x) does NOT do this. Words on the same
    printed line rarely share an exact top coordinate - baselines wobble
    by a pixel or two after rectification, and a word with an ascender
    ('Qty') starts a few pixels higher than one without ('Net'). Sorting
    on raw y therefore interleaves words from adjacent lines, and every
    downstream stage that reconstructs lines from span order silently
    produces garbage.

    That failure is quiet and nasty: OCR looks perfect when you print the
    spans, extraction still returns fields, and the values are simply
    wrong. So: bucket spans into rows by vertical overlap first, then
    order within each row by x.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: s.bbox.y)
    # Typical text height sets the row tolerance. Median is robust to a
    # stray full-height box from a logo or a border artefact.
    heights = sorted(s.bbox.h for s in ordered)
    median_h = heights[len(heights) // 2] or 1.0
    tol = median_h * 0.6

    rows: list[list[TextSpan]] = []
    for span in ordered:
        placed = False
        for row in rows:
            # Compare against the row's running centre, not its first
            # member, so a row that drifts slightly still accumulates.
            row_cy = sum(s.bbox.cy for s in row) / len(row)
            if abs(span.bbox.cy - row_cy) <= tol:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    rows.sort(key=lambda r: sum(s.bbox.cy for s in r) / len(r))
    out: list[TextSpan] = []
    for row in rows:
        out.extend(sorted(row, key=lambda s: s.bbox.x))
    return out


def _crop(image: np.ndarray, box: BBox, pad: int = 4) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, int(box.x) - pad)
    y0 = max(0, int(box.y) - pad)
    x1 = min(w, int(box.x2) + pad)
    y1 = min(h, int(box.y2) + pad)
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]
