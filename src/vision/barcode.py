"""
Barcode / GTIN reading.

WHY THIS MATTERS FOR RULE 18(2A)
--------------------------------
Rule 18(2A) prohibits different MRPs on an IDENTICAL pre-packaged
commodity. Deciding "identical" from OCR alone is guesswork: brand plus
net quantity plus commodity name will happily group a 500g pack of one
variant with a 500g pack of another, and every such finding has to be
hedged and handed to a human.

A GTIN settles it. Two packages with the same barcode are the same
product, full stop. That moves a dual-MRP finding from "inferred,
confirm before acting" to something an inspector can act on directly -
which is the difference between a lead and evidence.

VALIDATION IS NOT OPTIONAL
--------------------------
A misread barcode is worse than no barcode: it silently groups unrelated
products, and the resulting "identical commodity, two prices" finding
would be confidently wrong. GTIN check digits exist precisely for this,
so every read is verified before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core.schema import BBox


@dataclass
class BarcodeRead:
    # bbox is needed downstream: a barcode is a high-contrast rectangle
    # and the sticker detector will happily report it as a pasted label
    # unless the region is masked out.
    value: str
    format: str
    valid_checksum: bool
    confidence: float
    bbox: Optional[BBox] = None

    @property
    def trustworthy(self) -> bool:
        return self.valid_checksum and len(self.value) in (8, 12, 13, 14)


def gtin_checksum_valid(code: str) -> bool:
    """
    Verify a GTIN-8/12/13/14 check digit.

    Weights alternate 3 and 1 from the rightmost digit before the check
    digit. Anything failing this is a misread, not a product.
    """
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in code]
    check = digits[-1]
    body = digits[:-1][::-1]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(body))
    return (10 - total % 10) % 10 == check


def read_barcode(image: np.ndarray) -> Optional[BarcodeRead]:
    """
    Detect and decode a barcode, trying progressively harder.

    Packaging barcodes are often printed small, on curved surfaces, or
    at low contrast against a coloured background, so a single pass at
    native resolution misses many of them. Upscaling helps more than
    anything else because the detector needs enough pixels per bar.
    """
    if image is None or image.size == 0:
        return None

    # -- primary: zbar --------------------------------------------
    zbar_available()
    # OpenCV's BarcodeDetector is kept below as a fallback, but it
    # failed to decode cleanly rendered EAN-13 at every size tested
    # here - 318px through 1216px wide - including the standalone
    # barcode image with no packaging around it. zbar decodes the same
    # images first try, both standalone and composited onto a label.
    try:
        from pyzbar.pyzbar import decode as _zbar_decode

        for d in _zbar_decode(image):
            value = d.data.decode("utf-8", "ignore").strip()
            if not value or not gtin_checksum_valid(value):
                continue
            _r = d.rect
            return BarcodeRead(
                value=value,
                format=str(d.type),
                valid_checksum=True,
                confidence=1.0,
                bbox=BBox(float(_r.left), float(_r.top),
                          float(_r.width), float(_r.height)),
            )
    except ImportError:
        # No reader installed: Rule 18(2A) degrades to inferred product
        # matching. That is a weaker finding, not a broken scan.
        pass
    except Exception:
        pass

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.barcode.BarcodeDetector()

    attempts = [
        gray,
        cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.equalizeHist(gray),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]

    for i, view in enumerate(attempts):
        try:
            ok, decoded, types, _ = detector.detectAndDecodeWithType(view)
        except cv2.error:
            continue
        if not ok or not decoded:
            continue
        for value, kind in zip(decoded, types):
            value = (value or "").strip()
            if not value:
                continue
            valid = gtin_checksum_valid(value)
            if not valid:
                # A failed check digit means a misread. Grouping
                # unrelated products under a bad code would produce a
                # confidently wrong dual-price finding, so drop it.
                continue
            return BarcodeRead(
                value=value,
                format=kind or "unknown",
                valid_checksum=True,
                # Later attempts needed more help, so trust them slightly less.
                confidence=max(0.6, 1.0 - 0.1 * i),
            )
    return None


_ZBAR_STATE: dict = {}


def zbar_available() -> bool:
    """True when the zbar barcode reader loads. On Windows, pyzbar's DLL
    needs the Microsoft Visual C++ 2013 Redistributable (x64); without it
    the import fails and GTIN reading silently falls back to OpenCV, which
    rarely decodes. Warns once, loudly, so a missing reader is noticed."""
    if "ok" not in _ZBAR_STATE:
        try:
            from pyzbar.pyzbar import decode  # noqa: F401
            _ZBAR_STATE["ok"] = True
        except Exception as e:  # ImportError, FileNotFoundError / OSError (DLL)
            _ZBAR_STATE["ok"] = False
            import sys
            print("!! Barcode reader (zbar) not loaded: %s: %s. GTIN will rarely be "
                  "read. Fix: pip install pyzbar; on Windows also install the Microsoft "
                  "Visual C++ 2013 Redistributable (x64)." % (type(e).__name__, e),
                  file=sys.stderr)
    return _ZBAR_STATE["ok"]
