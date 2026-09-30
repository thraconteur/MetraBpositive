"""
Tight ink extents for OCR boxes.

PaddleOCR's detector deliberately pads every text box: DB detection
predicts a shrunken text kernel and then "unclips" it outward, so the
returned polygon sits some way outside the actual ink - typically 10-25%
of the line height on each side. That is right for recognition and wrong
for anything measured in the gaps BETWEEN lines. Rule 8 clear space is
exactly such a measurement: two padded boxes eat 0.2-0.5 numeral heights
out of the gap, which turned compliant labels into "insufficient clear
space" findings on real photographs.

`ink_extent` shrinks a box to the ink it actually contains. It can only
ever SHRINK the box (the result lies inside the input), and it returns
None whenever the crop does not look like dark-on-light or
light-on-dark text, so the failure mode is "measure from the OCR box as
before", never "invent a gap".
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..core.schema import BBox


def ink_extent(gray: np.ndarray, box: BBox, min_side: int = 4) -> Optional[BBox]:
    H, W = gray.shape[:2]
    x0, y0 = max(0, int(np.floor(box.x))), max(0, int(np.floor(box.y)))
    x1, y1 = min(W, int(np.ceil(box.x2))), min(H, int(np.ceil(box.y2)))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    crop = gray[y0:y1, x0:x1]
    if crop.dtype != np.uint8:
        crop = np.clip(crop, 0, 255).astype(np.uint8)
    if int(crop.max()) - int(crop.min()) < 25:
        return None                      # no contrast: nothing to trust

    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = bw > 0
    frac = float(ink.mean())
    if frac > 0.5:                       # light text on a dark ground
        ink = ~ink
        frac = 1.0 - frac
    if frac < 0.005 or frac > 0.6:
        return None

    h, w = ink.shape
    row_mass = ink.sum(axis=1)
    # A low bar: the top rows of a line hold only the tips of the tallest
    # glyphs (a bracket, a 'Q' tail), and a 1%-of-width bar clipped them.
    on = row_mass >= max(1, int(0.002 * w))
    runs = _runs(on)
    if not runs:
        return None
    # The padded box usually catches the descenders of the line above or
    # the caps of the line below. Those arrive as a separate band of ink
    # rows TOUCHING the crop edge; keep them and the "tight" box grows
    # into the neighbour, which is precisely the gap we want to measure.
    # Drop edge-touching bands unless they are the dominant one.
    main = max(runs, key=lambda r: row_mass[r[0]:r[1]].sum())
    keep = [r for r in runs
            if r == main or not (r[0] == 0 or r[1] == h)]
    ty0, ty1 = min(r[0] for r in keep), max(r[1] for r in keep)
    cols = np.where(ink[ty0:ty1].sum(axis=0) >= 1)[0]
    if cols.size == 0:
        return None
    tx0, tx1 = int(cols[0]), int(cols[-1]) + 1
    if ty1 - ty0 < 2 or tx1 - tx0 < 2:
        return None
    return BBox(float(x0 + tx0), float(y0 + ty0), float(tx1 - tx0), float(ty1 - ty0))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True values."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out
