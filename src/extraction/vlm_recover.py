"""
Second pass: ask a vision-language model for the declarations OCR missed,
then check what it says against the pixels and against OCR.

A VLM read becomes a Declaration only through the SAME extractor as OCR
text (so "₹5" is parsed exactly as it would be from OCR), carries the
box the model pointed at (so the report shows a crop, and glyph
measurement runs on real pixels), and is marked:

    source "vlm+ocr"  OCR read matching text inside that box
    source "vlm"      the model alone - the rules engine will not base a
                      verdict on it (findings become INDETERMINATE)
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.schema import BBox, Declaration, TextSpan

logger = logging.getLogger(__name__)

# Fields worth a second read. Country of origin only matters for imports
# and common_name is often on another panel; both are still asked for when
# missing, but they are last so a truncated reply loses them first.
RECOVERABLE = ["net_quantity", "retail_sale_price", "manufacture_date",
               "unit_sale_price", "consumer_care", "manufacturer_details",
               "common_name", "country_of_origin"]


def _agrees(vlm_text: str, box: BBox, spans: list[TextSpan], value=None) -> Optional[str]:
    """
    Did OCR read the same thing inside the model's box?

    The OCR lines inside the box are joined and compared WHOLE with the
    model's text - a one-character fragment ("2") must not "confirm"
    "MAR/26-NOV/26-D". Failing that, the parsed value itself ("0.83")
    appearing in the OCR text there is enough.
    """
    import re

    from . import fuzzy as _fz

    inside = [sp for sp in spans
              if box.contains(sp.bbox, tol=0.25 * max(sp.bbox.h, 1.0)) or box.iou(sp.bbox) > 0.2]
    if not inside:
        return None
    inside.sort(key=lambda sp: (sp.bbox.cy, sp.bbox.x))
    joined = " ".join(sp.text for sp in inside)
    a, b = _fz._norm(joined), _fz._norm(vlm_text)
    if a and b and min(len(a), len(b)) >= 0.6 * max(len(a), len(b)):
        if _fz.partial_match(b, a)[0] >= 75 or _fz.partial_match(a, b)[0] >= 75:
            return joined
    if value is not None:
        v = str(value)
        if len(re.sub(r"\D", "", v)) >= 3 and v in joined.replace(" ", ""):
            return joined
    return None


def recover_with_vlm(reader, image, declarations: list[Declaration],
                     spans: list[TextSpan], extractor, calibration=None) -> dict:
    """
    Fill in missing declarations from `reader` (e.g. GeminiReader).
    Mutates `declarations`; returns stats for the report.
    """
    from .fields import _CANONICAL_LABEL, PATTERNS

    by_id = {d.field_id: d for d in declarations}
    missing = [f for f in RECOVERABLE if f in by_id and not by_id[f].present]
    stats = {"asked": missing, "found": [], "confirmed_by_ocr": [], "model": reader.label()}
    if not missing:
        return stats

    try:
        reads = reader.read_fields(image, missing)
    except Exception as exc:
        # Reported, not swallowed: the report says the second reader failed.
        logger.warning("Vision-model read failed: %s", exc)
        stats["error"] = str(exc)
        return stats

    H, W = image.shape[:2]
    for item in reads:
        field, text, box = item["field"], item["text"], item["box"]
        bbox = BBox(box[0], box[1], box[2] - box[0], box[3] - box[1]) if box \
            else BBox(0.0, 0.0, float(W), float(H))
        candidates = [text]
        label = _CANONICAL_LABEL.get(field)
        if label and not PATTERNS[field][0].search(text):
            candidates.append(f"{label} {text}")
        decl = None
        for cand in candidates:
            span = TextSpan(text=cand, bbox=bbox, confidence=0.5, source_engine=reader.label())
            got = {d.field_id: d for d in extractor.extract(
                [span], image=image if box else None, calibration=calibration)}
            if field in got and got[field].present:
                decl = got[field]
                break
        if decl is None:
            continue

        match = _agrees(text, bbox, spans, decl.value) if box else None
        decl.source = "vlm+ocr" if match else "vlm"
        decl.extraction_confidence = 0.7 if match else 0.45
        decl.notes.append(
            f"Read by {reader.label()} (a vision-language model), not by OCR. "
            + (f"OCR read '{match}' at the same place, so the reading is confirmed."
               if match else
               "OCR did not read matching text there - confirm on the pack."
               if box else
               "The model gave no location, so there is no crop to check against.")
        )
        idx = next(i for i, d in enumerate(declarations) if d.field_id == field)
        declarations[idx] = decl
        stats["found"].append(field)
        if match:
            stats["confirmed_by_ocr"].append(field)
    return stats
