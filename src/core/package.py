"""
One PACKAGE, several photos.

An inspector has to check the whole package, and packs spread their
declarations over every side: the address and consumer care on the back,
the net quantity on the front, and the MRP, date and batch stamped on the
top of a carton, the base of a can or the crimp of a tube - usually with a
printed line elsewhere saying "For MRP (incl. of all taxes) ... see base".
A single photo of any one side can only say "not visible here".

scan_package() scans each photo on its own, then decides the package from
all of them together:

  * each declaration is taken from the photo that read it best, and says
    which photo that was;
  * every text line from every photo is available to the text checks, so
    "For MRP (incl. of all taxes) ... see top" on the side satisfies the
    qualifier for the price stamped on the top;
  * the photos are laid one under another on one canvas, so every box
    still points at the right pixels (evidence crops, clear space) and no
    measurement ever mixes two photos;
  * with every side photographed (coverage_complete), a declaration that
    no photo shows is a real finding, not "not visible in this image".
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .schema import BBox, Declaration, ScanResult

GAP_PX = 80


def _shift(b: Optional[BBox], dy: float) -> Optional[BBox]:
    return None if b is None else BBox(b.x, b.y + dy, b.w, b.h)


def _score(d: Declaration) -> tuple:
    # OCR over a vision-model-only read; a parsed value over none; then
    # extraction confidence.
    return (getattr(d, "source", "ocr") != "vlm", d.value is not None,
            float(d.extraction_confidence or 0.0))


def merge_package(parts: list[ScanResult], engine, coverage_complete: bool = True,
                  names: Optional[list[str]] = None) -> ScanResult:
    names = names or [Path(p.image_path or f"photo{i+1}").name for i, p in enumerate(parts)]

    # -- one canvas, photos stacked ---------------------------------
    imgs = []
    for p in parts:
        im = p.analysed_image
        if im is None and p.image_path:
            im = cv2.imread(str(p.image_path))
        if im is None:
            im = np.full((10, 10, 3), 255, np.uint8)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        imgs.append(im)
    W = max(im.shape[1] for im in imgs)
    offsets, y = [], 0
    for im in imgs:
        offsets.append(y)
        y += im.shape[0] + GAP_PX
    canvas = np.full((max(1, y - GAP_PX), W, 3), 255, np.uint8)
    for im, oy in zip(imgs, offsets):
        canvas[oy:oy + im.shape[0], :im.shape[1]] = im

    # -- spans, moved onto the canvas --------------------------------
    moved: dict[int, object] = {}
    spans = []
    for k, (p, oy) in enumerate(zip(parts, offsets)):
        for sp in p.spans or []:
            c = copy.copy(sp)
            c.bbox = _shift(sp.bbox, oy)
            c.ink_bbox = _shift(getattr(sp, "ink_bbox", None), oy)
            c.photo = k                      # which photo of the package
            moved[id(sp)] = c
            spans.append(c)

    # -- best declaration per field ----------------------------------
    best: dict[str, tuple] = {}
    field_order: list[str] = []
    for k, (p, oy) in enumerate(zip(parts, offsets)):
        for d in p.declarations:
            if d.field_id not in field_order:
                field_order.append(d.field_id)
            if not d.present:
                continue
            cur = best.get(d.field_id)
            if cur is None or _score(d) > _score(cur[0]):
                best[d.field_id] = (d, k, oy)
    decls = []
    for fid in field_order:
        if fid not in best:
            decls.append(Declaration(field_id=fid, present=False))
            continue
        d, k, oy = best[fid]
        c = copy.copy(d)
        c.bbox = _shift(d.bbox, oy)
        c.spans = [moved.get(id(s), s) for s in d.spans]
        c.notes = list(d.notes) + [f"Read from photo {k + 1} ({names[k]})."]
        decls.append(c)

    # -- the rest of the result --------------------------------------
    main = max(range(len(parts)),
               key=lambda i: sum(1 for d in parts[i].declarations if d.present))
    ctx = copy.copy(parts[main].context)
    if getattr(ctx, "pdp_bbox", None) is not None:
        ctx.pdp_bbox = _shift(ctx.pdp_bbox, offsets[main])
    calib = next((p.calibration for p in parts if p.calibration.available),
                 parts[main].calibration)

    merged = ScanResult(image_path=" + ".join(str(p.image_path) for p in parts))
    merged.calibration = calib
    merged.context = ctx
    merged.declarations = decls
    merged.spans = spans
    # Overlays (stickers) from the photo that supplied the price.
    price = best.get("retail_sale_price")
    src = price[1] if price else main
    ov = parts[src].overlays
    if ov:
        ov = [copy.copy(o) for o in ov]
        for o in ov:
            if getattr(o, "bbox", None) is not None:
                o.bbox = _shift(o.bbox, offsets[src])
    merged.overlays = ov
    merged.overlay_detection_failed = parts[src].overlay_detection_failed
    merged.coverage_complete = coverage_complete
    merged.image_quality_ok = any(p.image_quality_ok for p in parts)
    merged.quality_notes = "; ".join(
        f"photo {i + 1}: {p.quality_notes}" for i, p in enumerate(parts) if p.quality_notes)
    merged.analysed_image = canvas
    merged.ocr_stats = {
        "package_photos": names,
        "per_photo": [dict(p.ocr_stats or {}) for p in parts],
        "regions": len(spans),
    }
    merged.package_photos = names
    conflict = _identity_conflict(parts, names)
    if conflict:
        # Photos of two different products: nothing may be pooled into a
        # verdict - a qualifier or a declaration from one product must not
        # clear or condemn the other, and absence proves nothing.
        merged.coverage_complete = False
        merged.ocr_stats["pack_conflict"] = conflict
    res = engine.evaluate(merged)
    if conflict:
        from .schema import Finding, Outcome, Severity
        res.findings.insert(0, Finding(
            rule_id="package.same_product", citation="package scan",
            outcome=Outcome.INDETERMINATE, severity=Severity.CRITICAL,
            message=(f"These photos do not look like one pack: {conflict}. "
                     f"Scan each product separately."),
            confidence=0.9))
    return res


def _identity_conflict(parts: list[ScanResult], names: list[str]) -> Optional[str]:
    """Different barcodes, net quantities or MRPs across the photos of one
    'pack' mean two products were mixed (a real test: a churna pouch and a
    biscuit pack merged into one record, which hid the pouch's violation)."""
    def per(fid):
        out = []
        for k, p in enumerate(parts):
            d = next((d for d in p.declarations if d.field_id == fid and d.present
                      and d.value is not None), None)
            if d is None:
                continue
            try:
                v = float(d.value)
            except (TypeError, ValueError):
                continue
            u = (d.unit or "").lower()
            if u in ("kg", "l", "ltr", "litre"):
                v *= 1000.0
            out.append((k, v))
        return out

    codes = {}
    for k, p in enumerate(parts):
        c = getattr(getattr(p, "context", None), "barcode", None)
        if c:
            codes.setdefault(c, k)
    if len(codes) > 1:
        a, b = list(codes.items())[:2]
        return f"barcode {a[0]} ({names[a[1]]}) vs {b[0]} ({names[b[1]]})"
    for fid, label, tol in (("net_quantity", "net quantity", 0.01), ("retail_sale_price", "MRP", 0.001)):
        vals = per(fid)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                (ka, va), (kb, vb) = vals[i], vals[j]
                if abs(va - vb) > tol * max(va, vb, 1.0):
                    return f"{label} {va:g} ({names[ka]}) vs {vb:g} ({names[kb]})"
    return None


_SEE = __import__("re").compile(
    r"\bsee\s+(?:on\s+)?(?:the\s+)?(?:(?:top|base|bottom|side|back)\s+of\s+(?:the\s+)?)?"
    r"(top|base|bottom|lid|cap|crimp|side|back|carton|coding\s+area)", __import__("re").I)


def capture_advice(scan: ScanResult) -> list[str]:
    """
    Guided capture: what to photograph next. Packs say where the rest is
    ("For MRP ... see base of can", "See on Crimp", "see Cap/Bottom",
    "SEE LID/BASE") - so the app can tell the inspector which side to shoot
    instead of leaving a declaration as "not visible".
    """
    if scan.package_photos:
        return []
    # (A common name is rarely labelled, so its absence says nothing
    # about where to point the camera.)
    missing = {f.field_id for f in scan.indeterminate
               if f.rule_id.endswith(".presence") and f.field_id != "common_name"}
    if not missing:
        return []
    import re

    sides = []
    for sp in scan.spans or []:
        for m in _SEE.finditer(sp.text or ""):
            side = re.sub(r"\s+", " ", m.group(1).lower())
            if side not in sides:
                sides.append(side)
    names = ", ".join(sorted(m.replace("_", " ") for m in missing))
    if sides:
        return [f"The pack says to see the {' / '.join(sides)} - photograph that side too "
                f"and scan all photos together as one pack (not read yet: {names})."]
    return [f"Not on this side: {names}. Photograph the other sides and scan them "
            f"together as one pack."]
