"""
The pipeline orchestrator.

Wires every stage into one call:

    image -> quality gate -> calibration -> rectify -> preprocess
          -> OCR -> extraction -> classification -> rules engine -> ScanResult

DESIGN RULES WORTH KEEPING
--------------------------
* Every stage is INJECTABLE. The OCR backend, the extractor and the
  classifier are constructor arguments, not imports buried in a method.
  That is what makes the ablation table in your evaluation harness
  possible: hold everything constant, swap one stage, re-measure.

* The pipeline NEVER raises on a bad image. It returns a ScanResult with
  image_quality_ok=False and an explanation. A crash during a live demo
  looks like incompetence; a polite "retake with better lighting" looks
  like a product.

* Calibration is attempted but never required. Without it you still get
  every presence check, every wording check, and the two ratio-based
  presentation checks. Only the millimetre-height rule goes
  INDETERMINATE. Degrade, do not fail.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np

from .rules_engine import RulesEngine
from .schema import (
    BBox,
    Calibration,
    Declaration,
    PackageContext,
    PackageGeometry,
    ScanResult,
    TextSpan,
)

# ---------------------------------------------------------------------
# Stage protocols
# ---------------------------------------------------------------------

class Extractor(Protocol):
    def extract(self, spans, image=None, calibration=None) -> list[Declaration]: ...


def _keyword_in(word: str, text: str) -> bool:
    """
    Whole-word keyword match (a trailing plural or suffix is allowed for
    longer stems such as "moistur" or "noodle").

    Plain substring matching classified a real Streax HAIR SERUM as an
    alcoholic beverage: "rum" is inside "serum". The alcohol category then
    deferred the price rules to State excise law and the MRP was never
    checked. Short keywords ("rum", "oil", "tea", "dal") must stand alone.
    """
    w = word.strip().lower()
    if len(w) <= 4:
        pat = rf"(?<![a-z]){re.escape(w)}(?:s|es)?(?![a-z])"
    else:
        pat = rf"(?<![a-z]){re.escape(w)}"
    return re.search(pat, text) is not None


class Classifier(Protocol):
    def classify(self, image, declarations, spans, marker_corners=None) -> PackageContext: ...


# ---------------------------------------------------------------------
# Default classifier
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "food_packaged": ["biscuit", "chips", "rice", "flour", "atta", "oil", "milk",
                      "snack", "namkeen", "cereal", "noodle", "tea", "coffee",
                      "sugar", "salt", "spice", "masala", "dal", "pulse"],
    "beverage_non_alcoholic": ["water", "juice", "beverage", "soda", "drink", "cola"],
    "alcoholic_beverage": ["beer", "whisky", "whiskey", "rum", "vodka", "wine", "liquor"],
    "cosmetic": ["cream", "lotion", "shampoo", "hair oil", "face wash", "moistur",
                 "cosmetic", "talc", "perfume", "deodorant", "serum", "hair serum",
                 "conditioner", "hair colour", "hair color", "sunscreen"],
    "soap": ["soap", "bathing bar"],
    "detergent_household": ["detergent", "washing powder", "dishwash", "cleaner",
                            "phenyl", "bleach"],
    "pharmaceutical": ["tablet", "capsule", "syrup", "ointment", "mg ", "pharma"],
    "textile": ["cotton", "fabric", "shirt", "saree", "textile"],
    # Added: five categories referenced by rules in exemptions.yaml and
    # lmpc_2011.yaml that the classifier could never previously assign.
    # A category the classifier does not know is a category no exemption
    # or rule keyed to it can ever fire, no matter how correctly the
    # rules themselves were transcribed from the gazette - the medical-
    # device and pan-masala rules pulled from the 2025/2026 amendments
    # were, until now, encoded but permanently dead code.
    "medical_device": ["thermometer", "glucometer", "bp monitor",
                       "blood pressure monitor", "medical device",
                       "surgical", "syringe", "nebulizer", "oximeter",
                       "sphygmomanometer", "hearing aid"],
    "pan_masala": ["pan masala", "gutkha", "supari", "zarda"],
    "agricultural_unprocessed": ["wheat", "paddy", "unprocessed", "raw grain",
                                 "farm produce", "mandi"],
    "restaurant_prepared": ["restaurant", "eatery", "prepared fresh",
                            "ready to eat", "hot food", "takeaway"],
    "notified_essential": ["essential commodity", "govt notified price",
                           "government notified", "pds", "ration", "fair price"],
    # Named explicitly in Rule 3(b) alongside agricultural produce, but
    # never given their own keywords - the >50kg bag exemption could
    # fire for farm produce and never for the two commodities the rule
    # actually leads with. Caught by test_all_referenced_categories_are_known.
    "cement": ["cement", "opc", "ppc", "portland"],
    "fertilizer": ["fertiliser", "fertilizer", "urea", "dap", "npk"],
}

# Phrases that set a context FLAG rather than a category - Rule 3(c)'s
# exemption and the export carve-out both key off wording on the pack,
# not off what kind of commodity it is.
FLAG_PHRASES = {
    # Rule 2(bc) defines an institutional consumer's purchase by this
    # exact phrase; Rule 3(c) then exempts it from Chapter II entirely.
    "industrial_or_institutional": [
        "not for retail sale", "for industrial use", "for institutional use",
    ],
    "for_export": ["for export only", "not for sale in india", "export only"],
}


class KeywordClassifier:
    """
    Baseline commodity classifier over the OCR text.

    Deliberately simple and fully inspectable. Two upgrade paths when you
    have time, in order of value:

      1. Barcode / GTIN lookup. If you can read the barcode you can look
         the product up and get category, net quantity and often the
         declared MRP - which turns several "did we OCR it right" questions
         into a database join. Highest value per hour of work.
      2. A small image classifier fine-tuned on your collected photos.

    Whatever you use, it MUST run before the rules engine. Getting the
    category wrong means applying the wrong rule subset, which produces
    exactly the false positives that make an inspector stop trusting the
    tool.
    """

    def classify(
        self,
        image: Optional[np.ndarray],
        declarations: list[Declaration],
        spans: list[TextSpan],
        marker_corners: Optional[np.ndarray] = None,
    ) -> PackageContext:
        text = " ".join(s.text for s in spans).lower()

        # Score by matched CHARACTERS, not match COUNT. A count-based
        # score ties "pan masala" against plain "masala": both hit once,
        # and with a strict > comparison the first-defined category wins
        # every tie regardless of specificity. That silently classified
        # every "Pan Masala Sachet" label as food_packaged instead of
        # pan_masala - the exact case the 2025 amendment singles out for
        # different treatment, misclassified by the one keyword the
        # amendment exists because of. Scoring by length lets a more
        # specific phrase outrank a shorter generic word it contains.
        category, best_score = "other", 0
        for cat, words in CATEGORY_KEYWORDS.items():
            score = sum(len(w) for w in words if _keyword_in(w, text))
            if score > best_score:
                category, best_score = cat, score
        best_hits = max(1, best_score // 6) if best_score else 0

        ctx = PackageContext(
            commodity_category=category,
            package_class="wholesale" if "wholesale" in text else "retail",
            geometry=PackageGeometry.RECTANGULAR,
            # "Imported & Marketed by", "Importer:", "Imported and Distributed by"
            # too - only "imported by" counted, so a missing country of origin
            # on those packs was "not applicable".
            # anchored like the label ("Made with imported cocoa. Manufactured
            # by ..." on a domestic pack is not an import declaration)
            is_imported=bool(re.search(
                r"imported\.?\s*(?:(?:,|&|and)\s*\w+\.?\s*){0,2}by\b|\bimporter\s*[:\-]|country of origin",
                text)),
            other_law_applies=category in ("food_packaged", "beverage_non_alcoholic",
                                           "pharmaceutical"),
            classifier_confidence=min(1.0, best_hits / 3.0) if best_hits else 0.2,
            industrial_or_institutional=any(
                p in text for p in FLAG_PHRASES["industrial_or_institutional"]
            ),
            for_export=any(p in text for p in FLAG_PHRASES["for_export"]),
        )

        # PDP comes from the PHYSICAL panel boundary, never from the
        # extent of the ink. See vision/panel.py for why that
        # distinction decides whether undersized text passes or fails.
        from ..vision.panel import detect_panel

        boxes = [s.bbox for s in spans if s.bbox]
        det = detect_panel(image, text_boxes=boxes, marker_corners=marker_corners)
        ctx.pdp_bbox = det.bbox
        ctx.pdp_quad = det.quad
        ctx.pdp_detection_method = det.method
        ctx.pdp_confidence = det.confidence

        return ctx


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

@dataclass
class PipelineConfig:
    marker_size_mm: float = 25.0
    rectify_to_marker: bool = True
    target_px_per_mm: float = 12.0
    run_preprocess: bool = True
    # OCR reads the rectified photo WITHOUT the deglare / CLAHE / sharpen
    # pass; measurement still uses the enhanced image. Measured on three
    # real photos run through PaddleOCR both ways: without the pass Paddle
    # read "Strong, Long, Thick Hair" (with it: "Brong, Long, Thick Hak"),
    # "(Rs.0.40/ml)" (with it: "(Rs.0.4 0/ml)"), "Toll Free" ("Tol Free"),
    # "INGREDIENTS" ("NGREDENTS"), the "6 g" beside NET QUANTITY (missed
    # with it), and mean line confidence rose 0.893 -> 0.916 on the Maggi
    # sachet. Paddle is trained on natural photos; a sharpened, CLAHE'd
    # image is out of its distribution.
    ocr_on_enhanced: bool = False
    enforce_quality_gate: bool = True
    min_blur_var: float = 60.0

    # Super-resolution before OCR. Off by default: it costs time on every
    # scan and the good adapter needs torch plus a weights file, so it is
    # opt-in rather than something a fresh clone silently depends on.
    #   CompliancePipeline(config=PipelineConfig(
    #       superres=True, sr_adapter=ESRGANAdapter()))
    # sr_adapter=None with superres=True uses Lanczos, which needs nothing.
    superres: bool = False
    sr_adapter: Optional[Any] = None


def _user_scale(ctx, panel_size_mm, reference):
    """
    A scale from something the inspector measured, never from a guess.

    This is what a pasted-in ArUco marker was trying to achieve - but a
    marker drawn onto the photo afterwards has whatever size it was drawn
    at, not 25 mm in the scene, so every millimetre derived from it was
    invented. A ruler measurement of the real panel is evidence; the
    report names it as such and carries its (wider) uncertainty.
    """
    from ..vision.calibration import calibrate_from_known_dimension

    if reference:
        px, mm = reference
        cal = calibrate_from_known_dimension(float(px), float(mm), assumed_error_mm=1.0)
        cal.notes = (f"Inspector-supplied reference: {px:.0f} px = {mm:g} mm "
                     f"(assumed +/-1 mm). Weaker than a printed marker.")
        return cal, None

    w_mm, h_mm = (float(panel_size_mm[0]), float(panel_size_mm[1] or 0.0))
    box = ctx.pdp_bbox
    reliable = box is not None and getattr(ctx, "pdp_confidence", 0.0) >= 0.5 \
        and getattr(ctx, "pdp_detection_method", "") != "text_extent"
    if not reliable:
        return Calibration(notes=(
            f"Panel size {w_mm:g} x {h_mm:g} mm was given, but the panel's edges "
            f"were not found reliably in the photo, so no scale was taken. Retake "
            f"with the whole panel flat and its edges visible, or use the marker.")), None
    if h_mm <= 0:
        # Width only: it is the width as photographed. Matching it to the
        # LONG side (a 180 x 240 panel) gave a scale 34% too high and a
        # false font-size violation.
        scales = [box.w / w_mm]
    else:
        # Long side to long side: the inspector may give width/height either way.
        long_px, short_px = max(box.w, box.h), min(box.w, box.h)
        long_mm, short_mm = max(w_mm, h_mm), min(w_mm, h_mm)
        scales = [long_px / long_mm, short_px / short_mm]
    ppm = sum(scales) / len(scales)
    spread = (max(scales) - min(scales)) / ppm if len(scales) == 2 else 0.0
    if spread > 0.15:
        return Calibration(notes=(
            f"The panel found in the photo ({box.w:.0f} x {box.h:.0f} px) does not "
            f"have the proportions of the size given ({w_mm:g} x {h_mm:g} mm): the "
            f"detected edges are probably not the panel's. No scale was taken.")), None
    ref_px, ref_mm = (box.w, w_mm) if h_mm <= 0 else (max(box.w, box.h), max(w_mm, h_mm))
    cal = calibrate_from_known_dimension(ref_px, ref_mm, assumed_error_mm=1.0)
    cal.px_per_mm = ppm
    cal.uncertainty_px_per_mm = ppm * max(1.0 / ref_mm, spread / 2, 0.02)
    cal.notes = (f"Scale from the inspector-supplied panel size {w_mm:g}"
                 + (f" x {h_mm:g}" if h_mm else "") + f" mm, matched to the panel edges "
                 f"found in the photo ({box.w:.0f} x {box.h:.0f} px). Weaker than a "
                 f"printed marker; uncertainty widened accordingly.")
    area = (w_mm * h_mm) / 100.0 if h_mm else None
    return cal, area


def _apply_scale_to_glyphs(declarations, calib) -> None:
    for d in declarations:
        g = d.glyph
        if g is None or not g.cap_height_px:
            continue
        g.cap_height_mm = calib.px_to_mm(g.cap_height_px)
        g.cap_height_mm_uncertainty = max(calib.mm_uncertainty(g.cap_height_px),
                                          0.5 / calib.px_per_mm)


class CompliancePipeline:

    def __init__(
        self,
        ocr=None,
        extractor: Optional[Extractor] = None,
        classifier: Optional[Classifier] = None,
        engine: Optional[RulesEngine] = None,
        config: Optional[PipelineConfig] = None,
        vlm=None,
    ):
        """
        vlm: optional second reader for declarations OCR missed, e.g.
             src.vision.ocr.vlm_gemini.GeminiReader(). Its reads are
             marked on the report and never decide a verdict alone.
        """
        self.vlm = vlm
        from ..extraction.fields import RegexExtractor
        from ..vision.ocr.backends import build_default_ocr

        self.config = config or PipelineConfig()
        self.ocr = ocr or build_default_ocr()
        self.extractor = extractor or RegexExtractor()
        self.classifier = classifier or KeywordClassifier()
        self.engine = engine or RulesEngine()

    # -----------------------------------------------------------------
    def scan(
        self,
        image_path: str | Path,
        inspector_id: Optional[str] = None,
        known_dimension_mm: Optional[float] = None,
        hints: Optional[dict] = None,
        panel_size_mm: Optional[tuple[float, float]] = None,
        reference: Optional[tuple[float, float]] = None,
        _orientation: Optional[tuple[int, float]] = None,
    ) -> ScanResult:
        """
        panel_size_mm  (width, height) of the photographed panel, measured
                       with a ruler, when no ArUco marker is in the frame.
        reference      (length_px, length_mm) of any one thing in the photo
                       - e.g. two points the inspector tapped in the app.
        known_dimension_mm  older form of panel_size_mm: the width only.

        These give a scale that is honest about being weaker than a
        marker: the report names the method and the uncertainty is wider,
        so borderline heights come back INDETERMINATE.
        """
        if panel_size_mm is None and known_dimension_mm:
            panel_size_mm = (float(known_dimension_mm), 0.0)
        from ..vision.calibration import calibrate_from_image, rectify_to_marker_plane
        from ..vision.preprocess import assess_quality, preprocess

        result = ScanResult(image_path=str(image_path), inspector_id=inspector_id)

        # IMREAD_COLOR applies the phone's EXIF rotation (portrait photos);
        # only a PNG with transparency is re-read with its alpha channel.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None and str(image_path).lower().endswith(".png"):
            full = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if full is not None and full.ndim == 3 and full.shape[2] == 4:
                if full.dtype != np.uint8:
                    full = (full / 257.0).astype(np.uint8)
                # Transparent background -> white, as printed (it loaded black).
                a_ = full[:, :, 3:4].astype(np.float32) / 255.0
                image = (full[:, :, :3].astype(np.float32) * a_ + 255.0 * (1 - a_)).astype(np.uint8)
        if image is None:
            result.image_quality_ok = False
            result.quality_notes = f"Could not read image at {image_path}."
            return self.engine.evaluate(result)

        # ---- 0. which way up ----------------------------------------
        # A pack photographed sideways, upside down or at a slant is turned
        # upright FIRST, so reading, measuring, evidence crops and the
        # report all work on the straightened photo. Row and label/value
        # grouping assume level text: a can base turned 270 degrees read
        # 1 of its 4 declarations, upright all 4.
        orient_note = ""
        if _orientation is not None:
            turn, tilt = _orientation
            self.ocr.note_orientation(turn, tilt)
        else:
            try:
                turn, tilt = self.ocr.upright_rotation(image, source=str(image_path))
            except Exception:
                logger.warning("Orientation check failed", exc_info=True)
                turn, tilt = 0, 0.0
        if turn or tilt:
            from ..vision.ocr.paddle import turn_upright

            image = turn_upright(image, turn, tilt)
            parts = ([f"turned {turn} degrees clockwise"] if turn else []) + (
                [f"straightened by {abs(tilt):.0f} degrees"] if tilt else [])
            orient_note = "Photo " + " and ".join(parts) + " to read it."

        # ---- 1. quality gate ----------------------------------------
        quality = assess_quality(image, min_blur_var=self.config.min_blur_var)
        result.quality_notes = "; ".join(x for x in (orient_note, quality.notes) if x)
        # The gate no longer refuses to READ. Sharpness is measured over the
        # whole frame, so a label shot sharp against a blurred background
        # scores "soft": a real Amul carton top (sharpness 34) and Monster
        # can base (25) were refused outright although every character on
        # them is legible. OCR runs anyway; the photo is sent back for a
        # retake only if the text itself could not be read (step 4b).
        quality_suspect = self.config.enforce_quality_gate and not quality.ok

        # ---- 2. calibration + rectification -------------------------
        working = image
        calib = Calibration()

        if self.config.rectify_to_marker:
            rectified, rcal = rectify_to_marker_plane(
                image,
                marker_size_mm=self.config.marker_size_mm,
                target_px_per_mm=self.config.target_px_per_mm,
            )
            if rectified is not None:
                # Rectified to the marker plane: scale is now uniform
                # across the whole image, so measurement is trivial and
                # perspective error is gone rather than merely small.
                working, calib = rectified, rcal

        if not calib.available:
            calib = calibrate_from_image(
                working, marker_size_mm=self.config.marker_size_mm
            )

        result.calibration = calib

        # ---- 3. preprocessing ---------------------------------------
        read_image = working
        if self.config.run_preprocess:
            working, _stages = preprocess(working)
            if self.config.ocr_on_enhanced:
                read_image = working

        # ---- 4. OCR --------------------------------------------------
        # Super-resolution, when enabled, feeds the READ path only. OCR
        # sees the upscaled copy; every box is scaled back, and glyph
        # measurement, clear space and evidence crops run on `working`.
        # A generative upscaler can invent plausible text that was never
        # legible, and a millimetre measured off invented glyphs would be
        # evidence of nothing. (It used to replace `working` itself, so
        # measurement ran on the upscaled pixels despite saying it did not.)
        ocr_input, sx, sy = read_image, 1.0, 1.0
        if self.config.superres:
            from ..vision.preprocess import ClassicalSR

            adapter = self.config.sr_adapter or ClassicalSR()
            try:
                ocr_input = adapter(read_image)
                sy = ocr_input.shape[0] / read_image.shape[0]
                sx = ocr_input.shape[1] / read_image.shape[1]
            except Exception:
                logger.warning("Super-resolution failed; OCR reads the original",
                               exc_info=True)
                ocr_input, sx, sy = read_image, 1.0, 1.0

        spans = self.ocr.recognise(ocr_input, source=str(image_path))
        # The full read says the photo lies on its side / is tilted, which the
        # quick check could not see (faint print): scan again, turned.
        late = (getattr(self.ocr, "last_stats", None) or {}).get("upright_after")
        if late and _orientation is None and (turn, tilt) == (0, 0.0):
            return self.scan(image_path, inspector_id=inspector_id,
                             known_dimension_mm=known_dimension_mm, hints=hints,
                             panel_size_mm=panel_size_mm, reference=reference,
                             _orientation=(int(late.get("turn", 0)), float(late.get("tilt", 0.0))))
        if sx != 1.0 or sy != 1.0:
            for sp in spans:
                sp.bbox = BBox(sp.bbox.x / sx, sp.bbox.y / sy,
                               sp.bbox.w / sx, sp.bbox.h / sy)
        # Tight ink boxes for gap measurement - OCR boxes are padded.
        try:
            from ..vision.ink import ink_extent

            _gray = (cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                     if working.ndim == 3 else working)
            for sp in spans:
                sp.ink_bbox = ink_extent(_gray, sp.bbox)
        except Exception:
            logger.warning("Ink extent measurement failed", exc_info=True)
        result.spans = spans
        result.analysed_image = working
        result.ocr_stats = dict(getattr(self.ocr, "last_stats", {}) or {})

        # ---- 4b. quality verdict, judged on what was actually read ----
        if quality_suspect:
            good = [sp for sp in spans if sp.confidence >= 0.8 and len(sp.text.strip()) >= 3]
            if len(good) < 3:
                result.image_quality_ok = False
                return self.engine.evaluate(result)
            result.quality_notes = (
                f"Photo flagged as {quality.notes}, but the text was read "
                f"({len(good)} clear lines). Retake only if something printed is missing.")

        # ---- 5. extraction ------------------------------------------
        declarations = self.extractor.extract(
            spans, image=working, calibration=calib
        )
        result.declarations = declarations

        # ---- 5b. second reader for what OCR missed (optional) --------
        if self.vlm is not None:
            from ..extraction.vlm_recover import recover_with_vlm

            result.ocr_stats["vlm"] = recover_with_vlm(
                self.vlm, read_image, declarations, spans, self.extractor, calib)

        # ---- 6. classification (BEFORE the rules engine) ------------
        # Re-detect the marker on the WORKING image: after rectification
        # the original corner coordinates no longer correspond to this
        # frame, and the panel detector needs to mask the marker out in
        # the same space it is searching.
        from ..vision.calibration import detect_markers

        try:
            _dets = detect_markers(working)
            marker_corners = (
                max(_dets, key=lambda d: d.area_px).corners if _dets else None
            )
        except Exception:
            marker_corners = None

        ctx = self.classifier.classify(working, declarations, spans, marker_corners)

        # ---- 6a. scale without a marker (inspector-supplied size) ----
        if not calib.available and (panel_size_mm or reference):
            calib, area = _user_scale(ctx, panel_size_mm, reference)
            result.calibration = calib
            if calib.available:
                _apply_scale_to_glyphs(declarations, calib)
                if area:
                    ctx.pdp_area_cm2 = area

        if ctx.pdp_area_cm2 is None and calib.available and ctx.pdp_bbox:
            from ..vision.panel import PanelDetection, panel_area_cm2

            det = PanelDetection(
                bbox=ctx.pdp_bbox,
                quad=getattr(ctx, "pdp_quad", None),
                method=getattr(ctx, "pdp_detection_method", "text_extent"),
                confidence=getattr(ctx, "pdp_confidence", 0.0),
            )
            # Only publish an area we actually believe. A low-confidence
            # detection leaves pdp_area_cm2 as None, and the rules engine
            # then abstains on the height check rather than applying a
            # Table-I band derived from a guess. Under-reporting the area
            # picks a smaller minimum height, so a wrong guess here fails
            # OPEN - undersized text would pass as compliant.
            if det.reliable:
                ctx.pdp_area_cm2 = panel_area_cm2(
                    det, calib, geometry=ctx.geometry.value
                    if ctx.geometry.value in ("rectangular", "cylindrical") else "rectangular"
                )
        # ---- 6b. overlay detection (Rule 6(3)) ----------------------
        from ..vision.overlay import detect_overlays

        try:
            result.overlays = detect_overlays(
                working, panel_bbox=ctx.pdp_bbox, marker_corners=marker_corners
            )
        except Exception:
            # NOT an empty list. An empty list means "the detector ran
            # and this package has no pasted label", which the sticker
            # rule reads as evidence of compliance - so a crash in the
            # detector was silently reported as a clean pass on a legal
            # check. Record the failure and let the engine abstain.
            logger.warning("Overlay detection failed", exc_info=True)
            result.overlays = None
            result.overlay_detection_failed = True

        # ---- 6c. printed vs stickered price -------------------------
        # A sticker that revises the price DOWNWARD is lawful; one that
        # raises it is not. Deciding that needs both numbers, so pull
        # every price-like token and split them by whether they fall
        # inside a detected overlay. The price on the sticker is the one
        # inside it; the manufacturer's printed price is the one outside.
        import re as _re

        # Price tokens, from line-level spans (PaddleOCR returns
        # "MRP Rs. 145.00 (inclusive of all taxes)" as ONE span) and from
        # split spans ("Rs." and "145.00" as neighbours on a row). This
        # used to handle only the split form, which is what Tesseract
        # produced; with Paddle no printed or sticker price was ever
        # found, so a sticker RAISING the price could not be caught.
        _PRICE_IN = _re.compile(
            r"(?:rs\.?|inr|\u20b9)\s*([0-9]{1,6}(?:\.[0-9]{1,2})?)", _re.I
        )
        _CUR = _re.compile(r"^(?:rs\.?|inr|\u20b9)$", _re.I)
        _NUM = _re.compile(r"^([0-9]{1,6}(?:\.[0-9]{1,2})?)$")

        def _row_text(anchor):
            return " ".join(
                o.text for o in spans
                if abs(o.bbox.cy - anchor.bbox.cy) <= max(anchor.bbox.h, 1) * 0.8
            ).lower()

        def _is_unit_price(txt: str) -> bool:
            # The unit sale price must never be compared with a sticker
            # MRP: a true verdict for the wrong reason is still wrong on
            # the report.
            return "unit" in txt or "per " in txt or "/g" in txt or "/kg" in txt

        def _price_tokens():
            for cur in spans:
                t = cur.text.strip()
                m = _PRICE_IN.search(t)
                if m and not _CUR.match(t):
                    if _is_unit_price(t.lower()):
                        continue
                    yield float(m.group(1)), cur.bbox
                    continue
                if not _CUR.match(t):
                    continue
                best, best_dx = None, 1e9
                for other in spans:
                    if other is cur or other.bbox.x < cur.bbox.x2 - 2:
                        continue
                    if abs(other.bbox.cy - cur.bbox.cy) > max(cur.bbox.h, 1) * 0.8:
                        continue
                    mm = _NUM.match(other.text.strip())
                    if not mm:
                        continue
                    dx = other.bbox.x - cur.bbox.x2
                    if 0 <= dx < best_dx:
                        best, best_dx = (float(mm.group(1)), other.bbox), dx
                if best and not _is_unit_price(_row_text(cur)):
                    yield best

        inside_vals, outside_vals = [], []
        for val, box in _price_tokens():
            in_overlay = any(
                o.bbox.contains(box, tol=6.0) or o.bbox.iou(box) > 0.3
                for o in (result.overlays or [])
            )
            (inside_vals if in_overlay else outside_vals).append(val)

        if inside_vals:
            ctx.sticker_price = max(inside_vals)
        if outside_vals:
            ctx.printed_price = max(outside_vals)

        # ---- 6d. barcode / GTIN -------------------------------------
        # Read from the ORIGINAL image: rectification resamples the fine
        # bar pattern and decoding rates drop sharply.
        from ..vision.barcode import read_barcode

        try:
            _bc = read_barcode(image)
            if _bc and _bc.trustworthy:
                ctx.barcode = _bc.value
                _barcode_bbox = _bc.bbox
        except Exception:
            pass

        # Rule 6(7): find the "GM" mark. Checked on the rectified view
        # AND on the original, because rectification resamples small
        # marks near the panel edge badly - a 3mm "GM" at the top of the
        # panel came back as "|". Geometry comes from whichever view
        # found it, so the top-of-panel test stays meaningful.
        from ..vision.overlay import find_top_token

        # BOTH calls sit inside the guard. The first was outside it, so
        # an exception there escaped the whole scan rather than being
        # confined to this one rule - a crash in an optional GM check
        # took down compliance evaluation for the entire package.
        # The second, full OCR pass on the original image is only worth
        # its cost when Rule 6(7) can actually apply. It used to run on
        # every scan without a "GM" mark - i.e. nearly every scan - which
        # doubled OCR time per photo for a rule that is NOT_APPLICABLE to
        # anything that is not genetically modified food.
        _gm_relevant = bool(
            (hints or {}).get("genetically_modified_food")
            or "GENETICALLY MODIFIED" in " ".join(sp.text for sp in spans).upper()
        )
        _present, _at_top = False, False
        try:
            _present, _at_top = find_top_token(spans, "GM", ctx.pdp_bbox)
            if not _present and _gm_relevant:
                # source=None: this pass must not overwrite the photo's
                # OCR dump with a second read of different pixels.
                _raw_spans = self.ocr.recognise(image, source=None)
                _present, _at_top = find_top_token(_raw_spans, "GM", None)
                if _present:
                    # No panel box in original coordinates, so fall back
                    # to the top third of the image itself.
                    _h = image.shape[0]
                    _gm_word = _re.compile(r"(?<![A-Za-z0-9])GM(?![A-Za-z0-9])", _re.I)
                    _at_top = any(
                        sp.bbox.cy <= _h * 0.33
                        for sp in _raw_spans
                        if _gm_word.search(sp.text)
                    )
        except Exception:
            # Leaving _present False would assert "no GM mark on the
            # package", i.e. a VIOLATION we never actually established.
            # Flag the failure and abstain instead.
            logger.warning("GM mark detection failed", exc_info=True)
            ctx.gm_detection_failed = True
        ctx.gm_mark_present = _present
        ctx.gm_mark_at_top = _at_top

        # Rule 6(7) only bites on GM food, so the flag has to be set
        # before the engine runs its exemption gate.
        # NOTE: whether a product IS genetically modified cannot be
        # inferred from whether it carries the "GM" mark - that is
        # circular, and it makes the one case the rule exists to catch
        # (GM food with no mark) undetectable by construction. In the
        # field this flag comes from the inspector or a product
        # database. `hints` lets either supply it.
        _text = " ".join(s.text for s in spans).upper()
        ctx.genetically_modified_food = bool(
            (hints or {}).get("genetically_modified_food")
            or "GENETICALLY MODIFIED" in _text
        )

        result.context = ctx

        # ---- 7. rules engine ----------------------------------------
        return self.engine.evaluate(result)

    # -----------------------------------------------------------------
    def scan_package(self, image_paths, coverage_complete: bool = True, **kw) -> ScanResult:
        """
        Several photos of ONE package (front, back, top, base, crimp ...),
        decided together. See core/package.py.
        """
        from .package import merge_package

        parts = [self.scan(str(p), **kw) for p in image_paths]
        return merge_package(parts, self.engine, coverage_complete=coverage_complete)

    # -----------------------------------------------------------------
    def scan_array(
        self,
        image: np.ndarray,
        inspector_id: Optional[str] = None,
    ) -> ScanResult:
        """In-memory variant, for the API path where nothing hits disk."""
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            cv2.imwrite(fh.name, image)
        try:
            return self.scan(fh.name, inspector_id=inspector_id)
        finally:
            try:
                os.unlink(fh.name)
            except OSError:
                pass
