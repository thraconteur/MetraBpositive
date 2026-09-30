"""
Core data model for the compliance pipeline.

Every stage of the pipeline speaks in these types. Keeping them in one
place means the OCR backend, the rules engine, the report builder and
the API never have to guess at each other's shapes.

Design note on `Finding`: the engine emits four outcomes, not two.
COMPLIANT / VIOLATION / INDETERMINATE / UNVERIFIED_RULE. Most teams
build a binary classifier and then have nothing sensible to say when
the image is bad or the rule isn't confirmed. The extra two states are
what make the output usable as evidence.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates of the rectified image."""
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    def contains(self, other: "BBox", tol: float = 0.0) -> bool:
        return (
            self.x - tol <= other.x
            and self.y - tol <= other.y
            and self.x2 + tol >= other.x2
            and self.y2 + tol >= other.y2
        )

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


# ---------------------------------------------------------------------
# OCR output
# ---------------------------------------------------------------------

@dataclass
class TextSpan:
    """One recognised piece of text with its location and confidence."""
    text: str
    bbox: BBox
    confidence: float = 1.0
    source_engine: str = "unknown"
    language: str = "en"
    # Text running top-to-bottom (an inkjet date code printed at 90
    # degrees). Kept off the horizontal line grouping, which would
    # otherwise splice it into whatever row shares its centre.
    vertical: bool = False
    # The box shrunk to the ink it contains (see vision/ink.py). OCR
    # detectors pad their boxes; gap measurements (Rule 8 clear space)
    # use this when it is available.
    ink_bbox: Optional[BBox] = None
    # Slope of the text line in degrees (0 = level). A tilted photo makes
    # the gaps between axis-aligned boxes meaningless (Rule 8 clear space).
    angle: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = self.bbox.to_dict()
        d["ink_bbox"] = self.ink_bbox.to_dict() if self.ink_bbox else None
        return d


# ---------------------------------------------------------------------
# Calibration and measurement
# ---------------------------------------------------------------------

class CalibrationMethod(str, Enum):
    FIDUCIAL = "fiducial_marker"
    AR_DEPTH = "ar_depth"
    USER_DIMENSION = "user_supplied_dimension"
    NONE = "none"


@dataclass
class Calibration:
    """
    Pixel-to-millimetre scale for ONE specific photograph.

    This is per-image, never global: the scale changes every time the
    user moves the camera. `px_per_mm` is only meaningful alongside the
    image it was derived from.
    """
    px_per_mm: Optional[float] = None
    method: CalibrationMethod = CalibrationMethod.NONE
    # Standard deviation of the estimate, in px/mm, where recoverable.
    uncertainty_px_per_mm: float = 0.0
    marker_id: Optional[int] = None
    marker_size_mm: Optional[float] = None
    notes: str = ""
    # 1.0 = the card seen square-on. Below ~0.95 the card is at an angle to
    # the camera - often flat on the TABLE beside an upright pack, whose
    # plane it does not share; mm checks then abstain.
    squareness: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.px_per_mm is not None and self.px_per_mm > 0

    def px_to_mm(self, px: float) -> Optional[float]:
        if not self.available:
            return None
        return px / self.px_per_mm

    def mm_uncertainty(self, px: float) -> float:
        """Propagate scale uncertainty into a millimetre error bar."""
        if not self.available or self.uncertainty_px_per_mm <= 0:
            return 0.0
        # d(px/s)/ds = -px/s^2
        return abs(px / (self.px_per_mm ** 2)) * self.uncertainty_px_per_mm

    def to_dict(self) -> dict:
        d = asdict(self)
        d["method"] = self.method.value
        return d


@dataclass
class GlyphMetrics:
    """
    Measured geometry of the characters in one declaration.

    cap_height_px is measured by connected-component analysis on the
    binarised glyph, NOT taken from an OCR bounding box. See
    vision/glyph.py for why that distinction matters.
    """
    cap_height_px: float = 0.0
    mean_width_px: float = 0.0
    cap_height_mm: Optional[float] = None
    cap_height_mm_uncertainty: float = 0.0
    width_over_height: Optional[float] = None
    n_glyphs_measured: int = 0
    measured_characters: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# Extraction output
# ---------------------------------------------------------------------

@dataclass
class Declaration:
    """
    One extracted mandatory declaration, bound to a field id from
    rules/lmpc_2011.yaml.
    """
    field_id: str
    raw_text: str = ""
    value: Any = None                 # parsed value (float price, qty tuple...)
    unit: Optional[str] = None
    bbox: Optional[BBox] = None
    spans: list[TextSpan] = field(default_factory=list)
    glyph: Optional[GlyphMetrics] = None
    extraction_confidence: float = 0.0
    present: bool = False
    # Anything an inspector should know about HOW the value was read,
    # e.g. "unit 'g' inferred from a '9' glyph". Shown on the report.
    notes: list[str] = field(default_factory=list)
    # Where the text came from: "ocr", or "vlm" (a vision-language model,
    # e.g. Gemini) - and "vlm+ocr" when OCR independently read the same
    # text in the same place. The rules engine treats a VLM-only read as
    # something to confirm, never as grounds for a verdict.
    source: str = "ocr"

    def to_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "raw_text": self.raw_text,
            "value": self.value,
            "unit": self.unit,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "glyph": self.glyph.to_dict() if self.glyph else None,
            "extraction_confidence": self.extraction_confidence,
            "present": self.present,
            "notes": list(self.notes),
            "source": self.source,
        }


# ---------------------------------------------------------------------
# Package classification
# ---------------------------------------------------------------------

class PackageGeometry(str, Enum):
    RECTANGULAR = "rectangular"
    CYLINDRICAL = "cylindrical"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class PackageContext:
    """
    What the classifier determined about this package BEFORE any rule
    was evaluated. Exemption resolution depends entirely on this.
    """
    commodity_category: str = "other"
    package_class: str = "retail"
    geometry: PackageGeometry = PackageGeometry.UNKNOWN
    is_imported: bool = False
    is_embossed: bool = False          # blown/formed/moulded/embossed printing
    other_law_applies: bool = False    # e.g. FSSAI-governed food
    for_export: bool = False
    # Set when the label prints "not for retail sale" - the exact phrase
    # Rule 2(bc) uses to define an institutional consumer's purchase.
    # Chapter II does not apply to such packages (Rule 3(c)), but that
    # exemption can only fire if something actually sets this.
    industrial_or_institutional: bool = False
    capacity_cc: Optional[float] = None
    pdp_area_cm2: Optional[float] = None
    pdp_bbox: Optional[BBox] = None
    classifier_confidence: float = 0.0
    # How the panel was found and how much we trust it. Surfaced in the
    # report so an inspector can see whether the area behind a font-size
    # ruling was measured or approximated.
    pdp_detection_method: str = "none"
    pdp_confidence: float = 0.0
    genetically_modified_food: bool = False
    gm_mark_present: bool = False
    gm_mark_at_top: bool = False
    gm_detection_failed: bool = False
    barcode: Optional[str] = None
    # Set when a package shows both a printed and a stickered price.
    printed_price: Optional[float] = None
    sticker_price: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["geometry"] = self.geometry.value
        d["pdp_bbox"] = self.pdp_bbox.to_dict() if self.pdp_bbox else None
        return d


# ---------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------

class Outcome(str, Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    INDETERMINATE = "INDETERMINATE"        # measured, but inside the abstain band
    UNVERIFIED_RULE = "UNVERIFIED_RULE"    # rule config not human-verified
    NOT_APPLICABLE = "NOT_APPLICABLE"      # exempted
    SUPPRESSED = "SUPPRESSED"              # Rule 33 relaxation on record


class Severity(str, Enum):
    # Upper case, like Outcome, so the app and the API see one convention.
    # The rules YAML and older stored scans say "major": _missing_ maps any
    # case to the same member.
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value == value.strip().upper():
                    return member
        return None


class PresenceState(str, Enum):
    """
    What the system can say about whether a declaration is on the pack.

    Kept apart from the finding outcome on purpose: one photo that does
    not show a declaration is NOT_VISIBLE, and only a scan that covers
    every panel can make that CONFIRMED_ABSENT.
    """
    PRESENT = "PRESENT"                     # read on the pack
    NOT_VISIBLE = "NOT_VISIBLE"             # not in the photos taken so far
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"   # all panels covered, not there


@dataclass
class Finding:
    """
    The atomic unit of output. One rule evaluated against one package.

    Everything an inspector needs to act is on this object: what rule,
    what was measured, what was required, the evidence crop, and how
    confident the system is. If a field here is empty, the report will
    be missing something a human needs.
    """
    rule_id: str
    citation: str = ""
    outcome: Outcome = Outcome.INDETERMINATE
    severity: Severity = Severity.MINOR
    message: str = ""
    field_id: Optional[str] = None
    measured_value: Optional[float] = None
    required_value: Optional[float] = None
    unit: str = ""
    uncertainty: float = 0.0
    evidence_bbox: Optional[BBox] = None
    evidence_crop_path: Optional[str] = None
    confidence: float = 0.0
    rule_verified: bool = True
    exemption_applied: Optional[str] = None
    # Set by an inspector in the UI, never by the model.
    human_override: Optional[str] = None
    human_note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["severity"] = self.severity.value
        d["evidence_bbox"] = self.evidence_bbox.to_dict() if self.evidence_bbox else None
        return d


@dataclass
class ScanResult:
    """Everything produced by one scan of one package."""
    scan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.UTC).isoformat())
    image_path: Optional[str] = None
    inspector_id: Optional[str] = None

    calibration: Calibration = field(default_factory=Calibration)
    context: PackageContext = field(default_factory=PackageContext)
    declarations: list[Declaration] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    # EVERY text span the OCR stage produced, not just the ones that
    # became declarations. Rule 8 asks whether the space around the net
    # quantity is free of "other printed matter" - and most other
    # printed matter (marketing copy, ingredients, a barcode caption) is
    # precisely the text that never becomes a declaration. Judging clear
    # space from declarations alone therefore reports a violation as
    # compliant, which is the wrong way to be wrong.
    spans: list[TextSpan] = field(default_factory=list)
    # Candidate pasted regions from vision/overlay.py. None means the
    # detector never ran; an empty list means it ran and found nothing.
    overlays: Optional[list] = None
    # Distinguishes "detector crashed" from "detector found nothing".
    # Both previously collapsed to an empty list, and an empty list is
    # read downstream as evidence of compliance.
    overlay_detection_failed: bool = False

    # Do we know this scan saw the WHOLE package? A single photograph
    # shows one face. Mandatory declarations are routinely split across
    # panels - the name and quantity on the front, the address, date and
    # consumer care on the back. So a field missing from one view is not
    # evidence it is missing from the PACKAGE, and reporting it as a
    # violation is a confident false accusation. Only a multi-image scan
    # covering every panel, or an explicit assertion by the inspector,
    # can establish absence.
    coverage_complete: bool = False
    # Set when this result decides a PACKAGE from several photos
    # (core/package.py): the photo names, in order.
    package_photos: list = field(default_factory=list)

    image_quality_ok: bool = True
    quality_notes: str = ""
    # Timing and cache info from the OCR engine for this scan (model load,
    # predict, refinement, cache hit). Diagnostic only.
    ocr_stats: dict = field(default_factory=dict)
    # The image every box on this result refers to (the photo after
    # marker rectification). In memory only - evidence crops are cut from
    # it; cutting them from the original photo put them in the wrong place
    # whenever a marker had rectified the frame.
    analysed_image: Any = field(default=None, repr=False, compare=False)
    pipeline_version: str = "0.3.0-scaffold"

    # -- convenience views -------------------------------------------

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.outcome is Outcome.VIOLATION]

    @property
    def indeterminate(self) -> list[Finding]:
        return [f for f in self.findings if f.outcome is Outcome.INDETERMINATE]

    @property
    def unverified(self) -> list[Finding]:
        return [f for f in self.findings if f.outcome is Outcome.UNVERIFIED_RULE]

    @property
    def is_compliant(self) -> Optional[bool]:
        """
        None means "cannot say" - which is a legitimate and important
        answer. Do not collapse this to False.
        """
        if not self.image_quality_ok:
            return None
        if self.violations:
            return False
        if self.indeterminate or self.unverified:
            return None
        return True

    @property
    def abstained(self) -> bool:
        return self.is_compliant is None

    def declaration(self, field_id: str) -> Optional[Declaration]:
        for d in self.declarations:
            if d.field_id == field_id:
                return d
        return None

    def presence_state(self, field_id: str) -> "PresenceState":
        d = self.declaration(field_id)
        if d is not None and d.present:
            return PresenceState.PRESENT
        return (PresenceState.CONFIRMED_ABSENT if self.coverage_complete
                else PresenceState.NOT_VISIBLE)

    @property
    def suppressed(self) -> list["Finding"]:
        return [f for f in self.findings if f.outcome is Outcome.SUPPRESSED]

    @property
    def escalatable(self) -> list["Finding"]:
        """
        Violations the ruleset marks as requiring escalation - critical
        and major severities. The `escalate` flag lived in the config
        and was read by nothing, so a report gave an inspector a count
        of violations but no signal about which ones actually demand
        action versus which are formatting defects.
        """
        return [
            f for f in self.violations
            if f.severity in (Severity.CRITICAL, Severity.MAJOR)
        ]

    def summary(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "compliant": self.is_compliant,
            "violations": len(self.violations),
            "indeterminate": len(self.indeterminate),
            "unverified_rules": len(self.unverified),
            "critical": sum(
                1 for f in self.violations if f.severity is Severity.CRITICAL
            ),
            # Surfaced so a dashboard can rank by what needs action, not
            # merely by how many boxes are unticked.
            "escalatable": len(self.escalatable),
            "suppressed_by_relaxation": len(self.suppressed),
        }

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "inspector_id": self.inspector_id,
            "calibration": self.calibration.to_dict(),
            "context": self.context.to_dict(),
            "declarations": [d.to_dict() for d in self.declarations],
            "findings": [f.to_dict() for f in self.findings],
            "image_quality_ok": self.image_quality_ok,
            "quality_notes": self.quality_notes,
            "summary": self.summary(),
            "pipeline_version": self.pipeline_version,
            # What the OCR engine actually read, persisted so a report
            # recalled from the database can still show it - the first
            # thing to check when a finding looks wrong.
            "spans": [s.to_dict() for s in self.spans],
            "ocr_stats": self.ocr_stats,
            "coverage_complete": self.coverage_complete,
            "package_photos": list(self.package_photos or []),
        }
