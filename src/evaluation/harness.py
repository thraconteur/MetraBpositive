"""
Evaluation harness.

This produces the table you put on your final slide. It is the single
most important thing separating your submission from the ones that
demo a live scan and say "look, it works."

WHAT IT MEASURES
----------------
  1. Character error rate for OCR, per preprocessing configuration, so
     you can show each stage earns its place (the ablation table).
  2. Per-declaration extraction precision / recall / F1, so you can show
     WHICH fields are reliable rather than one averaged number that
     hides a field you always miss.
  3. Millimetre measurement MAE against ground truth, the number that
     validates your hardest claim.
  4. End-to-end violation detection precision / recall, WITH the
     abstention rate reported alongside.

WHY ABSTENTION MUST BE REPORTED
-------------------------------
A system that refuses to answer half the time can post a beautiful
precision figure. Reporting precision without the abstention rate is
the most common way to accidentally mislead a judging panel, and if
someone on that panel notices, everything else you claimed comes into
question. Report both, always, and treat the pair as the real result.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..core.schema import Outcome, ScanResult

# ---------------------------------------------------------------------
# String metrics
# ---------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = "".join((reference or "").split())
    hyp = "".join((hypothesis or "").split())
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


# ---------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------

@dataclass
class PRCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvaluationReport:
    n_images: int = 0
    n_abstained: int = 0
    extraction: dict[str, dict] = field(default_factory=dict)
    measurement_mae_mm: Optional[float] = None
    measurement_max_err_mm: Optional[float] = None
    measurement_n: int = 0
    measurement_within_tolerance: Optional[float] = None
    violation_detection: dict = field(default_factory=dict)
    per_rule_detection: dict[str, dict] = field(default_factory=dict)
    ocr_cer: Optional[float] = None
    calibration_success_rate: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def abstention_rate(self) -> float:
        return self.n_abstained / self.n_images if self.n_images else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["abstention_rate"] = round(self.abstention_rate, 4)
        return d

    def render_table(self) -> str:
        """Plain-text summary. Paste straight onto the slide."""
        L = []
        L.append("=" * 66)
        L.append("  COMPLIANCE PIPELINE EVALUATION")
        L.append("=" * 66)
        L.append(f"  Images evaluated            : {self.n_images}")
        L.append(f"  Abstention rate             : {self.abstention_rate*100:.1f}%")
        if self.calibration_success_rate is not None:
            L.append(f"  Calibration success         : {self.calibration_success_rate*100:.1f}%")
        if self.ocr_cer is not None:
            L.append(f"  OCR character error rate    : {self.ocr_cer*100:.2f}%")
        L.append("")

        if self.measurement_mae_mm is not None:
            L.append("  MILLIMETRE MEASUREMENT (vs ground truth)")
            L.append(f"    n                         : {self.measurement_n}")
            L.append(f"    mean absolute error       : {self.measurement_mae_mm:.4f} mm")
            L.append(f"    worst case                : {self.measurement_max_err_mm:.4f} mm")
            if self.measurement_within_tolerance is not None:
                L.append(f"    within 0.30 mm tolerance  : "
                         f"{self.measurement_within_tolerance*100:.1f}%")
            L.append("")

        if self.extraction:
            L.append("  DECLARATION EXTRACTION")
            L.append(f"    {'field':<24}{'P':>7}{'R':>7}{'F1':>7}{'n':>6}")
            for fid, m in sorted(self.extraction.items()):
                L.append(f"    {fid:<24}{m['precision']:>7.3f}"
                         f"{m['recall']:>7.3f}{m['f1']:>7.3f}"
                         f"{m['tp']+m['fn']:>6}")
            L.append("")

        if self.violation_detection:
            v = self.violation_detection
            L.append("  VIOLATION DETECTION (end to end)")
            L.append(f"    precision                 : {v['precision']:.3f}")
            L.append(f"    recall                    : {v['recall']:.3f}")
            L.append(f"    F1                        : {v['f1']:.3f}")
            L.append(f"    tp/fp/fn                  : {v['tp']}/{v['fp']}/{v['fn']}")
            L.append("")

        if self.per_rule_detection:
            L.append("  VIOLATION DETECTION BY RULE")
            L.append(f"    {'rule':<28}{'P':>7}{'R':>7}{'n':>6}")
            for rid, m in sorted(self.per_rule_detection.items()):
                L.append(f"    {rid:<28}{m['precision']:>7.3f}"
                         f"{m['recall']:>7.3f}{m['tp']+m['fn']:>6}")
            L.append("")

        if self.notes:
            L.append("  NOTES")
            for n in self.notes:
                L.append(f"    - {n}")
        L.append("=" * 66)
        return "\n".join(L)


# ---------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------

# Maps an injected synthetic violation to the rule ids that should fire.
VIOLATION_TO_RULES = {
    "numeral_height":        {"numeral_height"},
    "missing_mrp":           {"retail_sale_price.presence"},
    "missing_consumer_care": {"consumer_care.presence"},
    "missing_manufacturer":  {"manufacturer_details.presence"},
    "missing_date":          {"manufacture_date.presence"},
    "mrp_no_tax_qualifier":  {"retail_sale_price.required_phrasing"},
    "banned_phrase":         {"net_quantity.banned_phrase"},
    "quantity_clear_space":  {"quantity_clear_space"},
    "glyph_aspect_ratio":    {"glyph_aspect_ratio.net_quantity"},
    # Rules added from later amendments. Without these entries the
    # injected kind maps to no rule, the detection is never scored, and
    # the rule silently contributes nothing to precision or recall - it
    # simply looks as though it was never tested.
    "sticker_over_mrp":      {"sticker_alteration"},
    "sticker_higher_price":  {"sticker_alteration"},
    "gm_not_declared":       {"gm_food_label"},
    "gm_not_at_top":         {"gm_food_label"},
}


class Evaluator:
    """
    Scores pipeline output against a ground-truth manifest.

    The manifest schema is the one emitted by synth/generator.py. When
    you replace synthetic images with real annotated photographs, emit
    the same schema and every metric here keeps working unchanged - that
    is the whole reason the schema is fixed rather than ad hoc.
    """

    def __init__(self, tolerance_mm: float = 0.30):
        self.tolerance_mm = tolerance_mm

    def evaluate(
        self,
        scans: Iterable[tuple[ScanResult, dict]],
    ) -> EvaluationReport:
        report = EvaluationReport()
        extraction: dict[str, PRCounts] = {}
        per_rule: dict[str, PRCounts] = {}
        overall = PRCounts()
        errors_mm: list[float] = []
        cers: list[float] = []
        calibrated = 0

        for scan, truth in scans:
            report.n_images += 1
            if scan.abstained:
                report.n_abstained += 1
            if scan.calibration.available:
                calibrated += 1

            gt_fields = {f["field_id"]: f for f in truth.get("fields", [])}

            # ---- extraction ------------------------------------------
            for decl in scan.declarations:
                fid = decl.field_id
                counts = extraction.setdefault(fid, PRCounts())
                expected = fid in gt_fields
                if decl.present and expected:
                    counts.tp += 1
                elif decl.present and not expected:
                    counts.fp += 1
                elif not decl.present and expected:
                    counts.fn += 1

            # ---- OCR CER ---------------------------------------------
            for decl in scan.declarations:
                gt = gt_fields.get(decl.field_id)
                if gt and decl.present and decl.raw_text:
                    cers.append(character_error_rate(gt["text"], decl.raw_text))

            # ---- millimetre measurement ------------------------------
            for decl in scan.declarations:
                gt = gt_fields.get(decl.field_id)
                if not (gt and decl.glyph and decl.glyph.cap_height_mm is not None):
                    continue
                errors_mm.append(abs(decl.glyph.cap_height_mm - gt["cap_height_mm"]))

            # ---- violation detection ---------------------------------
            expected_rules: set[str] = set()
            for kind in truth.get("injected_violations", []):
                expected_rules |= VIOLATION_TO_RULES.get(kind, set())

            detected_rules = {
                f.rule_id for f in scan.findings if f.outcome is Outcome.VIOLATION
            }

            # Only score rules the harness knows how to map, so an
            # unrelated true violation in the synthetic label does not
            # count against detection of the injected one.
            scorable = set().union(*VIOLATION_TO_RULES.values())
            detected_scorable = detected_rules & scorable

            for rid in scorable:
                counts = per_rule.setdefault(rid, PRCounts())
                exp, det = rid in expected_rules, rid in detected_scorable
                if exp and det:
                    counts.tp += 1
                    overall.tp += 1
                elif det and not exp:
                    counts.fp += 1
                    overall.fp += 1
                elif exp and not det:
                    counts.fn += 1
                    overall.fn += 1

        # ---- assemble -------------------------------------------------
        report.extraction = {k: v.to_dict() for k, v in extraction.items()}
        report.violation_detection = overall.to_dict()
        report.per_rule_detection = {
            k: v.to_dict() for k, v in per_rule.items() if (v.tp + v.fn + v.fp) > 0
        }

        if errors_mm:
            report.measurement_n = len(errors_mm)
            report.measurement_mae_mm = statistics.mean(errors_mm)
            report.measurement_max_err_mm = max(errors_mm)
            report.measurement_within_tolerance = sum(
                1 for e in errors_mm if e <= self.tolerance_mm
            ) / len(errors_mm)

        if cers:
            report.ocr_cer = statistics.mean(cers)
        if report.n_images:
            report.calibration_success_rate = calibrated / report.n_images

        report.notes.append(
            "Millimetre ground truth is exact only for fronto-parallel synthetic "
            "renders. Re-measure on real photographs before quoting these figures."
        )
        report.notes.append(
            "Rules marked unverified in the config cannot emit violations, so their "
            "recall is structurally zero until the gazette values are transcribed."
        )
        return report


# ---------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------

def run_ablation(
    manifest_path: str | Path,
    configs: dict[str, dict],
    limit: Optional[int] = None,
) -> dict[str, EvaluationReport]:
    """
    Run the pipeline under several preprocessing configurations and
    report each one's metrics.

    This is what produces the "CER before and after each stage" table.
    If a stage does not move the number, cut it: it is costing you
    latency and live-demo risk for nothing, and being able to say you
    measured and removed it is a stronger answer than a longer pipeline.
    """
    from ..core.pipeline import CompliancePipeline, PipelineConfig

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    records = manifest["records"][:limit] if limit else manifest["records"]

    out: dict[str, EvaluationReport] = {}
    for name, cfg in configs.items():
        pipeline = CompliancePipeline(config=PipelineConfig(**cfg))
        pairs = []
        for rec in records:
            if not Path(rec["image_path"]).exists():
                continue
            pairs.append((pipeline.scan(rec["image_path"]), rec))
        out[name] = Evaluator().evaluate(pairs)
    return out
