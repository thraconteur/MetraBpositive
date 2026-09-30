"""
The rules engine.

Reads rules/lmpc_2011.yaml and rules/exemptions.yaml, evaluates a set of
extracted declarations against them, and emits Findings with citations.

Three properties worth preserving if you rewrite this:

  1. NO LEGAL THRESHOLD IS HARDCODED HERE. Every number comes from the
     YAML. If you find yourself typing "2.1" into this file, stop.

  2. UNVERIFIED RULES CANNOT FIRE. If a rule's `verified` flag is false,
     the engine emits UNVERIFIED_RULE instead of VIOLATION. This is what
     stops a placeholder number from becoming a false accusation on an
     inspector's report.

  3. EXEMPTIONS RESOLVE BEFORE EVALUATION. classify -> exempt -> evaluate.
     Filtering violations after the fact produces subtly different (and
     wrong) results, because an exempt field should never have been
     measured against a threshold in the first place.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import (
    BBox,
    Declaration,
    Finding,
    Outcome,
    PackageContext,
    ScanResult,
    Severity,
)

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[2] / "rules"


_NUMBER_UNITS = {"piece", "pieces", "pc", "pcs", "no", "nos", "number", "unit",
                 "units", "n", "u", "set", "pair"}

# "alll" was read on a real sachet; "incl. all taxes" and "inc. of all taxes"
# are printed forms too (letters only: "inclalltaxes", "incofalltaxes").
_TAX_LETTERS = re.compile(r"inc(?:l[a-z]{0,7})?(?:of)?(?:al+)tax|incl[a-z]{0,7}of(?:al+)?tax")


def _ocr_conf(decl) -> float:
    """How cleanly the declaration's own text was read (0-1)."""
    if decl is None:
        return 0.0
    if decl.spans:
        return sum(sp.confidence for sp in decl.spans) / len(decl.spans)
    return min(1.0, (decl.extraction_confidence or 0.0) / 0.92)


# Absence of a word in OCR TEXT is only evidence of its absence on the
# PACK when that text was read cleanly. Below this, "not found in the
# text" is reported as something to confirm, never as a violation.
CLEAN_READ = 0.88


def _qualifier_resemblance(scan) -> tuple[bool, float, str]:
    """
    Does any read line look like a garbled tax qualifier?

    Two bars, both in rules/lexicon.yaml: a general one, and a lower one
    for a line that also names the MRP - "For MRP Rs. (nd of al aes),
    Batch No." is a real Dabur bottle printing "(incl. of all taxes)" by
    reference, and scores 65 where unrelated lines score about 50.
    """
    from ..extraction import fuzzy as _fz

    t = _fz.lexicon().get("thresholds") or {}
    general = float(t.get("qualifier_resemblance", 70))
    on_mrp_line = float(t.get("qualifier_resemblance_mrp_line", 60))
    best = (False, 0.0, "")
    for sp in scan.spans or []:
        low = (sp.text or "").lower()
        if not re.search(r"\bof\b|\(", low):
            continue
        mrp_line = bool(re.search(r"\bm\.?\s*r\.?\s*p\b|retail\s+price", low))
        score = max(_fz.resembles(sp.text, tgt, 0)[1]
                    for tgt in ("(incl. of all taxes)", "inclusive of all taxes"))
        # "of all tares)" / "of all taxes)": the qualifier's second line,
        # its "(incl." on the line above not read (real MHM, Drolia packs).
        # Judged on its own, since the full phrase dilutes the score.
        if re.search(r"\bof\s+a", low):
            score = max(score, _fz.resembles(sp.text, "of all taxes)", 0)[1])
        # The qualifier's first words with the rest lost to a fold or the
        # edge of the photo: "MRP.7 10/-INCL. OF" (real Haldiram crease).
        if re.search(r"\bincl(?:usive|\.)?\s*\.?\s*of\W*$", low):
            score = max(score, 90.0)
        hit = score >= (on_mrp_line if mrp_line else general)
        if (hit, score) > (best[0], best[1]):
            best = (hit, score, sp.text)
    return best


_CODE_TOKENS = re.compile(
    r"\b(?:usp|exp|pkd|mfd|mfg|use\s*by)\b|"
    r"\d{1,2}\s*[/\-.]\s*(?:\d{1,2}|[a-z]{3})\s*[/\-.]\s*\d{2,4}|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*[/\-.']?\s*\d{2}\b|"
    r"per\s*(?:g|ml|kg|l)\b|/\s*(?:g|ml|kg|l)\b", re.I)


# What only a printed code looks like: a unit price, a clock time, a
# month-name date ("03 JUL/26"). A printed label's "Mfd: 03/2026" is not.
_STAMP_TOKENS = re.compile(
    r"\busp\b|\b\d{1,2}:\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*[/\-.']?\s*\d{2}\b", re.I)


def _coded_price_line(decl, scan=None) -> bool:
    """Is the MRP part of a printed code - sharing its line with a date, a
    unit price or a batch number, or (on a sparse photo of a base, top or
    crimp) sitting right next to such a line?"""
    line = " ".join(sp.text for sp in (decl.spans or [])) or decl.raw_text or ""
    m = re.search(r"m\.?\s*r\.?\s*p", line, re.I)
    rest = line[:m.start()] + " " + line[m.end():] if m else line
    if _CODE_TOKENS.search(rest):
        return True
    spans = list(getattr(scan, "spans", None) or [])
    if not spans or decl.bbox is None:
        return False
    b, h = decl.bbox, max(8.0, decl.bbox.h)
    # Sparse = few lines AROUND the price (a base, top or crimp). Counted
    # locally, so a package scan - many photos on one canvas - still sees
    # the can base as the sparse photo it is.
    photo = next((getattr(sp, "photo", None) for sp in (decl.spans or [])), None)
    if photo is not None:                 # package scan: this photo only
        spans = [sp for sp in spans if getattr(sp, "photo", None) == photo]
    local = [sp for sp in spans if abs(sp.bbox.cy - b.cy) <= 25 * min(h, 40.0)]
    if len(local) >= 15:
        return False
    for sp in spans:
        if any(sp is d for d in (decl.spans or [])):
            continue
        sb = sp.bbox
        near = (abs(sb.cy - b.cy) <= 3 * h
                and min(sb.x2, b.x2 + 3 * h) - max(sb.x, b.x - 3 * h) > 0)
        if near and _STAMP_TOKENS.search(sp.text or ""):
            return True
    return False


_FIELD_LABELS = {
    "retail_sale_price": r"\bm\.?\s*r\.?\s*p\b|\bm\.\s*r\.\s*\.|retail\s+(?:sale\s+)?price",
    # ...or a quantity printed with no label at all ("200 g" on its own)
    "net_quantity": r"net\.?\s*(?:qty|quantity|wt|weight|content|vol)|^\W*\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l)\W*$",
    "manufacture_date": (r"\b(?:mfg|mfd|pkd)\b(?!\.?\s*(?:&\s*\w+\.?\s*)?by)|packed\s+on|"
                         r"date\s+of\s+(?:mfg|manuf|pack|import)|year\s+of\s+import|import(?:ed)?\s*(?:date|on)|"
                         r"manufactur\w*\s+(?:on|date)"),
    "unit_sale_price": r"\busp\b|unit\s+sale\s+price",
    "manufacturer_details": (r"(?:manufactured|marketed|packed|mfd|mkt\w*|imported|distributed|distd)\.?\s*"
                             r"(?:(?:,|&|and)\s*\w+\.?\s*){0,2}by|"
                             r"(?<![a-z])(?:manufacturer|packer|importer|marketer|mfr)\s*[:\-]"),
    # the heading, not the word: "Sparkle Consumer Products Pvt Ltd" is a name
    "consumer_care": (r"consumer\s*(?:care|cell|services?|helpline|complaints?|relations|affairs)|"
                      r"customer\s*(?:care|services?|support)|toll\s*free|helpline"),
    "country_of_origin": r"country\s+of\s+origin",
}


def _qualifier_by_reference(scan, variants) -> Optional[str]:
    """A line like "For MRP (incl. of all taxes) ... see base of can"."""
    from ..extraction import fuzzy as _fz

    for sp in scan.spans or []:
        low = (sp.text or "").lower()
        if not re.search(r"\bfor\s*m\.?\s*r\.?\s*p\b|\bm\.?r\.?p\b.*\bsee\b", low):
            continue
        if _has_tax_qualifier(sp.text, variants):
            return sp.text
        tail = low[low.find("mrp"):] if "mrp" in low else low
        # A clear read only: "(inci. of all taxes)" scores 95; "(nd of al
        # aes)" (75) is left to the resemblance check, which holds it for a
        # human instead of passing it.
        if _fz.resembles(tail[:40], "mrp (incl. of all taxes)", 0)[1] >= 85:
            return sp.text
    return None


def _poorly_read_photos(scan, max_weak: float = 0.25) -> list[int]:
    """Photos where more than a quarter of the text lines were read with
    low confidence - absence of a declaration there proves nothing."""
    groups: dict[int, list[float]] = {}
    for sp in scan.spans or []:
        if len((sp.text or "").strip()) < 3:
            continue
        groups.setdefault(getattr(sp, "photo", 0) or 0, []).append(sp.confidence)
    poor = {k for k, v in groups.items()
            if v and sum(1 for c in v if c < 0.8) / len(v) > max_weak}
    # In a package, a photo that yielded (almost) no text - a can base whose
    # stamp was not detected - proves nothing about what is printed on it.
    for k in range(len(getattr(scan, "package_photos", None) or []) or 1):
        if len(groups.get(k, [])) < 3:
            poor.add(k)
    return sorted(poor)


def _label_seen(field_id: str, scan) -> bool:
    pat = _FIELD_LABELS.get(field_id)
    if not pat:
        return False
    rx = re.compile(pat, re.I)
    return any(rx.search(sp.text or "") for sp in (scan.spans or []))


def _hold_vlm_only(scan, findings):
    """
    A declaration only a vision-language model read (source "vlm") cannot
    carry a verdict: the model may have read words that are not printed.
    Its findings become INDETERMINATE, keeping the message and the crop so
    the officer can confirm in seconds. Reads OCR confirmed ("vlm+ocr")
    are left alone.
    """
    vlm_only = {d.field_id for d in scan.declarations
                if d.present and getattr(d, "source", "ocr") == "vlm"}
    if not vlm_only:
        return findings
    for f in findings:
        if f.field_id in vlm_only and f.outcome in (Outcome.COMPLIANT, Outcome.VIOLATION):
            f.message = (f"[Vision-model read, not confirmed by OCR - would be "
                         f"{f.outcome.value}] {f.message}")
            f.outcome = Outcome.INDETERMINATE
            f.confidence = min(f.confidence, 0.45)
    return findings


# Every form names the TAX: "Batteries not included" or "Marketed exclusively
# by" beside an MRP are not a statement about taxes (false violation).
_TAX_OBJ = r"\.?\s*(?:of\s+)?(?:all\s+)?(?:applicable\s+)?(?:local\s+)?tax"
_TAX_NEGATED = re.compile(
    r"(?:\bnot\s+incl\w*" + _TAX_OBJ + r"|\bexcl\w*" + _TAX_OBJ + r"|\bwithout\s+(?:any\s+)?tax|"
    r"\bplus\s+(?:all\s+)?(?:applicable\s+)?tax|\+\s*(?:all\s+)?(?:applicable\s+)?tax|"
    r"\btax(?:es)?\s+(?:extra|additional|not\s+included))", re.I)


def _tax_negated(decl, scan) -> Optional[str]:
    """ "(not inclusive of all taxes)", "(excl. of all taxes)", "+ taxes" on
    the price's own line: the pack says the price is NOT tax-inclusive."""
    texts = [decl.raw_text or ""]
    b = decl.bbox
    if b is not None:
        h = max(b.h, 1.0)
        for sp in scan.spans or []:
            sb = sp.bbox
            if sb.y2 >= b.y - 0.5 * h and sb.y <= b.y2 + 0.5 * h and sb.x2 >= b.x - 2 * h and sb.x <= b.x2 + 2 * h:
                texts.append(sp.text or "")
    for t in texts:
        m = _TAX_NEGATED.search(t)
        if m:
            return t.strip()
    return None


def _has_tax_qualifier(text: str, variants) -> bool:
    """
    'Inclusive of all taxes' in any of the printed forms, compared on
    letters only so OCR spacing / punctuation loss does not matter.
    Also accepts the Hindi form (सभी करों सहित).
    """
    if not text:
        return False
    low = text.lower()
    letters = re.sub(r"[^a-z]", "", low)
    if _TAX_LETTERS.search(letters):
        return True
    for v in variants or []:
        vl = re.sub(r"[^a-z]", "", v.lower())
        if vl and vl in letters:
            return True
    dev = re.sub(r"\s+", "", text)
    return "\u0915\u0930\u094b\u0902\u0938\u0939\u093f\u0924" in dev or "\u0915\u0930\u094b\u0938\u0939\u093f\u0924" in dev


# =====================================================================
# Config loading
# =====================================================================

class RuleConfig:
    """Parsed view over the two YAML files."""

    def __init__(self, rules_dir: Path | str = DEFAULT_RULES_DIR):
        self.rules_dir = Path(rules_dir)
        with open(self.rules_dir / "lmpc_2011.yaml", encoding="utf-8") as fh:
            self.rules: dict = yaml.safe_load(fh)
        with open(self.rules_dir / "exemptions.yaml", encoding="utf-8") as fh:
            self.exemptions_doc: dict = yaml.safe_load(fh)

        self.declarations: list[dict] = self.rules.get("declarations", []) or []
        self.presentation: list[dict] = (
            (self.rules.get("presentation", []) or [])
            # Rules added by later amendments live in their own block but
            # evaluate identically - keep them in one list so no rule is
            # silently skipped because of where it sits in the file.
            + (self.rules.get("additional_rules", []) or [])
        )
        self.measurement: dict = self.rules.get("measurement", {}) or {}
        self.exemptions: list[dict] = self.exemptions_doc.get("exemptions", []) or []
        # Fail at load, with the offending rule named, rather than
        # mid-scan with a bare ValueError.
        self.validate(strict=True)

    # -- lookups ------------------------------------------------------

    def declaration(self, field_id: str) -> Optional[dict]:
        return next((d for d in self.declarations if d["id"] == field_id), None)

    def presentation_rule(self, rule_id: str) -> Optional[dict]:
        return next((p for p in self.presentation if p["id"] == rule_id), None)

    def exemption(self, ex_id: str) -> Optional[dict]:
        return next((e for e in self.exemptions if e["id"] == ex_id), None)

    @property
    def abstain_band_mm(self) -> float:
        return float(
            self.measurement.get("uncertainty", {}).get("abstain_band_mm", 0.3)
        )

    # Keys a rule entry may legitimately carry. Anything else is almost
    # certainly a typo, and a typo'd key in YAML is silently ignored -
    # the same failure mode that hid every other bug in this ruleset.
    _KNOWN_RULE_KEYS = {
        "id", "label", "short_label", "citation", "verified", "source", "severity", "notes",
        "checks", "check", "applies_unless", "applies_only_if", "formulas",
        "excluded_surfaces", "table_I", "table_II", "other_law_carve_out",
        "medical_device_carve_out", "requires_calibration", "proviso",
        "effective_from", "keyed_by", "bands", "boundary_reading",
    }

    def validate(self, strict: bool = True) -> list[str]:
        """
        Check the ruleset for typos and invalid values BEFORE it is used.

        Two failures motivated this. An invalid `severity` value raised a
        bare ValueError in the middle of evaluating a scan, so one typo
        took down every request with a stack trace that named neither the
        file nor the rule. And an unknown key - `verifed` for `verified` -
        was silently dropped by the YAML loader, quietly changing the
        rule's meaning with no error anywhere.
        """
        problems: list[str] = []
        valid_sev = {"critical", "major", "minor"}

        for item in self.declarations + self.presentation:
            rid = item.get("id", "<no id>")
            if "id" not in item:
                problems.append("a rule entry has no 'id'")
            if "citation" not in item:
                problems.append(f"{rid}: missing 'citation'")
            if "verified" not in item:
                problems.append(f"{rid}: missing 'verified' flag")
            sev = item.get("severity")
            if sev is not None and sev not in valid_sev:
                problems.append(
                    f"{rid}: severity {sev!r} is not one of {sorted(valid_sev)}"
                )
            unknown = set(item) - self._KNOWN_RULE_KEYS
            if unknown:
                problems.append(
                    f"{rid}: unrecognised key(s) {sorted(unknown)} - likely a typo, "
                    f"and unrecognised keys are silently ignored"
                )

        for ex in self.exemptions:
            if "id" not in ex:
                problems.append("an exemption entry has no 'id'")
            if "condition" not in ex and "relaxation" not in ex:
                problems.append(f"{ex.get('id','<no id>')}: missing 'condition'")

        if problems and strict:
            raise ValueError(
                "Invalid ruleset:\n  - " + "\n  - ".join(problems)
            )
        return problems

    def audit(self) -> dict:
        """
        Report which rules are still placeholders. Run this in CI and in
        your pitch - "17 of 19 rules verified against the gazette" is a
        much stronger claim than silence.
        """
        verified, unverified = [], []
        for item in self.declarations + self.presentation:
            (verified if item.get("verified") else unverified).append(item["id"])
        for ex in self.exemptions:
            (verified if ex.get("verified") else unverified).append(f"exemption:{ex['id']}")
        return {
            "verified": sorted(verified),
            "unverified": sorted(unverified),
            "coverage": len(verified) / max(1, len(verified) + len(unverified)),
        }


# =====================================================================
# Exemption resolution
# =====================================================================

class ExemptionResolver:
    """
    Decides which declarations do not apply to this package.

    Runs BEFORE evaluation. Returns a mapping field_id -> exemption_id
    so the engine can mark those findings NOT_APPLICABLE with a reason,
    rather than silently omitting them (an inspector needs to see that
    a field was considered and excused, not just missing from the list).
    """

    def __init__(self, config: RuleConfig):
        self.config = config

    def resolve(
        self,
        context: PackageContext,
        declarations: list[Declaration],
    ) -> dict[str, str]:
        exempted: dict[str, str] = {}
        by_id = {d.field_id: d for d in declarations}

        for ex in self.config.exemptions:
            if not self._condition_met(ex, context, by_id):
                continue
            for field_id in ex.get("exempts_declarations", []) or []:
                # First matching exemption wins; record which one.
                exempted.setdefault(field_id, ex["id"])
        return exempted

    def size_rule_relaxation(
        self,
        context: PackageContext,
        declarations: list[Declaration],
    ) -> tuple[bool, set[str], Optional[dict]]:
        """
        Rule 7(5): where the same information is required under ANOTHER
        law, the size rules in Rule 7(1)-(4) do not apply - EXCEPT for
        net quantity, retail sale price, expiry/best-before and consumer
        care, which always keep their minimum heights.

        Returns (relaxed, never_exempt_fields, exemption).

        This matters more than it looks. `other_law_applies` is set for
        every packaged food, beverage and pharmaceutical - i.e. most of
        a supermarket - because those are governed by FSSAI and the
        Drugs rules. The exemption resolved correctly and then did
        nothing, because `relax_size_rules` was never read by any code.
        Every one of those packages was being measured against Table-I
        for fields the Rules expressly relax.
        """
        by_id = {d.field_id: d for d in declarations}
        for ex in self.config.exemptions:
            eff = ex.get("effect") or {}
            if not eff.get("relax_size_rules"):
                continue
            if not self._condition_met(ex, context, by_id):
                continue
            never = set(eff.get("never_exempt_fields") or [])
            return True, never, ex
        return False, set(), None

    def deferred_to_other_law(
        self,
        context: PackageContext,
        declarations: list[Declaration],
    ) -> Optional[dict]:
        """
        Exemptions carrying `defer_to`: medical devices (Medical Devices
        Rules 2017) and alcoholic beverages (State Excise law). The size
        and declaration regime of a different statute governs, so
        asserting an LMPC violation would be wrong.

        Alcoholic beverages additionally carry `action:
        flag_for_human_review`, because whether State Excise actually
        provides for a price declaration is jurisdiction-dependent and
        not decidable from a photograph.
        """
        by_id = {d.field_id: d for d in declarations}
        for ex in self.config.exemptions:
            eff = ex.get("effect") or {}
            if not eff.get("defer_to"):
                continue
            if self._condition_met(ex, context, by_id):
                return ex
        return None

    def chapter_ii_exemption(
        self,
        context: PackageContext,
        declarations: list[Declaration],
    ) -> Optional[dict]:
        """
        Is this package outside Chapter II's scope ENTIRELY - Rule 3's
        >25kg/25L packages, cement/fertiliser/farm produce over 50kg
        bags, and goods for industrial or institutional consumers?

        This is a SCOPE question, not a per-declaration exemption, and
        it needed its own path. `resolve()` only ever populates fields
        listed under `exempts_declarations`, so an exemption whose only
        effect is `chapter_ii_not_applicable: true` and an empty
        `exempts_declarations` list - which is exactly how
        industrial_institutional, over_25kg_25l and farm_produce were
        written - resolved its CONDITION correctly and then exempted
        nothing at all. The condition fired; the effect was inert. All
        three exemptions were reachable and silently did nothing.

        Returns the matching exemption dict, or None if Chapter II
        applies normally.
        """
        by_id = {d.field_id: d for d in declarations}
        for ex in self.config.exemptions:
            if not (ex.get("effect") or {}).get("chapter_ii_not_applicable"):
                continue
            if self._condition_met(ex, context, by_id):
                return ex
        return None

    def _condition_met(
        self,
        ex: dict,
        ctx: PackageContext,
        by_id: dict[str, Declaration],
    ) -> bool:
        cond = ex.get("condition") or {}
        ctype = cond.get("type")

        if ctype == "net_quantity_at_most":
            # excluded_commodities sits on the EXEMPTION, not inside its
            # condition block, because it is a carve-out FROM the
            # exemption rather than a term of the quantity test itself.
            # GSR 881(E) (2025) added exactly one: pan masala loses the
            # <=10g/10ml exemption regardless of its actual size. This
            # branch previously ignored that field entirely, so the one
            # amendment clause specifically transcribed for this
            # exemption was silently never enforced.
            if ctx.commodity_category in (ex.get("excluded_commodities") or []):
                return False
            d = by_id.get("net_quantity")
            if not d or d.value is None or d.unit is None:
                return False
            qty = _normalise_quantity(d.value, d.unit)
            if qty is None:
                return False
            if not _same_dimension(d.unit, cond.get("units", [])):
                return False
            return qty <= float(cond["value"])

        if ctype == "package_capacity_at_most_cc":
            return ctx.capacity_cc is not None and ctx.capacity_cc <= float(cond["value"])

        if ctype == "package_capacity_greater_than_cc":
            return ctx.capacity_cc is not None and ctx.capacity_cc > float(cond["value"])

        if ctype == "package_class":
            return ctx.package_class == cond.get("value")

        if ctype == "commodity_category":
            return ctx.commodity_category == cond.get("value")

        if ctype == "flag":
            return bool(getattr(ctx, cond.get("value", ""), False))

        if ctype == "bag_weight_above_kg":
            # Rule 3(b): cement, fertiliser, agricultural farm produce in
            # bags above 50 kg. Referenced in exemptions.yaml since the
            # farm_produce entry was written but this branch was never
            # implemented - the exemption existed on paper and could
            # never actually fire, for any package, regardless of what
            # the classifier detected.
            if ctx.commodity_category not in (cond.get("commodities") or []):
                return False
            d = by_id.get("net_quantity")
            if not d or d.value is None or d.unit is None:
                return False
            qty_g = _normalise_quantity(d.value, d.unit)
            unit_l = _base_unit(d.unit)
            if qty_g is None or unit_l not in ("g", "kg"):
                return False
            return (qty_g / 1000.0) > float(cond["value"])

        if ctype == "net_quantity_above":
            # Rule 3(a): packages above 25 kg or 25 L. Same gap as above
            # - referenced, never implemented.
            d = by_id.get("net_quantity")
            if not d or d.value is None or d.unit is None:
                return False
            if not _same_dimension(d.unit, cond.get("units", [])):
                return False
            qty = _normalise_quantity(d.value, d.unit)
            if qty is None:
                return False
            # cond["value"] is in kg/L; qty is normalised to g/ml.
            return qty > float(cond["value"]) * 1000.0

        if ctype == "computed":
            # Only the one expression we actually support, evaluated
            # explicitly. Never eval() a config string.
            if cond.get("expr") == "unit_sale_price == retail_sale_price":
                usp, mrp = by_id.get("unit_sale_price"), by_id.get("retail_sale_price")
                # A pack of exactly ONE unit sold by number ("NET QUANTITY
                # 1 PIECE"): the price per unit IS the retail sale price,
                # so the second proviso applies whether or not a separate
                # unit price is printed. Flagging a wrist watch for not
                # printing "Rs. 1495.00 per piece" under its MRP of
                # Rs. 1495.00 is a finding no inspector would sign.
                nq = by_id.get("net_quantity")
                if (nq is not None and nq.present and nq.value is not None
                        and float(nq.value) == 1.0
                        and (nq.unit or "").lower().rstrip(".") in _NUMBER_UNITS
                        and (usp is None or usp.value is None)):
                    return True
                if not usp or not mrp:
                    return False
                if usp.value is None or mrp.value is None:
                    return False
                return abs(float(usp.value) - float(mrp.value)) < 1e-6
            return False

        return False


# =====================================================================
# The engine
# =====================================================================

# Check types the declaration evaluator implements. Anything outside
# this set is reported as unevaluated rather than skipped in silence.
_IMPLEMENTED_DECLARATION_CHECKS = {
    "presence", "banned_phrases", "required_phrasing", "unit_is_si",
    "decimal_places", "contact_reachable_format", "price_rounding",
    "currency_present", "date_plausible", "structure", "price_consistency",
    "not_expired",
}


class RulesEngine:

    def __init__(
        self,
        config: Optional[RuleConfig] = None,
        relaxation_lookup=None,
    ):
        self.config = config or RuleConfig()
        self.resolver = ExemptionResolver(self.config)
        # Rule 33 lets a manufacturer apply for relaxation from specific
        # requirements. A granted relaxation is package-specific and
        # time-bound, so it lives in the database, not in the YAML - the
        # config says `lookup: database` and nothing ever read it. The
        # API created a `relaxations` table and the engine never queried
        # it, so a manufacturer holding a valid relaxation was still
        # reported as violating.
        #
        # Injected rather than hardwired to SQLite: the engine must stay
        # runnable with no database at all (tests, batch jobs), and a
        # missing lookup means "no relaxations on record", never a crash.
        self.relaxation_lookup = relaxation_lookup

    def _relaxation_for(self, scan: ScanResult, rule_id: str) -> Optional[dict]:
        if self.relaxation_lookup is None:
            return None
        decl = scan.declaration("manufacturer_details")
        manufacturer = decl.raw_text if decl and decl.present else None
        if not manufacturer:
            return None
        try:
            return self.relaxation_lookup(manufacturer, rule_id)
        except Exception:
            # A failing lookup must never turn into a false compliance
            # result; treat it as "no relaxation on record".
            return None

    def _apply_relaxations(self, scan: ScanResult, findings: list[Finding]) -> list[Finding]:
        """
        Suppress violations covered by a Rule 33 relaxation.

        The finding is RECORDED as suppressed, never dropped: an
        inspector needs to see that a relaxation was claimed, by whom,
        and under what reference. Silently removing it would hide the
        claim from the person who has to sign off on the report.
        """
        if self.relaxation_lookup is None:
            return findings

        # Rule 33(2): where the Medical Devices Rules 2017 apply, the
        # Rule 33 relaxation does not. Without this check a medical
        # device manufacturer holding any relaxation record could
        # suppress findings the Rules say they cannot relax - a hole
        # opened by implementing relaxations at all.
        deferred = self.resolver.deferred_to_other_law(
            scan.context, scan.declarations
        )
        if deferred and (deferred.get("effect") or {}).get("relaxation_unavailable"):
            return findings

        for f in findings:
            if f.outcome is not Outcome.VIOLATION:
                continue
            rec = self._relaxation_for(scan, f.rule_id)
            if not rec:
                continue
            f.outcome = Outcome.SUPPRESSED
            ref = rec.get("reference") or "unreferenced"
            until = rec.get("valid_until") or "no expiry recorded"
            f.message += (
                f" [Suppressed under Rule 33 relaxation {ref}, valid until {until}."
                f" Recorded for inspector review, not dismissed.]"
            )
            f.exemption_applied = "relaxation_rule_33"
        return findings

    # -----------------------------------------------------------------
    def evaluate(self, scan: ScanResult) -> ScanResult:
        """Populate scan.findings. Mutates and returns the scan."""
        findings: list[Finding] = []

        if not scan.image_quality_ok:
            findings.append(
                Finding(
                    rule_id="image_quality_gate",
                    citation="measurement.image_quality_gate",
                    outcome=Outcome.INDETERMINATE,
                    severity=Severity.MINOR,
                    message=(
                        "Image quality below threshold - retake requested. "
                        f"{scan.quality_notes}"
                    ),
                    confidence=1.0,
                )
            )
            scan.findings = findings
            return scan

        # Scope gate BEFORE anything else. If this package is outside
        # Chapter II entirely - a >25kg/25L pack, a 50kg+ cement or
        # fertiliser bag, or a package for an industrial/institutional
        # consumer - evaluating individual rules against it is
        # meaningless, and doing so risks producing a violation report
        # for a package the Rules never governed in the first place.
        ch2 = self.resolver.chapter_ii_exemption(scan.context, scan.declarations)
        if ch2 is not None:
            findings.append(
                Finding(
                    rule_id="chapter_ii_scope",
                    citation=ch2.get("citation", "Rule 3"),
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=Severity.MINOR,
                    message=(
                        f"Package is outside the scope of Chapter II "
                        f"({ch2.get('label', ch2['id'])}); no declaration "
                        f"or presentation rule in this chapter applies."
                    ),
                    exemption_applied=ch2["id"],
                    rule_verified=bool(ch2.get("verified", False)),
                    confidence=1.0,
                )
            )
            scan.findings = findings
            return scan

        exempted = self.resolver.resolve(scan.context, scan.declarations)

        # -- Rule 6: presence and content of each declaration ----------
        for spec in self.config.declarations:
            findings.extend(self._eval_declaration(spec, scan, exempted))

        # -- Rules 7-9: presentation --------------------------------
        for spec in self.config.presentation:
            if "check" not in spec:
                continue  # e.g. pdp_area_formula is data, not a check
            findings.extend(
                self._eval_presentation(spec, scan, exempted, findings)
            )

        findings = _hold_vlm_only(scan, findings)
        f2 = _second_mrp(scan)
        if f2 is not None:
            findings.append(f2)
        # The project's own safety rule, applied once for every check: a
        # rule not verified against the gazette text never asserts a
        # violation (it said so for banned phrases only; the expiry check
        # slipped through).
        for f in findings:
            if f.outcome == Outcome.VIOLATION and not f.rule_verified:
                f.outcome = Outcome.UNVERIFIED_RULE
                if f.rule_id == "expiry_date.not_expired":
                    f.message += (" Selling expired stock is an offence under food-safety law, "
                                  "not the LMPC Rules (Rule 6(1)(da) requires the date to be "
                                  "DECLARED) - shown for the inspector, not counted as an "
                                  "LMPC violation.")
                else:
                    f.message += " (Rule not yet verified against the gazette text - not asserted.)"
        scan.findings = self._apply_relaxations(scan, findings)
        return scan

    # -----------------------------------------------------------------
    def _eval_declaration(
        self,
        spec: dict,
        scan: ScanResult,
        exempted: dict[str, str],
    ) -> list[Finding]:
        field_id = spec["id"]
        citation = spec.get("citation", "")
        severity = Severity(spec.get("severity", "minor"))
        verified = bool(spec.get("verified", False))
        out: list[Finding] = []

        # applies_only_if gate (e.g. country_of_origin for imports)
        only_if = spec.get("applies_only_if") or []
        if "imported_goods" in only_if and not scan.context.is_imported:
            return [
                Finding(
                    rule_id=field_id,
                    citation=citation,
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=severity,
                    field_id=field_id,
                    message="Not an imported package.",
                    rule_verified=verified,
                    confidence=1.0,
                )
            ]

        if field_id in exempted:
            ex_id = exempted[field_id]
            ex = self.config.exemption(ex_id) or {}

            # Some exemptions cannot be resolved from a photograph.
            # Alcoholic beverages defer to State Excise law, and whether
            # that law provides for a price declaration is
            # jurisdiction-dependent - so the config marks them
            # `action: flag_for_human_review`. Auto-clearing them would
            # silently pass packages nobody actually checked; that key
            # existed and was never read.
            if ex.get("action") == "flag_for_human_review":
                return [
                    Finding(
                        rule_id=field_id,
                        citation=citation,
                        outcome=Outcome.INDETERMINATE,
                        severity=severity,
                        field_id=field_id,
                        message=(
                            f"{ex.get('label', ex_id)} ({ex.get('citation','')}): "
                            f"cannot be decided from the package alone - depends on "
                            f"the law of the manufacturing State. Referred for "
                            f"inspector review rather than cleared."
                        ),
                        exemption_applied=ex_id,
                        rule_verified=verified and bool(ex.get("verified", False)),
                        confidence=0.5,
                    )
                ]

            return [
                Finding(
                    rule_id=field_id,
                    citation=citation,
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=severity,
                    field_id=field_id,
                    message=f"Exempt: {ex.get('label', ex_id)} ({ex.get('citation','')}).",
                    exemption_applied=ex_id,
                    rule_verified=verified and bool(ex.get("verified", False)),
                    confidence=1.0,
                )
            ]

        decl = scan.declaration(field_id)

        for check in spec.get("checks", []) or []:
            ctype = check.get("type")
            # A check may carry its own verified flag (e.g. banned_phrases).
            check_verified = verified and bool(check.get("verified", True))
            check_citation = check.get("citation", citation)

            if ctype == "presence":
                present = bool(decl and decl.present)

                # Every side photographed, and the pack still names this
                # declaration ("MRP", "Mfg. Date", "For MRP ... see base") -
                # but no value was read: the value is printed and the OCR
                # missed it (a smudged sticker, an inkjet code). And a
                # common name is rarely labelled, so its absence is never
                # provable from text. Neither is a violation.
                poorly_read = _poorly_read_photos(scan) if (not present and scan.coverage_complete) else []
                if not present and scan.coverage_complete and poorly_read:
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.presence",
                            citation=check_citation,
                            outcome=Outcome.INDETERMINATE,
                            severity=severity,
                            field_id=field_id,
                            message=(
                                f"{spec.get('short_label') or spec['label']} not read on any "
                                f"photo, but {'photo ' + ', '.join(str(i + 1) for i in poorly_read) if len(poorly_read) < 9 else 'some photos'} "
                                f"had text that could not be read clearly (small, curved, "
                                f"glossy or in another script), so it may be printed there. "
                                f"Check by eye or retake those sides."
                            ),
                            confidence=0.5,
                            rule_verified=check_verified,
                        )
                    )
                    break
                if not present and scan.coverage_complete and (
                        field_id == "common_name" or _label_seen(field_id, scan)):
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.presence",
                            citation=check_citation,
                            outcome=Outcome.INDETERMINATE,
                            severity=severity,
                            field_id=field_id,
                            message=(
                                f"{spec.get('short_label') or spec['label']}: the pack "
                                f"refers to it, but its value was not read on any "
                                f"photo. Check by eye (stamped codes and small "
                                f"stickers are often unreadable to OCR)."
                                if field_id != "common_name" else
                                "Common name not identified from the text read "
                                "(it is rarely labelled). Check by eye."
                            ),
                            confidence=0.5,
                            rule_verified=check_verified,
                        )
                    )
                    break

                # Absence cannot be established from a partial view.
                if not present and not scan.coverage_complete:
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.presence",
                            citation=check_citation,
                            outcome=Outcome.INDETERMINATE,
                            severity=severity,
                            field_id=field_id,
                            message=(
                                f"{spec['label']} not visible in this image. A "
                                f"single view cannot establish that it is absent "
                                f"from the package - declarations are routinely "
                                f"split across panels. Capture the remaining "
                                f"panels to decide this."
                            ),
                            confidence=0.9 if scan.image_quality_ok else 0.4,
                            rule_verified=check_verified,
                        )
                    )
                    break

                out.append(
                    Finding(
                        rule_id=f"{field_id}.presence",
                        citation=check_citation,
                        outcome=Outcome.COMPLIANT if present else Outcome.VIOLATION,
                        severity=severity,
                        field_id=field_id,
                        # PRESENCE ONLY. The old wording ("Retail sale
                        # price (MRP), inclusive of all taxes present.")
                        # sat in the same report as "Price stated without
                        # the 'inclusive of all taxes' qualifier" and read
                        # as a contradiction. This row only says the
                        # declaration was LOCATED; its content is judged
                        # by the separate checks.
                        message=(
                            f"{spec.get('short_label') or spec['label']} located. "
                            f"(Presence only - its content is assessed by the "
                            f"other checks on this field.)"
                            if present
                            else f"{spec['label']} not found on the package."
                        ),
                        evidence_bbox=decl.bbox if decl else None,
                        # Confidence in the FINDING, not in an extraction
                        # that by definition did not happen. When a field
                        # is absent the declaration object still exists
                        # with extraction_confidence 0.0, so a critical
                        # violation was rendered as "Conf. 0.00" - which
                        # reads to an inspector as "the system is unsure",
                        # exactly inverting the meaning. Absence is a
                        # confident finding when the image was readable;
                        # it is the IMAGE we might doubt, not the result.
                        confidence=(
                            decl.extraction_confidence
                            if (present and decl)
                            else (0.9 if scan.image_quality_ok else 0.4)
                        ),
                        rule_verified=check_verified,
                    )
                )
                if not present:
                    # No point running content checks on an absent field.
                    break

            elif ctype == "banned_phrases" and decl and not _find_banned_phrase(
                    decl.raw_text, check.get("phrases", [])) and self._when_packed_finding(
                    check, scan, decl, field_id, check_citation, check_verified):
                out.append(self._when_packed_finding(check, scan, decl, field_id,
                                                     check_citation, check_verified))
            elif ctype == "banned_phrases" and decl:
                hit = _find_banned_phrase(decl.raw_text, check.get("phrases", []))
                allowed = self._conditional_phrase_allowed(check, scan.context, decl.raw_text)
                # "15 g (Approx. 40 tablets)" (real Tansukh jar): the word sits
                # in a bracket about a COUNT, and the declared 15 g is not
                # qualified. Whether that is Rule 12(6) is for an officer.
                full = " ".join(sp.text for sp in (decl.spans or [])) or decl.raw_text
                aside = hit and _qualifier_only_in_aside(full, hit, decl.value)
                if hit and not allowed and aside:
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.banned_phrase",
                            citation=check_citation,
                            outcome=Outcome.INDETERMINATE,
                            severity=Severity.MINOR,
                            field_id=field_id,
                            message=(
                                f"'{hit}' appears only in a bracketed note beside the "
                                f"declared quantity (\"{full.strip()[:60]}\"), not on "
                                f"the quantity itself. Check by eye whether it qualifies "
                                f"the net quantity."
                            ),
                            evidence_bbox=decl.bbox,
                            confidence=0.5,
                            rule_verified=check_verified,
                        )
                    )
                elif hit and not allowed:
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.banned_phrase",
                            citation=check_citation,
                            outcome=Outcome.VIOLATION if check_verified else Outcome.UNVERIFIED_RULE,
                            severity=Severity.MAJOR,
                            field_id=field_id,
                            message=(
                                f"Quantity declaration contains the qualifying "
                                f"expression '{hit}', which creates an exaggerated "
                                f"or inadequate impression of quantity."
                            ),
                            evidence_bbox=decl.bbox,
                            confidence=0.95,
                            rule_verified=check_verified,
                        )
                    )
                else:
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.banned_phrase",
                            citation=check_citation,
                            outcome=Outcome.COMPLIANT,
                            severity=Severity.MINOR,
                            field_id=field_id,
                            message="No prohibited qualifying expression found.",
                            confidence=0.9,
                            rule_verified=check_verified,
                        )
                    )

            elif ctype == "required_phrasing" and decl:
                # Matched on LETTERS ONLY. PaddleOCR returns whole lines
                # and routinely drops or merges the spaces and dots in
                # small print: "(Incl.of all taxes)", "Incl.ofalltaxes",
                # "INCL OF ALL TAXES)". A literal substring test missed
                # every one of those on a real Maggi sachet and reported
                # a missing qualifier that was plainly printed.
                negated = _tax_negated(decl, scan)
                ok = (not negated) and _has_tax_qualifier(decl.raw_text, check.get("any_of", []))
                elsewhere = (not ok) and (not negated) and any(
                    _has_tax_qualifier(sp.text, check.get("any_of", []))
                    for sp in (scan.spans or [])
                )
                looks_like, resemblance, resembling = (False, 0.0, "")
                if not ok and not elsewhere:
                    looks_like, resemblance, resembling = _qualifier_resemblance(scan)
                by_ref = None if ok else _qualifier_by_reference(scan, check.get("any_of", []))
                if negated:
                    outcome, conf = Outcome.VIOLATION, 0.85 if scan.image_quality_ok else 0.5
                    msg = (f"The price is stated as NOT inclusive of taxes: "
                           f"'{negated[:90]}'. The MRP must be inclusive of all taxes.")
                elif ok:
                    outcome, conf = Outcome.COMPLIANT, 0.9
                    msg = "Price carries the inclusive-of-all-taxes qualifier."
                elif by_ref:
                    # "For MRP (incl. of all taxes), Date of Packaging &
                    # Batch No, see top of the pack" - the qualifier is
                    # declared once, for the price stamped elsewhere.
                    outcome, conf = Outcome.COMPLIANT, 0.75
                    msg = (f"Qualifier declared by reference on the pack: "
                           f"'{by_ref.strip()[:90]}'.")
                elif looks_like:
                    # "For MRP Rs. (nd of al aes)" - a real bottle printing
                    # "(incl. of all taxes)" in 5-point type. Probably the
                    # qualifier, garbled by OCR; not ours to call either way.
                    outcome, conf = Outcome.INDETERMINATE, 0.5
                    msg = (f"A line that looks like the tax qualifier was read as "
                           f"'{resembling.strip()}' ({resemblance:.0f}% similar). "
                           f"Confirm on the pack that 'inclusive of all taxes' is printed.")
                elif _ocr_conf(decl) < CLEAN_READ:
                    outcome, conf = Outcome.INDETERMINATE, 0.5
                    msg = ("No 'inclusive of all taxes' qualifier was read, but the "
                           "price was read with low OCR confidence "
                           f"({_ocr_conf(decl):.2f}). Confirm on the pack.")
                elif elsewhere:
                    # The words are on the panel, just not on a line the
                    # extractor tied to the price (wrapped into another
                    # column, or read as a separate text block). Not a
                    # violation we can stand behind.
                    outcome, conf = Outcome.INDETERMINATE, 0.6
                    msg = ("An 'inclusive of all taxes' qualifier was read on the "
                           "panel but not on the price line. Confirm visually that "
                           "it belongs to the MRP.")
                elif _coded_price_line(decl, scan):
                    # An inkjet / crimp code: "L9 MRP ₹26; USP ₹0.10/mL"
                    # (Amul carton top), "MRP Rs.125/-" beside EXP (Monster
                    # base). Packs print the qualifier ONCE on the label -
                    # "For MRP (incl. of all taxes) ... see top of the pack"
                    # - and code the number where it is stamped. From the
                    # coded side alone its absence proves nothing.
                    outcome, conf = Outcome.INDETERMINATE, 0.5
                    msg = ("Price is in a printed code (with the date / unit price) "
                           "and the tax qualifier is not on this side. Packs usually "
                           "print 'For MRP (incl. of all taxes) ... see top/base/crimp' "
                           "on the label - photograph that side to decide.")
                else:
                    outcome, conf = Outcome.VIOLATION, 0.9 if scan.image_quality_ok else 0.5
                    msg = ("Price stated without the required "
                           "'inclusive of all taxes' qualifier.")
                out.append(
                    Finding(
                        rule_id=f"{field_id}.required_phrasing",
                        citation=check_citation,
                        outcome=outcome,
                        severity=severity,
                        field_id=field_id,
                        message=msg,
                        evidence_bbox=decl.bbox,
                        confidence=conf,
                        rule_verified=check_verified,
                    )
                )

            elif ctype == "unit_is_si" and decl and _unit_finding(decl, check_citation, check_verified):
                out.append(_unit_finding(decl, check_citation, check_verified))
            elif ctype == "unit_is_si" and decl:
                allowed = check.get("allowed_units", [])
                unit_ok = decl.unit in allowed or _base_unit(decl.unit or "") in allowed
                out.append(
                    Finding(
                        rule_id=f"{field_id}.unit_is_si",
                        citation=check_citation,
                        outcome=Outcome.COMPLIANT if unit_ok else Outcome.VIOLATION,
                        severity=Severity.MAJOR,
                        field_id=field_id,
                        message=(
                            f"Quantity declared in '{decl.unit}'."
                            if unit_ok
                            else f"Quantity unit '{decl.unit}' is not a permitted SI unit."
                        ),
                        evidence_bbox=decl.bbox,
                        confidence=0.9,
                        rule_verified=check_verified,
                    )
                )

            elif ctype == "decimal_places" and decl and decl.raw_text:
                # At most two places (rupees and paise): "0.071" is not rounded to
                # paise; "Rs. 25 per 100 g" and "₹250/kg" are whole rupees - a
                # "violation" for those was a false one on ordinary packs.
                want = int(check.get("exactly", 2))
                m = re.search(r"\d+\.(\d+)", decl.raw_text)
                ok = not m or len(m.group(1)) <= want
                out.append(
                    Finding(
                        rule_id=f"{field_id}.decimal_places",
                        citation=check_citation,
                        outcome=Outcome.COMPLIANT if ok else Outcome.VIOLATION,
                        severity=Severity.MINOR,
                        field_id=field_id,
                        message=(
                            "Unit sale price stated in rupees and paise."
                            if ok
                            else f"Unit sale price stated to {len(m.group(1))} decimal places; "
                                 f"rupees and paise take at most {want}."
                        ),
                        evidence_bbox=decl.bbox,
                        confidence=0.85,
                        rule_verified=check_verified,
                    )
                )

            elif ctype == "not_expired" and decl and decl.present:
                import datetime as _dt

                parts = [int(x) for x in str(decl.value).split("/")]
                if len(parts) == 3:
                    d, mo, yr = parts
                else:
                    import calendar

                    mo, yr = parts
                    d = calendar.monthrange(yr, mo)[1]      # "use by 03/2027" = end of month
                exp = _dt.date(yr, mo, d)
                today = today_date()
                expired = exp < today
                derived = any("Computed" in n or "earlier than the manufacture date" in n
                              for n in (decl.notes or []))
                out.append(Finding(
                    rule_id="expiry_date.not_expired", citation=check_citation,
                    outcome=(Outcome.COMPLIANT if not expired else
                             Outcome.INDETERMINATE if derived or _ocr_conf(decl) < CLEAN_READ
                             else Outcome.VIOLATION),
                    severity=severity, field_id="expiry_date",
                    message=(f"Use-by / best-before date {decl.value} has not passed "
                             f"(checked {today.isoformat()})." if not expired else
                             f"Pack is past its use-by / best-before date {decl.value} "
                             f"(checked {today.isoformat()}): expired stock on sale."
                             + (" The date was computed or read unclearly - confirm on the pack."
                                if derived or _ocr_conf(decl) < CLEAN_READ else "")),
                    evidence_bbox=decl.bbox, confidence=0.85 if not derived else 0.5,
                    rule_verified=check_verified))

            elif ctype == "price_consistency" and decl and decl.present:
                f = _price_consistency(scan, decl, check, severity, check_citation,
                                       check_verified)
                if f is not None:
                    out.append(f)

            elif ctype == "price_rounding" and decl and decl.raw_text:
                # Rule 6(1)(e) as amended 2017: price in rupees and paise
                # rounded to the nearest rupee or 50 paise. So .00 and
                # .50 are compliant and Rs.119.99 is not. Transcribed
                # from the gazette and then never implemented - the
                # engine's elif chain simply fell through on an unknown
                # check type and emitted no finding at all.
                allowed = check.get("allowed_paise_endings", [0, 50])
                m = re.search(r"(\d+)\.(\d{1,2})(?!\d)", decl.raw_text)
                if m:
                    paise = int(m.group(2).ljust(2, "0"))          # "49.3" = 30 paise
                    ok = paise in allowed
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.price_rounding",
                            citation=check_citation,
                            outcome=Outcome.COMPLIANT if ok else Outcome.VIOLATION,
                            severity=Severity.MINOR,
                            field_id=field_id,
                            message=(
                                "Price rounded to the nearest rupee or 50 paise."
                                if ok else
                                f"Price ends in {paise:02d} paise; the Rules require "
                                f"rounding to the nearest rupee or 50 paise."
                            ),
                            evidence_bbox=decl.bbox,
                            confidence=0.9,
                            rule_verified=check_verified,
                        )
                    )

            elif ctype == "currency_present" and decl:
                symbols = check.get("symbols", ["Rs", "Rs.", "INR", "\u20b9"])
                low = decl.raw_text.lower()
                ok = any(sym.lower() in low for sym in symbols) or bool(
                    re.search(r"\u0930\u0941|\u0930\u0942", decl.raw_text)  # रु / रू
                )
                # ABSENCE IS NOT ASSERTED. The rupee sign is the glyph OCR
                # handles worst: PaddleOCR drops it, or reads it as '?',
                # 'F', 'Z', 'R' or even a leading '7' / '2'. On a real
                # sachet printing "MRP \u20b910.00" this check reported a
                # MAJOR violation for a symbol that was there. A missing
                # currency in the TEXT is therefore reported as something
                # for the officer to confirm, never as a violation.
                out.append(
                    Finding(
                        rule_id=f"{field_id}.currency_present",
                        citation=check_citation,
                        outcome=Outcome.COMPLIANT if ok else Outcome.INDETERMINATE,
                        severity=Severity.MAJOR,
                        field_id=field_id,
                        message=(
                            "Price carries a currency indication."
                            if ok else
                            "No currency indication (Rs., INR or \u20b9) was read in "
                            "the price text. OCR frequently drops or misreads the "
                            "\u20b9 sign, so this is not asserted as a violation - "
                            "confirm on the pack. If \u20b9 is present, also check "
                            "the price was not misread with an extra leading digit."
                        ),
                        evidence_bbox=decl.bbox,
                        confidence=0.85 if ok else 0.5,
                        rule_verified=check_verified,
                    )
                )

            elif ctype == "date_plausible" and decl and decl.raw_text:
                # A future manufacture date is prima facie non-compliant.
                import datetime as _dt

                allow_future = bool(check.get("allow_future", False))
                # The extracted VALUE is the manufacture date. Re-parsing
                # raw_text picked the use-by date out of "03/2026 (B4) /
                # 03/2029" on a real bottle and called it "in the future".
                m = re.fullmatch(r"(\d{1,2})/(\d{4})", str(decl.value or "")) or \
                    re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{4})", decl.raw_text)
                if m:
                    mon, yr = int(m.group(1)), int(m.group(2))
                    now = _dt.datetime.now(_dt.UTC)
                    valid_month = 1 <= mon <= 12
                    future = (yr, mon) > (now.year, now.month)
                    ok = valid_month and (allow_future or not future)
                    out.append(
                        Finding(
                            rule_id=f"{field_id}.date_plausible",
                            citation=check_citation,
                            outcome=Outcome.COMPLIANT if ok else Outcome.VIOLATION,
                            severity=Severity.MAJOR,
                            field_id=field_id,
                            message=(
                                f"Date {mon:02d}/{yr} is plausible."
                                if ok else
                                f"Date {mon:02d}/{yr} is not plausible"
                                + (" (month out of range)." if not valid_month
                                   else " (dated in the future).")
                            ),
                            evidence_bbox=decl.bbox,
                            confidence=0.85,
                            rule_verified=check_verified,
                        )
                    )

            elif ctype == "structure" and decl:
                # A complete address under Rule 10 needs enough parts to
                # actually locate the manufacturer, not just a company
                # name. Counts commas plus any PIN code as components.
                need = int(check.get("min_components", 2))
                # Judge the address itself (the value), not the label, and
                # drop parts that locate nobody: a bare country ("KING
                # CORPORATION, INDIA" - real) or a web / e-mail address.
                addr = str(decl.value or decl.raw_text)
                parts = [x for x in re.split(r"[,\n]", addr) if x.strip()
                         and not re.fullmatch(r"\W*(?:india|bharat)\W*", x.strip(), re.I)
                         and not re.search(r"www\.|@|https?:", x, re.I)]
                # A 6-digit PIN, or the short postal-district form
                # "Kolkata-73" that older Kolkata / Mumbai packs print.
                has_pin = bool(re.search(r"\b\d{6}\b|\b\d{3}\s\d{3}\b|[A-Za-z]\s*-\s*\d{1,3}\b", addr))
                found = len(parts) + (1 if has_pin else 0)
                ok = found >= need
                clean_read = _ocr_conf(decl) >= CLEAN_READ
                out.append(
                    Finding(
                        rule_id=f"{field_id}.structure",
                        citation=check_citation,
                        outcome=(Outcome.COMPLIANT if ok else
                                 Outcome.VIOLATION if clean_read else
                                 Outcome.INDETERMINATE),
                        severity=severity,
                        field_id=field_id,
                        message=(
                            "Address appears to be a complete postal address."
                            if ok else
                            "Address appears incomplete - a complete postal "
                            "address is required so a consumer can locate the "
                            "manufacturer, packer or importer."
                            if clean_read else
                            "No complete postal address (PIN code, locality) was "
                            "read in the address block, which was read with low "
                            "OCR confidence. Confirm on the pack."
                        ),
                        evidence_bbox=decl.bbox,
                        confidence=0.7,
                        rule_verified=check_verified,
                    )
                )

            elif ctype == "contact_reachable_format" and decl:
                # REGRESSION FOUND ON A REAL PHOTO. The old pattern only
                # matched a single unbroken 10-digit run or exactly ONE
                # separator, which fails on the printed format almost
                # every Indian toll-free number actually uses: grouped
                # digits with a space or hyphen between EACH group, e.g.
                # "1800 103 1947" or "1800-103-1947" (4-3-4). Neither
                # matched, and this check fell back to whether an email
                # happened to survive in the same window - so a page
                # printing a perfectly legible toll-free number could
                # still be reported as lacking a usable phone.
                # Digits, not word boundaries, delimit it: OCR glues words
                # on ("ADAT1800-203-0515ORWRTE", real Paper Boat), and a
                # 14-digit FSSAI licence number is not a phone.
                has_phone = bool(re.search(
                    # "(020) 2567 8901": a bracketed STD code
                    r"(?<!\d)(?:\+?91[\s-]?)?\(?\d(?:[\s\-()]{0,2}\d){9,11}(?![\d])", decl.raw_text
                ))
                # OCR puts spaces round "@" and ".": "care @ brand . com"
                has_email = bool(re.search(r"[\w.+-]+\s?@\s?[\w-]+\s?\.\s?[a-z]{2,}", decl.raw_text, re.I))
                ok = has_phone or has_email
                # A number or address that stops short - "033-40",
                # "importinfo.mhm@" (real MHM pack, the rest under a thumb)
                # - was printed and cut off in the photo, not left out.
                truncated = not ok and (any("could not read" in n for n in (decl.notes or []))
                                        or bool(
                    re.search(r"[\w.+-]+\s?@(?!\s?[\w-]+\s?\.\s?[a-z]{2,})", decl.raw_text, re.I)
                    or re.search(r"(?<![\w.])\d{2,5}[\s-]\d{1,6}\b", decl.raw_text)))
                if ok:
                    msg = "Consumer care contact present ("
                    msg += ", ".join(
                        x for x, present in
                        [("phone", has_phone), ("email", has_email)] if present
                    ) + ")."
                else:
                    msg = (
                        "Consumer care details lack a phone number or email "
                        "in a recognisable format. If a phone or email is "
                        "visible on the pack, this may be an extraction "
                        "issue rather than an absent declaration - check "
                        "against the raw OCR text."
                    )
                clean_read = _ocr_conf(decl) >= CLEAN_READ and not truncated
                if truncated:
                    msg = ("Consumer care contact looks cut off in this photo "
                           f"(read: '{decl.raw_text[-60:]}'). Confirm the full "
                           "phone number or e-mail on the pack.")
                out.append(
                    Finding(
                        rule_id=f"{field_id}.contact_format",
                        citation=check_citation,
                        outcome=(Outcome.COMPLIANT if ok else
                                 Outcome.VIOLATION if clean_read else
                                 Outcome.INDETERMINATE),
                        severity=severity,
                        field_id=field_id,
                        message=msg,
                        evidence_bbox=decl.bbox,
                        confidence=0.85 if ok else 0.5,
                        rule_verified=check_verified,
                    )
                )


            elif ctype not in _IMPLEMENTED_DECLARATION_CHECKS:
                # An unrecognised check type used to fall straight
                # through this chain and emit nothing - the rule looked
                # implemented in the config, carried a citation, and
                # produced no finding whatsoever. Four real checks
                # (price_rounding, currency_present, date_plausible,
                # structure) sat dead this way. Fail loudly instead.
                out.append(
                    Finding(
                        rule_id=f"{field_id}.{ctype}",
                        citation=check_citation,
                        outcome=Outcome.UNVERIFIED_RULE,
                        severity=Severity.MINOR,
                        field_id=field_id,
                        message=(
                            f"Check type '{ctype}' is declared in the ruleset "
                            f"but not implemented by the engine, so this "
                            f"requirement was NOT evaluated."
                        ),
                        rule_verified=False,
                        confidence=0.0,
                    )
                )
        return out

    def _conditional_phrase_allowed(
        self, check: dict, ctx: PackageContext, text: str
    ) -> bool:
        for cp in check.get("conditional_phrases", []) or []:
            if cp["phrase"].lower() in text.lower():
                if ctx.commodity_category in (cp.get("allowed_for_categories") or []):
                    return True
        return False

    def _when_packed_finding(self, check, scan, decl, field_id, citation, verified):
        """ "(when packed)" after a net quantity is allowed only for the Third
        Schedule commodities - soaps, lotions, creams. It was silently
        accepted on anything (a namkeen pack). The product type is a guess
        from the label text, so a mismatch is held for the officer."""
        full = " ".join(sp.text for sp in (decl.spans or [])) + " " + (decl.raw_text or "")
        if not re.search(r"(?<![a-z])(?:when|hen|wen)\s*pa?c?ked\b", full, re.I):
            return None
        allow = [c for cp in check.get("conditional_phrases", []) or []
                 for c in (cp.get("allowed_for_categories") or [])]
        text = " ".join(sp.text for sp in (scan.spans or [])).lower()
        food = (scan.context.commodity_category or "") in ("food_packaged", "beverage_non_alcoholic")
        third = scan.context.commodity_category in allow or (
            not food and re.search(r"\bsoaps?\b|\blotions?\b|\bcreams?\b", text)
            and not re.search(r"ice\s*cream|biscuit|cookie|wafer|cream\s*(?:&|and)\s*herb", text))
        if third:
            return None
        return Finding(
            rule_id=f"{field_id}.banned_phrase", citation="Rule 11(4), Third Schedule",
            outcome=Outcome.INDETERMINATE, severity=Severity.MINOR, field_id=field_id,
            message=("'when packed' after the net quantity is permitted only for soaps, "
                     "lotions and creams (Third Schedule). This does not look like one of "
                     "them - check the product type."),
            evidence_bbox=decl.bbox, confidence=0.6, rule_verified=verified)

    # -----------------------------------------------------------------
    def _eval_presentation(
        self,
        spec: dict,
        scan: ScanResult,
        exempted: dict[str, str],
        findings_so_far: list[Finding] = (),
    ) -> list[Finding]:
        rule_id = spec["id"]
        citation = spec.get("citation", "")
        severity = Severity(spec.get("severity", "minor"))
        verified = bool(spec.get("verified", False))
        check = spec.get("check", {})
        ctype = check.get("type")

        # -- Rule 7(3): width >= 1/3 height. No calibration needed. ----
        if ctype == "aspect_ratio":
            return self._eval_aspect_ratio(spec, check, scan, verified, citation, severity)

        # -- Rule 8: clear space. No calibration needed. ---------------
        if ctype == "clear_space":
            return self._eval_clear_space(spec, check, scan, verified, citation, severity)

        # -- Rule 7(2): minimum height in mm. Needs calibration. -------
        if ctype == "min_height_mm":
            return self._eval_min_height(spec, scan, verified, citation, severity)

        # -- Rule 7: grouping on the PDP -------------------------------
        if ctype == "all_declarations_within_region":
            return self._eval_pdp_grouping(
                spec, scan, verified, citation, severity, findings_so_far
            )

        if ctype == "overlay_detection":
            return self._eval_sticker(spec, scan, verified, citation, severity)

        if ctype == "token_at_pdp_top":
            return self._eval_gm_label(spec, check, scan, verified, citation, severity)

        return []

    # -- Rule 6(3): sticker altering a declaration ---------------------

    def _eval_sticker(self, spec, scan, verified, citation, severity):
        """
        A sticker is unlawful only if it conceals the printed MRP or
        raises the price. The proviso permits a downward revision that
        leaves the original visible, so this must not fire on every
        pasted label - see vision/overlay.py.
        """
        from ..vision.overlay import assess_price_overlay

        if getattr(scan, "overlay_detection_failed", False):
            return [
                Finding(
                    rule_id="sticker_alteration",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id="retail_sale_price",
                    message=(
                        "Sticker detection failed on this image; whether a "
                        "pasted label conceals the price could not be "
                        "determined. Not asserting compliance."
                    ),
                    rule_verified=verified,
                    confidence=0.0,
                )
            ]

        overlays = getattr(scan, "overlays", None)
        if overlays is None:
            return []

        mrp = scan.declaration("retail_sale_price")
        mrp_bbox = mrp.bbox if mrp and mrp.present else None

        violation, reason, ov = assess_price_overlay(
            overlays,
            mrp_bbox,
            printed_price=getattr(scan.context, "printed_price", None),
            sticker_price=getattr(scan.context, "sticker_price", None),
            price_value_boxes=([s.bbox for s in mrp.spans if re.search(r"\d", s.text or "")]
                               if mrp and mrp.present else None),
            price_label_boxes=[sp.bbox for sp in (scan.spans or [])
                               if _MRP_LABEL_ONLY.search(sp.text or "")],
        )
        return [
            Finding(
                rule_id="sticker_alteration",
                citation=citation,
                outcome=Outcome.VIOLATION if violation else Outcome.COMPLIANT,
                severity=severity,
                field_id="retail_sale_price",
                message=reason,
                evidence_bbox=(ov.bbox if ov else mrp_bbox),
                confidence=(ov.confidence if ov else 0.7),
                rule_verified=verified,
            )
        ]

    # -- Rule 6(7): 'GM' at the top of the PDP -------------------------

    def _eval_gm_label(self, spec, check, scan, verified, citation, severity):
        """
        Reads facts the vision stage already established. Detection lives
        in the pipeline because the mark has to be looked for on both the
        rectified and the original image - rectification resamples a 3mm
        mark near the panel edge badly enough to turn "GM" into "|".
        """
        only_if = check.get("applies_only_if", []) or []
        if "genetically_modified_food" in only_if and not getattr(
            scan.context, "genetically_modified_food", False
        ):
            return [
                Finding(
                    rule_id="gm_food_label",
                    citation=citation,
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=severity,
                    message="Not declared as genetically modified food.",
                    rule_verified=verified,
                    confidence=1.0,
                )
            ]

        token = check.get("token", "GM")

        if getattr(scan.context, "gm_detection_failed", False):
            return [
                Finding(
                    rule_id="gm_food_label",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    message=(
                        f"Detection of the '{check.get('token', 'GM')}' mark "
                        f"failed on this image; absence could not be "
                        f"established. Not asserting a violation."
                    ),
                    rule_verified=verified,
                    confidence=0.0,
                )
            ]

        present = getattr(scan.context, "gm_mark_present", False)
        at_top = getattr(scan.context, "gm_mark_at_top", False)

        if not present:
            outcome = Outcome.VIOLATION
            msg = (f"Genetically modified food does not bear the mark "
                   f"'{token}' on the principal display panel.")
        elif not at_top:
            # A distinct finding from absence: the mark exists but is in
            # the wrong place, and an inspector needs to know which.
            outcome = Outcome.VIOLATION
            msg = (f"The mark '{token}' is present but not at the top of "
                   f"the principal display panel as required.")
        else:
            outcome = Outcome.COMPLIANT
            msg = f"'{token}' declared at the top of the principal display panel."

        return [
            Finding(
                rule_id="gm_food_label",
                citation=citation,
                outcome=outcome,
                severity=severity,
                message=msg,
                evidence_bbox=scan.context.pdp_bbox,
                confidence=0.8,
                rule_verified=verified,
            )
        ]

    # -- individual presentation checks --------------------------------

    def _eval_aspect_ratio(self, spec, check, scan, verified, citation, severity):
        out = []
        min_ratio = float(check.get("min_width_over_height", 1 / 3))
        for field_id in check.get("applies_to_fields", []):
            if field_id in exempt_set(scan, self.config):
                continue
            decl = scan.declaration(field_id)
            if not decl or not decl.glyph or decl.glyph.width_over_height is None:
                continue
            ratio = decl.glyph.width_over_height
            n_glyphs = decl.glyph.n_glyphs_measured

            # PLAUSIBILITY FLOOR. Printed text does not get narrower than
            # roughly 0.15 - even aggressively condensed type measured
            # 0.25 in controlled tests, and the rule's own threshold is
            # 0.33. A real photograph produced 0.08, i.e. characters
            # twelve times taller than wide, which is not text at all but
            # a merged stroke, a box spanning two lines, or noise. The
            # engine asserted a Rule 7(3) VIOLATION from that reading.
            # An implausible measurement is a failure to measure, not
            # evidence of an offence.
            if not _glyph_measurement_plausible(decl.glyph):
                out.append(
                    Finding(
                        rule_id=f"glyph_aspect_ratio.{field_id}",
                        citation=citation,
                        outcome=Outcome.INDETERMINATE,
                        severity=severity,
                        field_id=field_id,
                        measured_value=round(ratio, 4),
                        required_value=round(min_ratio, 4),
                        unit="width/height",
                        message=(
                            f"Text is only {decl.glyph.cap_height_px:.0f} px tall in "
                            f"this photo - too small to measure character shape "
                            f"reliably. Retake this panel closer."
                            if decl.glyph.cap_height_px < MIN_CAP_PX else
                            f"Character shape could not be measured reliably "
                            f"(ratio {ratio:.2f} from {n_glyphs} glyph(s)). "
                            f"Printed text does not fall below about "
                            f"{IMPLAUSIBLE_RATIO}, so this is a measurement "
                            f"failure rather than a finding. Retake this panel "
                            f"closer or in better focus."
                        ),
                        evidence_bbox=decl.bbox,
                        confidence=0.0,
                        rule_verified=verified,
                    )
                )
                continue

            ok = ratio >= min_ratio
            # Measurement tolerance: +-1 px on a glyph's width and height,
            # plus up to ~8% from the camera not being square to the pack
            # when there is no marker to rectify it. 0.32 against 0.33 on
            # a real Gulal pack is inside that - not a finding either way.
            h_px = max(1.0, decl.glyph.cap_height_px)
            w_px = max(1.0, ratio * h_px)
            tol = ratio * (1.0 / w_px + 1.0 / h_px)
            rectified = scan.calibration is not None and getattr(scan.calibration, "available", False)
            # The 8% also on a calibrated photo: a marker flattens the card's
            # plane, not a curved label (a can's edge narrowed an MRP to 0.27).
            tol += 0.08 * ratio
            # Unidentified blobs are trusted only on a marker-rectified
            # photo (the field workflow, validated on the synthetic set);
            # on a plain phone photo they have not earned a violation.
            unaligned = ("unaligned" in (decl.glyph.measured_characters or "")
                         and not rectified)
            if not ok and (ratio + tol >= min_ratio or unaligned):
                out.append(
                    Finding(
                        rule_id=f"glyph_aspect_ratio.{field_id}",
                        citation=citation,
                        outcome=Outcome.INDETERMINATE,
                        severity=severity,
                        field_id=field_id,
                        measured_value=round(ratio, 4),
                        required_value=round(min_ratio, 4),
                        unit="width/height",
                        uncertainty=round(tol, 3),
                        message=(
                            f"Character width/height ratio measured {ratio:.2f} "
                            f"against a minimum of {min_ratio:.2f}"
                            + (f" - within measurement tolerance (±{tol:.2f}). "
                               if ratio + tol >= min_ratio else
                               ", but from shapes that could not be matched to the "
                               "letters read (a narrow '1', 'l' or 't' looks the "
                               "same). ")
                            + "Not asserted either way; check by eye."
                        ),
                        evidence_bbox=decl.bbox,
                        confidence=0.4,
                        rule_verified=verified,
                    )
                )
                continue
            out.append(
                Finding(
                    rule_id=f"glyph_aspect_ratio.{field_id}",
                    citation=citation,
                    outcome=Outcome.COMPLIANT if ok else Outcome.VIOLATION,
                    severity=severity,
                    field_id=field_id,
                    measured_value=round(ratio, 4),
                    required_value=round(min_ratio, 4),
                    unit="width/height",
                    message=(
                        f"Character width/height ratio {ratio:.2f} meets the "
                        f"one-third minimum."
                        if ok
                        else f"Character width/height ratio {ratio:.2f} is below "
                             f"the required minimum of {min_ratio:.2f}."
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=0.85,
                    rule_verified=verified,
                )
            )
        return out

    def _eval_clear_space(self, spec, check, scan, verified, citation, severity):
        field_id = check.get("target_field", "net_quantity")
        decl = scan.declaration(field_id)
        if not decl or not decl.bbox or not decl.glyph or decl.glyph.cap_height_px <= 0:
            return []

        if not _glyph_measurement_plausible(decl.glyph):
            return [
                Finding(
                    rule_id="quantity_clear_space",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id=field_id,
                    message=(
                        "Clear space could not be assessed: it is measured in "
                        "multiples of the numeral height, and that height was "
                        "not measured reliably on this image"
                        + (f" (numerals only {decl.glyph.cap_height_px:.0f} px "
                           f"tall - retake closer)."
                           if decl.glyph.cap_height_px < MIN_CAP_PX else ".")
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=0.0,
                    rule_verified=verified,
                )
            ]

        rectified = scan.calibration is not None and getattr(scan.calibration, "available", False)
        if "unaligned" in (decl.glyph.measured_characters or "") and not rectified:
            # Clear space is counted in numeral heights. When the blobs
            # could not be matched to the digits read, the "numeral
            # height" may be the small letters beside them (real Gulal
            # pack: 19.5 px measured on "100g" digits about 40 px tall),
            # which would pass or fail the check on a wrong yardstick.
            return [
                Finding(
                    rule_id="quantity_clear_space",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id=field_id,
                    message=(
                        "Clear space not assessed: it is measured in numeral "
                        "heights, and the numerals could not be picked out from "
                        "the other characters in this photo. Retake the panel "
                        "square-on and closer, or check by eye."
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=0.0,
                    rule_verified=verified,
                )
            ]

        margins = check.get("margins", {})
        h = decl.glyph.cap_height_px
        required_px = {k: float(v) * h for k, v in margins.items()}

        # Distance to the nearest other text in each direction.
        uncertain: set = set()
        extra_tol: dict = {}
        actual_px = _nearest_neighbour_gaps(decl, scan, uncertain, extra_tol)

        # Measurement tolerance. A gap is measured between ink edges that
        # blur and anti-alias over a pixel or two, and the numeral height
        # that the gap is divided by carries its own pixel of error. A
        # real Maggi sachet measured 0.97 numeral heights against a
        # required 1.0 - inside that error, so not a finding either way.
        tol_px = max(1.5, 0.08 * h)
        worst_side, worst_deficit = None, 0.0
        marginal = None
        for side, req in required_px.items():
            act = actual_px.get(side)
            if act is None:
                continue  # nothing on that side - clear by definition
            deficit = req - act
            if deficit <= 0:
                continue
            best_case = (act + tol_px + extra_tol.get(side, 0.0)) / max(1.0, h - 1.0)
            if best_case >= float(margins[side]):
                if marginal is None or deficit > marginal[1]:
                    marginal = (side, deficit)
                continue
            if deficit > worst_deficit:
                worst_deficit, worst_side = deficit, side

        unsure = [side for side in uncertain if side in required_px
                  and (actual_px.get(side) is None or actual_px[side] >= required_px[side])]
        if worst_side is None and unsure:
            return [
                Finding(
                    rule_id="quantity_clear_space",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id=field_id,
                    message=(
                        f"Other print sits close to the {', '.join(sorted(unsure))} of the "
                        f"net quantity declaration, but the gap could not be measured "
                        f"reliably (the ink edges merge with the background, or the "
                        f"print is tilted or curved in the photo). Check by eye, or retake square-on."
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=0.3,
                    rule_verified=verified,
                )
            ]

        if worst_side is None and marginal is not None:
            side = marginal[0]
            return [
                Finding(
                    rule_id="quantity_clear_space",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id=field_id,
                    measured_value=round(actual_px[side] / h, 2),
                    required_value=float(margins[side]),
                    unit="numeral heights",
                    uncertainty=round(tol_px / h, 2),
                    message=(
                        f"Clear space to the {side} of the net quantity declaration "
                        f"measures {actual_px[side]/h:.2f} numeral heights against a "
                        f"required {margins[side]} - within measurement tolerance "
                        f"(±{(tol_px + extra_tol.get(side, 0.0)) / h:.2f}"
                        f"{', the photo is tilted' if extra_tol.get(side) else ''}). "
                        f"Not asserted either way; check by eye."
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=0.5,
                    rule_verified=verified,
                )
            ]

        if worst_side is None:
            return [
                Finding(
                    rule_id="quantity_clear_space",
                    citation=citation,
                    outcome=Outcome.COMPLIANT,
                    severity=severity,
                    field_id=field_id,
                    message="Required clear space around the quantity declaration is present.",
                    evidence_bbox=decl.bbox,
                    confidence=0.85,
                    rule_verified=verified,
                )
            ]

        return [
            Finding(
                rule_id="quantity_clear_space",
                citation=citation,
                outcome=Outcome.VIOLATION,
                severity=severity,
                field_id=field_id,
                measured_value=round(actual_px[worst_side] / h, 2),
                required_value=float(margins[worst_side]),
                unit="numeral heights",
                message=(
                    f"Insufficient clear space to the {worst_side} of the net "
                    f"quantity declaration: {actual_px[worst_side]/h:.2f} numeral "
                    f"heights against a required {margins[worst_side]}."
                ),
                evidence_bbox=decl.bbox,
                confidence=0.8,
                rule_verified=verified,
            )
        ]

    def _eval_min_height(self, spec, scan, verified, citation, severity):
        """
        Rule 7(2) / Table-I. The only check that needs real-world scale.

        Emits UNVERIFIED_RULE while the table values are placeholders,
        which is the state this scaffold ships in.
        """
        decl = scan.declaration("net_quantity")
        if not decl or not decl.glyph:
            return []

        if not verified:
            return [
                Finding(
                    rule_id="numeral_height",
                    citation=citation,
                    outcome=Outcome.UNVERIFIED_RULE,
                    severity=severity,
                    field_id="net_quantity",
                    measured_value=decl.glyph.cap_height_mm,
                    required_value=None,
                    unit="mm",
                    uncertainty=decl.glyph.cap_height_mm_uncertainty,
                    message=(
                        "Numeral height measured but not evaluated: Table-I "
                        "thresholds in rules/lmpc_2011.yaml are placeholders. "
                        "Transcribe the gazette values and set verified: true."
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=0.0,
                    rule_verified=False,
                )
            ]

        if not scan.calibration.available or decl.glyph.cap_height_mm is None:
            return [
                Finding(
                    rule_id="numeral_height",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id="net_quantity",
                    message=(
                        "Cannot measure numeral height: no scale reference in "
                        "frame. Retake with the calibration card visible."
                    ),
                    evidence_bbox=decl.bbox,
                    confidence=1.0,
                    rule_verified=verified,
                )
            ]

        # Rule 7(5) / medical-device proviso: the size rules may not
        # apply to this field at all. Checked BEFORE looking up a
        # threshold, because measuring against a threshold that does not
        # govern the package is how a compliant pack gets a violation.
        relaxed, never_exempt, relax_ex = self.resolver.size_rule_relaxation(
            scan.context, scan.declarations
        )
        if relaxed and "net_quantity" not in never_exempt:
            return [
                Finding(
                    rule_id="numeral_height",
                    citation="Rule 7(5)",
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=severity,
                    field_id="net_quantity",
                    message=(
                        f"Size requirements relaxed: this information is also "
                        f"required under another law "
                        f"({relax_ex.get('label', relax_ex['id'])})."
                    ),
                    exemption_applied=relax_ex["id"],
                    rule_verified=verified,
                    confidence=1.0,
                )
            ]

        deferred = self.resolver.deferred_to_other_law(
            scan.context, scan.declarations
        )
        if deferred is not None:
            target = (deferred.get("effect") or {}).get("defer_to")
            return [
                Finding(
                    rule_id="numeral_height",
                    citation=deferred.get("citation", "Rule 7(2) proviso"),
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=severity,
                    field_id="net_quantity",
                    message=(
                        f"Numeral and letter height for this package are "
                        f"governed by {target}, not by Table-I."
                    ),
                    exemption_applied=deferred["id"],
                    rule_verified=bool(deferred.get("verified", False)),
                    confidence=1.0,
                )
            ]

        required = self._lookup_required_height(spec, scan, decl)
        if required is None:
            return [
                Finding(
                    rule_id="numeral_height",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    field_id="net_quantity",
                    message="No applicable Table band matched this package.",
                    confidence=0.5,
                    rule_verified=verified,
                )
            ]

        measured = decl.glyph.cap_height_mm
        band = max(self.config.abstain_band_mm, decl.glyph.cap_height_mm_uncertainty)

        why_not = _mm_abstain_reason(scan, decl)
        if why_not:
            return [Finding(
                rule_id="numeral_height", citation=citation, outcome=Outcome.INDETERMINATE,
                severity=severity, field_id="net_quantity", measured_value=round(measured, 3),
                required_value=required, unit="mm",
                message=f"Numeral height {measured:.2f} mm not judged: {why_not}",
                evidence_bbox=decl.bbox, confidence=0.5, rule_verified=verified)]

        # The panel area carries the same scale error (squared) plus a few
        # edge pixels: 99.2 cm2 read as 100.3 moved a pack into the next
        # Table-I band. A violation stands against the LOWEST requirement
        # the area could plausibly give, a pass against the HIGHEST.
        cal = scan.calibration
        rel = (cal.uncertainty_px_per_mm / cal.px_per_mm) if cal.px_per_mm else 0.05
        area = scan.context.pdp_area_cm2
        u_area = 2.0 * rel + 0.02
        req_lo = self._lookup_required_height(spec, scan, decl, area=area * (1 - u_area)) or required
        req_hi = self._lookup_required_height(spec, scan, decl, area=area * (1 + u_area)) or required
        aligned = "unaligned" not in (decl.glyph.measured_characters or "")

        if measured + band < req_lo:
            outcome, msg = Outcome.VIOLATION, (
                f"Numeral height {measured:.2f} mm is below the required "
                f"{req_lo:.2f} mm (uncertainty +/-{band:.2f} mm)."
            )
        elif measured - band > req_hi and aligned:
            outcome, msg = Outcome.COMPLIANT, (
                f"Numeral height {measured:.2f} mm meets the required "
                f"{req_hi:.2f} mm."
            )
        elif measured - band > req_hi:
            outcome, msg = Outcome.INDETERMINATE, (
                f"Characters {measured:.2f} mm tall were measured, but the digits "
                f"themselves could not be picked out of the line (a larger caption "
                f"may have been measured). Check the numerals by eye.")
        elif req_lo != req_hi:
            outcome, msg = Outcome.INDETERMINATE, (
                f"Numeral height {measured:.2f} mm; the panel area "
                f"({area:.0f} cm2 +/-{100 * u_area:.0f}%) is at a Table-I band edge, "
                f"so the requirement is {req_lo:.1f} or {req_hi:.1f} mm. Not asserting "
                f"a violation.")
        else:
            outcome, msg = Outcome.INDETERMINATE, (
                f"Numeral height {measured:.2f} mm is within measurement "
                f"uncertainty (+/-{band:.2f} mm) of the {required:.2f} mm "
                f"threshold. Not asserting a violation."
            )

        return [
            Finding(
                rule_id="numeral_height",
                citation=citation,
                outcome=outcome,
                severity=severity,
                field_id="net_quantity",
                measured_value=round(measured, 3),
                required_value=required,
                unit="mm",
                uncertainty=round(band, 3),
                message=msg,
                evidence_bbox=decl.bbox,
                confidence=0.8,
                rule_verified=verified,
            )
        ]

    def _lookup_required_height(self, spec, scan, decl, area=None) -> Optional[float]:
        """
        Table-I lookup, keyed to PRINCIPAL DISPLAY PANEL AREA.

        This used to branch on the unit: Table-I by net quantity for
        weight/volume goods, Table-II by PDP area for length/area/number
        goods. That branch is gone. G.S.R. 629(E) (2017) substituted
        Table-I to key on PDP area for everything and omitted Table-II
        outright.

        The practical consequence is that there is no longer a shortcut
        path where the threshold can be read off the declared net
        quantity. Every font-size evaluation now needs a real-world
        panel area, so calibration plus a fully visible PDP is a hard
        prerequisite rather than an optimisation.
        """
        col = (
            "min_height_mm_embossed"
            if scan.context.is_embossed
            else "min_height_mm_printed"
        )

        if area is None:
            area = scan.context.pdp_area_cm2
        if area is None:
            return None

        for b in spec.get("table_I", {}).get("bands", []) or []:
            cap = b.get("max_area_cm2")
            if cap is None or area <= float(cap):
                return b.get(col)
        return None

    def _eval_pdp_grouping(self, spec, scan, verified, citation, severity,
                           findings_so_far=()):
        pdp = scan.context.pdp_bbox
        if pdp is None:
            return []
        required = [
            d["id"]
            for d in self.config.declarations
            if d.get("severity") in ("critical", "major") and d.get("verified", False)
        ]
        found = {d.field_id for d in scan.declarations if d.present}

        # Only fields that actually APPLY to this package. A domestic
        # pack has no country-of-origin declaration to group, and an
        # exempt field is not missing - it is not required. Without this
        # the grouping check demanded declarations the Rules never asked
        # for and could never return COMPLIANT on an ordinary package.
        not_applicable = {
            f.field_id for f in findings_so_far
            if f.outcome in (Outcome.NOT_APPLICABLE, Outcome.SUPPRESSED)
            and f.field_id
        }
        missing = [
            f for f in required
            if f not in found and f not in not_applicable
        ]
        if missing:
            # Grouping only inspected fields that were FOUND, so it
            # reported "all declarations grouped on the panel" on the
            # same report that said those declarations were missing -
            # two findings flatly contradicting each other. You cannot
            # judge the arrangement of declarations you never located.
            return [
                Finding(
                    rule_id="pdp_grouping",
                    citation=citation,
                    outcome=Outcome.INDETERMINATE,
                    severity=severity,
                    message=(
                        "Cannot assess grouping: "
                        + ", ".join(missing)
                        + " not located in this image."
                    ),
                    evidence_bbox=pdp,
                    confidence=0.9,
                    rule_verified=verified,
                )
            ]

        outside = [
            d.field_id
            for d in scan.declarations
            if d.present and d.bbox and d.field_id in required and not pdp.contains(d.bbox, tol=4.0)
        ]
        ok = not outside
        # A close-up can hide the pack's edge and leave an inner printed box
        # as "the panel" (audit: false grouping violation). Only a boundary
        # that holds essentially all the text is trusted for a violation.
        trusted = (getattr(scan.context, "pdp_detection_method", "") == "contour"
                   and getattr(scan.context, "pdp_confidence", 0.0) >= 0.95)
        if not ok and not trusted:
            return [Finding(
                rule_id="pdp_grouping", citation=citation, outcome=Outcome.INDETERMINATE,
                severity=severity,
                message=("Declarations appear outside the detected panel ("
                         + ", ".join(outside) + "), but the panel's edge was not found "
                         "reliably. Check the grouping by eye."),
                evidence_bbox=pdp, confidence=0.5, rule_verified=verified)]
        return [
            Finding(
                rule_id="pdp_grouping",
                citation=citation,
                outcome=Outcome.COMPLIANT if ok else Outcome.VIOLATION,
                severity=severity,
                message=(
                    "All mandatory declarations are grouped on the principal display panel."
                    if ok
                    else "Declarations found outside the principal display panel: "
                         + ", ".join(outside)
                ),
                evidence_bbox=pdp,
                confidence=0.75,
                rule_verified=verified,
            )
        ]


# =====================================================================
# Helpers
# =====================================================================

_UNIT_ALIASES = {
    "gm": "g", "gms": "g", "gram": "g", "grams": "g", "g": "g",
    "kg": "kg", "kgs": "kg", "kilogram": "kg",
    "ml": "ml", "mls": "ml", "millilitre": "ml", "milliliter": "ml",
    "l": "l", "ltr": "l", "litre": "l", "liter": "l",
    "n": "N", "u": "U",
}



_MASS_UNITS = {"g", "kg"}
_VOLUME_UNITS = {"ml", "l"}


def _same_dimension(unit: str, limit_units: list[str]) -> bool:
    """
    Does `unit` measure the same physical dimension as the threshold?

    The previous check asked whether the unit STRING appeared in the
    threshold's unit list, so "26000 g" failed a ">25 kg" test purely
    because the label said grams - the exemption compared labels rather
    than quantities. A 26 kg sack was in scope or out of it depending on
    which unit the packer chose to print.
    """
    u = _base_unit(unit)
    limits = {_base_unit(x) for x in limit_units}
    if u in limits:
        return True
    if u in _MASS_UNITS and limits & _MASS_UNITS:
        return True
    if u in _VOLUME_UNITS and limits & _VOLUME_UNITS:
        return True
    return False

# Printed text does not get narrower than roughly 0.15 width/height -
# aggressively condensed type measured 0.25 in controlled tests, and the
# rule's own threshold is 0.33. A real photograph produced 0.08, i.e.
# characters twelve times taller than wide, which is a merged stroke or
# a box spanning two lines, not text.
IMPLAUSIBLE_RATIO = 0.15
MIN_GLYPHS = 3
MIN_CAP_PX = 16.0


def _label_curved(scan) -> bool:
    """Text lines on a curved surface (bottle, can) lean by different
    amounts; on a flat panel, even a tilted photo, they agree."""
    angs = []
    for sp in scan.spans or []:
        b = sp.bbox
        if b and b.w >= 3 * max(b.h, 1.0) and getattr(sp, "angle", None) is not None:
            angs.append(float(sp.angle))
    if len(angs) < 5:
        return False
    import statistics
    med = statistics.median(angs)
    return statistics.median(abs(a - med) for a in angs) > 2.0


def _mm_abstain_reason(scan, decl) -> Optional[str]:
    """Why a millimetre verdict cannot be trusted on this photo, or None."""
    cal = scan.calibration
    sq = getattr(cal, "squareness", None)
    if sq is not None and sq < 0.95:
        return (f"the calibration card was seen at an angle (squareness {sq:.2f}); it "
                f"must lie flat ON the label face, photographed square-on.")
    g = decl.glyph
    if g is not None and g.cap_height_px and g.cap_height_px < MIN_CAP_PX:
        return (f"the numerals are only {g.cap_height_px:.0f} px tall in the photo - too "
                f"few pixels to measure in millimetres. Retake closer.")
    if (g is not None and g.width_over_height and g.width_over_height > 1.1
            and "unaligned" not in (g.measured_characters or "")):
        return ("the digits measure wider than tall, so the card is probably not in the "
                "label's plane (card on the table, label upright).")
    if _label_curved(scan):
        return ("the label is curved (bottle / can): a flat card does not give its "
                "size, and the display-panel area of a cylinder needs its diameter.")
    return None


def _glyph_measurement_plausible(glyph) -> bool:
    """
    Did we actually measure glyphs, or produce a number from noise?

    Used by BOTH Rule 7(3) and Rule 8, because clear space is expressed
    in multiples of the numeral height - so the same bad cap height that
    yields an impossible width ratio also corrupts the clear-space
    denominator. Judging them independently let one rule abstain on a
    failed measurement while the other asserted a violation from it.
    """
    if glyph is None or not glyph.cap_height_px:
        return False
    if glyph.n_glyphs_measured < MIN_GLYPHS:
        return False
    # Below ~16 px a character is a handful of pixels: one pixel is 6%+
    # of its height, and neighbouring glyphs merge when binarised. On
    # real 720x1280 phone photos of a bottle the net quantity is ~11 px
    # tall and clear space measured 0.57 numeral heights where the print
    # shows about 1.1. Ask for a closer photo instead of a finding.
    if glyph.cap_height_px < MIN_CAP_PX:
        return False
    ratio = glyph.width_over_height
    if ratio is not None and ratio < IMPLAUSIBLE_RATIO:
        return False
    return True


_USP_BASIS = re.compile(
    r"(?:/|per)\s*(\d+(?:\.\d+)?)?\s*(kg|kgs|kilogram|gm|gms|g|ml|l|ltr|litre|liter)\b", re.I)


def today_date():
    """Today, or SIH_TODAY=YYYY-MM-DD - so a scored photo set gives the same
    result next month (a pack that expires later must not turn into a
    "false violation" in the tests)."""
    import datetime as _dt
    import os

    v = os.environ.get("SIH_TODAY")
    if v:
        try:
            return _dt.date.fromisoformat(v)
        except ValueError:
            pass
    return _dt.date.today()


def _price_consistency(scan, usp, check, severity, citation, verified):
    """
    MRP / net quantity against the printed unit sale price.

    On a real Troovy pack OCR read the MRP "65.00" as "55.00": the unit price
    "0.93/g" and "70 g" show at once that one of the three numbers is wrong.
    When the three disagree:
      * all three read cleanly and far apart (>10%) -> VIOLATION: the unit
        price printed on the pack is wrong;
      * otherwise -> INDETERMINATE, naming the numbers to check by eye.
    """
    mrp, qty = scan.declaration("retail_sale_price"), scan.declaration("net_quantity")
    if not (mrp and mrp.present and qty and qty.present):
        return None
    try:
        price, usp_v = float(mrp.value), float(usp.value)
    except (TypeError, ValueError):
        return None
    grams = _normalise_quantity(qty.value, qty.unit or "")
    if not grams or price <= 0 or usp_v <= 0:
        return None
    # The unit the USP is quoted per: "0.93/g", "per 100 g", "/kg", "/mL".
    m = _USP_BASIS.search(" ".join(sp.text for sp in (usp.spans or [])) or usp.raw_text or "")
    if m:
        per = float(m.group(1) or 1.0) * (1000.0 if _base_unit(m.group(2)) in ("kg", "l") else 1.0)
    else:
        # No basis printed next to it: assume per g / ml, else per kg / l,
        # whichever the number fits.
        per = 1.0 if abs(price / grams - usp_v) <= abs(price / grams * 1000 - usp_v) else 1000.0
    expected = price / grams * per
    diff = abs(expected - usp_v)
    tol = max(float(check.get("tolerance_abs", 0.011)),
              float(check.get("tolerance_rel", 0.02)) * expected)
    basis = ("g/ml" if per == 1 else f"{per:g} g/ml")
    detail = (f"MRP {price:g} / net quantity {qty.value:g} {qty.unit or ''} = "
              f"{expected:.3f} per {basis}; unit sale price read as {usp_v:g}.")
    if diff <= tol:
        return Finding(rule_id="unit_sale_price.price_consistency", citation=citation,
                       outcome=Outcome.COMPLIANT, severity=severity, field_id="unit_sale_price",
                       message=f"Unit sale price agrees with the MRP and net quantity. {detail}",
                       evidence_bbox=usp.bbox, confidence=0.9, rule_verified=verified)
    # The WEAKEST piece of each read counts, not the average: a label read
    # at 1.00 must not carry a value read at 0.90 ("55.00" for 65.00 on a
    # real Troovy pouch - one digit, and the three numbers disagree).
    def weakest(d):
        return min((sp.confidence for sp in d.spans), default=_ocr_conf(d))
    clean = all(weakest(d) >= 0.95 for d in (mrp, qty, usp)) and not any(
        d.notes for d in (mrp, qty, usp))
    if clean and diff > float(check.get("violation_rel", 0.10)) * expected:
        return Finding(rule_id="unit_sale_price.price_consistency", citation=citation,
                       outcome=Outcome.VIOLATION, severity=severity, field_id="unit_sale_price",
                       measured_value=round(usp_v, 3), required_value=round(expected, 3),
                       message=f"Unit sale price does not match the MRP and net quantity. {detail}",
                       evidence_bbox=usp.bbox, confidence=0.8, rule_verified=verified)
    return Finding(rule_id="unit_sale_price.price_consistency", citation=citation,
                   outcome=Outcome.INDETERMINATE, severity=severity, field_id="unit_sale_price",
                   measured_value=round(usp_v, 3), required_value=round(expected, 3),
                   message=(f"MRP, net quantity and unit sale price do not agree - one of them "
                            f"was probably misread. {detail} Check the three numbers on the pack."),
                   evidence_bbox=usp.bbox, confidence=0.5, rule_verified=verified)


def _base_unit(unit: str) -> str:
    return _UNIT_ALIASES.get((unit or "").strip().lower(), (unit or "").strip())


def _normalise_quantity(value: Any, unit: str) -> Optional[float]:
    """Convert to grams or millilitres so table bands can be compared."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = _base_unit(unit)
    if u == "kg":
        return v * 1000.0
    if u == "l":
        return v * 1000.0
    if u in ("g", "ml"):
        return v
    return None


def _qualifier_only_in_aside(text: str, phrase: str, value) -> bool:
    """Is every use of `phrase` inside brackets that do not hold the declared
    quantity? "15 g (Approx. 40 tablets)" -> True; "(approx.) 15 g" or
    "Net wt. approx 15 g" -> False."""
    low = (text or "").lower()
    groups = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\([^()]*\)", low)]
    num = None
    if isinstance(value, (int, float)):
        num = f"{value:g}"
    hits = [m.start() for m in re.finditer(rf"\b{re.escape(phrase.lower())}\b", low)]
    if not hits or not groups:
        return False
    for h in hits:
        g = next((g for g in groups if g[0] < h < g[1]), None)
        if g is None:
            return False
        inner = g[2]
        # The bracket must hold something else counted ("40 tablets"), and not
        # the declared number itself.
        if num and re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", inner):
            return False
        if not re.search(r"\d", inner):
            return False
        # ...and the quantity must stand outside the bracket.
        outside = low[:g[0]] + low[g[1]:]
        if num and not re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", outside):
            return False
    return True


_IMPERIAL = re.compile(r"\d\s*(fl\.?\s*oz|oz|lbs?|cc|pints?|gallons?|inch(?:es)?)(?![a-z])", re.I)
_METRIC = re.compile(r"\d\s*(?:g|gm|gms|kg|kgs|ml|mls|l|ltrs?|litres?|liters?|grams?)(?![a-z])", re.I)
_NONSTD_SYMBOL = {"gm": "g", "gms": "g", "grm": "g", "grms": "g", "gram": "g", "grams": "g",
                  "kgs": "kg", "ltr": "l", "ltrs": "l", "lt": "l", "lts": "l", "mls": "ml"}


def _unit_finding(decl, citation, verified) -> Optional[Finding]:
    """The unit as PRINTED (the declaration's unit is already normalised).
    None = the plain allowed-unit check decides."""
    raw = decl.raw_text or ""
    if _IMPERIAL.search(raw) and not _METRIC.search(raw):
        u = _IMPERIAL.search(raw).group(1)
        return Finding(rule_id="net_quantity.unit_is_si", citation=citation,
                       outcome=Outcome.VIOLATION, severity=Severity.MAJOR,
                       field_id="net_quantity",
                       message=f"Quantity declared in '{u}', not in a metric (SI) unit.",
                       evidence_bbox=decl.bbox, confidence=0.85, rule_verified=verified)
    m = re.search(r"\d\s*([A-Za-z]+)\.?(?![A-Za-z])", raw)
    tok = m.group(1).lower() if m else (decl.unit or "").lower()
    if tok in _NUMBER_UNITS and tok not in ("n", "u"):
        return Finding(rule_id="net_quantity.unit_is_si", citation=citation,
                       outcome=Outcome.INDETERMINATE, severity=Severity.MAJOR,
                       field_id="net_quantity",
                       message=(f"Declared by count ('{m.group(1) if m else tok}'). The Rules give "
                                f"the symbol N or U for goods sold by number; confirm whether "
                                f"this printed form is accepted."),
                       evidence_bbox=decl.bbox, confidence=0.6, rule_verified=verified)
    if tok in _NONSTD_SYMBOL and tok not in ("gram", "grams"):
        return Finding(rule_id="net_quantity.unit_is_si", citation=citation,
                       outcome=Outcome.INDETERMINATE, severity=Severity.MINOR,
                       field_id="net_quantity",
                       message=(f"Unit printed as '{m.group(1)}', not the SI symbol "
                                f"'{_NONSTD_SYMBOL[tok]}'. Confirm whether the inspecting "
                                f"office treats this as a unit violation."),
                       evidence_bbox=decl.bbox, confidence=0.6, rule_verified=verified)
    return None


_MRP_TAG = re.compile(r"(?<![A-Za-z])m\.?\s*r\.?\s*p\b\.?|max(?:imum)?\.?\s*retail\s+price", re.I)
_AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"(?!\s*(?:/\s*-?\s*[a-z]|per\b|%|\d|[a-z]))", re.I)


_AMOUNT_AFTER_TAG = re.compile(
    r"\W{0,4}(?:₹|rs\.?|inr)?\s*[:.\-]?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"(?![\d.,]|\s*/\s*(?:\d+\s*)?(?:g|gm|kg|ml|l|pc|pcs|unit|units|n|u|piece)\b|\s*per\b)", re.I)


def _second_mrp(scan) -> Optional[Finding]:
    """Two different MRPs printed on one pack (a sticker over the old
    price, a second price line): the extractor keeps the first, so the
    second one passed silently. Every amount after an MRP label is
    collected (not per-unit prices); a second value is held for a human."""
    decl = next((d for d in scan.declarations
                 if d.field_id == "retail_sale_price" and d.present and d.value is not None), None)
    if decl is None:
        return None
    try:
        mine = float(decl.value)
    except (TypeError, ValueError):
        return None
    others = []
    for sp in scan.spans or []:
        t = sp.text or ""
        if sp.confidence < 0.8:
            continue
        for m in _MRP_TAG.finditer(t):
            # Only the amount right after each MRP label - not dates, batch
            # numbers or the unit price further along the line.
            a = _AMOUNT_AFTER_TAG.match(t[m.end():])
            if not a:
                continue
            try:
                v = float(a.group(1).replace(",", ""))
            except ValueError:
                continue
            if v >= 1 and abs(v - mine) > 0.005 and v not in others:
                others.append(v)
    if not others:
        return None
    vals = ", ".join(f"{v:g}" for v in [mine] + others)
    return Finding(
        rule_id="retail_sale_price.single_mrp", citation="Rule 6(1)(e)",
        outcome=Outcome.INDETERMINATE, severity=Severity.CRITICAL,
        field_id="retail_sale_price",
        message=(f"More than one MRP read on this pack ({vals}). A pack carries one "
                 f"retail sale price; check for a price sticker or a second price line."),
        evidence_bbox=decl.bbox, confidence=0.6, rule_verified=True)


def _find_banned_phrase(text: str, phrases: list[str]) -> Optional[str]:
    low = (text or "").lower()
    for p in phrases:
        # Letter edges, not \b: "\bmin.\b" needed a letter AFTER the dot,
        # so "200 g (Min.)" passed; "±" and "+/-" have no word edge at all.
        left = r"(?<![a-z0-9])" if p[:1].isalpha() else ""      # "5min noodles" is not "min"
        if re.search(rf"{left}{re.escape(p.lower())}(?![a-z])", low):
            return p
    return None


def _union(boxes: list[BBox]) -> BBox:
    x0 = min(b.x for b in boxes); y0 = min(b.y for b in boxes)
    x1 = max(b.x2 for b in boxes); y1 = max(b.y2 for b in boxes)
    return BBox(x0, y0, x1 - x0, y1 - y0)


# "(When Packed)" / "(at the time of packing)" under a net quantity is
# part of that declaration, not "other printed matter" crowding it.
_QTY_SUBCAPTION = re.compile(
    r"^\W*(?:w?h?en\s*pa?c?ked|at\s+the\s+time\s+of\s+pack\w*)\W*$", re.I)


# A printed MRP label (with or without its value on the same span).
_MRP_LABEL_ONLY = re.compile(r"(?<![A-Za-z])M\.?\s*R\.?\s*P\b|max(?:imum)?\.?\s*retail\s+price", re.I)


def _ink_is_tight(sp) -> bool:
    """Did vision.ink actually find the ink inside this OCR box? When it
    could not (busy background), ink_bbox is just the padded box."""
    ib = getattr(sp, "ink_bbox", None)
    return ib is not None and (ib.h < sp.bbox.h - 1.0 or ib.w < sp.bbox.w - 1.0)


def _nearest_neighbour_gaps(decl: Declaration, scan: ScanResult,
                            uncertain: Optional[set] = None,
                            extra_tol: Optional[dict] = None) -> dict[str, float]:
    """
    Distance in px from the declaration box to the nearest other printed
    text on each side. Returns only sides where something was found.

    Sides where padded boxes overlap but the ink inside them could not be
    separated go into `uncertain` instead of reading as a zero gap.
    """
    gaps: dict[str, float] = {}
    if decl.bbox is None:
        return gaps
    # Measure between INK, not between OCR boxes: PaddleOCR pads every
    # box outward, and two padded boxes swallow a large part of the very
    # gap Rule 8 is about (see vision/ink.py).
    own_ink = [s.ink_bbox or s.bbox for s in decl.spans]
    box = _union(own_ink) if own_ink else decl.bbox

    others: list[BBox] = []

    # Prefer the full OCR span set. Rule 8 is about "other printed
    # matter", which is mostly text that never became a declaration -
    # marketing copy, ingredients, a barcode caption. Searching only
    # declarations makes a crowded label look compliant.
    if scan.spans:
        # Use the already-null-checked local, not a fresh read of
        # decl.bbox. Re-reading defeats the guard four lines above and
        # leaves an AttributeError one refactor away.
        own = decl.bbox
        for sp in scan.spans:
            # Skip the spans that make up the declaration itself.
            if any(sp is s for s in decl.spans):
                continue
            if decl.field_id == "net_quantity" and _QTY_SUBCAPTION.match(sp.text or ""):
                continue
            if own.iou(sp.bbox) > 0.5 or own.contains(sp.bbox, tol=2.0):
                continue
            others.append(sp.ink_bbox or sp.bbox)
    else:
        for d in scan.declarations:
            if d.field_id == decl.field_id or not d.bbox:
                continue
            others.append(d.bbox)
            for s in d.spans:
                others.append(s.bbox)

    own_tight = all(_ink_is_tight(s) for s in decl.spans) if decl.spans else False
    tight_of = {id(sp.ink_bbox or sp.bbox): _ink_is_tight(sp) for sp in (scan.spans or [])}
    angle_of = {id(sp.ink_bbox or sp.bbox): getattr(sp, "angle", 0.0) for sp in (scan.spans or [])}
    own_angles = [getattr(s, "angle", 0.0) for s in decl.spans if not getattr(s, "vertical", False)]
    own_angle = sum(own_angles) / len(own_angles) if own_angles else 0.0
    # Pieces of ONE printed line sloping different ways = the line is an
    # arc (a label round a jar: "Net Quantity:" +1.7 deg, "100g" -3.2 deg
    # on a real Zydus jar). No straight band fits it.
    own_curved = len(own_angles) >= 2 and max(own_angles) - min(own_angles) > 2.0
    t_own = math.tan(math.radians(own_angle))
    for other in others:
        v_overlap = min(box.y2, other.y2) - max(box.y, other.y)
        h_overlap = min(box.x2, other.x2) - max(box.x, other.x)
        # Above or below (not beside): the only pairs tilt and curvature
        # distort, since the gap runs across the lines.
        vertical_pair = h_overlap > 0 and (
            v_overlap <= 0
            or abs(other.cy - box.cy) / max(box.h, 1.0) >= abs(other.cx - box.cx) / max(box.w, 1.0))

        if uncertain is not None and vertical_pair and own_curved:
            uncertain.add("bottom" if other.cy > box.cy else "top")
            continue

        # A tilted photo: every line is a sloped band, and its level box is
        # taller than the print by width x tan(tilt). On a real Rajam jar
        # 3 degrees off level, the box of "NET. WEIGHT: 500 g" ended 25 px
        # below the digits (at the far end of the line), and the gap to
        # "USE BY" measured 0.16 numeral heights where the print shows
        # about 0.9. So above/below neighbours of a tilted line are
        # measured DESKEWED: each line as a band of slope tan(angle)
        # through its box centre, the gap taken where the two overlap.
        t_oth = math.tan(math.radians(angle_of.get(id(other), 0.0)))
        skew = box.w * abs(t_own) + other.w * abs(t_oth)
        if vertical_pair and skew > 3.0:
            th_o = max(1.0, box.h - box.w * abs(t_own))
            th_n = max(1.0, other.h - other.w * abs(t_oth))
            xa, xb = max(box.x, other.x), min(box.x2, other.x2)

            def mid(b, t, x):
                return b.cy + t * (x - b.cx)

            xm = (xa + xb) / 2
            if mid(other, t_oth, xm) > mid(box, t_own, xm):
                g = min(mid(other, t_oth, x) - th_n / 2 - (mid(box, t_own, x) + th_o / 2)
                        for x in (xa, xb))
                side = "bottom"
            else:
                g = min(mid(box, t_own, x) - th_o / 2 - (mid(other, t_oth, x) + th_n / 2)
                        for x in (xa, xb))
                side = "top"
            if g <= 0 and uncertain is not None and not (own_tight and tight_of.get(id(other), True)):
                uncertain.add(side)
                continue
            g = max(0.0, g)
            if g < gaps.get(side, 1e9):
                gaps[side] = g
                if extra_tol is not None:
                    # The band model is approximate: a fifth of the skew,
                    # plus the descenders - a straight band cannot tell
                    # "NET." (none) from "500 g" (the g), so under the
                    # letters without one it places the bottom too low by
                    # up to a quarter of the line's thickness.
                    extra_tol[side] = 0.2 * skew + 0.25 * min(th_o, th_n)
            continue

        if v_overlap > 0 and h_overlap > 0:
            # The other print actually touches or overlaps the
            # declaration. These pairs used to fall through both branches
            # below and be ignored - so the MOST crowded case of all
            # measured as "no neighbour, compliant". Zero gap, on the side
            # its centre lies.
            dx = (other.cx - box.cx) / max(box.w, 1.0)
            dy = (other.cy - box.cy) / max(box.h, 1.0)
            if abs(dy) >= abs(dx):
                side = "bottom" if dy > 0 else "top"
            else:
                side = "right" if dx > 0 else "left"
            if uncertain is not None and not (own_tight and tight_of.get(id(other), True)):
                # Two PADDED boxes overlapping is what Paddle's padding
                # does to any two close lines; without the ink edges it
                # says nothing about the printed gap (real Drolia Gulal:
                # "0.00 numeral heights" between lines plainly apart).
                uncertain.add(side)
                continue
            gaps[side] = 0.0
            continue

        if v_overlap > 0:  # same horizontal band -> left/right neighbour
            if other.x2 <= box.x:
                gaps["left"] = min(gaps.get("left", 1e9), box.x - other.x2)
            elif other.x >= box.x2:
                gaps["right"] = min(gaps.get("right", 1e9), other.x - box.x2)
        if h_overlap > 0:  # same vertical band -> top/bottom neighbour
            if other.y2 <= box.y:
                gaps["top"] = min(gaps.get("top", 1e9), box.y - other.y2)
            elif other.y >= box.y2:
                gaps["bottom"] = min(gaps.get("bottom", 1e9), other.y - box.y2)

    return {k: v for k, v in gaps.items() if v < 1e9}


def exempt_set(scan: ScanResult, config: RuleConfig) -> set[str]:
    return set(ExemptionResolver(config).resolve(scan.context, scan.declarations).keys())
