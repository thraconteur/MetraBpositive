"""
Listing mode - compliance checking for e-commerce product listings.

WHY A SECOND INPUT PATH
-----------------------
Rule 6(10) requires an e-commerce entity to display the SAME mandatory
declarations on the listing page that Rule 6(1) requires on the package
itself, with one exception: the month and year of manufacture need not
be shown. Rule 6(10A), inserted 2026 and effective 01.07.2027, adds a
searchable and sortable country-of-origin filter for imported products.

None of that involves a photograph. The input is text scraped from a
listing, so the whole vision half of the pipeline - calibration,
rectification, glyph measurement - simply does not apply. Rules keyed to
how a declaration is PRINTED cannot be evaluated on a web page at all,
and pretending otherwise would produce confident findings about
millimetre heights that have no meaning in HTML.

So listing mode reuses the extractor and the rules engine, skips the
vision stages, and marks the presentation rules NOT_APPLICABLE with a
reason rather than silently omitting them.

WHAT THIS UNLOCKS
-----------------
Three rules that were previously config-only, and a demonstration path
that needs no camera, no marker card and no calibration - useful when
the packages are not to hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import (
    BBox,
    Declaration,
    Finding,
    Outcome,
    PackageContext,
    ScanResult,
    Severity,
    TextSpan,
)


# Rules that describe the physical printing of a label. A web listing has
# no panel, no glyph height and no clear space, so these are reported as
# not applicable rather than passed over in silence.
PRINT_ONLY_RULES = {
    "numeral_height": "minimum numeral height is a property of the printed panel",
    "glyph_aspect_ratio": "character shape is a property of the printed panel",
    "quantity_clear_space": "clear space is measured on the printed panel",
    "pdp_grouping": "there is no principal display panel on a web listing",
    "sticker_alteration": "a listing cannot bear a pasted sticker",
    "gm_food_label": "mark placement is a property of the printed panel",
}


@dataclass
class Listing:
    """One e-commerce product listing."""
    url: str = ""
    title: str = ""
    text: str = ""
    platform: str = ""
    # Rule 6(10A): does the platform offer a searchable AND sortable
    # country-of-origin filter on imported product listings?
    has_searchable_coo_filter: bool = False
    has_sortable_coo_filter: bool = False
    is_imported: bool = False
    seller: str = ""

    def all_text(self) -> str:
        return "\n".join(x for x in (self.title, self.text) if x)


def _spans_from_text(text: str) -> list[TextSpan]:
    """
    Turn listing text into spans the existing extractor understands.

    Each line becomes a row of word spans with synthetic geometry. The
    coordinates are fabricated and are used ONLY so the extractor's line
    grouping works; nothing downstream measures them, because every rule
    that would measure them is excluded above.
    """
    spans: list[TextSpan] = []
    for row, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        x = 10.0
        for word in line.split():
            w = max(6.0, len(word) * 8.0)
            spans.append(
                TextSpan(text=word, bbox=BBox(x, 10.0 + row * 24.0, w, 16.0),
                         confidence=1.0, source_engine="listing")
            )
            x += w + 6.0
    return spans


class ListingScanner:
    """
    Evaluates a listing against the declaration rules.

    Deliberately reuses the same extractor and the same rules engine as
    the image path. A separate implementation would drift: the two paths
    would start disagreeing about what counts as a valid MRP, and the
    citations on a listing report would stop matching those on a package
    report for the same rule.
    """

    def __init__(self, extractor=None, engine=None):
        from ..extraction.fields import RegexExtractor
        from .rules_engine import RulesEngine

        self.extractor = extractor or RegexExtractor()
        self.engine = engine or RulesEngine()

    # -----------------------------------------------------------------
    def scan(self, listing: Listing, inspector_id: Optional[str] = None) -> ScanResult:
        result = ScanResult(inspector_id=inspector_id)
        result.image_path = listing.url or "(listing)"

        spans = _spans_from_text(listing.all_text())
        result.spans = spans
        result.declarations = self.extractor.extract(spans)

        # On a listing the TITLE is the common or generic name. A package
        # heads it "COMMODITY:" or prints it under the brand, but no
        # e-commerce title does that, so without this every listing would
        # be reported as missing its Rule 6(1)(b) declaration.
        if listing.title.strip():
            existing = result.declaration("common_name")
            if existing is None or not existing.present:
                name = re.sub(r"\b\d+(\.\d+)?\s*(g|kg|ml|l|gm|litre|piece|pcs)\b",
                              "", listing.title, flags=re.I).strip(" -,|")
                decl = Declaration(
                    field_id="common_name", raw_text=listing.title,
                    value=name or listing.title, present=True,
                    extraction_confidence=0.75,
                    bbox=BBox(10.0, 10.0, 200.0, 16.0),
                )
                result.declarations = [
                    d for d in result.declarations if d.field_id != "common_name"
                ] + [decl]

        # A listing is a complete view of what the seller published, so
        # unlike a single photograph it CAN establish that a declaration
        # is absent - there is no second panel to turn over.
        result.coverage_complete = True

        ctx = PackageContext()
        ctx.is_imported = listing.is_imported or bool(
            re.search(r"country\s+of\s+origin|imported\s+by",
                      listing.all_text(), re.I)
        )
        text = listing.all_text().lower()
        from .pipeline import CATEGORY_KEYWORDS

        best, score = "other", 0
        for cat, words in CATEGORY_KEYWORDS.items():
            sc = sum(len(w) for w in words if w in text)
            if sc > score:
                best, score = cat, sc
        ctx.commodity_category = best
        result.context = ctx

        self.engine.evaluate(result)
        result.findings = self._adjust_for_listing(result.findings, listing)
        result.findings.extend(self._platform_findings(listing))
        return result

    # -----------------------------------------------------------------
    def _adjust_for_listing(self, findings: list[Finding],
                            listing: Listing) -> list[Finding]:
        """
        Convert print-only findings to NOT_APPLICABLE, and apply the
        Rule 6(10) carve-out for the manufacture date.
        """
        out: list[Finding] = []
        for f in findings:
            base = f.rule_id.split(".")[0]

            if base in PRINT_ONLY_RULES:
                f.outcome = Outcome.NOT_APPLICABLE
                f.message = (
                    f"Not applicable to an e-commerce listing: "
                    f"{PRINT_ONLY_RULES[base]}."
                )
                f.confidence = 1.0
                out.append(f)
                continue

            # Rule 6(10) requires everything in Rule 6(1) EXCEPT the
            # month and year of manufacture.
            if base == "manufacture_date":
                f.outcome = Outcome.NOT_APPLICABLE
                f.message = (
                    "Rule 6(10) excludes the month and year of manufacture "
                    "from the declarations required on a listing."
                )
                f.citation = "Rule 6(10)"
                f.confidence = 1.0
                out.append(f)
                continue

            # Everything else keeps its verdict, but is cited as the
            # listing-mode obligation.
            if f.outcome is Outcome.VIOLATION:
                # The engine's wording is written for a photographed
                # package. On this path the subject is a web page, and a
                # report that says "not found on the package" about a URL
                # reads as though the wrong thing was inspected.
                f.message = f.message.replace(
                    "not found on the package", "not shown on the listing")
                f.message = f.message.replace("on the package", "on the listing")
                f.message += "  Required under Rule 6(10)."
            out.append(f)
        return out

    # -----------------------------------------------------------------
    def _platform_findings(self, listing: Listing) -> list[Finding]:
        """
        Rule 6(10A) - a platform-level duty, not a product-level one.

        Inserted by G.S.R. 128(E) (2026) and substituted by G.S.R. 312(E),
        effective 01.07.2027: every e-commerce entity offering an imported
        product must provide a searchable and sortable country-of-origin
        filter. Reported as a forward-looking obligation rather than a
        present violation, because the duty has not yet commenced.
        """
        if not listing.is_imported:
            return [
                Finding(
                    rule_id="ecommerce_coo_filter",
                    citation="Rule 6(10A)",
                    outcome=Outcome.NOT_APPLICABLE,
                    severity=Severity.MAJOR,
                    message="Not an imported product listing.",
                    rule_verified=True,
                    confidence=1.0,
                )
            ]

        missing = []
        if not listing.has_searchable_coo_filter:
            missing.append("searchable")
        if not listing.has_sortable_coo_filter:
            missing.append("sortable")

        if not missing:
            msg = ("Platform provides a searchable and sortable "
                   "country-of-origin filter.")
            outcome = Outcome.COMPLIANT
        else:
            msg = (
                f"Country-of-origin filter is not {' and not '.join(missing)}. "
                f"Required for every e-commerce entity offering imported "
                f"products from 01.07.2027 - reported now as a forthcoming "
                f"obligation, not a present violation."
            )
            outcome = Outcome.INDETERMINATE

        return [
            Finding(
                rule_id="ecommerce_coo_filter",
                citation="Rule 6(10A)",
                outcome=outcome,
                severity=Severity.MAJOR,
                message=msg,
                rule_verified=True,
                confidence=0.9,
            )
        ]


def listing_from_text(text: str, url: str = "", **kw) -> Listing:
    """Convenience constructor for a pasted listing."""
    return Listing(url=url, text=text, **kw)
