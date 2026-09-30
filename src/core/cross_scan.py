"""
Cross-scan analysis - Rule 18(2A).

THE RULE
--------
Rule 18(2A), inserted by G.S.R. 629(E) (2017): no manufacturer, packer
or importer shall declare DIFFERENT maximum retail prices on an
IDENTICAL pre-packaged commodity by adopting restrictive or unfair trade
practices.

WHY THIS ONE IS DIFFERENT FROM EVERY OTHER CHECK
------------------------------------------------
Every other rule in this system can be decided from a single photograph.
This one cannot. A package showing Rs. 120 is perfectly compliant on its
own; it becomes evidence only when a second, identical package shows
Rs. 145. The violation lives in the COMPARISON, not in either image.

That has a nice consequence for the pitch. The problem statement asks
for a "repository of scanned products and inspection history" and lists
it among the plain-software requirements, alongside dashboards and
report export. Rule 18(2A) is the reason that repository is not
plumbing: it is the only thing that makes this rule checkable at all.
A team that builds the database as a storage feature will not spot this;
a team that builds it as an evidence base will.

IDENTITY IS THE HARD PART
-------------------------
"Identical pre-packaged commodity" has to be decided from OCR output.
A barcode or GTIN is definitive when present. Absent that we fall back
to brand plus net quantity plus commodity name, normalised - which is
weaker, so those findings carry lower confidence and should be surfaced
for human confirmation rather than asserted.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from .schema import Finding, Outcome, ScanResult, Severity


@dataclass
class ProductKey:
    """How we decide two scans show the same commodity."""
    barcode: Optional[str] = None
    brand: Optional[str] = None
    common_name: Optional[str] = None
    net_quantity: Optional[float] = None
    unit: Optional[str] = None

    @property
    def definitive(self) -> bool:
        """A barcode identifies a product; the rest is inference."""
        return bool(self.barcode)

    def key(self) -> Optional[tuple]:
        if self.barcode:
            return ("gtin", self.barcode)
        if self.net_quantity is not None and (self.brand or self.common_name):
            return (
                "attrs",
                _norm(self.brand or ""),
                _norm(self.common_name or ""),
                round(float(self.net_quantity), 3),
                (self.unit or "").lower(),
            )
        return None

    def label(self) -> str:
        if self.barcode:
            return f"GTIN {self.barcode}"
        # brand and common_name often resolve to the same extracted text
        # (a package rarely separates them cleanly), so de-duplicate
        # rather than printing it twice in the violation report.
        bits: list[str] = []
        for b in (self.brand, self.common_name):
            if b and _norm(b) not in {_norm(x) for x in bits}:
                bits.append(b)
        q = f"{self.net_quantity:g}{self.unit or ''}" if self.net_quantity else ""
        return " ".join(bits + ([q] if q else [])) or "unidentified product"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def product_key(scan: ScanResult) -> ProductKey:
    qty = scan.declaration("net_quantity")
    name = scan.declaration("common_name")
    return ProductKey(
        barcode=getattr(scan.context, "barcode", None),
        brand=(name.raw_text if name and name.present else None),
        common_name=(name.raw_text if name and name.present else None),
        net_quantity=(qty.value if qty and qty.present else None),
        unit=(qty.unit if qty and qty.present else None),
    )


@dataclass
class DualPriceGroup:
    key_label: str
    definitive: bool
    prices: dict[float, list[str]] = field(default_factory=dict)  # price -> scan_ids

    @property
    def is_violation(self) -> bool:
        return len(self.prices) > 1


class CrossScanAnalyzer:
    """
    Compares a set of scans and reports Rule 18(2A) findings.

    Runs over the repository, not inside the per-image pipeline, because
    it needs more than one image by definition.
    """

    CITATION = "Rule 18(2A)"

    def __init__(self, price_tolerance: float = 0.005):
        # Two readings of the same printed price can differ by a paise
        # through OCR noise; that is not two different declared prices.
        self.price_tolerance = price_tolerance

    def analyse(self, scans: Iterable[ScanResult]) -> list[DualPriceGroup]:
        buckets: dict[tuple, DualPriceGroup] = {}

        for scan in scans:
            pk = product_key(scan)
            k = pk.key()
            if k is None:
                continue
            mrp = scan.declaration("retail_sale_price")
            if not mrp or not mrp.present or mrp.value is None:
                continue

            price = round(float(mrp.value), 2)
            grp = buckets.setdefault(
                k, DualPriceGroup(key_label=pk.label(), definitive=pk.definitive)
            )

            merged = next(
                (p for p in grp.prices if abs(p - price) <= self.price_tolerance),
                None,
            )
            grp.prices.setdefault(merged if merged is not None else price, []).append(
                scan.scan_id
            )

        return [g for g in buckets.values() if g.is_violation]

    def findings(self, scans: Iterable[ScanResult]) -> list[Finding]:
        out: list[Finding] = []
        for grp in self.analyse(scans):
            listed = ", ".join(
                f"Rs. {p:.2f} ({len(ids)} scan{'s' if len(ids) != 1 else ''})"
                for p, ids in sorted(grp.prices.items())
            )
            out.append(
                Finding(
                    rule_id="dual_mrp_prohibited",
                    citation=self.CITATION,
                    outcome=Outcome.VIOLATION,
                    severity=Severity.CRITICAL,
                    field_id="retail_sale_price",
                    message=(
                        f"Different maximum retail prices declared on an identical "
                        f"pre-packaged commodity ({grp.key_label}): {listed}."
                        + (
                            ""
                            if grp.definitive
                            else " Product identity inferred from brand, name and net "
                            "quantity rather than a barcode - confirm the packages "
                            "are genuinely identical before acting."
                        )
                    ),
                    confidence=0.9 if grp.definitive else 0.55,
                    rule_verified=True,
                )
            )
        return out
