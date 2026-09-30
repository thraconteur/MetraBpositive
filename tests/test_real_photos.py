import pytest
"""
Real photographs, real PaddleOCR output (replayed - no model needed).

tests/fixtures/real holds phone photos of real packs and the OCR a
teammate's PaddleOCR 3.7 produced for them, with ground_truth.yaml saying
what is actually printed. These tests pin the two things that matter most
on real packaging:

  * nothing is read WRONGLY (a wrong value is worse than a missing one),
  * no violation is reported on a pack that is compliant.

All 17 photos have been used to fix extraction and rules (photo1/2/5 first,
the other 14 on 28 Sept 2026), so the score here is a regression floor, not
an accuracy claim. An unseen-photo number needs NEW photos.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from real_eval import evaluate_dir  # noqa: E402
from src.core.schema import Outcome  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "real"
FLOOR = 103
HIRES = ROOT / "tests" / "fixtures" / "real_hires"
HIRES_FLOOR = 70


def _rows(variant="default"):
    out = evaluate_dir(FIX, verbose=False)
    # Every real photo is a regression gate now (set 2 was scored once
    # before any fix - that number is in docs/results_log.md).
    return [r for r in out["rows"] if r["variant"] == variant]


def test_real_photos_nothing_read_wrongly():
    rows = _rows()
    assert rows, "no real OCR fixtures found"
    wrong = [w for r in rows for w in r["wrong"]]
    assert wrong == [], wrong


def test_real_photos_no_false_violations():
    fv = [(r["photo"], v) for r in _rows() for v in r["false_violations"]]
    assert fv == [], fv


def test_real_photos_never_call_an_unreadable_declaration_missing():
    fa = [(r["photo"], f) for r in _rows() for f in r["false_absent"]]
    assert fa == [], fa


def test_real_photos_reading_floor():
    rows = _rows()
    correct = sum(len(r["correct"]) for r in rows)
    total = correct + sum(len(r["missed"]) + len(r["wrong"]) for r in rows)
    # 101 of 125 on 38 photos (29 Sept 2026 evening; 102 that morning - the
    # MyFitness side strip on the small copy is now unreadable fragments, see
    # docs/results_log.md; 98 on 28 Sept). Misses are
    # text Paddle did not read at all: smudged stickers, inkjet on curved or
    # metal surfaces, a sideways edge print, red-on-foil lines.
    assert correct >= FLOOR, (correct, total)


def test_real_photos_catch_the_real_violations():
    rows = _rows()
    missed = [(r["photo"], v) for r in rows for v in r["not_caught"]]
    assert missed == [], missed


def test_hair_serum_is_not_an_alcoholic_beverage():
    """'rum' is inside 'serum': substring matching deferred the MRP rules to
    State excise law for a hair serum."""
    from src.core.pipeline import CompliancePipeline
    from src.vision.ocr.paddle import ReplayOCR

    res = CompliancePipeline(ocr=ReplayOCR(FIX / "photo1.paddle.json")).scan(
        str(FIX / "photo1.jpg"))
    assert res.context.commodity_category == "cosmetic"
    assert not any("Alcoholic" in f.message for f in res.findings)


def test_every_real_photo_has_ground_truth_and_valid_rule_ids():
    """A new photo without a ground-truth entry is silently skipped by the
    eval, and a typo in a rule id would make a real violation look 'not
    caught' forever."""
    import yaml

    truth = yaml.safe_load((FIX / "ground_truth.yaml").read_text(encoding="utf-8"))
    photos = {p.stem for p in FIX.glob("*.jpg")}
    assert photos <= set(truth), sorted(photos - set(truth))
    truth = {k: v for k, v in truth.items() if isinstance(v, dict)}

    rules = yaml.safe_load((ROOT / "rules" / "lmpc_2011.yaml").read_text(encoding="utf-8"))
    decl_ids = {d["id"] for d in rules["declarations"]}
    checks = {d["id"]: {c["type"] for c in d.get("checks", [])}
              for d in rules["declarations"]}
    for key, gt in truth.items():
        if not isinstance(gt, dict):          # "as_of" and other settings
            continue
        assert (FIX / f"{key}.jpg").exists(), key
        for rid in (gt.get("expected_violations") or []) + (gt.get("possible_violations") or []):
            field, _, check = rid.partition(".")
            assert field in decl_ids and check in checks[field], (key, rid)
        for field in (gt.get("fields") or {}):
            assert field in decl_ids, (key, field)


def _package(ns):
    from src.core.package import merge_package
    from src.core.pipeline import CompliancePipeline
    from src.vision.ocr.paddle import ReplayOCR

    parts = [CompliancePipeline(ocr=ReplayOCR(FIX / f"photo{n}.paddle.json")).scan(
        str(FIX / f"photo{n}.jpg")) for n in ns]
    return merge_package(parts, CompliancePipeline(ocr=ReplayOCR(
        FIX / f"photo{ns[0]}.paddle.json")).engine, coverage_complete=True)


def test_package_scan_takes_the_qualifier_declared_by_reference():
    """Amul carton: price stamped on the top, "For MRP (incl. of all taxes)
    ... see top of the pack" printed on the side. One pack, no violation."""
    res = _package([19, 20, 22])
    q = next(f for f in res.findings if f.rule_id == "retail_sale_price.required_phrasing")
    assert q.outcome == Outcome.COMPLIANT
    assert res.violations == []
    assert res.declaration("retail_sale_price").value == 26
    assert res.declaration("net_quantity").value == 250


def test_package_scan_keeps_the_real_violations():
    """Kesar Chandan jar, every side photographed: no tax qualifier, an
    address that is a company name and a country, and no net quantity,
    date or consumer care on any side - all true of the jar in hand."""
    res = _package([8, 9, 15])
    ids = {f.rule_id for f in res.violations}
    assert {"retail_sale_price.required_phrasing", "manufacturer_details.structure",
            "net_quantity.presence", "consumer_care.presence"} <= ids


def _scan(n):
    from src.core.pipeline import CompliancePipeline
    from src.vision.ocr.paddle import ReplayOCR

    return CompliancePipeline(ocr=ReplayOCR(FIX / f"photo{n}.paddle.json")).scan(
        str(FIX / f"photo{n}.jpg"))


def test_unit_price_agrees_with_mrp_and_quantity_on_real_packs():
    # Nat Habit: MRP 130.00, 40 gm, "RS: 3.25/G"; Haldiram: Rs 10, 35 g, Rs 0.29/g
    for n in (37, 3):
        f = next(x for x in _scan(n).findings if x.rule_id.endswith("price_consistency"))
        assert f.outcome == Outcome.COMPLIANT, (n, f.message)


def test_unit_price_that_does_not_fit_is_flagged():
    """The same real Nat Habit read with its unit price changed: 2.25/g
    cannot be 130.00 for 40 g."""
    from src.core.rules_engine import RuleConfig, RulesEngine

    scan = _scan(37)
    scan.declaration("unit_sale_price").value = 2.25
    scan.findings = []
    RulesEngine(RuleConfig()).evaluate(scan)
    f = next(x for x in scan.findings if x.rule_id.endswith("price_consistency"))
    assert f.outcome in (Outcome.VIOLATION, Outcome.INDETERMINATE)
    assert "3.250" in f.message


def test_expired_pack_is_flagged_and_dates_are_read():
    """Nat Habit sachet: "Mfg:10MAY26", "EXP:07AUG26" - expired on the day the
    photos were scored. Amul carton top: "Exp:18/FEB/27" - not expired."""
    import os

    os.environ["SIH_TODAY"] = "2026-09-28"
    try:
        nat, amul = _scan(37), _scan(19)
    finally:
        os.environ.pop("SIH_TODAY", None)
    assert nat.declaration("expiry_date").value == "07/08/2026"
    # Shown to the inspector, not counted: selling expired stock is a
    # food-safety offence, not an LMPC one (Rule 6(1)(da) requires the date
    # to be DECLARED) - the rule is unverified, so it is never a violation.
    exp = [f for f in nat.findings if f.rule_id == "expiry_date.not_expired"]
    assert exp and exp[0].outcome.value == "UNVERIFIED_RULE" and "expired" in exp[0].message
    assert not any(f.rule_id.startswith("expiry") for f in nat.violations)
    assert amul.declaration("expiry_date").value == "18/02/2027"
    assert not any(f.rule_id.startswith("expiry") for f in amul.violations)


# ---------------------------------------------------------------------
# The same packs photographed at full resolution (1856x4096): photos
# 18-38 as the phone took them, not the WhatsApp copies. First read
# before any fix: 55/76, 6 false violations (29 Sept 2026).
# ---------------------------------------------------------------------

def _hires_rows():
    return [r for r in evaluate_dir(HIRES, verbose=False)["rows"] if r["variant"] == "default"]


def test_sharp_photos_read_nothing_wrongly_and_raise_no_false_violation():
    rows = _hires_rows()
    assert len(rows) == 21, len(rows)
    assert [w for r in rows for w in r["wrong"]] == []
    assert [(r["photo"], v) for r in rows for v in r["false_violations"]] == []
    assert [(r["photo"], f) for r in rows for f in r["false_absent"]] == []


def test_sharp_photos_reading_floor_and_violations():
    rows = _hires_rows()
    correct = sum(len(r["correct"]) for r in rows)
    assert correct >= HIRES_FLOOR, correct
    assert [(r["photo"], v) for r in rows for v in r["not_caught"]] == []


# ---------------------------------------------------------------------
# SET 3: 13 packs first seen 29 Sept 2026 (first read 31/40, 4 false
# violations; after fixes 36/40), and 12 rotated copies of three of them.
# ---------------------------------------------------------------------

def _rows_of(folder):
    return [r for r in evaluate_dir(ROOT / "tests" / "fixtures" / folder, verbose=False)["rows"]
            if r["variant"] == "default"]


@pytest.mark.parametrize("folder,floor,n", [("set3", 36, 13), ("rotated", 52, 12)])
def test_unseen_and_rotated_sets(folder, floor, n):
    rows = _rows_of(folder)
    assert len(rows) == n, len(rows)
    assert [w for r in rows for w in r["wrong"]] == []
    assert [(r["photo"], v) for r in rows for v in r["false_violations"]] == []
    assert [(r["photo"], v) for r in rows for v in r["not_caught"]] == []
    assert sum(len(r["correct"]) for r in rows) >= floor
