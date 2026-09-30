#!/usr/bin/env python3
"""
Score the system on REAL photos, from saved PaddleOCR output.

    python scripts/real_eval.py                  # tests/fixtures/real
    python scripts/real_eval.py --dir my_photos  # any folder laid out the same way

A folder holds, per photo: `<name>.jpg`, one or more `<name>*.paddle.json`
OCR dumps (as written by scan_photo.py into data/ocr_dumps/), and one
`ground_truth.yaml` saying what is really printed on each photo (see the
one in tests/fixtures/real for the format).

No OCR model is loaded: each dump is replayed through the full pipeline
(extraction, classification, rules). So this measures what the system
does with what Paddle actually read on real packaging - the numbers you
can put on a slide, with the sample size next to them.

Reported per photo and in total:
  read correctly   - field found with the right value
  read wrongly     - field found with a WRONG value (the dangerous kind)
  missed           - printed and legible, but not found (INDETERMINATE)
  false violations - violations a careful human would not raise
  violations caught - real violations (expected_violations) the system raised

Photos in ground_truth.yaml with no dump yet are listed at the end, so you
know which ones still need `scan_photo.py` run on them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "real"


def _matches(expected, decl) -> bool:
    if decl is None or not decl.present:
        return False
    if expected == "present":
        return True
    if isinstance(expected, (int, float)):
        try:
            return abs(float(decl.value) - float(expected)) < 0.011
        except (TypeError, ValueError):
            return False
    exp = str(expected).strip().lower()
    parts = exp.split()
    if len(parts) == 2 and parts[0].replace(".", "", 1).isdigit():
        # "45 ml" -> value + unit
        try:
            # "gm" is the same quantity as "g" (whether "gm" is an
            # acceptable symbol is a rule question, not a reading one).
            unit = {"gm": "g", "gms": "g", "gram": "g", "grams": "g"}.get(
                (decl.unit or "").lower(), (decl.unit or "").lower())
            return (abs(float(decl.value) - float(parts[0])) < 0.011
                    and unit == parts[1])
        except (TypeError, ValueError):
            return False
    return exp in str(decl.value).lower() or exp in (decl.raw_text or "").lower()


def _natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def evaluate_dir(folder: Path, verbose: bool = True) -> dict:
    import os

    import yaml

    truth_head = (folder / "ground_truth.yaml").read_text(encoding="utf-8")
    m = re.search(r"^as_of:\s*(\d{4}-\d{2}-\d{2})", truth_head, re.M)
    if m:
        # Score as of the day the ground truth was written, so a pack that
        # expires later stays correct in the numbers.
        os.environ.setdefault("SIH_TODAY", m.group(1))

    from src.core.pipeline import CompliancePipeline, PipelineConfig
    from src.vision.ocr.paddle import ReplayOCR

    truth = yaml.safe_load((folder / "ground_truth.yaml").read_text(encoding="utf-8"))
    rows = []
    for dump in sorted(folder.glob("*.paddle.json"), key=_natural):
        stem = dump.name[: -len(".paddle.json")]
        photo_key = stem.split(".")[0]
        variant = stem[len(photo_key) + 1:] or "default"
        gt = truth.get(photo_key)
        img = folder / f"{photo_key}.jpg"
        if not isinstance(gt, dict) or not img.exists():
            continue
        # "<photo>.paddle.json" is the default read (photo as taken);
        # "<photo>.enhanced.paddle.json" was read after our
        # deglare/contrast/sharpen pass (PipelineConfig.ocr_on_enhanced).
        cfg = PipelineConfig(ocr_on_enhanced=(variant == "enhanced"))
        try:
            ocr = ReplayOCR(dump)
        except Exception as e:                 # one bad file must not stop the score
            print(f"SKIPPED {dump.name}: unreadable dump ({e.__class__.__name__}). "
                  f"Delete it and re-run scripts/dump_real.py.")
            continue
        res = CompliancePipeline(ocr=ocr, config=cfg).scan(str(img))

        row = {"photo": photo_key, "variant": variant, "correct": [], "wrong": [],
               "missed": [], "false_absent": [], "false_violations": [],
               "caught": [], "not_caught": [], "judgement_calls": []}
        for field, exp in (gt.get("fields") or {}).items():
            d = res.declaration(field)
            if exp in ("not_legible", "not_on_panel"):
                if any(f.field_id == field and f.outcome.value == "VIOLATION"
                       and f.rule_id.endswith("presence") for f in res.findings):
                    row["false_absent"].append(field)
                continue
            if _matches(exp, d):
                row["correct"].append(field)
            elif d is not None and d.present:
                row["wrong"].append(f"{field}={d.value!r} (expected {exp!r})")
            else:
                row["missed"].append(field)
        expected_v = set(gt.get("expected_violations") or [])
        possible_v = set(gt.get("possible_violations") or [])
        raised = {f.rule_id for f in res.violations}
        row["caught"] = sorted(expected_v & raised)
        row["not_caught"] = sorted(expected_v - raised)
        row["judgement_calls"] = sorted(possible_v & raised)
        row["false_violations"] = [f"{f.rule_id}: {f.message[:80]}"
                                   for f in res.violations
                                   if f.rule_id not in expected_v | possible_v]
        row["heldout"] = bool(gt.get("set2") or gt.get("heldout"))
        rows.append(row)

    tot = {k: sum(len(r[k]) for r in rows)
           for k in ("correct", "wrong", "missed", "false_absent", "false_violations",
                     "caught", "not_caught", "judgement_calls")}
    have = {r["photo"] for r in rows}
    waiting = sorted((k for k, v in truth.items()
                      if isinstance(v, dict) and k not in have and (folder / f"{k}.jpg").exists()),
                     key=lambda k: _natural(Path(k)))
    if verbose:
        for r in rows:
            print(f"\n{r['photo']} [{r['variant']}]  correct {len(r['correct'])}, "
                  f"wrong {len(r['wrong'])}, missed {len(r['missed'])}, "
                  f"false violations {len(r['false_violations'])}")
            for k in ("wrong", "missed", "false_absent", "false_violations",
                      "caught", "not_caught", "judgement_calls"):
                for item in r[k]:
                    print(f"   {k:17s} {item}")
        for variant in sorted({r["variant"] for r in rows}):
            sub = [r for r in rows if r["variant"] == variant]
            c = sum(len(r["correct"]) for r in sub)
            w = sum(len(r["wrong"]) for r in sub)
            m = sum(len(r["missed"]) for r in sub)
            fv = sum(len(r["false_violations"]) for r in sub)
            ca = sum(len(r["caught"]) for r in sub)
            ev = ca + sum(len(r["not_caught"]) for r in sub)
            n = c + w + m
            print(f"\nTOTAL [{variant}] over {len(sub)} real photo(s): "
                  f"{c}/{n} legible declarations read correctly, {w} read wrongly, "
                  f"{m} missed; {fv} false violation(s)"
                  + (f"; {ca}/{ev} real violation(s) caught." if ev else "."))
        ho = [r for r in rows if r["heldout"] and r["variant"] == "default"]
        if ho:
            c = sum(len(r["correct"]) for r in ho)
            n = c + sum(len(r["wrong"]) + len(r["missed"]) for r in ho)
            w = sum(len(r["wrong"]) for r in ho)
            fv = sum(len(r["false_violations"]) for r in ho)
            ca = sum(len(r["caught"]) for r in ho)
            ev = ca + sum(len(r["not_caught"]) for r in ho)
            print(f"\nSET 2 (photo18-38; its first read, before any fix, is in "
                  f"docs/results_log.md) over {len(ho)} photo(s): "
                  f"{c}/{n} read correctly, {w} read wrongly; {fv} false violation(s)"
                  + (f"; {ca}/{ev} real violation(s) caught." if ev else "."))
        if waiting:
            print(f"\nNo OCR dump yet for {len(waiting)} photo(s): {', '.join(waiting)}")
            print("  Run scripts/scan_photo.py on them and copy "
                  "data/ocr_dumps/<name>.paddle.json into this folder.")
    return {"rows": rows, "totals": tot, "waiting": waiting}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--json", default=None, help="Also write the results here")
    args = ap.parse_args()
    out = evaluate_dir(Path(args.dir))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
