#!/usr/bin/env python3
"""
Score the pipeline against a ground-truth manifest.

    python scripts/make_dataset.py --out data/synth
    python scripts/evaluate.py --manifest data/synth/manifest.json

The manifest schema is fixed on purpose: when you replace synthetic
images with real annotated photographs, emit the same schema and every
metric here keeps working unchanged.
"""
import os
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.pipeline import CompliancePipeline
from src.evaluation.harness import Evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/synth/manifest.json")
    ap.add_argument("--tolerance-mm", type=float, default=0.30)
    ap.add_argument("--ocr", default="paddle", choices=["paddle", "synthetic"],
                    help="paddle = real end-to-end numbers (default). synthetic = "
                         "perfect OCR from the renderer's own text, which isolates "
                         "extraction / measurement / rules from OCR errors. "
                         "Report both: the gap between them is what OCR costs.")
    a = ap.parse_args()
    os.environ["SIH_OCR_BACKEND"] = a.ocr

    man = json.load(open(a.manifest, encoding="utf-8"))
    records = man["records"] if isinstance(man, dict) else man

    pipeline = CompliancePipeline()
    # Pass through any context the image cannot carry (e.g. whether the
    # food is genetically modified). Without this the GM rule returns
    # NOT_APPLICABLE and every GM case scores as a miss.
    pairs = [
        (pipeline.scan(r["image_path"], hints=r.get("hints") or None), r)
        for r in records
    ]
    rep = Evaluator(tolerance_mm=a.tolerance_mm).evaluate(pairs)

    print(f"\nimages={rep.n_images}  abstained={rep.n_abstained}  "
          f"calibrated={rep.calibration_success_rate*100:.0f}%")

    print("\nEXTRACTION (per declaration)")
    print(f"  {'field':<24}{'P':>6}{'R':>6}{'F1':>7}")
    for k, v in sorted(rep.extraction.items()):
        if v["tp"] + v["fn"] == 0:
            continue
        print(f"  {k:<24}{v['precision']:>6.2f}{v['recall']:>6.2f}{v['f1']:>7.2f}")

    d = rep.violation_detection
    print(f"\nVIOLATION DETECTION  P={d['precision']:.2f}  "
          f"R={d['recall']:.2f}  F1={d['f1']:.2f}")
    print(f"  {'rule':<40}{'tp':>4}{'fp':>4}{'fn':>4}")
    for k, v in sorted(rep.per_rule_detection.items()):
        print(f"  {k:<40}{v['tp']:>4}{v['fp']:>4}{v['fn']:>4}")

    print(f"\nMEASUREMENT  MAE={rep.measurement_mae_mm:.3f} mm  "
          f"max={rep.measurement_max_err_mm:.3f} mm  "
          f"within {a.tolerance_mm} mm: {rep.measurement_within_tolerance*100:.0f}%")

    if rep.notes:
        print("\nCAVEATS")
        for n in rep.notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
