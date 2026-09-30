#!/usr/bin/env python3
"""
Try several PaddleOCR settings on the same photo(s) and compare what
each one extracted - so tuning is a table you read, not guesswork.

    python scripts/paddle_probe.py back_9.jpeg
    python scripts/paddle_probe.py photos/ --out data/probe.json

For every photo and every setting it runs the full pipeline and prints
the value of each declaration, the number of violations, and the OCR
time. Results are cached per setting, so a second run is fast.

Settings compared (edit PROBES below to add your own):
  default     - the shipped configuration
  det1920     - larger detection input (small print missed?)
  unclip2     - looser boxes (descenders clipped: "6 g" read as "69"?)
  unclip13    - tighter boxes (neighbouring lines merged?)
  lowthresh   - more sensitive detector (faint print missed?)
  v5          - PP-OCRv5 models instead of the default v6
  norefine    - without the second look at weak lines (is it helping?)
  noprep      - OCR on the photo without our deglare/CLAHE/sharpen pass
                (Paddle is trained on natural images; is the pass helping?)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROBES = {
    "default": {},
    "det1920": {"text_det_limit_side_len": 1920},
    "unclip2": {"text_det_unclip_ratio": 2.0},
    "unclip13": {"text_det_unclip_ratio": 1.3},
    "lowthresh": {"text_det_thresh": 0.2, "text_det_box_thresh": 0.45},
    "v5": {"ocr_version": "PP-OCRv5"},
    "norefine": {"refine_below_score": 0.0},
    "noprep": {"_pipeline": {"run_preprocess": False}},
}

FIELDS = ["net_quantity", "retail_sale_price", "unit_sale_price", "manufacture_date",
          "manufacturer_details", "consumer_care", "country_of_origin", "common_name"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"Subset of settings to run: {', '.join(PROBES)}")
    ap.add_argument("--secondary-lang", default=None, choices=["hi"])
    ap.add_argument("--mkldnn", action="store_true")
    ap.add_argument("--out", default=None, help="Also save everything as JSON")
    args = ap.parse_args()

    from scan_photo import _photos  # same folder
    from src.core.pipeline import CompliancePipeline, PipelineConfig
    from src.vision.ocr.paddle import PaddleConfig, PaddleOCRBackend

    names = args.only or list(PROBES)
    photos = _photos(args.photos)
    table = {}

    for name in names:
        probe = dict(PROBES[name])
        pipe_cfg = PipelineConfig(**probe.pop("_pipeline", {}))
        cfg = PaddleConfig(secondary_lang=args.secondary_lang,
                           enable_mkldnn=args.mkldnn, **probe)
        ocr = PaddleOCRBackend(config=cfg, dump_dir=None)
        if not ocr.available():
            sys.exit("PaddleOCR is not installed: pip install paddlepaddle paddleocr")
        pipe = CompliancePipeline(ocr=ocr, config=pipe_cfg)
        for photo in photos:
            t0 = time.perf_counter()
            res = pipe.scan(str(photo))
            st = dict(res.ocr_stats or {})
            row = {
                "ocr_s": "cached" if st.get("cache_hit") else st.get("total_s"),
                "lines": len(res.spans),
                "mean_conf": round(sum(s.confidence for s in res.spans) / max(1, len(res.spans)), 3),
                "violations": [f.rule_id for f in res.violations],
                "total_s": round(time.perf_counter() - t0, 1),
            }
            for f in FIELDS:
                d = res.declaration(f)
                row[f] = (None if not d or not d.present else
                          (f"{d.value}{(' ' + d.unit) if d.unit else ''}"
                           if d.value is not None else d.raw_text[:40]))
            table.setdefault(str(photo), {})[name] = row

    for photo, rows in table.items():
        print(f"\n=== {photo}")
        print(f"  {'':22s}" + "".join(f"{n:>18s}" for n in rows))
        for key in ["lines", "mean_conf", "ocr_s"] + FIELDS:
            print(f"  {key:22s}" + "".join(
                f"{str(rows[n][key])[:17]:>18s}" for n in rows))
        print(f"  {'violations':22s}" + "".join(
            f"{len(rows[n]['violations']):>18d}" for n in rows))

    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved {args.out}")
    print("\nPick the setting whose row matches the pack. Differences in a single "
          "field point at the knob that matters for that field.")


if __name__ == "__main__":
    main()
