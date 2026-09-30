#!/usr/bin/env python3
"""
Build a labelled synthetic dataset with exact millimetre ground truth.

    python scripts/make_dataset.py --out data/synth -c 40 -v 40

This is the fastest way to get ground truth for the ONE quantity that is
otherwise almost impossible to label by hand: the true cap height of
printed text in millimetres. Use it to develop and regression-test the
measurement stage, then re-measure on real photographs before quoting
any number publicly.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synth.generator import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("-c", "--compliant", type=int, default=25)
    ap.add_argument("-v", "--violating", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args()
    m = generate_dataset(a.out, a.compliant, a.violating, a.dpi, a.seed,
                         add_noise=not a.no_noise)
    print(f"Manifest: {m}")


if __name__ == "__main__":
    main()
