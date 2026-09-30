#!/usr/bin/env python3
"""
One command for the real-photo test set: read every photo in
tests/fixtures/real that has no OCR dump yet, copy the dumps in, and score.

    python scripts/dump_real.py              # only photos without a dump
    python scripts/dump_real.py --mkldnn     # any extra flags go to scan_photo.py
    python scripts/dump_real.py --all        # re-read every photo (overwrites dumps)
    python scripts/dump_real.py --dir tests/fixtures/real_hires   # another photo set

Needs PaddleOCR (this is the step that runs the model). Afterwards anyone can
re-score without Paddle: `python scripts/real_eval.py`.

HTML reports for each photo land in data/reports/ as usual.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "real"
DUMPS = ROOT / "data" / "ocr_dumps"


def main():
    global FIX
    args = sys.argv[1:]
    redo = "--all" in args
    args = [a for a in args if a != "--all"]
    if "--dir" in args:
        i = args.index("--dir")
        FIX = Path(args[i + 1]).resolve()
        del args[i:i + 2]

    photos = sorted(FIX.glob("photo*.jpg"),
                    key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    todo = [p for p in photos if redo or not (FIX / f"{p.stem}.paddle.json").exists()]
    if not todo:
        print("Every photo already has a dump. Use --all to re-read them.")
    else:
        print(f"Reading {len(todo)} photo(s): {', '.join(p.stem for p in todo)}\n")
        cmd = [sys.executable, str(ROOT / "scripts" / "scan_photo.py"),
               *map(str, todo), *args]
        started = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        copied = []
        for p in todo:
            src = DUMPS / f"{p.stem}.paddle.json"
            # Only dumps written by THIS run: a stale one from an older run
            # would be scored as if it were a fresh read.
            if src.exists() and src.stat().st_mtime >= started - 1:
                shutil.copyfile(src, FIX / src.name)
                copied.append(p.stem)
        print(f"\nCopied {len(copied)} dump(s) into {FIX.relative_to(ROOT)}")
        missing = [p.stem for p in todo if p.stem not in copied]
        if missing:
            print(f"No dump produced for: {', '.join(missing)}")
            if "--mkldnn" in args:
                print("  If the error above says 'ConvertPirAttribute2RuntimeAttribute', "
                      "your CPU can't use --mkldnn. Run again without it.")
        if rc and not copied:
            sys.exit(rc)

    print("\n" + "=" * 70 + "\nScore on real photos\n" + "=" * 70)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    from real_eval import evaluate_dir

    evaluate_dir(FIX)


if __name__ == "__main__":
    main()
