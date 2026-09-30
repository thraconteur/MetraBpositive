#!/usr/bin/env python3
"""
Delete scans older than N days, with their photos, evidence crops and
reports (and OCR cache files not used for N days).

    python scripts/cleanup_data.py --days 90          # show what would go
    python scripts/cleanup_data.py --days 90 --yes    # delete it

Or let the API do it at every start: set SIH_RETENTION_DAYS=90.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.retention import cleanup  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, required=True, help="keep scans newer than this")
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--db", default=str(ROOT / "data" / "compliance.db"))
    a = ap.parse_args()
    r = cleanup(Path(a.db), a.days, data=ROOT / "data", dry_run=not a.yes)
    verb = "Would remove" if r["dry_run"] else "Removed"
    print(f"{verb} {r['scans']} scan(s), {r['files']} file(s), {r['bytes'] / 1e6:.1f} MB.")
    if r["dry_run"] and (r["scans"] or r["files"]):
        print("Run again with --yes to delete.")


if __name__ == "__main__":
    main()
