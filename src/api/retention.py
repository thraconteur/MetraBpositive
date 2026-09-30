"""
Retention: delete scans older than N days, with everything stored for them.

Every scan keeps a viewing copy of the photo (data/scans), evidence crops
(data/evidence/<scan_id>/) and generated reports (data/reports/<scan_id>.*).
Nothing removed them, so the data folder grew without bound. This removes a
scan's record (scan and findings; officers' actions are kept as the audit
trail) together with its files, so
the history never lists a scan whose photo or report is gone.

    python scripts/cleanup_data.py --days 90          # what would go
    python scripts/cleanup_data.py --days 90 --yes    # delete

The API runs it at start-up when SIH_RETENTION_DAYS is set.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("data")


def _files_of(scan_id: str, data: Path) -> list[Path]:
    out = [data / "scans" / f"{scan_id}.jpg", data / "evidence" / scan_id]
    out += list((data / "reports").glob(f"{scan_id}.*")) if (data / "reports").exists() else []
    return [p for p in out if p.exists()]


def old_scans(db_path: Path, days: float) -> list[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if not Path(db_path).exists():
        return []
    with closing(sqlite3.connect(db_path)) as conn:
        try:
            return [r[0] for r in conn.execute(
                "SELECT scan_id FROM scans WHERE created_at < ?", (cutoff,))]
        except sqlite3.OperationalError:          # no scans table yet
            return []


def cleanup(db_path: Path, days: float, data: Path = DATA, dry_run: bool = True) -> dict:
    """Remove scans older than `days` and their files; also cached OCR reads
    and dumps not touched for `days`. Returns what was (or would be) removed."""
    ids = old_scans(db_path, days)
    files = [p for sid in ids for p in _files_of(sid, data)]
    limit = time.time() - days * 86400
    stale = []
    for sub in (".ocr_cache", "ocr_dumps"):
        d = data / sub
        if d.exists():
            stale += [p for p in d.iterdir() if p.is_file() and p.stat().st_mtime < limit]
    freed = sum((p.stat().st_size if p.is_file() else
                 sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))
                for p in files + stale)
    if not dry_run:
        for p in files + stale:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        if ids:
            with closing(sqlite3.connect(db_path)) as conn:
                q = ",".join("?" * len(ids))
                # Officers' decisions (actions) are the audit trail and are
                # kept: only the scan record and its findings go.
                for table in ("findings", "scans"):
                    try:
                        conn.execute(f"DELETE FROM {table} WHERE scan_id IN ({q})", ids)
                    except sqlite3.OperationalError:
                        pass
                conn.commit()
    return {"scans": len(ids), "files": len(files) + len(stale), "bytes": freed,
            "dry_run": dry_run}
