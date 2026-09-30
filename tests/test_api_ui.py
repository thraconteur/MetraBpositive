"""
The inspector app (templates/index.html) on top of the engine, through the
API: a pack of three real photos scanned in one request, what the app is
told to show, and the stored photo / reports it links to.

OCR is replayed from the stored PaddleOCR reads of the real photos, so no
model is needed.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HIRES = ROOT / "tests" / "fixtures" / "real_hires"
AMUL = [19, 20, 22]          # top (MRP, dates), side (address, care, qty), front


class _ReplayByImage:
    """Replays the stored read of whichever real photo was uploaded."""

    name = "paddleocr"

    def __init__(self, numbers):
        from src.vision.ocr.paddle import ReplayOCR

        self.by_hash = {}
        for n in numbers:
            img = cv2.imread(str(HIRES / f"photo{n}.jpg"))
            self.by_hash[self._h(img)] = ReplayOCR(HIRES / f"photo{n}.paddle.json")
        self.last_stats = {}

    @staticmethod
    def _h(img):
        return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()

    def available(self):
        return True

    def recognise(self, image, source=None):
        rep = self.by_hash[self._h(image)]
        spans = rep.recognise(image, source=source)
        self.last_stats = getattr(rep, "last_stats", {})
        return spans

    def __getattr__(self, k):             # anything else the pipeline asks of a backend
        return getattr(next(iter(self.by_hash.values())), k)


@pytest.fixture
def client(tmp_path, monkeypatch):
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient

    import src.api.main as api
    from src.core.pipeline import CompliancePipeline
    from src.core.rules_engine import RulesEngine

    monkeypatch.chdir(tmp_path)                       # data/scans, data/evidence here
    monkeypatch.setenv("SIH_TODAY", "2026-09-28")
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "t.db")
    api._SCHEMA_READY.clear()
    api._pipeline = CompliancePipeline(ocr=_ReplayByImage(AMUL),
                                       engine=RulesEngine(relaxation_lookup=api.relaxation_lookup))
    yield TestClient(api.app)
    api._pipeline = None


H = {"Authorization": "Bearer demo-inspector"}


def _pack(client):
    files = [("files", (f"amul_{i}.jpg", (HIRES / f"photo{n}.jpg").read_bytes(), "image/jpeg"))
             for i, n in enumerate(AMUL)]
    r = client.post("/scan_package?product_name=Amul", files=files, headers=H)
    assert r.status_code == 200, r.text
    return r.json()


def test_pack_scan_reads_each_declaration_from_the_right_photo(client):
    d = _pack(client)
    app = d["app"]
    decl = {x["field"]: x for x in app["declarations"]}
    assert decl["retail_sale_price"]["value"] == 26.0
    assert any("photo 1" in n for n in decl["retail_sale_price"]["notes"])
    assert decl["net_quantity"]["value"] == 250.0
    assert "Gujarat Co-operative" in decl["manufacturer_details"]["value"]
    assert app["package_photos"] == ["amul_0.jpg", "amul_1.jpg", "amul_2.jpg"]
    assert app["verdict_note"]
    assert [f for f in app["findings"] if f["outcome"] == "VIOLATION"] == []


def test_stored_photo_opens_from_a_plain_img_link_and_reports_stay_small(client):
    sid = _pack(client)["scan_id"]
    # An <img src> cannot send a header: the token rides in the URL.
    assert client.get(f"/scans/{sid}/image").status_code == 401
    r = client.get(f"/scans/{sid}/image?token=demo-inspector")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert len(r.content) < 3_000_000                 # a viewing copy, not the 27 MB canvas
    html = client.get(f"/scans/{sid}/report?fmt=html", headers=H)
    assert html.status_code == 200 and len(html.content) < 8_000_000
    pdf = client.get(f"/scans/{sid}/report?fmt=pdf", headers=H)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_app_page_needs_no_internet_for_its_styles(client):
    page = client.get("/").text
    assert "cdn.tailwindcss.com" not in page
    assert "/assets/tailwind.css" in page
    assert client.get("/assets/tailwind.css").status_code == 200
    # No invented product or brand when nothing was read.
    assert "Atta Maggi" not in page and "'NESTLE'" not in page


def test_retention_removes_old_scans_and_their_files(tmp_path):
    """Old scans go with their photo, evidence and reports; new ones stay."""
    import sqlite3
    from datetime import datetime, timedelta, timezone

    from src.api import main as api
    from src.api.retention import cleanup

    db = tmp_path / "c.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(api.SCHEMA)
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        new = datetime.now(timezone.utc).isoformat()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(scans)")]
        for sid, at in (("aaaaaaaaaaaa", old), ("bbbbbbbbbbbb", new)):
            vals = {c: "" for c in cols}
            vals.update(scan_id=sid, created_at=at, payload="{}")
            conn.execute(f"INSERT INTO scans ({','.join(vals)}) VALUES ({','.join('?' * len(vals))})",
                         list(vals.values()))
    data = tmp_path / "data"
    for sid in ("aaaaaaaaaaaa", "bbbbbbbbbbbb"):
        (data / "scans").mkdir(parents=True, exist_ok=True)
        (data / "scans" / f"{sid}.jpg").write_bytes(b"x" * 10)
        (data / "evidence" / sid).mkdir(parents=True, exist_ok=True)
        (data / "evidence" / sid / "e.png").write_bytes(b"x")
        (data / "reports").mkdir(parents=True, exist_ok=True)
        (data / "reports" / f"{sid}.r2.pdf").write_bytes(b"x")
    dry = cleanup(db, 90, data=data, dry_run=True)
    assert dry["scans"] == 1 and (data / "scans" / "aaaaaaaaaaaa.jpg").exists()
    cleanup(db, 90, data=data, dry_run=False)
    assert not (data / "scans" / "aaaaaaaaaaaa.jpg").exists()
    assert not (data / "evidence" / "aaaaaaaaaaaa").exists()
    assert not (data / "reports" / "aaaaaaaaaaaa.r2.pdf").exists()
    assert (data / "scans" / "bbbbbbbbbbbb.jpg").exists()
    with sqlite3.connect(db) as conn:
        assert [r[0] for r in conn.execute("SELECT scan_id FROM scans")] == ["bbbbbbbbbbbb"]
