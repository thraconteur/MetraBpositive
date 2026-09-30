"""
FastAPI service layer.

Covers the PS functional requirements that are plain software rather
than ML: image upload, scan history, the product repository, role-based
access, the dashboard aggregation, search, and report export.

SCOPE NOTE
----------
Auth here is a demonstration of the ROLE MODEL, not production security.
Tokens are static strings in a dict. Before this is deployed anywhere
real you need proper user storage, password hashing, expiring JWTs and
HTTPS. It is written this way deliberately so the role logic is readable
in one screen; do not ship it as-is and do not claim it is secure.

THE ROLE MODEL IS THE POINT
---------------------------
An inspector can scan and can propose a finding. Only a supervisor can
CONFIRM or OVERRIDE one, and every such action is recorded with who did
it and when. That separation is what makes the output usable in an
enforcement context: the machine never has the last word, a named human
does.
"""

from __future__ import annotations
import logging
import os
import re
import json
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..core.pipeline import CompliancePipeline
from ..core.rules_engine import RuleConfig
from ..core.schema import ScanResult

DB_PATH = Path("data/compliance.db")
REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory audit log ledger to record all official actions
audit_logs: list[dict] = []


# =====================================================================
# Roles
# =====================================================================

ROLES = {
    "guest": {"read"},
    "inspector": {"scan", "read"},
    "supervisor": {"scan", "read", "confirm", "override"},
    "admin": {"scan", "read", "confirm", "override", "manage", "audit"},
}

# Static mock user database mapping auth tokens to profile data
USER_DB = {
    "demo-guest": {
        "user_id": "guest-001",
        "name": "Guest User",
        "role": "Guest",
        "email": "guest@metra.gov.in",
        "phone": "+91 90000 00000",
        "uid": "",
        "is_verified": False,
    },
    "demo-inspector": {
        "user_id": "insp-001",
        "name": "Sankalp Dige",
        "role": "Inspector",
        "email": "sankalp@gov.in",
        "phone": "+91 98765 43210",
        "uid": "INSP-101",
        "is_verified": True,
    },
    "demo-supervisor": {
        "user_id": "supv-001",
        "name": "Rajesh Sharma",
        "role": "Supervisor",
        "email": "rajesh.sharma@metra.gov.in",
        "phone": "+91 98111 22334",
        "uid": "SUPV-201",
        "is_verified": True,
    },
    "demo-admin": {
        "user_id": "admin-001",
        "name": "System Administrator",
        "role": "Supervisor",
        "email": "admin@metra.gov.in",
        "phone": "+91 98999 88888",
        "uid": "SUPV-203",
        "is_verified": True,
    },
}

# Authoritative Government Registry Directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = PROJECT_ROOT / "data" / "registry"


def load_registry(role: str) -> list[dict]:
    r = role.strip().lower()
    if r in ("inspector", "insp"):
        filename = "inspectors.json"
    elif r in ("supervisor", "supv", "admin"):
        filename = "supervisors.json"
    else:
        return []

    path = REGISTRY_DIR / filename
    if not path.exists():
        path = Path("data/registry") / filename
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading registry {path}: {e}")
        return []

# DEMO ONLY. Replace with real user storage.
REPORT_VERSION = "r2"      # bump when the report layout changes (r2: Hindi, text read)

TOKENS = {
    "demo-guest": ("guest-001", "guest"),
    "demo-inspector": ("insp-001", "inspector"),
    "demo-supervisor": ("supv-001", "supervisor"),
    "demo-admin": ("admin-001", "admin"),
}


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Extract token string from Authorization header (supports 'Bearer <token>')."""
    if not authorization or not isinstance(authorization, str):
        return ""
    auth = authorization.strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth


class User(BaseModel):
    user_id: str
    role: str

    def can(self, action: str) -> bool:
        return action in ROLES.get(self.role, set())


def current_user(authorization: Optional[str] = Header(None)) -> User:
    token = extract_bearer_token(authorization)
    if not token or token not in TOKENS:
        raise HTTPException(401, "Invalid or missing token.")
    uid, role = TOKENS[token]
    return User(user_id=uid, role=role)


def current_user_or_query(authorization: Optional[str] = Header(None),
                          token: Optional[str] = Query(None, include_in_schema=False)) -> User:
    # ?token= ONLY for GET image / evidence / report links: a plain <img src>
    # cannot send a header. Every other endpoint takes the header only, so a
    # token never has to appear in a URL that changes anything.
    token = extract_bearer_token(authorization) or (token or "")
    if not token or token not in TOKENS:
        raise HTTPException(401, "Invalid or missing token.")
    uid, role = TOKENS[token]
    return User(user_id=uid, role=role)


def require_q(action: str):
    def dep(user: User = Depends(current_user_or_query)) -> User:
        if not user.can(action):
            raise HTTPException(403, f"Role '{user.role}' is not permitted to '{action}'.")
        return user
    return dep


# The role each demo token was issued with. A profile edit or an official
# verification may lower a token's role or restore it, never raise it above
# this: a self-service form must not hand out supervisor rights.
_ROLE_RANK = {"guest": 0, "inspector": 1, "supervisor": 2, "admin": 3}
BASE_ROLE = {t: r for t, (_, r) in TOKENS.items()}


def _allowed_role(token: str, wanted: str) -> bool:
    base = BASE_ROLE.get(token, "guest")
    return _ROLE_RANK.get(wanted, 99) <= _ROLE_RANK.get(base, 0)


def require(action: str):
    def dep(user: User = Depends(current_user)) -> User:
        if not user.can(action):
            raise HTTPException(
                403, f"Role '{user.role}' is not permitted to '{action}'."
            )
        return user
    return dep


# =====================================================================
# Storage
# =====================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id      TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    inspector_id TEXT,
    image_path   TEXT,
    product_name TEXT,
    category     TEXT,
    verdict      TEXT,           -- compliant / non_compliant / indeterminate
    n_violations INTEGER DEFAULT 0,
    payload      TEXT NOT NULL   -- full ScanResult as JSON
);
CREATE TABLE IF NOT EXISTS findings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id    TEXT NOT NULL,
    rule_id    TEXT NOT NULL,
    citation   TEXT,
    outcome    TEXT,
    severity   TEXT,
    message    TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);
-- Every human decision on a machine finding, kept forever.
CREATE TABLE IF NOT EXISTS actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id    TEXT NOT NULL,
    rule_id    TEXT NOT NULL,
    action     TEXT NOT NULL,   -- confirmed / overridden
    user_id    TEXT NOT NULL,
    note       TEXT,
    at         TEXT NOT NULL
);
-- Rule 33 relaxations are package-specific and time-bound, so they
-- live here rather than in the rules YAML.
CREATE TABLE IF NOT EXISTS relaxations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer TEXT NOT NULL,
    rule_id      TEXT NOT NULL,
    reference    TEXT,
    valid_until  TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at);
CREATE INDEX IF NOT EXISTS idx_findings_scan ON findings(scan_id);
"""


_SCHEMA_READY: set = set()


def db():
    """
    Open a connection, creating the schema on first use.

    Schema creation is done here rather than only in a startup hook
    because the hook does not run in every context that touches the
    database - test clients, one-off scripts and worker processes all
    import this module and call db() directly. Relying on the hook alone
    produced "no such table" the first time the API was exercised
    outside a full server boot.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    key = str(Path(DB_PATH).resolve())
    if key not in _SCHEMA_READY:          # per database file, not per process
        conn.executescript(SCHEMA)
        conn.commit()
        _SCHEMA_READY.add(key)
    return conn


def init_db():
    with closing(db()) as conn:
        conn.commit()


def verdict_of(res: ScanResult) -> str:
    c = res.is_compliant
    return "compliant" if c is True else "non_compliant" if c is False else "indeterminate"


def persist(res: ScanResult, product_name: str = "") -> None:
    with closing(db()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scans (scan_id, created_at, inspector_id, "
            "image_path, product_name, category, verdict, n_violations, payload) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                res.scan_id, res.created_at, res.inspector_id, res.image_path,
                product_name or (res.declaration("common_name").raw_text
                                 if res.declaration("common_name") else ""),
                res.context.commodity_category, verdict_of(res),
                len(res.violations), json.dumps(res.to_dict()),
            ),
        )
        for f in res.findings:
            conn.execute(
                "INSERT INTO findings (scan_id, rule_id, citation, outcome, "
                "severity, message) VALUES (?,?,?,?,?,?)",
                (res.scan_id, f.rule_id, f.citation, f.outcome.value,
                 f.severity.value, f.message),
            )
        conn.commit()


# =====================================================================
# App
# =====================================================================

app = FastAPI(
    title="LMPC Compliance Checker",
    description=(
        "Legal Metrology (Packaged Commodities) Rules, 2011 compliance "
        "checking. Reference implementation - see README for scope limits. "
        "Mobile-app integration: docs/app_integration.md."
    ),
    version="0.3.0-scaffold",
)

# The inspector app calls this API from a phone (native) or a browser
# (web build / Expo web). Browsers need CORS; native apps ignore it.
# SIH_CORS_ORIGINS="https://app.example,http://localhost:8081" to lock down.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    # Same-origin by default (the METRA page is served by this API; native
    # apps ignore CORS). Web builds on another origin: SIH_CORS_ORIGINS.
    allow_origins=[o.strip() for o in os.environ.get("SIH_CORS_ORIGINS", "").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the METRA inspector frontend.  The templates/ directory sits at
# the project root, two levels above src/api/.
_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
_ASSETS_DIR = _TEMPLATE_DIR / "assets"
if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the METRA mobile-first inspector UI."""
    return _templates.TemplateResponse(request=request, name="index.html")

_pipeline: Optional[CompliancePipeline] = None
# One scan at a time through the shared OCR engine: PaddleOCR's predictor
# is not safe to call from two threads at once. Other requests (history,
# reports) keep being served while a scan runs.
import threading  # noqa: E402

_scan_lock = threading.Lock()


def relaxation_lookup(manufacturer: str, rule_id: str) -> Optional[dict]:
    """
    Rule 33 relaxations, matched by manufacturer and rule.

    Matched on a substring of the extracted manufacturer text rather
    than equality: the stored name is typed by an official, while the
    scanned one comes from OCR of a real package, so they will rarely
    be byte-identical.
    """
    with closing(db()) as conn:
        rows = conn.execute(
            "SELECT manufacturer, reference, valid_until FROM relaxations "
            "WHERE rule_id = ?", (rule_id,)
        ).fetchall()
    hay = (manufacturer or "").lower()
    for r in rows:
        name = (r["manufacturer"] or "").lower().strip()
        if name and name in hay:
            if r["valid_until"]:
                # An expired relaxation is not a relaxation.
                if str(r["valid_until"]) < datetime.now(timezone.utc).date().isoformat():
                    continue
            return dict(r)
    return None


def pipeline() -> CompliancePipeline:
    global _pipeline
    if _pipeline is None:
        from ..core.rules_engine import RulesEngine

        vlm = None
        if os.environ.get("SIH_VLM", "").lower() == "gemini":
            from ..vision.ocr.vlm_gemini import GeminiReader

            vlm = GeminiReader()
            vlm.check()        # fail at startup with the exact fix, not per request
        _pipeline = CompliancePipeline(
            engine=RulesEngine(relaxation_lookup=relaxation_lookup), vlm=vlm,
        )
    return _pipeline


@app.on_event("startup")
def _startup():          # kept for real server boots; db() self-heals anyway
    init_db()
    # Optional retention: SIH_RETENTION_DAYS=90 removes older scans and their
    # photos / evidence / reports at every start (see src/api/retention.py).
    days = os.environ.get("SIH_RETENTION_DAYS", "").strip()
    if days:
        try:
            from .retention import cleanup
            r = cleanup(DB_PATH, float(days), data=Path("data"), dry_run=False)
            logging.getLogger(__name__).info(
                "Retention %s days: removed %d scan(s), %.1f MB", days, r["scans"], r["bytes"] / 1e6)
        except Exception:
            logging.getLogger(__name__).exception("Retention cleanup failed")
    # Load the OCR models in the background, so the FIRST scan from the
    # app does not sit for a minute and hit the phone's request timeout.
    if (os.environ.get("SIH_OCR_BACKEND", "paddle") == "paddle"
            and os.environ.get("SIH_WARMUP", "1") != "0"):
        def _warm():
            try:
                warm = getattr(pipeline().ocr, "warm_up", None)
                if warm:
                    warm()
            except Exception:
                pass
        threading.Thread(target=_warm, daemon=True).start()


@app.get("/health")
def health():
    audit = RuleConfig().audit()
    return {
        "status": "ok",
        "rules_verified": len(audit["verified"]),
        "rules_unverified": len(audit["unverified"]),
        "coverage": round(audit["coverage"], 3),
        "ocr_backend": os.environ.get("SIH_OCR_BACKEND", "paddle"),
    }


class OfficialVerificationRequest(BaseModel):
    role: str
    name: str
    uid: str


@app.post("/verify-official")
def verify_official(
    req: OfficialVerificationRequest,
    authorization: Optional[str] = Header(None),
):
    token = extract_bearer_token(authorization)
    if not token or token not in USER_DB:
        raise HTTPException(401, "Sign in before verifying official credentials.")
    """
    Verify official credentials against authoritative government registry files.
    Matches name.strip().lower() and uid.strip().upper() against inspectors.json or supervisors.json.
    """
    clean_role_lower = req.role.strip().lower()
    if clean_role_lower in ("inspector", "insp"):
        clean_role = "Inspector"
    elif clean_role_lower in ("supervisor", "supv", "admin"):
        clean_role = "Supervisor"
    else:
        return {
            "verified": False,
            "message": f"Verification is not applicable for role '{req.role.strip()}'",
        }

    clean_name = req.name.strip().lower()
    clean_uid = req.uid.strip().upper()
    records = load_registry(clean_role)
    matched_record = None
    for entry in records:
        entry_uid = str(entry.get("uid", "")).strip().upper()
        entry_name = str(entry.get("name", "")).strip().lower()
        if entry_uid == clean_uid and entry_name == clean_name:
            matched_record = entry
            break

    if not matched_record:
        return {
            "verified": False,
            "message": "Record not found or UID mismatch",
        }

    if str(matched_record.get("status", "")).lower() != "active":
        return {
            "verified": False,
            "message": f"Official credential for UID {clean_uid} is inactive or revoked",
        }

    # The credential proves who the person is; it does not raise what this
    # sign-in may do. Supervisor rights come with a supervisor sign-in.
    if not _allowed_role(token, clean_role.lower()):
        raise HTTPException(403, f"This sign-in cannot act as {clean_role}.")
    USER_DB[token]["is_verified"] = True
    USER_DB[token]["uid"] = clean_uid
    USER_DB[token]["name"] = req.name.strip()
    if BASE_ROLE.get(token) != "admin":          # an admin stays admin
        USER_DB[token]["role"] = clean_role
        TOKENS[token] = (USER_DB[token]["user_id"], clean_role.lower())

    return {
        "verified": True,
        "message": "Official credentials verified",
        "official": {
            "uid": clean_uid,
            "name": matched_record.get("name"),
            "status": matched_record.get("status"),
        },
    }


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    uid: Optional[str] = None
    is_verified: Optional[bool] = None


@app.get("/profile")
def get_profile(authorization: Optional[str] = Header(None)):
    """
    Get profile information for the authenticated user.
    Reads incoming Authorization: Bearer <token> header, looks up user in USER_DB,
    and returns profile JSON (or raises 401 Unauthorized if token does not exist).
    """
    token = extract_bearer_token(authorization)
    if not token or token not in USER_DB:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token.",
        )
    return dict(USER_DB[token])


@app.put("/profile")
def update_profile(
    body: ProfileUpdate,
    authorization: Optional[str] = Header(None),
):
    """
    Update profile details for the authenticated user.
    """
    token = extract_bearer_token(authorization)
    if not token or token not in USER_DB:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token.",
        )

    user = USER_DB[token]
    if body.role is not None and body.role.strip():
        for vr in ["Guest", "Inspector", "Supervisor"]:
            if vr.lower() == body.role.strip().lower() and not _allowed_role(token, vr.lower()):
                raise HTTPException(403, f"This sign-in cannot act as {vr}.")
    if body.name is not None and body.name.strip():
        user["name"] = body.name.strip()
    if body.role is not None and body.role.strip():
        for vr in ["Guest", "Inspector", "Supervisor"]:
            if vr.lower() == body.role.strip().lower():
                if not _allowed_role(token, vr.lower()):
                    raise HTTPException(403, f"This sign-in cannot act as {vr}.")
                if BASE_ROLE.get(token) != "admin":
                    user["role"] = vr
                    TOKENS[token] = (user["user_id"], vr.lower())
                    if vr == "Guest":
                        user["is_verified"] = False
                break
    if body.email is not None:
        user["email"] = body.email.strip()
    if body.phone is not None:
        user["phone"] = body.phone.strip()
    if body.uid is not None:
        user["uid"] = body.uid.strip().upper()
    # is_verified is set only by /verify-official, never by the user.
    if body.is_verified is False or user["role"] == "Guest":
        user["is_verified"] = False

    return dict(user)


@app.get("/qr.svg")
def qr_svg(data: str = Query(..., max_length=300)):
    """A QR code drawn here (the ID card used an outside website, which then
    received the officer's name and UID with every request)."""
    import io
    import segno
    from fastapi.responses import Response

    buf = io.BytesIO()
    segno.make(data, error="m").save(buf, kind="svg", scale=4, border=1)
    return Response(buf.getvalue(), media_type="image/svg+xml")


@app.get("/rules/audit")
def rules_audit(user: User = Depends(require("read"))):
    """
    Which rules are backed by verified gazette values and which are
    still placeholders. Expose this rather than hiding it: a tool that
    is honest about its coverage is far more credible than one that
    implies it checks everything.
    """
    return RuleConfig().audit()


# ---------------------------------------------------------------------
# Uploads: read, check and decode a photo safely
# ---------------------------------------------------------------------
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000          # 50 MP: every phone camera is below; a PNG
                                 # "bomb" (30000x30000 in 2.6 MB) is refused
                                 # before it is decoded into ~3 GB of RAM
MAX_PACK_PHOTOS = 8
MAX_PACK_BYTES = 100 * 1024 * 1024


async def _read_upload(up: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read at most limit+1 bytes: an oversized upload is refused without
    holding all of it in memory."""
    chunks, n = [], 0
    while True:
        chunk = await up.read(1 << 20)
        if not chunk:
            break
        n += len(chunk)
        if n > limit:
            raise HTTPException(413, f"{up.filename or 'Photo'}: over the "
                                     f"{limit // (1024 * 1024)} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_photo(raw: bytes, name: str = "Photo") -> np.ndarray:
    """Bytes -> 8-bit BGR image, or an HTTP error that says what to do."""
    if not raw:
        raise HTTPException(400, f"{name}: file is empty.")
    head = raw[:32]
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypmsf1", b"ftypavif"):
        raise HTTPException(415, f"{name}: HEIC/AVIF photos are not supported. Set the "
                                 "camera to JPEG ('Most Compatible' on iPhone) or share as JPEG.")
    try:
        from PIL import Image, ImageFile
        import io
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
                fmt = im.format
                if w * h > MAX_PIXELS:
                    raise HTTPException(413, f"{name}: {w}x{h} is too large (max 50 megapixels).")
                if fmt == "JPEG":
                    # A JPEG cut off in transit decodes with the missing part
                    # black, and the scan would report the declarations there as
                    # absent. Decoding it fully says so. (Checking for the end
                    # marker instead refused Samsung / Google "motion photos",
                    # which carry data after it.)
                    ImageFile.LOAD_TRUNCATED_IMAGES = False
                    try:
                        im.load()
                    except OSError:
                        raise HTTPException(400, f"{name}: the photo arrived incomplete. Upload it again.")
    except HTTPException:
        raise
    except Exception as exc:
        if type(exc).__name__ == "DecompressionBombError":   # PIL's own, far above 50 MP
            raise HTTPException(413, f"{name}: the image is too large (max 50 megapixels).")
        # PIL cannot read it; OpenCV decides - but never beyond 50 MP either.
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
            w, h = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
            if w * h > MAX_PIXELS:
                raise HTTPException(413, f"{name}: {w}x{h} is too large (max 50 megapixels).")
    try:
        # IMREAD_COLOR applies the phone's EXIF rotation (portrait photos),
        # and turns grey / 16-bit images into ordinary 8-bit colour.
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        arr = None
    if arr is None:
        raise HTTPException(400, f"{name}: could not decode the image (use JPEG or PNG).")
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) > 25 and raw[25] in (4, 6):
        # PNG with transparency: background -> white, as printed (it loaded black).
        full = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if full is not None and full.ndim == 3 and full.shape[2] == 4:
            if full.dtype != np.uint8:
                full = (full / 257.0).astype(np.uint8)
            a = full[:, :, 3:4].astype(np.float32) / 255.0
            arr = (full[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    h, w = arr.shape[:2]
    if max(h, w) > 4096:
        # Full resolution is kept up to phone-camera size: the OCR engine
        # sizes detection itself and reads from the full-resolution pixels.
        s_ = 4096 / max(h, w)
        arr = cv2.resize(arr, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)
    return arr


def _write_temp_png(arr: np.ndarray) -> str:
    fh = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fh.close()
    cv2.imwrite(fh.name, arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fh.name


def _unlink(paths) -> None:
    for p_ in paths:
        try:
            os.unlink(p_)
        except OSError:
            pass


@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    product_name: str = Query(""),
    panel_width_mm: Optional[float] = Query(None, gt=0, lt=5000, description=(
        "Real panel width in mm (ruler), for mm checks when no marker is in frame")),
    panel_height_mm: Optional[float] = Query(None, gt=0, lt=5000),
    coverage_complete: bool = Query(False, description=(
        "The photos of this pack cover every panel: absence becomes a violation")),
    user: User = Depends(require("scan")),
):
    from fastapi.concurrency import run_in_threadpool

    raw = await _read_upload(file)
    # Decoding a 50 MP photo takes seconds: off the event loop, so the server
    # keeps answering (health checks, other users) meanwhile.
    arr = await run_in_threadpool(_decode_photo, raw, file.filename or "Photo")
    del raw

    panel = (panel_width_mm, panel_height_mm or 0.0) if panel_width_mm else None

    def _run(path: str) -> ScanResult:
        with _scan_lock:
            r = pipeline().scan(path, inspector_id=user.user_id, panel_size_mm=panel)
            if coverage_complete:
                r.coverage_complete = True
                r.findings = []
                pipeline().engine.evaluate(r)
            return r

    tmp = await run_in_threadpool(_write_temp_png, arr)
    del arr
    try:
        res = await run_in_threadpool(_run, tmp)
        return await run_in_threadpool(_finish_single, res, product_name)
    finally:
        _unlink([tmp])


def _finish_single(res: ScanResult, product_name: str) -> JSONResponse:
    # Cropped evidence per finding. The problem statement asks for
    # "attachment of photographs and supporting evidence", and every
    # finding already carried a bounding box - but nothing ever called
    # the crop extractor, so no report ever contained a single evidence
    # image. A violation report that cites a rule and shows nothing is
    # not evidence an inspector can act on.
    from ..report.builder import extract_evidence_crops

    try:
        crop_dir = Path("data/evidence") / res.scan_id
        extract_evidence_crops(res, crop_dir)
    except Exception:
        # Evidence generation must never fail the scan itself.
        pass

    try:
        _save_view_copy(res)
    except Exception:
        pass
    persist(res, product_name)
    payload = res.to_dict()
    payload["app"] = app_summary(payload, product_name)
    _add_guidance(payload, res)
    return JSONResponse(payload)


def _save_view_copy(res: ScanResult, src: Optional[str] = None) -> None:
    """
    Keep a viewing copy of the photo(s) with the scan record: JPEG, long
    side <= 2000 px, in data/scans/. The analysis ran on the full-resolution
    upload; the record, the app and the report only need to SHOW it (a pack
    of three sharp photos is 27 MB as PNG). Also, the upload itself sat in a
    temp folder and was gone after a restart.
    """
    img = getattr(res, "analysed_image", None)
    if img is None and (src or res.image_path):
        img = cv2.imread(src or res.image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    # A pack is several photos stacked: bound the width, let it be tall.
    s_ = min(1.0, 2000.0 / max(h, w)) if h <= 2.5 * w else min(1.0, 1100.0 / w)
    if s_ < 1.0:
        img = cv2.resize(img, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)
    out = Path("data/scans")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{res.scan_id}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    res.image_path = str(path)


def _add_guidance(payload: dict, res: ScanResult) -> None:
    """
    What the engine says the inspector should do next, in the fields the
    app already shows: "photograph the base / crimp / lid" (the pack says
    the MRP is there) joins the retake advice, and the one-line reason for
    an undecided verdict goes in verdict_note.
    """
    from ..core.package import capture_advice
    from ..report.builder import _verdict

    app_ = payload.setdefault("app", {})
    advice = capture_advice(res)
    app_["capture_advice"] = advice
    app_["retake_advice"] = list(dict.fromkeys((app_.get("retake_advice") or []) + advice))[:4]
    app_["verdict_note"] = _verdict(res)[2]
    app_["package_photos"] = payload.get("package_photos") or []


@app.post("/scan_package")
async def scan_package(
    files: list[UploadFile] = File(..., description="Every side of ONE pack"),
    product_name: str = Query(""),
    user: User = Depends(require("scan")),
):
    """
    Several photos of one pack (front, back, top, base, crimp ...) decided
    together - the way an inspector checks a pack. Every side is taken as
    photographed, so a declaration that no photo shows is a finding, not
    "not visible in this image". Each declaration is read from the photo
    that shows it best, and the report says which.
    """
    from fastapi.concurrency import run_in_threadpool

    if not files:
        raise HTTPException(400, "No photos uploaded.")
    if len(files) > MAX_PACK_PHOTOS:
        raise HTTPException(413, f"At most {MAX_PACK_PHOTOS} photos per pack.")
    paths: list[str] = []
    total = 0
    try:
        for up in files:
            raw = await _read_upload(up)
            total += len(raw)
            if total > MAX_PACK_BYTES:
                raise HTTPException(413, "The photos of one pack add up to over 100 MB.")
            arr = await run_in_threadpool(_decode_photo, raw, up.filename or "Photo")
            del raw
            paths.append(await run_in_threadpool(_write_temp_png, arr))
            del arr
    except BaseException:
        _unlink(paths)
        raise

    from ..core.package import merge_package

    def _run() -> ScanResult:
        with _scan_lock:
            parts = [pipeline().scan(p, inspector_id=user.user_id) for p in paths]
            return merge_package(parts, pipeline().engine, coverage_complete=True,
                                 names=[f.filename or f"photo{i+1}" for i, f in enumerate(files)])

    try:
        res = await run_in_threadpool(_run)
        res.inspector_id = user.user_id
        return await run_in_threadpool(_finish_pack, res, product_name)
    finally:
        _unlink(paths)


def _finish_pack(res: ScanResult, product_name: str) -> JSONResponse:
    from ..report.builder import extract_evidence_crops

    try:
        extract_evidence_crops(res, Path("data/evidence") / res.scan_id)
    except Exception:
        pass
    # One image for the scan record: the photos one under another, as analysed.
    try:
        _save_view_copy(res)
    except Exception:
        pass
    persist(res, product_name)
    payload = res.to_dict()
    payload["app"] = app_summary(payload, product_name)
    _add_guidance(payload, res)
    return JSONResponse(payload)


# =====================================================================
# Mobile app view
# =====================================================================

_SEV_ORDER = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
_OUT_ORDER = {"VIOLATION": 0, "INDETERMINATE": 1, "UNVERIFIED_RULE": 2,
              "COMPLIANT": 3, "NOT_APPLICABLE": 4, "SUPPRESSED": 5}


def app_summary(payload: dict, product_name: str = "") -> dict:
    """
    What the inspector app's screens need, in one flat object.

    Counts are plain counts, not a score: the scan page shows "3 critical,
    4 major" and the officer knows exactly what that means. A "3/10
    compliance rating" would need a definition someone can defend in a
    hearing - if the app wants one, show `checks` (passed of assessed).
    """
    findings = payload.get("findings") or []
    decls = payload.get("declarations") or []
    sid = payload.get("scan_id", "")
    viol = [f for f in findings if f.get("outcome") == "VIOLATION"]
    sev = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}
    for f in viol:
        k = str(f.get("severity", "MINOR")).upper()
        sev[k] = sev.get(k, 0) + 1
    to_confirm = [f for f in findings if f.get("outcome") == "INDETERMINATE"]
    passed = [f for f in findings if f.get("outcome") == "COMPLIANT"]
    c = (payload.get("summary") or {}).get("compliant")
    # "undecided" = the system abstained (unreadable, or not every panel
    # seen). The app should show it as "needs another photo", not a pass.
    verdict = "compliant" if c is True else "non_compliant" if c is False else "undecided"
    common = next((d for d in decls if d.get("field_id") == "common_name" and d.get("present")), None)
    cal = payload.get("calibration") or {}

    def ev_url(f):
        p = f.get("evidence_crop_path")
        return f"/scans/{sid}/evidence/{Path(p).name}" if p else None

    retake = []
    if not payload.get("image_quality_ok", True):
        retake.append(payload.get("quality_notes") or "Image quality too low - retake.")
    for f in to_confirm:
        m = f.get("message", "")
        if "retake" in m.lower() or "closer" in m.lower():
            retake.append(m)

    return {
        "scan_id": sid,
        "created_at": payload.get("created_at"),
        "verdict": verdict,
        "product": {
            "name": product_name or (common or {}).get("raw_text") or "",
            "category": (payload.get("context") or {}).get("commodity_category", ""),
        },
        "counts": {"violations": sev, "to_confirm": len(to_confirm),
                   "passed": len(passed)},
        "checks": {"passed": len(passed), "failed": len(viol),
                   "to_confirm": len(to_confirm)},
        "findings": [
            {
                "rule_id": f.get("rule_id"), "citation": f.get("citation"),
                "outcome": f.get("outcome"), "severity": str(f.get("severity", "")).upper(),
                "field": f.get("field_id"), "message": f.get("message"),
                "measured": f.get("measured_value"), "required": f.get("required_value"),
                "unit": f.get("unit"), "confidence": f.get("confidence"),
                "evidence_url": ev_url(f),
            }
            for f in sorted(findings, key=lambda f: (
                _OUT_ORDER.get(f.get("outcome"), 9),
                _SEV_ORDER.get(str(f.get("severity", "")).upper(), 9)))
            if f.get("outcome") != "NOT_APPLICABLE"
        ],
        "declarations": [
            {"field": d.get("field_id"), "value": d.get("value"), "unit": d.get("unit"),
             "text": d.get("raw_text"), "source": d.get("source", "ocr"),
             "notes": d.get("notes") or []}
            for d in decls if d.get("present")
        ],
        "context": payload.get("context") or {},
        "summary": payload.get("summary") or {},
        "calibration": {"method": cal.get("method"), "px_per_mm": cal.get("px_per_mm"),
                        "note": cal.get("notes", "")},
        "retake_advice": list(dict.fromkeys(retake))[:3],
        "report_urls": {fmt: f"/scans/{sid}/report?fmt={fmt}"
                        for fmt in ("pdf", "html", "docx")},
    }


@app.get("/scans/{scan_id}/summary")
def scan_summary(scan_id: str, user: User = Depends(require("read"))):
    """The app view of a stored scan (History screen -> detail)."""
    with closing(db()) as conn:
        row = conn.execute("SELECT payload, product_name FROM scans WHERE scan_id = ?",
                           (scan_id,)).fetchone()
    if not row:
        raise HTTPException(404, "No such scan.")
    return app_summary(json.loads(row["payload"]), row["product_name"] or "")


@app.get("/scans/{scan_id}/evidence/{name}")
def evidence_image(scan_id: str, name: str, user: User = Depends(require_q("read"))):
    """An evidence crop, for the app to show next to its finding."""
    if not re.fullmatch(r"[0-9a-f]{6,32}", scan_id) or not re.fullmatch(r"[\w.-]+\.png", name):
        raise HTTPException(404, "No such evidence image.")
    base = (Path("data/evidence") / scan_id).resolve()
    path = (base / name).resolve()
    if base not in path.parents or not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(404, "No such evidence image.")
    return FileResponse(path, media_type="image/png")


@app.get("/scans/{scan_id}/image")
def scan_image(scan_id: str, user: User = Depends(require_q("read"))):
    """The photograph of the package captured during the scan."""
    with closing(db()) as conn:
        row = conn.execute("SELECT image_path FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    if not row or not row["image_path"]:
        raise HTTPException(404, "No image for this scan.")
    p = Path(row["image_path"]).resolve()
    if not p.exists():
        raise HTTPException(404, "Image file not found on disk.")
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        p.suffix.lower(), "image/png")
    return FileResponse(p, media_type=media)


@app.get("/scans")
def list_scans(
    q: str = Query("", description="search product name or category"),
    barcode: str = Query("", description="search barcode or UPC"),
    verdict: str = Query(""),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(require("read")),
):
    sql = "SELECT scan_id, created_at, product_name, category, verdict, n_violations, image_path FROM scans WHERE 1=1"
    args: list = []
    if q:
        sql += " AND (product_name LIKE ? OR category LIKE ? OR payload LIKE ?)"
        args += [f"%{q}%", f"%{q}%", f"%{q}%"]
    if barcode:
        sql += " AND payload LIKE ?"
        args.append(f"%{barcode}%")
    if verdict:
        sql += " AND verdict = ?"
        args.append(verdict)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    with closing(db()) as conn:
        scans = [dict(r) for r in conn.execute(sql, args)]
        for s in scans:
            if s.get("image_path") and Path(s["image_path"]).exists():
                s["image_url"] = f"/scans/{s['scan_id']}/image"
        return scans


# =====================================================================
# Manual UPC / barcode lookup: real scans only
# =====================================================================
# (A hard-coded "registry" of three products with invented results used to
# answer here first, and was shown to inspectors as real inspection
# records - removed.)


@app.get("/scans/upc/{upc_code}")
def get_scan_by_upc(upc_code: str, user: User = Depends(require("read"))):
    """
    The latest real scan of a product, found by its barcode (GTIN read from
    the pack) or by scan id. 404 when this product has not been scanned.
    """
    clean_upc = upc_code.strip()
    with closing(db()) as conn:
        row = conn.execute(
            "SELECT payload, product_name FROM scans WHERE scan_id = ? "
            "OR json_extract(payload, '$.context.barcode') = ? "
            "OR ltrim(json_extract(payload, '$.context.barcode'), '0') = ltrim(?, '0') "
            "ORDER BY created_at DESC LIMIT 1",
            (clean_upc, clean_upc, clean_upc),
        ).fetchone()
        if row:
            payload = json.loads(row["payload"])
            payload["product_name"] = row["product_name"] or ""
            payload["app"] = app_summary(payload, row["product_name"] or "")
            payload["human_actions"] = [dict(r) for r in conn.execute(
                "SELECT rule_id, action, user_id, note, at FROM actions "
                "WHERE scan_id = ? ORDER BY at", (payload.get("scan_id"),))]
            return payload

    raise HTTPException(status_code=404, detail="No scan of this product yet.")


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str, user: User = Depends(require("read"))):
    with closing(db()) as conn:
        row = conn.execute(
            "SELECT payload, product_name FROM scans WHERE scan_id = ?", (scan_id,)
        ).fetchone()
        acts = [dict(r) for r in conn.execute(
            "SELECT rule_id, action, user_id, note, at FROM actions "
            "WHERE scan_id = ? ORDER BY at", (scan_id,))]
    if not row:
        raise HTTPException(404, "No such scan.")
    payload = json.loads(row["payload"])
    payload["product_name"] = row["product_name"] or ""
    payload["app"] = app_summary(payload, row["product_name"] or "")
    payload["human_actions"] = acts
    return payload


class ActionIn(BaseModel):
    rule_id: str
    action: str        # "confirmed" | "overridden"
    note: str = ""


class FlagIn(BaseModel):
    reason: Optional[str] = "Flagged for compliance review"
    rule_id: Optional[str] = "general_compliance"


class ReviewIn(BaseModel):
    action: str        # "FLAG_APPROVED" | "FLAG_REJECTED" | "approve" | "reject"
    note: Optional[str] = ""
    rule_id: Optional[str] = "general_compliance"


@app.post("/scans/{scan_id}/flag")
def flag_scan(
    scan_id: str,
    body: Optional[FlagIn] = None,
    authorization: Optional[str] = Header(None),
):
    """
    Inspector raises a flag on a product scan.
    Requires verified user (is_verified is True).
    """
    token = extract_bearer_token(authorization)
    if not token or token not in USER_DB:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token.")

    user = USER_DB[token]
    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=403,
            detail="User must be verified to perform official actions.",
        )
    if "scan" not in ROLES.get(TOKENS.get(token, ("", "guest"))[1], set()):
        raise HTTPException(403, "This role cannot raise flags.")
    _scan_must_exist(scan_id)

    note = (body.reason if body and body.reason else "Flagged for compliance review").strip()
    rule_id = (body.rule_id if body and body.rule_id else "general_compliance").strip()
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "timestamp": timestamp,
        "user_id": user["user_id"],
        "user_name": user["name"],
        "role": user["role"],
        "action": "FLAG_RAISED",
        "scan_id": scan_id,
        "note": note,
    }
    audit_logs.append(entry)

    try:
        with closing(db()) as conn:
            conn.execute(
                "INSERT INTO actions (scan_id, rule_id, action, user_id, note, at) VALUES (?,?,?,?,?,?)",
                (scan_id, rule_id, "flagged", user["user_id"], note, timestamp),
            )
            conn.commit()
    except Exception:
        pass

    return {
        "ok": True,
        "action": "FLAG_RAISED",
        "scan_id": scan_id,
        "recorded_by": user["name"],
        "log": entry,
    }


@app.post("/scans/{scan_id}/review")
def review_scan(
    scan_id: str,
    body: ReviewIn,
    authorization: Optional[str] = Header(None),
):
    """
    Supervisor reviews, approves, or rejects a flagged scan.
    Requires verified user (is_verified is True).
    """
    token = extract_bearer_token(authorization)
    if not token or token not in USER_DB:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token.")

    user = USER_DB[token]
    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=403,
            detail="User must be verified to perform official actions.",
        )

    # Approving or rejecting a flag is a supervisor decision, like an
    # override in /action; any other action name is refused, so this route
    # cannot be used to write an "overridden" record past that check.
    if "confirm" not in ROLES.get(TOKENS.get(token, ("", "guest"))[1], set()):
        raise HTTPException(403, "Only a supervisor can review flags.")
    _scan_must_exist(scan_id)
    act_str = (body.action or "").strip().lower()
    if act_str in ("approve", "approved", "flag_approved"):
        action_name = "FLAG_APPROVED"
    elif act_str in ("reject", "disapprove", "rejected", "disapproved", "flag_rejected"):
        action_name = "FLAG_REJECTED"
    elif act_str in ("", "review", "reviewed", "flag_reviewed"):
        action_name = "FLAG_REVIEWED"
    else:
        raise HTTPException(400, "action must be approve, reject or review.")
    with closing(db()) as conn:
        acts = [r["action"] for r in conn.execute(
            "SELECT action FROM actions WHERE scan_id = ? ORDER BY at", (scan_id,))]
    last_flag = max((i for i, a_ in enumerate(acts) if a_ == "flagged"), default=-1)
    open_flag = last_flag >= 0 and not any(
        a_ in ("flag_approved", "flag_rejected") for a_ in acts[last_flag + 1:])
    if not open_flag:
        raise HTTPException(409, "This scan has no open flag to review.")

    note = (body.note or f"Action: {action_name}").strip()
    rule_id = (body.rule_id or "general_compliance").strip()
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "timestamp": timestamp,
        "user_id": user["user_id"],
        "user_name": user["name"],
        "role": user["role"],
        "action": action_name,
        "scan_id": scan_id,
        "note": note,
    }
    audit_logs.append(entry)

    try:
        with closing(db()) as conn:
            conn.execute(
                "INSERT INTO actions (scan_id, rule_id, action, user_id, note, at) VALUES (?,?,?,?,?,?)",
                (scan_id, rule_id, action_name.lower(), user["user_id"], note, timestamp),
            )
            conn.commit()
    except Exception:
        pass

    return {
        "ok": True,
        "action": action_name,
        "scan_id": scan_id,
        "recorded_by": user["name"],
        "log": entry,
    }


def _scan_must_exist(scan_id: str) -> None:
    with closing(db()) as conn:
        if conn.execute("SELECT 1 FROM scans WHERE scan_id = ?", (scan_id,)).fetchone() is None:
            raise HTTPException(404, "No such scan.")


@app.get("/audit-logs")
def get_audit_logs(user: User = Depends(require("confirm"))):
    """
    Return all official audit trail log records.
    """
    return {"total": len(audit_logs), "logs": audit_logs}


@app.post("/scans/{scan_id}/action")
def record_action(
    scan_id: str,
    body: ActionIn,
    authorization: Optional[str] = Header(None),
    user: User = Depends(require("confirm")),
):
    token = extract_bearer_token(authorization)
    if token in USER_DB and not USER_DB[token].get("is_verified", False):
        raise HTTPException(
            status_code=403,
            detail="User must be verified to perform official actions.",
        )
    """
    A named human confirms or overrides a machine finding.

    Overriding requires the 'override' permission specifically, so an
    inspector cannot quietly dismiss a finding their supervisor has not
    seen. Both actions are appended, never updated - the audit trail is
    the product.
    """
    if body.action not in ("confirmed", "overridden"):
        raise HTTPException(400, "action must be 'confirmed' or 'overridden'.")
    if body.action == "overridden" and not user.can("override"):
        raise HTTPException(403, "Overriding a finding requires a supervisor.")
    if body.action == "overridden" and not body.note.strip():
        raise HTTPException(400, "An override must carry a written reason.")

    with closing(db()) as conn:
        if not conn.execute("SELECT 1 FROM scans WHERE scan_id = ?",
                            (scan_id,)).fetchone():
            raise HTTPException(404, "No such scan.")
        conn.execute(
            "INSERT INTO actions (scan_id, rule_id, action, user_id, note, at) "
            "VALUES (?,?,?,?,?,?)",
            (scan_id, body.rule_id, body.action, user.user_id, body.note,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    return {"ok": True, "recorded_by": user.user_id}


@app.get("/analysis/dual-mrp")
def dual_mrp(user: User = Depends(require("read"))):
    """
    Rule 18(2A) across the whole repository.

    Every other endpoint judges one image. This one cannot: a package
    showing Rs.120 is compliant on its own and becomes evidence only
    next to an identical one showing Rs.145. It is the reason the
    "repository of scanned products" in the problem statement is an
    evidence base rather than storage.
    """
    from ..core.cross_scan import CrossScanAnalyzer
    from ..report.builder import scan_from_dict

    with closing(db()) as conn:
        rows = conn.execute("SELECT payload FROM scans").fetchall()

    scans = []
    for r in rows:
        try:
            scans.append(scan_from_dict(json.loads(r["payload"])))
        except Exception:
            continue

    findings = CrossScanAnalyzer().findings(scans)
    return {
        "scans_compared": len(scans),
        "violations": [f.to_dict() for f in findings],
        "note": (
            "Findings without a barcode rest on brand, name and net quantity "
            "matching. Confirm the packages are genuinely identical before acting."
        ),
    }


class RelaxationIn(BaseModel):
    manufacturer: str
    rule_id: str
    reference: str = ""
    valid_until: str = ""      # ISO date; empty means no recorded expiry


@app.post("/relaxations")
def add_relaxation(
    body: RelaxationIn,
    user: User = Depends(require("manage")),
):
    """
    Record a Rule 33 relaxation granted to a manufacturer.

    Admin-only: a relaxation suppresses violations, so the ability to
    create one is the ability to make findings disappear. It is recorded
    against a specific rule and a validity date, never as a blanket
    exemption for a manufacturer.
    """
    if not body.manufacturer.strip() or not body.rule_id.strip():
        raise HTTPException(400, "manufacturer and rule_id are required.")
    with closing(db()) as conn:
        conn.execute(
            "INSERT INTO relaxations (manufacturer, rule_id, reference, valid_until) "
            "VALUES (?,?,?,?)",
            (body.manufacturer.strip(), body.rule_id.strip(),
             body.reference.strip(), body.valid_until.strip() or None),
        )
        conn.commit()
    return {"ok": True, "recorded_by": user.user_id}


@app.get("/relaxations")
def list_relaxations(user: User = Depends(require("read"))):
    with closing(db()) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT id, manufacturer, rule_id, reference, valid_until FROM relaxations"
        )]


class ListingIn(BaseModel):
    url: str = ""
    title: str = ""
    text: str
    platform: str = ""
    is_imported: bool = False
    has_searchable_coo_filter: bool = False
    has_sortable_coo_filter: bool = False


@app.post("/scan/listing")
def scan_listing(body: ListingIn, user: User = Depends(require("scan"))):
    """
    Rule 6(10) compliance for an e-commerce listing.

    No image, no calibration, no marker card. The same extractor and the
    same rules engine run against listing text, so citations match those
    on a package report for the same rule - but every rule about how a
    declaration is PRINTED is reported NOT_APPLICABLE with a reason,
    because a web page has no panel to measure.
    """
    from ..core.listing import Listing, ListingScanner

    listing = Listing(
        url=body.url, title=body.title, text=body.text,
        platform=body.platform, is_imported=body.is_imported,
        has_searchable_coo_filter=body.has_searchable_coo_filter,
        has_sortable_coo_filter=body.has_sortable_coo_filter,
    )
    res = ListingScanner().scan(listing, inspector_id=user.user_id)
    persist(res, body.title)
    return JSONResponse(res.to_dict())


@app.get("/dashboard")
def dashboard(user: User = Depends(require("read"))):
    with closing(db()) as conn:
        by_verdict = {r["verdict"]: r["n"] for r in conn.execute(
            "SELECT verdict, COUNT(*) n FROM scans GROUP BY verdict")}
        by_rule = [dict(r) for r in conn.execute(
            "SELECT rule_id, citation, COUNT(*) n FROM findings "
            "WHERE outcome='VIOLATION' GROUP BY rule_id ORDER BY n DESC LIMIT 15")]
        by_category = [dict(r) for r in conn.execute(
            "SELECT category, COUNT(*) n FROM scans WHERE verdict='non_compliant' "
            "GROUP BY category ORDER BY n DESC")]
        trend = [dict(r) for r in conn.execute(
            "SELECT substr(created_at,1,10) day, COUNT(*) n, "
            "SUM(CASE WHEN verdict='non_compliant' THEN 1 ELSE 0 END) bad "
            "FROM scans GROUP BY day ORDER BY day DESC LIMIT 30")]
        total = conn.execute("SELECT COUNT(*) n FROM scans").fetchone()["n"]
    return {
        "total_scans": total,
        "by_verdict": by_verdict,
        "top_violated_rules": by_rule,
        "non_compliant_by_category": by_category,
        "trend": trend,
        # Surfaced so nobody reads the dashboard as a complete picture.
        "coverage_note": RuleConfig().audit(),
    }


@app.get("/scans/{scan_id}/report")
def report(
    scan_id: str,
    fmt: str = Query("pdf", pattern="^(pdf|html|json|docx)$"),
    user: User = Depends(require_q("read")),
):
    from ..report.builder import build_report

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_scan_id = Path(scan_id).name

    # 1. The scan must exist (a cached file of a deleted scan is not served)
    with closing(db()) as conn:
        row = conn.execute("SELECT payload FROM scans WHERE scan_id = ?",
                           (scan_id,)).fetchone()
    if not row:
        raise HTTPException(404, "No such scan.")
    payload = json.loads(row["payload"])

    # 2. Cache, keyed by the report layout version: files made by an older
    # builder (e.g. before Hindi text) are not served for a new request.
    if fmt == "pdf":
        cached_pdf = REPORTS_DIR / f"{safe_scan_id}.{REPORT_VERSION}.pdf"
        if cached_pdf.exists() and cached_pdf.stat().st_size > 0:
            return FileResponse(
                cached_pdf,
                media_type="application/pdf",
                filename=f"METRA_Report_{safe_scan_id}.pdf",
            )

    if fmt == "json":
        return payload

    if fmt == "docx":
        docx_path = REPORTS_DIR / f"{safe_scan_id}.{REPORT_VERSION}.docx"
        if docx_path.exists() and docx_path.stat().st_size > 0:
            return FileResponse(
                docx_path,
                media_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                filename=docx_path.name,
            )
        from ..report.builder import save_docx, scan_from_dict
        path = save_docx(scan_from_dict(payload), docx_path)
        if not path:
            raise HTTPException(
                501, "DOCX export unavailable: python-docx is not installed."
            )
        return FileResponse(
            path,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            filename=Path(path).name,
        )

    # 3. Generate report and physically save to data/reports/{scan_id}.{fmt}
    target_path = REPORTS_DIR / f"{safe_scan_id}.{REPORT_VERSION}.{fmt}"
    try:
        path = build_report(payload, target_path, fmt=fmt)
    except Exception:
        logging.getLogger(__name__).exception("Report generation failed for %s", scan_id)
        raise HTTPException(500, "Report generation failed; see the server log.")

    if fmt == "pdf":
        if not path or not Path(path).exists() or Path(path).suffix.lower() != ".pdf":
            raise HTTPException(500, "PDF generation failed to produce a valid PDF file.")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=f"METRA_Report_{safe_scan_id}.pdf",
        )

    return FileResponse(
        path,
        media_type="text/html",
        filename=Path(path).name,
    )
