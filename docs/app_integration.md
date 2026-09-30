# Connecting a mobile app to the METRA backend

The app takes the photo, the backend does everything else: OCR, rules, evidence crops, reports.
All of it is plain HTTP + JSON, so the stack doesn't matter (Flutter, React Native, Kotlin).

---

## 1. Run the backend so a phone can reach it

On the laptop (same Wi-Fi as the phone):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` is what makes it reachable from the phone. Without it only the laptop can connect.
- Find the laptop's IP: `ipconfig` (Windows) → "IPv4 Address", e.g. `192.168.1.23`.
- From the phone's browser, open `http://192.168.1.23:8000/health`. If that loads, the app can connect too.
- Windows asks whether to let Python through the firewall. Allow it on **Private** networks.
- Android emulator: use `http://10.0.2.2:8000`, not localhost.
- Android blocks plain `http://` by default. For development, add `android:usesCleartextTraffic="true"` to `<application>` in `AndroidManifest.xml`. For a demo on the open internet, use HTTPS instead, e.g. `cloudflared tunnel --url http://localhost:8000`.
- The OCR models load in the background when the server starts (~30–60 s the first time). A scan sent before then waits for them. A scan takes ~25–30 s on a fast CPU and longer on a laptop, so set the app's request timeout to **several minutes** (the web app uses 6).

Optional, set on the server:

```bash
set SIH_CORS_ORIGINS=https://your-web-app.example   # web builds on ANOTHER origin only; off by default
```

---

## 2. Auth

Every call needs a header:

```
Authorization: Bearer demo-inspector
```

The demo tokens are `demo-inspector` (scan + read), `demo-supervisor` (+ confirm / override) and `demo-admin`.
They demonstrate the role model; production sign-in replaces the `TOKENS` dict in `src/api/main.py`.

An `<img src>` cannot send a header, so image links take the token in the
URL instead: `/scans/{id}/image?token=demo-inspector` (the METRA web app does this).
Only the three GET downloads accept `?token=`: the scan image, evidence crops
and reports. Every other call needs the `Authorization: Bearer` header.

Roles cannot be raised from the app: a profile edit or an official
verification can lower a sign-in's role or restore it, never go above the
role the token was issued with. Approving / rejecting flags (`/review`)
needs a supervisor, and `/audit-logs` needs a supervisor sign-in.

---

## 3. Screens → endpoints

| App screen | Call | Notes |
|---|---|---|
| **Scan** → take photo | `POST /scan` (multipart, field `file`) | Returns the full result plus an `app` block, shaped for the screens (below) |
| Scan → "Enter manually" size | same call with `?panel_width_mm=95&panel_height_mm=70` | A ruler measurement of the panel gives a mm scale when there's no printed marker in the frame |
| Scan → all panels photographed | `?coverage_complete=true` | Only then can "missing" become a violation instead of "to confirm" |
| **Scan a whole pack** (front, back, top, base, crimp) | `POST /scan_package` (multipart, field `files`, one per photo) | The pack is decided from all photos together. Each declaration says which photo it came from. Use this for cans, cartons, tubes and jars |
| **Result** detail | the `app` block, or `GET /scans/{id}/summary` | |
| Evidence thumbnail | `GET {evidence_url}` | PNG crop. Send the auth header with the image request |
| **History** | `GET /scans?q=maggi&verdict=non_compliant&limit=50` | |
| **Dashboard** | `GET /dashboard` | Totals, top violated rules, 30-day trend |
| Report share / download | `GET /scans/{id}/report?fmt=pdf` (or `html`, `docx`) | |
| Supervisor confirm / override | `POST /scans/{id}/action` `{"rule_id": "...", "action": "confirmed" \| "overridden", "note": "..."}` | Needs the supervisor token. Every action is logged |

---

## 4. The `app` block

```json
{
  "scan_id": "660d13b51f73",
  "verdict": "non_compliant",
  "product": {"name": "Atta Maggi", "category": "food_packaged"},
  "counts": {"violations": {"CRITICAL": 1, "MAJOR": 2, "MINOR": 0},
             "to_confirm": 4, "passed": 11},
  "checks": {"passed": 11, "failed": 3, "to_confirm": 4},
  "findings": [
    {"rule_id": "retail_sale_price.required_phrasing", "citation": "Rule 6(1)(e) ...",
     "outcome": "VIOLATION", "severity": "CRITICAL", "field": "retail_sale_price",
     "message": "Price stated without the required 'inclusive of all taxes' qualifier.",
     "measured": null, "required": null, "unit": "", "confidence": 0.9,
     "evidence_url": "/scans/660d13b51f73/evidence/660d13b51f73_03_retail_sale_price_required_phrasing.png"}
  ],
  "declarations": [
    {"field": "net_quantity", "value": 6.0, "unit": "g", "text": "NET QUANTITY: 69",
     "source": "ocr", "notes": ["Unit 'g' inferred from a '9' glyph ... confirm on the pack."]}
  ],
  "calibration": {"method": "none", "px_per_mm": null, "note": "No calibration marker detected in frame."},
  "retake_advice": ["Text is only 11 px tall in this photo - ... Retake this panel closer."],
  "report_urls": {"pdf": "/scans/.../report?fmt=pdf", "html": "...", "docx": "..."}
}
```

How to show it:

- **`verdict`** is `compliant`, `non_compliant` or `undecided`. Show `undecided` as "needs another photo", never as a pass.
- **`outcome`** per finding:
  - `VIOLATION`: red.
  - `INDETERMINATE`: "confirm by eye", amber. The system could not decide from this photo. Show the message; it says why.
  - `COMPLIANT`: green.
- **`notes`** on a declaration: anything the reader was unsure of (a unit inferred from a digit, a repaired spelling). Show them under the value.
- **`retake_advice`**: show it as a banner on the result. It's the "guided capture" message.
- **`capture_advice`** (single-photo scans): what to photograph next, e.g. "The pack says to see the base - photograph that side too and scan all photos together as one pack". Show it with a button that adds a photo to the same pack and calls `/scan_package`.
- **`verdict_note`**: one line saying why there is or isn't a verdict ("No violation found. Not read on this photo: retail sale price. 4 size/spacing checks need the calibration card.").
- **Score**: there is deliberately no made-up score. If the design needs a number, use `checks.passed` of `passed + failed`, a count anyone can check.

---

## 5. Minimal client code

**Flutter** (`http` package):

```dart
final req = http.MultipartRequest('POST', Uri.parse('$base/scan?product_name=$name'))
  ..headers['Authorization'] = 'Bearer demo-inspector'
  ..files.add(await http.MultipartFile.fromPath('file', photo.path));
final res = await http.Response.fromStream(await req.send().timeout(const Duration(minutes: 6)));
final app = jsonDecode(res.body)['app'];

// evidence image
Image.network('$base${f['evidence_url']}', headers: {'Authorization': 'Bearer demo-inspector'});
```

**React Native / Expo**:

```js
const form = new FormData();
form.append('file', { uri: photo.uri, name: 'scan.jpg', type: 'image/jpeg' });
const res = await fetch(`${BASE}/scan?product_name=${encodeURIComponent(name)}`, {
  method: 'POST', headers: { Authorization: 'Bearer demo-inspector' }, body: form,
});
const { app } = await res.json();

<Image source={{ uri: BASE + f.evidence_url, headers: { Authorization: 'Bearer demo-inspector' } }} />
```

Don't set `Content-Type` yourself for multipart. The library adds the boundary.

---

## 6. Error codes the app should handle

| Code | Meaning | App action |
|---|---|---|
| 400 | Empty or undecodable image | "Couldn't read the photo, try again" |
| 401 / 403 | Missing/invalid token, or role not allowed | Back to login |
| 413 | Image over 25 MB or over 50 megapixels; a pack over 8 photos or 100 MB | Compress to JPEG ~2000-4000 px long side before upload (also makes scans faster) |
| 415 | HEIC / AVIF photo | Set the camera to JPEG ("Most Compatible" on iPhone), or convert before upload |
| 400 "arrived incomplete" | The JPEG was cut off in transit | Upload again |
| 404 | Unknown scan / evidence | Refresh history |
| 500 with "PaddleOCR is not installed" | Server set-up problem | Server-side: `pip install -r requirements.txt` |

---

## 7. Not provided by the backend

- **Weight checks** ("measured weight below declared"). A photo cannot weigh a pack; this needs a scale reading typed in or sent from a connected balance.
- **Expired stock as an LMPC violation.** When the use-by date has passed, the finding `expiry_date.not_expired` is returned with outcome `UNVERIFIED_RULE`. LMPC Rule 6(1)(da) requires the date to be declared; selling expired stock is a food-safety matter. Show it as "expired - refer to food safety".
- **A numeric risk score.** Use the verdict and counts (section 4).
