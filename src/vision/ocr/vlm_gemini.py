"""
Gemini as a second reader for declarations PaddleOCR missed.

What this is for: text OCR genuinely cannot read - wrapped round a
curved bottle, half in glare, in a font Paddle has never seen. A vision-
language model often can. What it must never do is decide compliance on
its own: a VLM can "read" words that are not printed (it knows MRP lines
usually end in "inclusive of all taxes"), and a report built on that is
a report of what packs usually say.

So every read comes back with WHERE it was seen (a box), is checked
against OCR at that spot, and is marked on the declaration:

    source = "vlm+ocr"  OCR read the same text in the same place
    source = "vlm"      only the model saw it - findings that depend on
                        it are INDETERMINATE, with the model's text and a
                        crop of the spot, for the officer to confirm

Works with either Google SDK:
    pip install google-genai          (new:  from google import genai)
    pip install google-generativeai   (old:  import google.generativeai)
Key from GEMINI_API_KEY (or GOOGLE_API_KEY). Model from SIH_VLM_MODEL,
default gemini-2.5-flash.

Needs internet. The rest of the pipeline does not.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FIELD_HINTS = {
    "retail_sale_price": "maximum retail price (MRP) with its currency and any tax wording printed next to it",
    "net_quantity": "net quantity with its unit (e.g. '6 g', '45 ml', '1 piece')",
    "unit_sale_price": "unit sale price (e.g. 'Rs.0.29/g', '(Rs.0.40/ml)')",
    "manufacture_date": "month and year of manufacture or packing, including coded inkjet dates",
    "manufacturer_details": "name and full address of the manufacturer / packer / importer",
    "consumer_care": "consumer care contact: address, phone number and/or e-mail",
    "common_name": "generic name of the product (not the brand)",
    "country_of_origin": "country of origin (imported goods)",
}

PROMPT = """You are reading the label on a photographed retail package for a legal
inspection. For each field below, copy EXACTLY the characters printed on the
package, including currency symbols and punctuation. Do not correct, complete,
translate or reword anything. Never add words that are not printed - in
particular do not add "inclusive of all taxes" unless you can see it. If a
field is not visible or not legible, use null.

Fields:
{fields}

Return ONLY a JSON array, one object per field:
[{{"field": "<field name>", "text": "<exact text or null>",
   "box_2d": [ymin, xmin, ymax, xmax]}}]
box_2d is where that text is in the image, each value 0-1000, or null."""


class VLMUnavailable(RuntimeError):
    pass


class GeminiReader:
    name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 client=None):
        self.model = model or os.environ.get("SIH_VLM_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY")
        self._client = client          # tests inject a fake with .generate(prompt, image)
        self._sdk = None

    # -- setup ------------------------------------------------------------
    def check(self) -> None:
        """Raise VLMUnavailable with the exact fix, instead of failing silently."""
        if self._client is not None:
            return
        if not self.api_key:
            raise VLMUnavailable(
                "Gemini was requested but no API key is set. Set GEMINI_API_KEY "
                "(PowerShell: $env:GEMINI_API_KEY=\"...\"; bash: export GEMINI_API_KEY=...).")
        try:
            from google import genai  # noqa: F401
            self._sdk = "genai"
            return
        except Exception:
            pass
        try:
            import google.generativeai  # noqa: F401
            self._sdk = "generativeai"
            return
        except Exception:
            pass
        raise VLMUnavailable(
            "No Google Gemini SDK is installed. Run:  pip install google-genai")

    def label(self) -> str:
        return f"{self.name}:{self.model}"

    # -- call -------------------------------------------------------------
    def _generate(self, prompt: str, image_bgr: np.ndarray) -> str:
        if self._client is not None:
            return self._client.generate(prompt, image_bgr)
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if self._sdk is None:
            self.check()
        if self._sdk == "genai":
            from google import genai

            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model, contents=[pil, prompt],
                config={"response_mime_type": "application/json", "temperature": 0},
            )
            return resp.text or ""
        import google.generativeai as genai_old

        genai_old.configure(api_key=self.api_key)
        model = genai_old.GenerativeModel(self.model)
        resp = model.generate_content(
            [prompt, pil],
            generation_config={"response_mime_type": "application/json", "temperature": 0},
        )
        return resp.text or ""

    def read_fields(self, image_bgr: np.ndarray, fields: list[str]) -> list[dict]:
        """
        [{"field", "text", "box": (x0, y0, x1, y1) in image pixels or None}]
        Fields the model could not see are left out.
        """
        if not fields:
            return []
        lines = "\n".join(f"- {f}: {FIELD_HINTS.get(f, f)}" for f in fields)
        raw = self._generate(PROMPT.format(fields=lines), image_bgr)
        items = _parse_json_list(raw)
        H, W = image_bgr.shape[:2]
        out = []
        for it in items:
            field = str(it.get("field", "")).strip()
            text = it.get("text")
            if field not in fields or not text or not str(text).strip():
                continue
            box = None
            b = it.get("box_2d")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    ymin, xmin, ymax, xmax = (float(v) for v in b)
                    x0, x1 = sorted((xmin / 1000 * W, xmax / 1000 * W))
                    y0, y1 = sorted((ymin / 1000 * H, ymax / 1000 * H))
                    if x1 - x0 >= 3 and y1 - y0 >= 3:
                        box = (max(0.0, x0), max(0.0, y0), min(W, x1), min(H, y1))
                except (TypeError, ValueError):
                    box = None
            out.append({"field": field, "text": str(text).strip(), "box": box})
        return out


def _parse_json_list(raw: str) -> list[dict]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            logger.warning("Gemini returned no JSON: %.200s", raw)
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.warning("Gemini returned unparseable JSON: %.200s", raw)
            return []
    if isinstance(data, dict):
        data = data.get("fields") or data.get("items") or [data]
    return [d for d in data if isinstance(d, dict)]
