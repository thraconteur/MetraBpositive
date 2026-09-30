"""
Test configuration.

The suite never loads PaddleOCR. OCR is replaced by SyntheticOCR, which
returns the exact lines the synthetic renderer drew (line-level, like
Paddle), so every test measures the pipeline DOWNSTREAM of OCR
deterministically and in seconds. How well Paddle itself reads real
packaging is measured on real photos with scripts/scan_photo.py, not
here.
"""

import os
import tempfile

os.environ.setdefault("SIH_OCR_BACKEND", "synthetic")
os.environ.setdefault(
    "SIH_SYNTH_REGISTRY", os.path.join(tempfile.gettempdir(), "sih_synth_registry_tests")
)
