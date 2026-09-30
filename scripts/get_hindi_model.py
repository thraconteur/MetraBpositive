#!/usr/bin/env python3
"""
Download the Hindi (Devanagari) text reader once, so the Hindi re-read works.

    python scripts/get_hindi_model.py

Why this exists: PaddleX checks each model host with a ONE-second ping before
downloading. On a slow connection every host "fails" that ping, PaddleX says
"No available model hosting platforms detected", and the scan quietly goes on
in English only. This script skips that ping and tries three sources in turn:

  1. Baidu's model store (a single .tar, the same file PaddleX would fetch)
  2. Hugging Face  (PaddlePaddle/devanagari_PP-OCRv5_mobile_rec)
  3. ModelScope    (same repository name)

The model lands where PaddleOCR looks for it:
    ~/.paddlex/official_models/devanagari_PP-OCRv5_mobile_rec
(on Windows: C:\\Users\\<you>\\.paddlex\\official_models\\...)

If all three fail, it prints the link to download by hand.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

MODEL = "devanagari_PP-OCRv5_mobile_rec"
BOS_URL = ("https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
           f"paddle3.0.0/{MODEL}_infer.tar")
HF_URL = f"https://huggingface.co/PaddlePaddle/{MODEL}/tree/main"
NEEDED = ("inference.pdiparams", "inference.yml")      # plus inference.json or .pdmodel


def target_dir() -> Path:
    home = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".paddlex")
    return Path(home) / "official_models" / MODEL


def complete(d: Path) -> bool:
    return (d.is_dir() and all((d / f).exists() for f in NEEDED)
            and ((d / "inference.json").exists() or (d / "inference.pdmodel").exists()))


def from_bos(dest: Path) -> None:
    import requests

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "model.tar"
        print(f"  downloading {BOS_URL}")
        with requests.get(BOS_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        out = Path(td) / "x"
        with tarfile.open(tar_path) as t:
            t.extractall(out)
        # The tar holds one folder (devanagari_..._infer); its files are the model.
        inner = next((p for p in out.rglob("inference.yml")), None)
        if inner is None:
            raise RuntimeError("the download has no inference.yml in it")
        shutil.copytree(inner.parent, dest, dirs_exist_ok=True)


def from_hf(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"  downloading PaddlePaddle/{MODEL} from Hugging Face")
    snapshot_download(repo_id=f"PaddlePaddle/{MODEL}", local_dir=str(dest))


def from_modelscope(dest: Path) -> None:
    from modelscope import snapshot_download

    print(f"  downloading PaddlePaddle/{MODEL} from ModelScope")
    snapshot_download(model_id=f"PaddlePaddle/{MODEL}", local_dir=str(dest))


def check_loads() -> None:
    import numpy as np
    from paddleocr import TextRecognition

    rec = TextRecognition(model_name=MODEL, enable_mkldnn=False)
    list(rec.predict([np.full((48, 160, 3), 255, np.uint8)]))


def main() -> int:
    dest = target_dir()
    print(f"Hindi reader: {MODEL}\nFolder      : {dest}\n")
    if complete(dest):
        print("Already downloaded.")
    else:
        errors = []
        for name, fn in (("Baidu", from_bos), ("Hugging Face", from_hf), ("ModelScope", from_modelscope)):
            print(f"Trying {name} ...")
            try:
                if dest.exists() and not complete(dest):
                    shutil.rmtree(dest)                # half-finished earlier try
                dest.parent.mkdir(parents=True, exist_ok=True)
                fn(dest)
                if complete(dest):
                    print(f"  OK from {name}\n")
                    break
                errors.append(f"{name}: files incomplete")
            except Exception as exc:                   # try the next source
                msg = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
                errors.append(f"{name}: {msg}")
                print(f"  failed: {msg}")
        else:
            if dest.exists() and not complete(dest):
                shutil.rmtree(dest, ignore_errors=True)
            print("\nCould not download it automatically:")
            for e in errors:
                print(f"  - {e}")
            print(f"""
Do it by hand (2 minutes):
  1. Open this link in the browser; it downloads a .tar file (~8 MB):
       {BOS_URL}
     (if that is slow or blocked, the same model is at {HF_URL} -
      download every file listed there)
  2. Open the .tar (Windows 11 opens it like a zip; or in cmd:
       tar -xf {MODEL}_infer.tar )
  3. Rename the folder inside from "{MODEL}_infer" to
       {MODEL}
     and move it into
       {dest.parent}
     so that {dest / 'inference.yml'} exists.
  4. Run this script again - it checks the model loads.""")
            return 1

    print("Checking the model loads ...")
    try:
        check_loads()
    except Exception as exc:
        print(f"  It is on disk but did not load: {str(exc).splitlines()[0][:200]}")
        print(f"  Delete {dest} and run this script again.")
        return 1
    print("  OK. The Hindi re-read is ready - every scan now uses it "
          "(look for '+ hi re-read' on the engine line).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
