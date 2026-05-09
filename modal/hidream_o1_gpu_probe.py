"""Tiny GPU dependency probe for HiDream-O1-Image on Modal.

This checks CUDA/Torch and critical Python imports only. It does not download
HiDream model weights and it does not run image generation.
"""

from __future__ import annotations

import json
import pathlib

import modal

APP_NAME = "jimsky-hidream-o1-gpu-probe"
REPO_DIR = pathlib.Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "transformers==4.57.1",
        "diffusers",
        "accelerate",
        "einops",
        "numpy",
        "pillow",
        "tqdm",
        "scipy",
    )
    .add_local_dir(str(REPO_DIR), remote_path="/workspace/HiDream-O1-Image", copy=True)
)
outputs = modal.Volume.from_name("outputs", create_if_missing=True)


@app.function(image=image, gpu="L4", cpu=2.0, memory=8192, timeout=600, volumes={"/outputs": outputs})
def gpu_probe() -> str:
    import datetime as _dt
    import importlib
    import os
    import pathlib as _pathlib
    import platform
    import subprocess
    import sys

    sys.path.insert(0, "/workspace/HiDream-O1-Image")
    import torch

    nvidia_smi = subprocess.check_output(
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
        shell=True,
        text=True,
    ).strip()
    imports = {}
    for name in [
        "transformers",
        "diffusers",
        "accelerate",
        "einops",
        "PIL",
        "models.pipeline",
        "models.qwen3_vl_transformers",
    ]:
        try:
            importlib.import_module(name)
            imports[name] = True
        except Exception as exc:  # return primitive, not exception object
            imports[name] = f"ERROR: {type(exc).__name__}: {str(exc)[:240]}"

    payload = {
        "ok": bool(torch.cuda.is_available()) and all(v is True for v in imports.values()),
        "app": APP_NAME,
        "purpose": "GPU dependency/import probe only; no model weight download and no image generation.",
        "utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi,
        "imports": imports,
        "repo_present": _pathlib.Path("/workspace/HiDream-O1-Image/inference.py").exists(),
        "gpu_started": True,
        "model_download_started": False,
        "generation_started": False,
        "secret_presence_only": {
            "HF_TOKEN": bool(os.getenv("HF_TOKEN")),
            "HUGGINGFACE_TOKEN": bool(os.getenv("HUGGINGFACE_TOKEN")),
        },
    }
    out = _pathlib.Path("/outputs/hidream_o1_gpu_probe.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs.commit()
    return json.dumps(payload)


@app.local_entrypoint()
def main():
    print(json.dumps(json.loads(gpu_probe.remote()), indent=2))
