"""CPU-only Modal readiness checks for HiDream-O1-Image.

This intentionally does not download model weights and does not request a GPU.
It verifies that Modal can run, the fork files are present, and Hugging Face
metadata for the requested checkpoint is reachable.
"""

from __future__ import annotations

import json
import pathlib

import modal

APP_NAME = "jimsky-hidream-o1-readiness"
DEFAULT_MODEL_ID = "HiDream-ai/HiDream-O1-Image-Dev"
REPO_DIR = pathlib.Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.25.0")
    .add_local_dir(str(REPO_DIR), remote_path="/workspace/HiDream-O1-Image", copy=True)
)
outputs = modal.Volume.from_name("outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    timeout=180,
    volumes={"/outputs": outputs, "/cache": hf_cache},
)
def readiness(model_id: str = DEFAULT_MODEL_ID) -> str:
    import datetime as _dt
    import importlib.util
    import os
    import platform
    from huggingface_hub import HfApi

    repo_root = pathlib.Path("/workspace/HiDream-O1-Image")
    required = [
        "README.md",
        "requirements.txt",
        "inference.py",
        "models/pipeline.py",
        "models/qwen3_vl_transformers.py",
    ]
    present = {item: (repo_root / item).exists() for item in required}

    api = HfApi(token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
    info = api.model_info(model_id, files_metadata=False)
    siblings = [s.rfilename for s in getattr(info, "siblings", []) or []]
    expected_config_files = [
        name for name in siblings if name.endswith(("config.json", "processor_config.json", "tokenizer_config.json"))
    ]

    payload = {
        "ok": all(present.values()),
        "app": APP_NAME,
        "purpose": "CPU-only fork/readiness check; no GPU and no model weight download.",
        "utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "python": platform.python_version(),
        "modal_volume_outputs_mounted": pathlib.Path("/outputs").exists(),
        "modal_volume_hf_cache_mounted": pathlib.Path("/cache").exists(),
        "repo_files_present": present,
        "model_id_checked": model_id,
        "hf_model_private": bool(getattr(info, "private", False)),
        "hf_model_sha": getattr(info, "sha", None),
        "hf_file_count": len(siblings),
        "hf_config_like_files": expected_config_files[:20],
        "secret_presence_only": {
            "HF_TOKEN": bool(os.getenv("HF_TOKEN")),
            "HUGGINGFACE_TOKEN": bool(os.getenv("HUGGINGFACE_TOKEN")),
        },
        "gpu_started": False,
        "model_download_started": False,
    }
    out = pathlib.Path("/outputs/hidream_o1_readiness.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs.commit()
    return json.dumps(payload)


@app.local_entrypoint()
def main(model_id: str = DEFAULT_MODEL_ID):
    print(json.dumps(json.loads(readiness.remote(model_id)), indent=2))
