"""Gated HiDream-O1-Image generation wrapper for Modal.

Default behavior is safe: the CLI refuses to download model weights or run
inference unless both --allow_model_download and --allow_generation are passed.
Use the separate readiness/gpu_probe scripts first.
"""

from __future__ import annotations

import json
import pathlib

import modal

APP_NAME = "jimsky-hidream-o1-generation"
DEFAULT_MODEL_ID = "HiDream-ai/HiDream-O1-Image-Dev"
REPO_DIR = pathlib.Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
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
        "huggingface_hub>=0.25.0",
    )
    .add_local_dir(str(REPO_DIR), remote_path="/workspace/HiDream-O1-Image", copy=True)
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
outputs = modal.Volume.from_name("outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8.0,
    memory=65536,
    timeout=1800,
    scaledown_window=60,
    volumes={"/cache": hf_cache, "/outputs": outputs},
)
def generate_smoke(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    width: int = 512,
    height: int = 512,
    seed: int = 111,
    allow_model_download: bool = False,
    allow_generation: bool = False,
) -> str:
    import datetime as _dt
    import hashlib
    import os
    import pathlib as _pathlib
    import subprocess
    import sys
    import time

    if not allow_model_download or not allow_generation:
        return json.dumps(
            {
                "ok": False,
                "app": APP_NAME,
                "refused": True,
                "reason": "Pass both allow_model_download=True and allow_generation=True for a real paid GPU/model run.",
                "gpu_started": True,
                "model_download_started": False,
                "generation_started": False,
            }
        )

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    os.environ.setdefault("HF_HUB_CACHE", "/cache/huggingface/hub")
    sys.path.insert(0, "/workspace/HiDream-O1-Image")

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from models.pipeline import DEFAULT_TIMESTEPS, generate_image
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    from inference import add_special_tokens, get_tokenizer

    t0 = time.time()
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=f"/cache/huggingface/snapshots/{model_id.replace('/', '__')}",
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )
    hf_cache.commit()

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cuda"
    ).eval()
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    is_dev = model_id.endswith("-Dev")
    image_obj: Image.Image = generate_image(
        model=model,
        processor=processor,
        prompt=prompt,
        ref_image_paths=[],
        height=height,
        width=width,
        num_inference_steps=28 if is_dev else 50,
        guidance_scale=0.0 if is_dev else 5.0,
        shift=1.0 if is_dev else 3.0,
        timesteps_list=DEFAULT_TIMESTEPS if is_dev else None,
        scheduler_name="flash" if is_dev else "default",
        seed=seed,
        keep_original_aspect=False,
    )

    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = _pathlib.Path("/outputs/hidream_o1") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke.png"
    image_obj.save(out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    probe = subprocess.check_output("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader", shell=True, text=True).strip()
    payload = {
        "ok": True,
        "app": APP_NAME,
        "model_id": model_id,
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "elapsed_seconds": round(time.time() - t0, 3),
        "output_path": str(out_path),
        "sha256": digest,
        "nvidia_smi": probe,
        "gpu_started": True,
        "model_download_started": True,
        "generation_started": True,
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs.commit()
    return json.dumps(payload)


@app.local_entrypoint()
def main(
    prompt: str = "A small gold robot DJ holding a sign that says BASS ORACLE 111, studio product photo, crisp typography",
    model_id: str = DEFAULT_MODEL_ID,
    width: int = 512,
    height: int = 512,
    seed: int = 111,
    allow_model_download: bool = False,
    allow_generation: bool = False,
):
    print(
        json.dumps(
            json.loads(
                generate_smoke.remote(
                    prompt=prompt,
                    model_id=model_id,
                    width=width,
                    height=height,
                    seed=seed,
                    allow_model_download=allow_model_download,
                    allow_generation=allow_generation,
                )
            ),
            indent=2,
        )
    )
