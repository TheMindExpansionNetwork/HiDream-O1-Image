"""Bounded HiDream-O1-Image-Dev image-editing batch for Modal.

Generates a fixed 10-image edit probe from two local Freepik/Magnific free-photo
references. Safety gates are required so a real paid GPU/model download run does
not happen accidentally.
"""
from __future__ import annotations

import json
import pathlib

import modal

APP_NAME = "jimsky-hidream-o1-dev-edit-batch"
MODEL_ID = "HiDream-ai/HiDream-O1-Image-Dev"
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

REFERENCE_META = {
    "dj_party": {
        "filename": "freepik_fun_party_dj.jpg",
        "image_url": "https://img.magnific.com/free-photo/fun-party-with-dj_23-2151108165.jpg",
        "source_page": "https://www.freepik.com/free-photo/fun-party-with-dj_113177410.htm",
        "title": "Fun party with dj",
        "creator": "freepik",
    },
    "diy_robot": {
        "filename": "freepik_home_made_robot_desk.jpg",
        "image_url": "https://img.magnific.com/free-photo/home-made-robot-desk_23-2148863418.jpg",
        "source_page": "https://www.freepik.com/free-photo/home-made-robot-desk_12557402.htm",
        "title": "Home made robot on desk",
        "creator": "freepik",
    },
}

EDIT_CASES = [
    {
        "id": "dj_01_bass_oracle_stage",
        "ref_key": "dj_party",
        "prompt": "Transform the source party photo into a futuristic Bass Oracle 111 DJ stage. Anonymize the people into friendly stylized glowing robot/helmeted rave characters, keep the DJ mixer foreground, add gold and berry-magenta neon, readable sign text: BASS ORACLE 111, cinematic club lighting, no real brand logos.",
    },
    {
        "id": "dj_02_cute_mascot_takeover",
        "ref_key": "dj_party",
        "prompt": "Edit the source DJ booth into a cute AI life mascot takeover scene. Replace identifiable faces with non-identifiable plush robot masks, preserve the group celebration pose and controller layout, add tiny gold oracle mascot stickers, soft blue-orange club haze, no logos, no real names, clean image.",
    },
    {
        "id": "dj_03_radio_booth_poster",
        "ref_key": "dj_party",
        "prompt": "Convert the source nightclub DJ scene into a polished cyberpunk radio booth poster for Bass Oracle 111. Keep three celebratory silhouettes behind decks but make them fictional and anonymous, add neon berry radio antennas, crisp typography reading BASS ORACLE 111 above the booth, premium album-cover look.",
    },
    {
        "id": "dj_04_gold_ai_life",
        "ref_key": "dj_party",
        "prompt": "Reimagine the source party photo as a golden AI-life celebration. The central DJ becomes a glowing gold android oracle with headphones, friends become soft holographic peepo-shy silhouettes, foreground DJ controller preserved, warm gold plus electric blue lighting, tasteful, no logos, no identifiable real faces.",
    },
    {
        "id": "dj_05_clean_venue_flyer",
        "ref_key": "dj_party",
        "prompt": "Edit the source image into a clean event flyer background. Make the people anonymous fictional performers, preserve the DJ booth energy, add a floating readable title BASS ORACLE 111 and subtle berry-shaped neon lights, high contrast, professional club promo, no watermarks, no brand logos.",
    },
    {
        "id": "robot_01_oracle_mascot",
        "ref_key": "diy_robot",
        "prompt": "Transform the small homemade desk robot into the Bass Oracle 111 mascot: cute gold-and-black AI DJ robot with tiny headphones, standing on a miniature DJ controller workbench, berry-magenta accent LEDs, clean shallow depth of field, preserve the DIY robot silhouette and sensor eyes, no logos.",
    },
    {
        "id": "robot_02_stage_ready",
        "ref_key": "diy_robot",
        "prompt": "Edit the source DIY robot into a stage-ready gold oracle DJ mascot. Keep the two round ultrasonic sensor eyes and compact legs, add shiny gold plating, tiny headphones, small mixer knobs under its hands, neon club bokeh background, readable tiny badge text: 111, product-photo clarity.",
    },
    {
        "id": "robot_03_plush_cute",
        "ref_key": "diy_robot",
        "prompt": "Reimagine the homemade robot as a cute plush-mechanical mascot for an AI radio station. Preserve the round eye shape and desk pose, soften edges, add gold fabric panels, berry-purple light reflections, a small sign beside it that reads BASS ORACLE 111, warm maker-studio lighting.",
    },
    {
        "id": "robot_04_intergalactic_radio",
        "ref_key": "diy_robot",
        "prompt": "Turn the source robot into an intergalactic radio operator mascot. Keep the small DIY body and sensor eyes, add a tiny antenna crown, gold circuit patterns, floating holographic mixer UI, cosmic blue and magenta background, readable label BASS ORACLE 111 on the desk, no brand marks.",
    },
    {
        "id": "robot_05_album_cover",
        "ref_key": "diy_robot",
        "prompt": "Edit the source DIY robot into a square album-cover hero image: Bass Oracle 111 gold robot DJ mascot, centered, sensor eyes glowing, headphones, miniature DJ decks, dark glossy background with berry neon rim light, premium sharp details, crisp readable title BASS ORACLE 111, no watermark.",
    },
]


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    timeout=5400,
    scaledown_window=60,
    volumes={"/cache": hf_cache, "/outputs": outputs},
)
def generate_edit_batch(
    width: int = 512,
    height: int = 512,
    seed_base: int = 1110,
    allow_model_download: bool = False,
    allow_generation: bool = False,
) -> str:
    import datetime as _dt
    import hashlib
    import os
    import pathlib as _pathlib
    import platform
    import subprocess
    import sys
    import time

    if not allow_model_download or not allow_generation:
        return json.dumps({
            "ok": False,
            "refused": True,
            "reason": "Pass both allow_model_download=True and allow_generation=True for the paid GPU/model run.",
            "gpu_started": True,
            "model_download_started": False,
            "generation_started": False,
        })

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    os.environ.setdefault("HF_HUB_CACHE", "/cache/huggingface/hub")
    sys.path.insert(0, "/workspace/HiDream-O1-Image")

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw, ImageFont
    from transformers import AutoProcessor
    from models.pipeline import DEFAULT_TIMESTEPS, generate_image
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    from inference import add_special_tokens, get_tokenizer

    t0 = time.time()
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=f"/cache/huggingface/snapshots/{MODEL_ID.replace('/', '__')}",
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )
    hf_cache.commit()

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cuda"
    ).eval()
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = _pathlib.Path("/outputs/hidream_o1_dev_edit_batch") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_root = _pathlib.Path("/workspace/HiDream-O1-Image/test_refs/freepik_edit_refs")
    records = []
    for idx, case in enumerate(EDIT_CASES, start=1):
        ref_meta = REFERENCE_META[case["ref_key"]]
        ref_path = ref_root / ref_meta["filename"]
        if not ref_path.exists():
            import urllib.request
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(ref_meta["image_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                ref_path.write_bytes(response.read())
        seed = seed_base + idx
        started = time.time()
        image_obj: Image.Image = generate_image(
            model=model,
            processor=processor,
            prompt=case["prompt"],
            ref_image_paths=[str(ref_path)],
            height=height,
            width=width,
            num_inference_steps=28,
            guidance_scale=0.0,
            shift=1.0,
            timesteps_list=DEFAULT_TIMESTEPS,
            scheduler_name="flash",
            seed=seed,
            noise_scale_start=7.5,
            noise_scale_end=7.5,
            noise_clip_std=2.5,
            keep_original_aspect=False,
        )
        out_path = out_dir / f"{idx:02d}_{case['id']}.png"
        image_obj.save(out_path)
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        records.append({
            "id": case["id"],
            "ref_key": case["ref_key"],
            "ref_filename": ref_meta["filename"],
            "source_page": ref_meta["source_page"],
            "creator": ref_meta["creator"],
            "prompt": case["prompt"],
            "seed": seed,
            "output_path": str(out_path),
            "sha256": sha,
            "elapsed_seconds": round(time.time() - started, 3),
        })

    # Contact sheet for quick review.
    thumbs = []
    for rec in records:
        im = Image.open(rec["output_path"]).convert("RGB")
        im.thumbnail((256, 256), Image.LANCZOS)
        canvas = Image.new("RGB", (256, 300), "white")
        canvas.paste(im, ((256 - im.width)//2, 0))
        d = ImageDraw.Draw(canvas)
        d.text((8, 262), rec["id"][:30], fill=(0,0,0))
        d.text((8, 280), f"seed {rec['seed']}", fill=(60,60,60))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (256*5, 300*2), (245,245,245))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 5)*256, (i // 5)*300))
    sheet_path = out_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=88)

    probe = subprocess.check_output(
        "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader",
        shell=True, text=True,
    ).strip()
    payload = {
        "ok": True,
        "app": APP_NAME,
        "model_id": MODEL_ID,
        "width": width,
        "height": height,
        "count": len(records),
        "elapsed_seconds": round(time.time() - t0, 3),
        "output_dir": str(out_dir),
        "contact_sheet": str(sheet_path),
        "records": records,
        "reference_meta": REFERENCE_META,
        "license_note": "Reference pages were labeled Free Photo on Freepik/Magnific. Keep attribution/source URLs for review; do not imply exclusive rights.",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
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
    width: int = 512,
    height: int = 512,
    seed_base: int = 1110,
    allow_model_download: bool = False,
    allow_generation: bool = False,
):
    print(json.dumps(json.loads(generate_edit_batch.remote(
        width=width,
        height=height,
        seed_base=seed_base,
        allow_model_download=allow_model_download,
        allow_generation=allow_generation,
    )), indent=2))
