# Jimsky Modal setup for HiDream-O1-Image

This fork adds Modal wrappers that make the repo reviewable and testable from Hermes without accidentally downloading 8B image weights or leaving GPUs running.

## Files added

- `modal/hidream_o1_readiness.py` — CPU-only readiness check. No GPU, no weights.
- `modal/hidream_o1_gpu_probe.py` — tiny L4 CUDA/import probe. No weights, no generation.
- `modal/hidream_o1_generation_app.py` — gated real image smoke wrapper. It refuses to run unless explicit approval flags are passed.
- `AGENTS.md` — operational guardrails for future agents.

## Modal resources

- CPU readiness app: `jimsky-hidream-o1-readiness`
- GPU probe app: `jimsky-hidream-o1-gpu-probe`
- Gated generation app: `jimsky-hidream-o1-generation`
- Volumes used:
  - `outputs` for JSON reports and generated proof images
  - `huggingface-cache` for future model caches

## Setup / readiness test

From Hermes:

```bash
cd /opt/data/hermes-agent
source venv/bin/activate
set -a; source /opt/data/.env; set +a
modal run /opt/data/workspace/projects/HiDream-O1-Image/modal/hidream_o1_readiness.py
```

Expected shape:

```json
{
  "ok": true,
  "purpose": "CPU-only fork/readiness check; no GPU and no model weight download.",
  "model_id_checked": "HiDream-ai/HiDream-O1-Image-Dev",
  "gpu_started": false,
  "model_download_started": false
}
```

## Tiny GPU dependency test

This starts a small L4 briefly and verifies CUDA/Torch/imports only:

```bash
cd /opt/data/hermes-agent
source venv/bin/activate
set -a; source /opt/data/.env; set +a
modal run /opt/data/workspace/projects/HiDream-O1-Image/modal/hidream_o1_gpu_probe.py
```

Expected shape:

```json
{
  "ok": true,
  "cuda_available": true,
  "cuda_device": "...",
  "imports": {
    "models.pipeline": true,
    "models.qwen3_vl_transformers": true
  },
  "model_download_started": false,
  "generation_started": false
}
```

## Gated real image smoke test

Only run after explicit approval for a paid GPU/model-download inference. First call without flags proves the guard is active and refuses the run:

```bash
modal run /opt/data/workspace/projects/HiDream-O1-Image/modal/hidream_o1_generation_app.py
```

To actually generate one tiny Dev-model proof image:

```bash
modal run /opt/data/workspace/projects/HiDream-O1-Image/modal/hidream_o1_generation_app.py \
  --prompt 'A cute gold robot DJ in a neon berry radio booth holding a sign that says BASS ORACLE 111, crisp readable text, product photo' \
  --model-id HiDream-ai/HiDream-O1-Image-Dev \
  --width 512 \
  --height 512 \
  --seed 111 \
  --allow-model-download \
  --allow-generation
```

Output is written to the Modal `outputs` volume under `/outputs/hidream_o1/<timestamp>/smoke.png` plus `result.json`. Pull it locally with:

```bash
modal volume get outputs hidream_o1/<timestamp>/smoke.png /opt/data/workspace/projects/HiDream-O1-Image/outputs/hidream-smoke.png
```

## Operational hygiene checks

After every Modal run:

```bash
modal app list --json | python3 -c 'import json,sys; apps=json.load(sys.stdin); print([a for a in apps if "hidream-o1" in str(a).lower() or "hidream" in str(a).lower()])'
modal billing report --for today --json | python3 -c 'import sys; raw=sys.stdin.read(); print(raw[:2000] if raw.strip() else "(no billing json returned)")'
```

All completed/probe apps should return to `Tasks: 0` / stopped after completion.

## Notes

- Upstream README says `flash-attn` is recommended. If not installed, `models/pipeline.py` line ~291 may need `use_flash_attn=False` for real inference. The dependency probe checks importability but does not validate a full denoising pass.
- The prompt agent local backend references `google/gemma-4-31B-it`; that is a separate large model and is not part of the default Modal tests.
- Use the Dev checkpoint first for bounded smoke tests before trying the full 50-step model.
