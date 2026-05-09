# HiDream-O1-Image Modal setup report — 2026-05-09

## Summary

Fork prepared for Modal-backed HiDream-O1-Image experiments under `TheMindExpansionNetwork/HiDream-O1-Image`.

## Added

- `AGENTS.md` with Modal/GPU/model-download guardrails.
- `modal/hidream_o1_readiness.py` CPU-only readiness wrapper.
- `modal/hidream_o1_gpu_probe.py` tiny L4 CUDA/import probe.
- `modal/hidream_o1_generation_app.py` gated real image smoke wrapper.
- `docs/JIMSKY_MODAL_SETUP.md` runbook with commands and approval gate.

## Tests run

### 1. CPU readiness

Command:

```bash
cd /opt/data/workspace/projects/HiDream-O1-Image
source /opt/data/hermes-agent/venv/bin/activate
set -a; source /opt/data/.env; set +a
modal run modal/hidream_o1_readiness.py
```

Result:

- App: `jimsky-hidream-o1-readiness`
- App ID: `ap-tv1oCOGcjuv9V5uN4x4y7w`
- Status: completed / stopped
- Tasks after run: `0`
- `ok`: `true`
- Checked model metadata: `HiDream-ai/HiDream-O1-Image-Dev`
- HF model SHA: `e3a13befd8d1b83bade417d90402600cb8503d3e`
- HF file count: `28`
- GPU started: `false`
- Model download started: `false`

Note: an earlier first run (`ap-5afBodSn17WtpsC55Nv9zI`) exposed a Modal import-packaging pitfall while `include_source=False` was set. The app was stopped, code was patched to use Modal's normal source mounting, and the second run passed.

### 2. GPU dependency/import probe

Command:

```bash
cd /opt/data/workspace/projects/HiDream-O1-Image
source /opt/data/hermes-agent/venv/bin/activate
set -a; source /opt/data/.env; set +a
modal run modal/hidream_o1_gpu_probe.py
```

Result:

- App: `jimsky-hidream-o1-gpu-probe`
- App ID: `ap-B9cMkxvd5y3Q8Xg6rqfqU3`
- Status: completed / stopped
- Tasks after run: `0`
- `ok`: `true`
- GPU: `NVIDIA L4, 23034 MiB, driver 580.95.05`
- Torch: `2.11.0+cu130`
- CUDA available: `true`
- Imports passed:
  - `transformers`
  - `diffusers`
  - `accelerate`
  - `einops`
  - `PIL`
  - `models.pipeline`
  - `models.qwen3_vl_transformers`
- Model download started: `false`
- Image generation started: `false`

## Modal hygiene

Filtered `modal app list --json` showed all HiDream apps stopped with `Tasks: 0`:

- `ap-B9cMkxvd5y3Q8Xg6rqfqU3` — `jimsky-hidream-o1-gpu-probe` — stopped — Tasks `0`
- `ap-tv1oCOGcjuv9V5uN4x4y7w` — `jimsky-hidream-o1-readiness` — stopped — Tasks `0`
- `ap-5afBodSn17WtpsC55Nv9zI` — earlier failed readiness run — stopped — Tasks `0`

Filtered billing rows for today:

- `jimsky-hidream-o1-readiness` / `ap-5afBodSn17WtpsC55Nv9zI`: `$0.00000768`
- `jimsky-hidream-o1-readiness` / `ap-tv1oCOGcjuv9V5uN4x4y7w`: `$0.00003568`
- `jimsky-hidream-o1-gpu-probe` / `ap-B9cMkxvd5y3Q8Xg6rqfqU3`: `$0.00191270`

## Real image smoke example, not run yet

The real generation wrapper is intentionally gated. It will refuse unless both approval flags are passed:

```bash
modal run modal/hidream_o1_generation_app.py \
  --prompt 'A cute gold robot DJ in a neon berry radio booth holding a sign that says BASS ORACLE 111, crisp readable text, product photo' \
  --model-id HiDream-ai/HiDream-O1-Image-Dev \
  --width 512 \
  --height 512 \
  --seed 111 \
  --allow-model-download \
  --allow-generation
```

This is the next approval gate because it may download model weights and run a paid A100 GPU inference.

## Known notes

- Upstream README says `flash-attn` is recommended. If real inference fails without it, patch the generation wrapper or upstream `models/pipeline.py` to set `use_flash_attn=False` for the first smoke run.
- The prompt agent's local backend is a separate large `google/gemma-4-31B-it` model and is not part of the default smoke lane.
