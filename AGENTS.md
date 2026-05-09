# Hermes / Jimsky Notes for HiDream-O1-Image

This fork is prepared for Modal-backed experiments without accidental spend.

## Safety defaults

- Do not commit credentials, `.env` files, Modal tokens, API keys, generated bulk outputs, model weights, or Hugging Face cache contents.
- Do not start model downloads or GPU inference unless a command explicitly passes an approval flag such as `--allow_model_download` / `--allow_generation`.
- Keep upstream compatibility: Modal wrappers live under `modal/` and docs under `docs/`.
- Prefer the Dev checkpoint for first real image tests: `HiDream-ai/HiDream-O1-Image-Dev`.
- Use small test dimensions first, then scale quality/resolution only after a successful smoke test.

## Modal app names

- CPU readiness: `jimsky-hidream-o1-readiness`
- GPU dependency probe: `jimsky-hidream-o1-gpu-probe`
- Gated generation wrapper: `jimsky-hidream-o1-generation`

