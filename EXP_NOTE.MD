# Experiment Notes (InstructPix2Pix)

Date: 2026-03-23

## Dataset generation

Files:
- [dataset_creation/generate_img_dataset.py](dataset_creation/generate_img_dataset.py)
- [scripts/generate_train.sh](scripts/generate_train.sh)

Changes made:
- reduced checkpoint RAM pressure by loading to CUDA when possible and cleaning temporary tensors
- added resume behavior: skip prompts with existing `metadata.jsonl`
- added true CLIP disable when all CLIP thresholds are `0`

Current command:

```bash
| Parameter | Value |
|-----------|-------|
| `prompts_file` | `data/gpt-generated-prompts.jsonl` |
| `steps` | `50` |
| `n-samples` | `10` |
| `max-out-samples` | `10` |
| `clip-threshold` | `0.2` |
| `clip-dir-threshold` | `0.2` |
| `clip-img-threshold` | `0.2` |
```

Current dataset size: **42034 image pairs**.
Generation time: 48 hours

## Training

File: [configs/train_small.yaml](configs/train_small.yaml)

Current settings:
- dataset: `data/generated_dataset`
- `batch_size: 8`
- `accumulate_grad_batches: 4`
- `precision: 16`
- `max_steps: 85000`

## Key fixes

Files:
- [main.py](main.py)
- [stable_diffusion/ldm/models/diffusion/ddpm_edit.py](stable_diffusion/ldm/models/diffusion/ddpm_edit.py)
- [stable_diffusion/ldm/modules/diffusionmodules/util.py](stable_diffusion/ldm/modules/diffusionmodules/util.py)

Applied fixes:
- guarded `trainer` access in error/finally paths
- disabled DDP for single-GPU runs
- switched default logger to TestTube
- set `num_sanity_val_steps = 0`
- fixed `persistent_workers` handling
- fixed `self.logvar` CPU/CUDA device mismatch
- fixed gradient checkpoint recomputation under fp16 autocast

## Inference / app

Files:
- [edit_cli.py](edit_cli.py)
- [edit_app.py](edit_app.py)

Notes:
- inference uses `--ckpt logs/<run>/checkpoints/last.ckpt`
- Gradio app updated for newer API
- app supports selectable examples from `imgs/examples/<name>/` with per-example `prompts.json`