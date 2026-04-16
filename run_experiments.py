"""
Run Level 1 (author dataset) and Level 2 (Kaggle dataset) experiments.
Three-phase approach per level to fit 10GB RAM:
  Phase 0: Prepare data (download/extract images to disk)
  Phase 1: Inference (load model only, read images from disk)
  Phase 2: CLIP metrics (load CLIP only, per-sample CSV for charts)

Usage:
  python3 run_experiments.py --level 1 --phase 0  # download author images
  python3 run_experiments.py --level 1 --phase 1  # inference
  python3 run_experiments.py --level 1 --phase 2  # CLIP metrics
  python3 run_experiments.py --level 2 --phase 1 --num-samples 1000
  python3 run_experiments.py --level 2 --phase 2
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from tqdm import tqdm


# ============================================================
# Shared: Model loading + inference
# ============================================================
def load_diffusion_model():
    """Load InstructPix2Pix model (memory-optimized for 10GB RAM)."""
    import einops
    import k_diffusion as K
    import torch.nn as nn
    from omegaconf import OmegaConf
    sys.path.append("./stable_diffusion")
    from stable_diffusion.ldm.util import instantiate_from_config

    class CFGDenoiser(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.inner_model = model
        def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
            cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
            cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
            cfg_cond = {
                "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
                "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
            }
            out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
            return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

    config = OmegaConf.load("configs/generate.yaml")
    print("Loading model (mmap + fp16)...")
    pl_sd = torch.load("checkpoints/instruct-pix2pix-00-22000.ckpt",
                        map_location="cpu", weights_only=False, mmap=True)
    sd = pl_sd["state_dict"]
    del pl_sd; gc.collect()

    model = instantiate_from_config(config.model)
    gc.collect()
    model.load_state_dict(sd, strict=False)
    del sd; gc.collect()
    # Match the proven working pattern from test_level2.py:
    # half() on CPU, then cuda() - with gc between each step
    model.eval()
    model = model.half()
    gc.collect()
    model = model.cuda()
    gc.collect(); torch.cuda.empty_cache()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    print("Model loaded.")
    return model, model_wrap, model_wrap_cfg, null_token


def run_inference_on_manifest(manifest_path, output_dir, steps=50,
                              cfg_text=7.5, cfg_image=1.5, seed=42, resolution=256):
    """Phase 1: load model, iterate manifest, save generated images."""
    import k_diffusion as K
    from torch import autocast

    output_dir = Path(output_dir)
    gen_dir = output_dir / "generated"
    examples_dir = output_dir / "examples"
    gen_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} samples")

    model, model_wrap, model_wrap_cfg, null_token = load_diffusion_model()

    for entry in tqdm(manifest, desc="Inference"):
        idx = entry["idx"]
        gen_path = gen_dir / f"{idx:04d}.png"
        if gen_path.exists():
            continue  # resume support

        img = Image.open(entry["input_img"]).convert("RGB")
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img_tensor = rearrange(
            2 * torch.tensor(np.array(img)).float() / 255 - 1, "h w c -> c h w"
        )

        torch.manual_seed(seed + idx)
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {
                "c_crossattn": [model.get_learned_conditioning([entry["edit"]])],
                "c_concat": [model.encode_first_stage(img_tensor[None].to(model.device)).mode()],
            }
            uncond = {
                "c_crossattn": [null_token],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])],
            }
            sigmas = model_wrap.get_sigmas(steps)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas,
                extra_args={"cond": cond, "uncond": uncond,
                            "text_cfg_scale": cfg_text, "image_cfg_scale": cfg_image})
            gen = model.decode_first_stage(z)[0]
            del z, cond, uncond
        torch.cuda.empty_cache()

        gen_img = Image.fromarray(
            (torch.clamp((gen.cpu() + 1) / 2, 0, 1).permute(1, 2, 0) * 255).byte().numpy()
        )
        gen_img.save(gen_path)
        entry["gen_img"] = str(gen_path)
        del gen

        if idx < 15:
            comp = Image.new("RGB", (resolution * 2, resolution))
            comp.paste(img, (0, 0))
            comp.paste(gen_img, (resolution, 0))
            comp.save(examples_dir / f"{idx:03d}.png")

        if idx % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()

    # Update manifest with gen_img paths
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Phase 1 done. Generated images in {gen_dir}")

    del model, model_wrap, model_wrap_cfg, null_token
    gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Level 1 Phase 0: Download author images to disk
# ============================================================
def level1_phase0(num_samples=1000, seed=42, resolution=256):
    """Download author dataset images to disk (no model loaded)."""
    from datasets import load_dataset

    output_dir = Path("outputs/level1_1k")
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LEVEL 1 PHASE 0: Download {num_samples} author images")
    print("=" * 60)

    print("Streaming from HuggingFace (timbrooks/instructpix2pix-clip-filtered)...")
    ds = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    manifest = []
    count = 0
    for sample in tqdm(ds, total=num_samples, desc="Downloading"):
        if count >= num_samples:
            break

        input_path = input_dir / f"{count:04d}.png"
        if not input_path.exists():
            img = sample["original_image"].convert("RGB").resize(
                (resolution, resolution), Image.Resampling.LANCZOS)
            img.save(input_path)

        manifest.append({
            "idx": count,
            "edit": sample["edit_prompt"],
            "input_prompt": sample["original_prompt"],
            "output_prompt": sample["edited_prompt"],
            "input_img": str(input_path),
            "gen_img": "",
        })
        count += 1

        # Free memory from dataset sample
        if count % 100 == 0:
            gc.collect()

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {len(manifest)} images saved to {input_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


# ============================================================
# Level 2 Phase 0: Prepare Kaggle manifest
# ============================================================
def level2_phase0(num_samples=1000, seed=42, resolution=256):
    """Create manifest from Kaggle dataset (images already on disk)."""
    data_path = "data/kaggle-level2"
    output_dir = Path("outputs/level2_1k")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LEVEL 2 PHASE 0: Prepare Kaggle manifest ({num_samples} samples)")
    print("=" * 60)

    with open(Path(data_path, "seeds.json")) as f:
        all_seeds = json.load(f)
    print(f"Dataset: {len(all_seeds)} prompts")

    torch.manual_seed(seed)
    perm = torch.randperm(len(all_seeds))[:num_samples]

    manifest = []
    for idx_i, dataset_idx in enumerate(tqdm(perm, desc="Building manifest")):
        name, seed_list = all_seeds[dataset_idx.item()]
        prompt_dir = Path(data_path, name)

        with open(prompt_dir / "prompt.json") as fp:
            prompt_data = json.load(fp)

        img_seed = seed_list[0]
        input_path = str(prompt_dir / f"{img_seed}_0.jpg")

        manifest.append({
            "idx": idx_i,
            "name": name,
            "edit": prompt_data["edit"],
            "input_prompt": prompt_data.get("input", ""),
            "output_prompt": prompt_data.get("output", ""),
            "input_img": input_path,
            "gen_img": "",
        })

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: {len(manifest)} entries -> {output_dir / 'manifest.json'}")


# ============================================================
# Phase 2: CLIP Metrics (shared for both levels)
# ============================================================
def run_phase2(output_dir, level_name, resolution=256):
    """Compute CLIP metrics with per-sample CSV logging for charts."""
    sys.path.append("./metrics")
    from clip_similarity import ClipSimilarity

    output_dir = Path(output_dir)
    print("\n" + "=" * 60)
    print(f"{level_name} PHASE 2: CLIP Metrics")
    print("=" * 60)

    with open(output_dir / "manifest.json") as f:
        manifest = json.load(f)
    print(f"Loaded: {len(manifest)} samples")

    print("Loading CLIP ViT-L/14...")
    clip_sim = ClipSimilarity("ViT-L/14").cuda()
    print("CLIP loaded.")

    csv_path = output_dir / "metrics_per_sample.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["idx", "edit", "sim_0", "sim_1", "sim_direction", "sim_image"])

    all_metrics = {"sim_0": [], "sim_1": [], "sim_direction": [], "sim_image": []}

    for entry in tqdm(manifest, desc="CLIP metrics"):
        img_0 = Image.open(entry["input_img"]).convert("RGB")
        img_0 = img_0.resize((resolution, resolution), Image.Resampling.LANCZOS)
        t0 = rearrange(torch.tensor(np.array(img_0)).float() / 255, "h w c -> 1 c h w")

        img_1 = Image.open(entry["gen_img"]).convert("RGB")
        t1 = rearrange(torch.tensor(np.array(img_1)).float() / 255, "h w c -> 1 c h w")

        with torch.no_grad():
            s0, s1, s_dir, s_img = clip_sim(
                t0.cuda(), t1.cuda(),
                [entry.get("input_prompt", "")],
                [entry.get("output_prompt", "")]
            )

        row = [s0.item(), s1.item(), s_dir.item(), s_img.item()]
        all_metrics["sim_0"].append(row[0])
        all_metrics["sim_1"].append(row[1])
        all_metrics["sim_direction"].append(row[2])
        all_metrics["sim_image"].append(row[3])

        writer.writerow([entry["idx"], entry["edit"][:80]] + [f"{v:.6f}" for v in row])

    csv_file.close()

    import statistics
    avg = {k: statistics.mean(v) for k, v in all_metrics.items()}
    std = {k: statistics.stdev(v) for k, v in all_metrics.items()}

    print("\n" + "=" * 60)
    print(f"{level_name} RESULTS ({len(manifest)} samples)")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*57}")
    for k in ["sim_image", "sim_direction", "sim_0", "sim_1"]:
        v = all_metrics[k]
        print(f"  {k:<25} {avg[k]:>8.4f} {std[k]:>8.4f} {min(v):>8.4f} {max(v):>8.4f}")

    results = {
        "level": level_name,
        "num_samples": len(manifest),
        "avg_metrics": avg,
        "std_metrics": std,
        "all_metrics": all_metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  CSV (per-sample): {csv_path}")
    print(f"  JSON (summary):   {output_dir / 'results.json'}")
    print("=" * 60)

    del clip_sim; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=[1, 2])
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    if args.level == 1:
        out = "outputs/level1_1k"
        if args.phase == 0:
            level1_phase0(num_samples=args.num_samples)
        elif args.phase == 1:
            run_inference_on_manifest(f"{out}/manifest.json", out)
        elif args.phase == 2:
            run_phase2(out, "LEVEL 1 (Author)")
    elif args.level == 2:
        out = "outputs/level2_1k"
        if args.phase == 0:
            level2_phase0(num_samples=args.num_samples)
        elif args.phase == 1:
            run_inference_on_manifest(f"{out}/manifest.json", out)
        elif args.phase == 2:
            run_phase2(out, "LEVEL 2 (Kaggle)")


if __name__ == "__main__":
    main()
