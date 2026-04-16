"""
Level 2: Inference on Kaggle dataset (ptran1203/instruct-p2p-generated-dataset)
using the author's pretrained checkpoint.

Two-phase approach to fit in 10GB RAM:
  Phase 1: Run inference, save generated images to disk
  Phase 2: Unload diffusion model, load CLIP, compute metrics

Usage:
  python3 test_level2.py                # Run both phases
  python3 test_level2.py --phase 1      # Inference only
  python3 test_level2.py --phase 2      # Metrics only (requires phase 1 output)
"""
from __future__ import annotations

import gc
import json
import math
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
# Phase 1: Inference
# ============================================================
def run_phase1(data_path, output_dir, num_samples=100, num_qualitative=10,
               resolution=256, steps=50, cfg_text=7.5, cfg_image=1.5, seed=42):
    """Run inference with diffusion model. No CLIP loaded."""
    import einops
    import k_diffusion as K
    import torch.nn as nn
    from omegaconf import OmegaConf
    from torch import autocast
    from PIL import ImageOps

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

    print("=" * 60)
    print("PHASE 1: Inference (diffusion model only)")
    print(f"  Samples: {num_samples} | Resolution: {resolution} | Steps: {steps}")
    print("=" * 60)

    # Load dataset index
    with open(Path(data_path, "seeds.json")) as f:
        all_seeds = json.load(f)
    print(f"Dataset: {len(all_seeds)} prompts")

    # Load model
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
    model = model.half().cuda().eval()
    gc.collect(); torch.cuda.empty_cache()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    print("Model loaded.")

    # Select random subset
    torch.manual_seed(seed)
    perm = torch.randperm(len(all_seeds))[:num_samples]

    # Prepare output dirs
    gen_dir = Path(output_dir, "generated")
    examples_dir = Path(output_dir, "examples")
    gen_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    manifest = []  # Track what we generated

    for idx_i, dataset_idx in enumerate(tqdm(perm, desc="Inference")):
        name, seed_list = all_seeds[dataset_idx.item()]
        prompt_dir = Path(data_path, name)

        with open(prompt_dir / "prompt.json") as fp:
            prompt_data = json.load(fp)

        img_seed = seed_list[0]
        input_image = Image.open(prompt_dir / f"{img_seed}_0.jpg").convert("RGB")
        input_image = input_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img_tensor = rearrange(
            2 * torch.tensor(np.array(input_image)).float() / 255 - 1,
            "h w c -> c h w"
        )

        # Inference
        torch.manual_seed(seed + idx_i)
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {
                "c_crossattn": [model.get_learned_conditioning([prompt_data["edit"]])],
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

        # Save generated image
        gen_img = Image.fromarray(
            (torch.clamp((gen.cpu() + 1) / 2, 0, 1).permute(1, 2, 0) * 255).byte().numpy()
        )
        gen_img.save(gen_dir / f"{idx_i:04d}.png")
        del gen

        # Save side-by-side for first N
        if idx_i < num_qualitative:
            comparison = Image.new("RGB", (resolution * 2, resolution))
            comparison.paste(input_image, (0, 0))
            comparison.paste(gen_img, (resolution, 0))
            comparison.save(examples_dir / f"{idx_i:03d}_{name}.png")

        manifest.append({
            "idx": idx_i,
            "dataset_idx": dataset_idx.item(),
            "name": name,
            "edit": prompt_data["edit"],
            "input_prompt": prompt_data.get("input", ""),
            "output_prompt": prompt_data.get("output", ""),
            "input_img": str(prompt_dir / f"{img_seed}_0.jpg"),
            "gen_img": str(gen_dir / f"{idx_i:04d}.png"),
        })

        if idx_i % 20 == 0:
            torch.cuda.empty_cache(); gc.collect()

    # Save manifest
    manifest_path = Path(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPhase 1 done. {len(manifest)} images generated.")
    print(f"Manifest: {manifest_path}")

    # Unload model to free memory for phase 2
    del model, model_wrap, model_wrap_cfg, null_token
    gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 2: CLIP Metrics
# ============================================================
def run_phase2(output_dir, resolution=256):
    """Load CLIP, compute metrics on saved images. No diffusion model."""
    sys.path.append("./metrics")
    from clip_similarity import ClipSimilarity

    print("\n" + "=" * 60)
    print("PHASE 2: CLIP Metrics (CLIP model only)")
    print("=" * 60)

    manifest_path = Path(output_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Loaded manifest: {len(manifest)} samples")

    # Load CLIP
    print("Loading CLIP ViT-L/14...")
    clip_sim = ClipSimilarity("ViT-L/14").cuda()
    print("CLIP loaded.")

    metrics = {"sim_0": [], "sim_1": [], "sim_direction": [], "sim_image": []}

    for entry in tqdm(manifest, desc="Computing CLIP metrics"):
        # Load input image
        input_image = Image.open(entry["input_img"]).convert("RGB")
        input_image = input_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img_0 = torch.tensor(np.array(input_image)).float() / 255
        img_0 = rearrange(img_0, "h w c -> 1 c h w")

        # Load generated image
        gen_image = Image.open(entry["gen_img"]).convert("RGB")
        img_1 = torch.tensor(np.array(gen_image)).float() / 255
        img_1 = rearrange(img_1, "h w c -> 1 c h w")

        with torch.no_grad():
            s0, s1, s_dir, s_img = clip_sim(
                img_0.cuda(), img_1.cuda(),
                [entry["input_prompt"]], [entry["output_prompt"]]
            )
        metrics["sim_0"].append(s0.item())
        metrics["sim_1"].append(s1.item())
        metrics["sim_direction"].append(s_dir.item())
        metrics["sim_image"].append(s_img.item())

    # Compute averages
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}

    print("\n" + "=" * 60)
    print("QUANTITATIVE RESULTS")
    print("=" * 60)
    print(f"  Samples:                  {len(manifest)}")
    print(f"  CLIP Image Similarity:    {avg['sim_image']:.4f}")
    print(f"  CLIP Direction Similarity:{avg['sim_direction']:.4f}")
    print(f"  CLIP Input-Text (sim_0):  {avg['sim_0']:.4f}")
    print(f"  CLIP Output-Text (sim_1): {avg['sim_1']:.4f}")

    # Save results
    results = {
        "config": {
            "dataset": "ptran1203/instruct-p2p-generated-dataset",
            "num_samples": len(manifest),
            "resolution": resolution,
        },
        "avg_metrics": avg,
        "all_metrics": metrics,
        "qualitative_examples": manifest[:10],
    }
    results_path = Path(output_dir, "level2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


def main():
    parser = ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="1=inference, 2=metrics, 0=both")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    data_path = "data/kaggle-level2"
    output_dir = "outputs/level2_test"
    os.makedirs(output_dir, exist_ok=True)

    if args.phase in (0, 1):
        run_phase1(data_path, output_dir, num_samples=args.num_samples)

    if args.phase in (0, 2):
        run_phase2(output_dir)


if __name__ == "__main__":
    main()
