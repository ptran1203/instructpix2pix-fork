"""
Level 1: Load pretrained InstructPix2Pix checkpoint and test on sample images.
Optimized for low-memory environments (10GB RAM container limit).

Key memory strategy:
  - Load checkpoint tensors directly to GPU via map_location="cuda"
  - This keeps CPU RAM usage minimal (~1-2GB for pickle overhead)
  - Model weights live on GPU VRAM (80GB H100), not CPU RAM (10GB limit)
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

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


def load_model_from_config(config, ckpt):
    """
    Memory-efficient model loading for 10GB RAM containers.

    Strategy: load state_dict directly to GPU to avoid CPU RAM pressure.
    Peak CPU RAM ~ 2GB (pickle overhead only), GPU VRAM ~ 7GB.
    """
    print(f"Loading model from {ckpt}")
    print("  Step 1: Memory-mapping checkpoint (minimal RAM usage)...")

    # mmap=True: memory-maps the file, only accessed pages are loaded into RAM
    # This keeps peak CPU RAM well under 10GB even for 7.2GB checkpoint
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if "global_step" in pl_sd:
        print(f"  Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    del pl_sd
    gc.collect()

    # Instantiate model on CPU, load mmap'd weights (low RAM due to mmap)
    print("  Step 2: Instantiating model on CPU...")
    model = instantiate_from_config(config.model)
    gc.collect()

    print("  Step 3: Loading mmap'd weights on CPU...")
    model.load_state_dict(sd, strict=False)
    del sd
    gc.collect()

    # Move to GPU as fp16 to halve VRAM (860M params: ~1.7GB fp16 vs ~3.4GB fp32)
    print("  Step 4: Moving model to GPU (fp16)...")
    model = model.half().cuda()
    gc.collect()
    torch.cuda.empty_cache()

    return model


def run_edit(model, model_wrap_cfg, null_token, input_path, output_path, edit_instruction,
             steps=50, resolution=256, seed=42, cfg_text=7.5, cfg_image=1.5):
    """Run a single image edit."""
    input_image = Image.open(input_path).convert("RGB")
    width, height = input_image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([edit_instruction])]
        input_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_tensor = rearrange(input_tensor, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_tensor).mode()]
        del input_tensor

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap_cfg.inner_model.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "image_cfg_scale": cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        del z

        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        del x, cond, uncond

    torch.cuda.empty_cache()
    gc.collect()

    edited_image.save(output_path)
    return edited_image


def main():
    test_cases = [
        ("imgs/examples/statue/example.jpg", "Turn him into a cyborg", "statue_cyborg"),
        ("imgs/examples/flower/image.png", "Swap sunflower with rose", "flower_rose"),
        ("imgs/examples/mona_lisa/image.png", "Make her smile more", "mona_lisa_smile"),
    ]

    output_dir = "outputs/level1_test"
    os.makedirs(output_dir, exist_ok=True)

    config_path = "configs/generate.yaml"
    ckpt_path = "checkpoints/instruct-pix2pix-00-22000.ckpt"

    print("=" * 60)
    print("Level 1: InstructPix2Pix Pretrained Model Testing")
    print(f"  Resolution: 256 | Steps: 50 | FP16: yes")
    print(f"  Memory strategy: GPU-direct loading (10GB RAM safe)")
    print("=" * 60)

    config = OmegaConf.load(config_path)

    t0 = time.time()
    model = load_model_from_config(config, ckpt_path)
    model.eval()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    load_time = time.time() - t0
    print(f"\nModel loaded in {load_time:.1f}s")
    print("=" * 60)

    results = []
    for i, (input_path, edit_instruction, output_name) in enumerate(test_cases):
        output_path = os.path.join(output_dir, f"{output_name}.png")
        print(f"\n[{i+1}/{len(test_cases)}] Input: {input_path}")
        print(f"  Instruction: \"{edit_instruction}\"")
        print(f"  Output: {output_path}")

        t_start = time.time()
        try:
            run_edit(
                model, model_wrap_cfg, null_token,
                input_path, output_path, edit_instruction,
                steps=50, resolution=256, seed=42,
                cfg_text=7.5, cfg_image=1.5,
            )
            elapsed = time.time() - t_start
            print(f"  Done in {elapsed:.1f}s")
            results.append((output_name, edit_instruction, elapsed, "OK"))
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  FAILED: {e}")
            results.append((output_name, edit_instruction, elapsed, f"FAIL: {e}"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Sample':<20} {'Instruction':<30} {'Time':>6} {'Status'}")
    print("-" * 70)
    for name, instr, t, status in results:
        print(f"{name:<20} {instr:<30} {t:>5.1f}s {status}")
    print("=" * 60)
    print(f"Output directory: {output_dir}/")
    total = len(results)
    passed = sum(1 for r in results if r[3] == "OK")
    print(f"Total: {total}, Passed: {passed}")


if __name__ == "__main__":
    main()
