# -*- coding: utf-8 -*-
"""
Ablation experiment: same base model, same seed, three prompt/LoRA variants at 640.
A: LoRA(37_best) + curated character prompt (correct identity tags + anime style)
B: LoRA(37_best) + old VLM-style prompt (hallucinated memory + photorealistic)
C: no LoRA + curated character prompt
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

BASE = r"D:\Python All\Pytorch Study\Endless_xxx\models\Illustrious-xl-early-release-v0\Illustrious-XL-v0.1.safetensors"
VAE = r"D:\Python All\Pytorch Study\Endless_xxx\models\sdxl-vae-fp16-fix\sdxl.vae.safetensors"
LORA = r"D:\Python All\Pytorch Study\Endless_xxx\output\train_models\37_best.safetensors"
OUT = r"D:\Python All\Pytorch Study\Endless_xxx\output"

def load(with_lora):
    pipe = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
    pipe.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
    if with_lora:
        pipe.load_lora_weights(LORA, adapter_name="default")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.to("cuda")
    return pipe

CURATED = ("ch37, 1girl, solo, masterpiece, best quality, highly detailed, "
           "pink hair, light pink hair, white hair, gradient hair, pink eyes, "
           "blue and white outfit, white shirt, blue skirt, hair ornament, "
           "four-leaf hair ornament, blue ribbon, anime style")

OLDVLM = ("ch37, 1girl, preserve original pose, preserve body proportions, character identity, "
          "light blue, light blue and pink and white signature accessory, "
          "Four-leaf hair clip signature symbol (teal) adapted as a medium hair clip on head "
          "in glossy plastic, detailed clothing, natural integration, photorealistic, "
          "preserve photographic appearance, high quality")

NEG = "worst quality, low quality, lowres, bad anatomy, bad hands, blurry, watermark, text, photorealistic, 3d, painting, oil painting, realistic"

def gen(pipe, prompt, path, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=25,
               guidance_scale=7.0, width=640, height=640, generator=g).images[0]
    img.save(path)
    print("saved", path)

print("=== A: LoRA + curated ===")
pa = load(True)
gen(pa, CURATED, os.path.join(OUT, "expA_lora_curated.png"))
del pa
torch.cuda.empty_cache()
print("=== B: LoRA + old VLM prompt ===")
pb = load(True)
gen(pb, OLDVLM, os.path.join(OUT, "expB_lora_oldprompt.png"))
del pb
torch.cuda.empty_cache()
print("=== C: no LoRA + curated ===")
pc = load(False)
gen(pc, CURATED, os.path.join(OUT, "expC_nolora_curated.png"))
del pc
print("DONE")
