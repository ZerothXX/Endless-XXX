# -*- coding: utf-8 -*-
"""
身份检查：用训练好的 LoRA + 卡片 prompt 文生图（无 ControlNet/img2img 干扰），
单看"触发词 ch37 是否绑定角色身份"。

用法：python _check_identity.py [lora_path] [out_path] [seed]
默认 lora = output/train_models/37_best.safetensors
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

BASE = r"D:\Python All\Pytorch Study\Endless_xxx\models\Illustrious-xl-early-release-v0\Illustrious-XL-v0.1.safetensors"
VAE = r"D:\Python All\Pytorch Study\Endless_xxx\models\sdxl-vae-fp16-fix\sdxl.vae.safetensors"
LORA = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\Python All\Pytorch Study\Endless_xxx\output\train_models\37_final.safetensors"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    r"D:\Python All\Pytorch Study\Endless_xxx\output\identity_check.png"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42

PROMPT = ("ch37, 1girl, solo, masterpiece, best quality, highly detailed, "
          "pink hair, light pink hair, white hair, gradient hair, pink eyes, "
          "blue and white outfit, white shirt, blue skirt, hair ornament, "
          "four-leaf hair ornament, blue ribbon, anime style")
NEG = ("worst quality, low quality, lowres, bad anatomy, bad hands, blurry, watermark, "
       "text, photorealistic, realistic, 3d, oil painting")

pipe = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
pipe.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
pipe.load_lora_weights(LORA, adapter_name="default")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.to("cuda")

g = torch.Generator(device="cuda").manual_seed(SEED)
img = pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=25,
           guidance_scale=7.0, width=640, height=640, generator=g).images[0]
img.save(OUT)
print("saved", OUT)
