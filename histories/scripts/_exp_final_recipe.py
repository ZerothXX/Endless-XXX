# -*- coding: utf-8 -*-
"""
P: img2img(0.72) + ControlNet-canny(0.35) 原生管线：
   姿势由 init 图像锚定（不依赖 canny 高 scale），canny 低 scale 保持清晰，
   identity 由 prompt+LoRA（强度 0.72 足以让粉发/蓝白服替换原画）
比较 O2(0.6+0.4)：更接近示例 37_1（全化身 + 保留姿势）
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

import image_utils
import config

BASE = config.BASE_MODEL_PATH
VAE = config.BASE_VAE_PATH
LORA = r"D:\Python All\Pytorch Study\Endless_xxx\output\train_models\37_best.safetensors"
OUT = r"D:\Python All\Pytorch Study\Endless_xxx\output"

PROMPT = ("ch37, 1girl, solo, masterpiece, best quality, highly detailed, "
          "pink hair, light pink hair, white hair, gradient hair, pink eyes, "
          "white shirt, blue skirt, blue and white outfit, hair ornament, "
          "four-leaf hair ornament, blue ribbon, anime style")
NEG = (config.NEGATIVE_PROMPT + ", " + config.ANIME_NEGATIVE_PROMPT)

img = image_utils.load_image(os.path.join(config.INPUT_DIR, "human.png"))
img = image_utils.resize_and_crop(img, 640)
canny = image_utils.extract_canny(img, config.CANNY_LOW, config.CANNY_HIGH)

p = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
p.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
p.load_lora_weights(LORA, adapter_name="default")
cn = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=torch.float16, variant="fp16")
pipe = StableDiffusionXLControlNetImg2ImgPipeline(**p.components, controlnet=cn)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.to("cuda")

for tag, strength, cn_scale, seed in [("P", 0.72, 0.35, 7), ("P2", 0.72, 0.35, 11), ("P3", 0.78, 0.35, 7)]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, control_image=canny,
               strength=strength, controlnet_conditioning_scale=cn_scale,
               num_inference_steps=36, guidance_scale=7.0,
               width=640, height=640, generator=g).images[0]
    out.save(os.path.join(OUT, f"exp{tag}_img2img{int(strength*100)}_cn35_s{seed}.png"))
    print("saved", tag)
print("DONE")
