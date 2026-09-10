# -*- coding: utf-8 -*-
"""
O2: 原生 StableDiffusionXLControlNetImg2ImgPipeline —— img2img(0.60) + ControlNet-canny(0.40)
   init 图像锚定姿势/背景 + canny 边缘保结构 + LoRA 身份（清晰画风）
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

g = torch.Generator(device="cuda").manual_seed(7)
out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, control_image=canny,
           strength=0.60, controlnet_conditioning_scale=0.40,
           num_inference_steps=36, guidance_scale=7.0,
           width=640, height=640, generator=g).images[0]
out.save(os.path.join(OUT, "expO2_mix_native.png"))
print("saved expO2_mix_native.png")
