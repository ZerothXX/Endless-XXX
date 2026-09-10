# -*- coding: utf-8 -*-
"""
M: ControlNet-canny 0.45（640）—— 0.35 清晰但姿势略松，0.45 找平衡
N: img2img strength 0.68（640）—— 强改写：身份完整 + 姿势由原图锚定
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
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


def gen(pipe, path, seed=7, **kw):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(prompt=PROMPT, negative_prompt=NEG,
               num_inference_steps=36, guidance_scale=7.0,
               width=640, height=640, generator=g, **kw).images[0]
    out.save(path)
    print("saved", path)


print("=== M: ControlNet-canny 0.45 ===")
p = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
p.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
p.load_lora_weights(LORA, adapter_name="default")
cn = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=torch.float16, variant="fp16")
mk = StableDiffusionXLControlNetPipeline(**p.components, controlnet=cn)
mk.enable_vae_slicing(); mk.enable_vae_tiling(); mk.to("cuda")
gen(mk, os.path.join(OUT, "expM_cn045.png"), image=canny,
    controlnet_conditioning_scale=0.45)
del mk
torch.cuda.empty_cache()

print("=== N: img2img strength 0.68 ===")
p = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
p.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
p.load_lora_weights(LORA, adapter_name="default")
mk = StableDiffusionXLImg2ImgPipeline(**p.components)
mk.enable_vae_slicing(); mk.enable_vae_tiling(); mk.to("cuda")
gen(mk, os.path.join(OUT, "expN_img2img68.png"), image=img, strength=0.68)
del mk
print("DONE")
