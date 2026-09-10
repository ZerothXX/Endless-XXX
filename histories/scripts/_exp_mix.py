# -*- coding: utf-8 -*-
"""
O: img2img(0.60) + ControlNet-canny(0.40) 混合（640）—— 标准角色重绘配方：
   init 潜变量锚定姿势/背景，canny 边缘保结构，LoRA+prompt 提供身份与清晰画风
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
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

p = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
p.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
p.load_lora_weights(LORA, adapter_name="default")
cn = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=torch.float16, variant="fp16")
pipe = StableDiffusionXLControlNetPipeline(**p.components, controlnet=cn)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.to("cuda")

# ---- img2img + ControlNet 混合：手工按 img2img 流程计算初始潜变量与时间步 ----
STRENGTH = 0.60
CN_SCALE = 0.40
STEPS = 36
CFG = 7.0
SEED = 7

import numpy as np
from PIL import Image
init_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
init_t = torch.from_numpy(init_np).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float16)
init_latents = pipe.vae.encode(init_t * 2.0 - 1.0).latent_dist.mode() * pipe.vae.config.scaling_factor
init_latents = init_latents.to(torch.float16)

g = torch.Generator(device="cuda").manual_seed(SEED)
noise = torch.randn(init_latents.shape, device=init_latents.device,
                    dtype=init_latents.dtype, generator=g)
pipe.scheduler.set_timesteps(STEPS, device="cuda")
timesteps = pipe.scheduler.timesteps
init_timestep = int(STEPS * STRENGTH)
t_start = len(timesteps) - init_timestep
timesteps = timesteps[t_start:]
noisy_latents = pipe.scheduler.add_noise(init_latents, noise, timesteps[:1]).to(torch.float16)

kw = dict(prompt=PROMPT, negative_prompt=NEG, image=canny,
          controlnet_conditioning_scale=CN_SCALE,
          num_inference_steps=STEPS, guidance_scale=CFG,
          width=640, height=640, generator=g, latents=noisy_latents,
          timesteps=timesteps.cpu(), return_dict=False)
img_out = pipe(**kw)[0]
while isinstance(img_out, (list, tuple)) and img_out:
    img_out = img_out[0]
if not isinstance(img_out, Image.Image):
    img_out = Image.fromarray(
        ((img_out.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()) * 255).astype(np.uint8))
img_out.save(os.path.join(OUT, "expO_mix.png"))
print("saved", os.path.join(OUT, "expO_mix.png"))
