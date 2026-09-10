# -*- coding: utf-8 -*-
"""
G: 768 不行再看：640 + ControlNet(canny cat) + LoRA 0.7（与 F 唯一差别：分辨率）
H: 768 无 ControlNet（纯文生图）+ LoRA（与 expA 唯一差别：分辨率/步数）
定位"朦胧"来自分辨率还是 ControlNet：
    G 清晰 -> 分辨率是根源（推理分辨率改 640，与训练一致）
    G 朦胧、H 清晰 -> ControlNet 是根源（降 scale / 换条件）
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline

import image_utils
import config

BASE = config.BASE_MODEL_PATH
VAE = config.BASE_VAE_PATH
LORA = r"D:\Python All\Pytorch Study\Endless_xxx\output\train_models\37_best.safetensors"
OUT = r"D:\Python All\Pytorch Study\Endless_xxx\output"

PROMPT = ("ch37, masterpiece, best quality, highly detailed, a cat, "
          "preserve cat anatomy, preserve original pose, "
          "blue and pink and white character color palette, "
          "four-leaf hair ornament signature symbol (blue and pink) adapted as a "
          "small collar charm on neck in soft fabric, anime style")
NEG = (config.NEGATIVE_PROMPT + ", " + config.ANIME_NEGATIVE_PROMPT
       + ", " + config.NON_HUMAN_NEGATIVE_PROMPT)

img = image_utils.load_image(os.path.join(config.INPUT_DIR, "cat.png"))


def gen(pipe, path, size, seed=7, lora_scale=None, canny=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if lora_scale is not None and hasattr(pipe, "set_adapters"):
        pipe.set_adapters("default", adapter_weights=lora_scale)
    kw = dict(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=36,
              guidance_scale=7.0, width=size, height=size, generator=g)
    if canny is not None:
        kw["image"] = canny
        kw["controlnet_conditioning_scale"] = 0.7
    out = pipe(**kw).images[0]
    out.save(path)
    print("saved", path)


print("=== G: 640 + ControlNet + LoRA 0.7 ===")
base = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
base.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
base.load_lora_weights(LORA, adapter_name="default")
cn = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=torch.float16, variant="fp16")
pg = StableDiffusionXLControlNetPipeline(**base.components, controlnet=cn)
pg.enable_vae_slicing(); pg.enable_vae_tiling(); pg.to("cuda")
img640 = image_utils.resize_and_crop(img, 640)
canny640 = image_utils.extract_canny(img640, config.CANNY_LOW, config.CANNY_HIGH)
gen(pg, os.path.join(OUT, "expG_cn640.png"), 640, lora_scale=0.7, canny=canny640)
del pg
torch.cuda.empty_cache()

print("=== H: 768 无 ControlNet + LoRA ===")
ph = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
ph.vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
ph.load_lora_weights(LORA, adapter_name="default")
ph.enable_vae_slicing(); ph.enable_vae_tiling(); ph.to("cuda")
gen(ph, os.path.join(OUT, "expH_t2i768.png"), 768, lora_scale=0.7)
del ph
print("DONE")
