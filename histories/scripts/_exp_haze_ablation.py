# -*- coding: utf-8 -*-
"""
朦胧画风消融实验（在真实推理路径 768 + ControlNet 上定位"雾感"来源）：
  D:  ControlNet(canny cat) + 卡片 prompt（猫模板） + 无 LoRA
  E:  ControlNet(canny cat) + 卡片 prompt（猫模板） + LoRA 0.7 + VAE 强制 fp32 解码
  F:  ControlNet(canny cat) + 卡片 prompt（猫模板） + LoRA 0.7（与 test.py 全等对照）

若 D 也朦胧 -> ControlNet/768 路径是雾感来源；
若 D 清晰而 E 朦胧 -> LoRA 是来源；
若 D/E 清晰而 F 朦胧 -> VAE fp16 解码是来源。
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
from PIL import Image

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


def build_pipe(use_lora=True, force_fp32_vae=False):
    pipe = StableDiffusionXLPipeline.from_single_file(BASE, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
    if force_fp32_vae:
        # 与标准 SDXL 一致：解码时按 config.force_upcast 自动切 fp32 并同步 cast latents
        vae.config.force_upcast = True
        print("VAE force_upcast=True（解码切 fp32）")
    pipe.vae = vae
    if use_lora:
        pipe.load_lora_weights(LORA, adapter_name="default")
    cn = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=torch.float16,
                                         variant="fp16")
    pipe = StableDiffusionXLControlNetPipeline(**pipe.components, controlnet=cn)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.to("cuda")
    return pipe


img = image_utils.load_image(os.path.join(config.INPUT_DIR, "cat.png"))
img = image_utils.resize_and_crop(img, 768)
canny = image_utils.extract_canny(img, config.CANNY_LOW, config.CANNY_HIGH)


def gen(pipe, path, seed=7, lora_scale=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if lora_scale is not None and hasattr(pipe, "set_adapters"):
        pipe.set_adapters("default", adapter_weights=lora_scale)
    out = pipe(prompt=PROMPT, negative_prompt=NEG, image=canny,
               controlnet_conditioning_scale=0.7,
               num_inference_steps=36, guidance_scale=7.0,
               width=768, height=768, generator=g).images[0]
    out.save(path)
    print("saved", path)


print("=== D: ControlNet, NO LoRA ===")
# D 已在上一轮跑通（expD_cn_nolora.png），本轮只跑 E（fp32 VAE 解码）与 F（对照）
# p = build_pipe(use_lora=False)
# gen(p, os.path.join(OUT, "expD_cn_nolora.png"))
# del p
# torch.cuda.empty_cache()
print("=== E: ControlNet + LoRA 0.7 + fp32 VAE 解码 ===")
p = build_pipe(use_lora=True, force_fp32_vae=True)
gen(p, os.path.join(OUT, "expE_cn_lora_fp32vae.png"), lora_scale=0.7)
del p
torch.cuda.empty_cache()
print("=== F: ControlNet + LoRA 0.7 + fp16 VAE (对照 test.py) ===")
p = build_pipe(use_lora=True, force_fp32_vae=False)
gen(p, os.path.join(OUT, "expF_cn_lora_fp16vae.png"), lora_scale=0.7)
del p
print("DONE")
