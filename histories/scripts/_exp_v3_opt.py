# -*- coding: utf-8 -*-
"""
v3 优化实验：
  R1: 蒙娜丽莎 + 新卡片（精确服饰标签）+ LoRA 1.2 + 混合路径 strength 0.72
  R2: 杯子 + 新卡片 object_extra + 768 分辨率 + strength 0.70 + canny 0.40
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
from character.semantic_memory import AccessoryAdaptationPlanner

LORA = r"D:\Python All\Pytorch Study\Endless_xxx\output\train_models\37_final.safetensors"
OUT = r"D:\Python All\Pytorch Study\Endless_xxx\output\experiments"
CARD = config.get_character_card("37")
planner = AccessoryAdaptationPlanner(client=None, memory={})
# 用与 test.py 一致的规则表规划（免 VLM）
SUBJ_HUMAN = {"subject_type": "human", "pose": "seated", "style": "realistic photo"}
SUBJ_CUP = {"subject_type": "object", "object_category": "cup", "material": "ceramic",
            "style": "photograph"}
PLAN_H = planner.plan(SUBJ_HUMAN, CARD["accessory"])
PLAN_C = planner.plan(SUBJ_CUP, CARD["accessory"])
PROMPT_H = planner.build_card_transfer_prompt(SUBJ_HUMAN, PLAN_H, CARD, "ch37", "1girl")
PROMPT_C = planner.build_card_transfer_prompt(SUBJ_CUP, PLAN_C, CARD, "ch37", "1girl")
NEG = config.NEGATIVE_PROMPT + ", " + config.ANIME_NEGATIVE_PROMPT
print("H prompt:", PROMPT_H)
print("C prompt:", PROMPT_C)


def build():
    p = StableDiffusionXLPipeline.from_single_file(config.BASE_MODEL_PATH, torch_dtype=torch.float16)
    p.vae = AutoencoderKL.from_single_file(config.BASE_VAE_PATH, torch_dtype=torch.float16)
    p.load_lora_weights(LORA, adapter_name="default")
    cn = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=torch.float16, variant="fp16")
    mk = StableDiffusionXLControlNetImg2ImgPipeline(**p.components, controlnet=cn)
    mk.enable_vae_slicing()
    mk.enable_vae_tiling()
    mk.to("cuda")
    return mk


# ---- R1: 蒙娜丽莎 640, LoRA 1.2 ----
mk = build()
mk.set_adapters("default", adapter_weights=1.2)
img = image_utils.load_image(os.path.join(config.INPUT_DIR, "human.png"))
img = image_utils.resize_and_crop(img, 640)
canny = image_utils.extract_canny(img, config.CANNY_LOW, config.CANNY_HIGH)
g = torch.Generator(device="cuda").manual_seed(7)
out = mk(prompt=PROMPT_H, negative_prompt=NEG, image=img, control_image=canny,
         strength=0.72, controlnet_conditioning_scale=0.35,
         num_inference_steps=36, guidance_scale=7.0,
         width=640, height=640, generator=g).images[0]
out.save(os.path.join(OUT, "expR1_human_lora120.png"))
print("saved R1")
del mk
torch.cuda.empty_cache()

# ---- R2: 杯子 768, LoRA 0.7, canny 0.40 ----
mk = build()
mk.set_adapters("default", adapter_weights=0.7)
imgc = image_utils.load_image(os.path.join(config.INPUT_DIR, "cup.jpg"))
imgc = image_utils.resize_and_crop(imgc, 768)
cannyc = image_utils.extract_canny(imgc, config.CANNY_LOW, config.CANNY_HIGH)
g = torch.Generator(device="cuda").manual_seed(7)
out = mk(prompt=PROMPT_C, negative_prompt=NEG, image=imgc, control_image=cannyc,
         strength=0.70, controlnet_conditioning_scale=0.40,
         num_inference_steps=36, guidance_scale=7.0,
         width=768, height=768, generator=g).images[0]
out.save(os.path.join(OUT, "expR2_cup768.png"))
print("saved R2")
print("DONE")
