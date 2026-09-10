# -*- coding: utf-8 -*-
"""
基线对比（“梯度是否真实下降”的铁证）：同一张固定评估图（dataset 006 脸特写）、
同一 prompt、同一噪声种子、同一组固定时间步 t∈{150,500,850}：
  A: 纯底模（无 LoRA）        B: 底模 + 37_final（训练后）
输出均值 MSE 对比 + 图表（output/experiments/eval_compare.png）
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\Python All\Pytorch Study\Endless_xxx")

import torch

import config
import dataset
import train_utils
from model_utils import load_train_pipeline, prepare_lora

FOLDER = "37"
IMAGE = r"D:\Python All\Pytorch Study\Endless_xxx\dataset\37\images\006.png"
PROMPT = "ch37, 1girl, masterpiece, best quality, front view, face close-up"

# 固定评估图：短边对齐 640 + 顶部偏置 40%（与训练 v3 策略一致，保证脸在窗内）
import random
from PIL import Image
IMG = Image.open(IMAGE).convert("RGB")
scale = 640 / min(IMG.size)
nw, nh = int(round(IMG.size[0] * scale)), int(round(IMG.size[1] * scale))
IMG = IMG.resize((nw, nh), Image.BILINEAR)
top = random.Random(0).randint(0, int((nh - 640) * 0.4))
IMG = IMG.crop((0, top, 640, top + 640)).convert("RGB")

import numpy as np
arr = torch.from_numpy(np.array(IMG).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def eval_on(pipe, unet):
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", num_train_timesteps=1000,
                              prediction_type="epsilon")
    prompt_embeds, pooled = train_utils.encode_prompt(
        pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, PROMPT)
    prompt_embeds = prompt_embeds.to("cuda", dtype=torch.float16)
    pooled = pooled.to("cuda", dtype=torch.float16)
    time_ids = train_utils.build_add_time_ids(1, (640, 640), (640, 640),
                                              device="cuda", dtype=torch.float16)
    out = {}
    with torch.no_grad():
        pixel = (arr.to("cuda", dtype=torch.float16) * 2.0 - 1.0)
        latents = pipe.vae.encode(pixel).latent_dist.mode() * pipe.vae.config.scaling_factor
        latents = latents.to(torch.float16)
        g = torch.Generator(device="cuda").manual_seed(1234)
        noise = torch.randn(latents.shape, device="cuda", dtype=latents.dtype, generator=g)
        for t in (150, 500, 850):
            tt = torch.full((1,), t, device="cuda", dtype=torch.long)
            noisy = scheduler.add_noise(latents, noise, tt)
            pred = unet(noisy, tt, encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                        return_dict=False)[0]
            out[t] = float(torch.nn.functional.mse_loss(pred.float(), noise.float()))
    mean = float(np.mean(list(out.values())))
    out['mean'] = mean
    return out


print("=== A: 纯底模 ===")
pipe = load_train_pipeline(device="cuda", dtype=torch.float16)
unet = pipe.unet
a = eval_on(pipe, unet)
print("A:", {k: round(v, 4) for k, v in a.items()})
del unet
torch.cuda.empty_cache()

print("=== B: 底模 + 37_final LoRA（与 test.py 相同的 load_lora_weights 方式）===")
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
pipe2 = StableDiffusionXLPipeline.from_single_file(config.BASE_MODEL_PATH, torch_dtype=torch.float16)
pipe2.vae = AutoencoderKL.from_single_file(config.BASE_VAE_PATH, torch_dtype=torch.float16)
pipe2.load_lora_weights(r"D:\Python All\Pytorch Study\Endless_xxx\output\train_models\37_final.safetensors",
                        adapter_name="default")
pipe2.enable_vae_slicing()
pipe2.enable_vae_tiling()
pipe2.to("cuda")
b = eval_on(pipe2, pipe2.unet)
print("B:", {k: round(v, 4) for k, v in b.items()})

# 图表
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ts = [150, 500, 850]
fig, ax = plt.subplots(figsize=(7, 4.5))
xs = list(range(len(ts)))
w = 0.35
ax.bar([x - w / 2 for x in xs], [a[t] for t in ts], w, label="base (no LoRA)", color="#888")
ax.bar([x + w / 2 for x in xs], [b[t] for t in ts], w, label="trained LoRA (37_final)", color="#3a86ff")
ax.set_xticks(xs)
ax.set_xticklabels([f"t={t}" for t in ts] + [])
ax.axhline(a['mean'], color="#888", linestyle=":", alpha=0.6)
ax.axhline(b['mean'], color="#3a86ff", linestyle=":", alpha=0.6)
ax.set_ylabel("MSE (noise prediction)")
ax.set_title(f"Fixed-image eval: base vs trained  (mean {a['mean']:.3f} -> {b['mean']:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig(r"D:\Python All\Pytorch Study\Endless_xxx\output\experiments\eval_compare.png", dpi=150)
print("chart saved")
