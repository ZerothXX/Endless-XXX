# -*- coding: utf-8 -*-
"""
train_utils.py —— 训练工具函数

职责（规格 §11 / §14 / §25 / §26）：
- set_seed            : 固定 random / numpy / torch / cuda 种子（规格 §26）
- make_noise_scheduler: 训练用 DDPM 加噪调度器（SDXL 标准参数，与 pipe 内 scheduler 无关）
- encode_prompt       : 双 text encoder 编码训练 prompt（与 SDXL pipeline 内部逻辑一致）
- build_add_time_ids  : SDXL 必需的 added_cond 时间条件（original_size/crops/target_size）
- train_step          : 标准噪声预测 loss（规格 §11.2），禁止像素级 loss / 生成图比对（规格 §28）
- get_vram_usage_gb   : 当前 CUDA 显存占用（GB），异常时返回 0.0
- log_training        : 训练日志打印（格式严格对齐规格 §25）
- record_curve        : 原始指标落盘 CSV（表头 step,epoch,loss,lr）
- save_curve_plots    : 由 CSV 生成三张曲线图（matplotlib Agg 后端，规格 §14）
- save_checkpoint     : 周期性保存 LoRA checkpoint（最新一次策略）
- print_oom_advice    : CUDA OOM 时的降级动作清单（规格 §25）
- get_steps_per_epoch : 计算每个 epoch 的优化步数（ceil(样本批数 / 梯度累积)）
"""

import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

import config
import model_utils


# ---------------------------------------------------------------------------
# 可复现性（规格 §26）
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定 Python random / NumPy / PyTorch / CUDA 全部随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 噪声调度器 / Prompt 编码 / 时间条件
# ---------------------------------------------------------------------------
def make_noise_scheduler():
    """训练用噪声调度器（DDPMScheduler，SDXL 标准参数）。

    注意：训练用 DDPM 调度器只负责"加噪"，与 pipe.scheduler（采样用）无关，
    两者相互独立（官方 SDXL LoRA 训练脚本同款做法）。
    """
    return DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",   # 标准噪声预测（规格 §11.2）
    )


def encode_prompt(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt):
    """双 text encoder 编码训练 prompt，返回 (prompt_embeds, pooled_prompt_embeds)。

    与 diffusers 0.31 SDXL pipeline.encode_prompt 同款逻辑（源码核实）：
    - 两个 TE 分别 tokenizer(padding="max_length", max_length=model_max_length, truncation=True)
    - prompt_embeds 取各自 hidden_states[-2]（倒数第二层），按 channel 拼接 -> [B, 77, 2048]
    - pooled_prompt_embeds 取 text_encoder_2 输出的 [0]：
      transformers 4.51 的 CLIPTextModelOutput 首字段是 text_projection 投影后的
      pooled 向量 [B, 1280]（即 SDXL 的 text_embeds，喂给 UNet 的 added_cond_kwargs）
    """
    # ---- text encoder 1（CLIPTextModel，openai/clip-vit-large-patch14） ----
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    te_outputs = text_encoder(text_input_ids, output_hidden_states=True)
    prompt_embeds_1 = te_outputs.hidden_states[-2]          # [B, 77, 768]

    # ---- text encoder 2（CLIPTextModelWithProjection） ----
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_2 = text_inputs_2.input_ids.to(text_encoder_2.device)
    te_outputs_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
    prompt_embeds_2 = te_outputs_2.hidden_states[-2]        # [B, 77, 1280]
    pooled_prompt_embeds = te_outputs_2[0]                  # [B, 1280]（投影后 pooled）

    # ---- 拼接两个 TE 的 prompt embeds ----
    prompt_embeds = torch.concat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # [B, 77, 2048]
    return prompt_embeds, pooled_prompt_embeds


def build_add_time_ids(batch_size: int, original_size, target_size, device, dtype) -> torch.Tensor:
    """构建 SDXL 必需的 added_cond 时间条件（规格 §11.2）。

    拼接顺序与 SDXL 训练脚本一致：
        original_size(2) + crops_coords_top_left(0, 0) + target_size(2) -> [B, 6]
    device / dtype 与 latents 保持一致（由调用方传入）。
    """
    if isinstance(original_size, int):
        original_size = (original_size, original_size)
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    add_time_ids = torch.tensor(
        [list(original_size) + [0, 0] + list(target_size)],
        dtype=dtype,
        device=device,
    )
    return add_time_ids.repeat(batch_size, 1)


# ---------------------------------------------------------------------------
# 单步训练
# ---------------------------------------------------------------------------
# 固定时间步评估（监控用）：在难度恒定的几个时间步上计算噪声预测误差，
# 用于观察"真实的梯度下降"——训练 loss（batch=1 + 随机 t）方差极大，
# 曲线呈锯齿状，肉眼无法判断是否在学习（此前用户反馈的"曲线没有下降"主因）。
EVAL_TIMESTEPS = (150, 500, 850)


def train_step(unet_lora, vae, noise_scheduler, pixel_values,
               prompt_embeds, pooled_prompt_embeds, time_ids) -> torch.Tensor:
    """标准噪声预测训练步（规格 §11.2），返回标量 loss。

    流程：VAE 编码 -> 潜变量采样 -> 随机 t 加噪 -> UNet 预测噪声 -> MSE。
    禁止像素级 loss / 生成图比对（规格 §28）。
    """
    dtype = prompt_embeds.dtype
    device = pixel_values.device

    # dataset 输出像素为 [0,1]（ToTensor），SDXL VAE 期望 [-1,1]，这里归一化
    pixel_values = pixel_values.to(device=device, dtype=dtype) * 2.0 - 1.0

    # ---- VAE 编码到潜空间并采样 ----
    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    latents = latents.to(dtype=dtype)

    # ---- 采样随机时间步并加噪 ----
    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=latents.device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # ---- UNet 预测噪声（SDXL 必须传 added_cond_kwargs） ----
    noise_pred = unet_lora(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        },
        return_dict=False,
    )[0]

    # ---- 噪声预测 MSE ----
    return F.mse_loss(noise_pred.float(), noise.float())


def eval_loss_fixed_timesteps(unet_lora, vae, noise_scheduler, pixel_values,
                              prompt_embeds, pooled_prompt_embeds, time_ids,
                              timesteps=EVAL_TIMESTEPS) -> torch.Tensor:
    """固定时间步评估 loss（torch.no_grad，不参与训练）。

    在 EVAL_TIMESTEPS 的每个 t 上，用同一份噪声计算噪声预测 MSE 后取均值：
    - 固定噪声与时间步便于比较同一参考的拟合变化，但不保证单调下降；
    - 与训练 loss 不同（无论是否在训练，t 若取到 1000 附近 loss 天然大，
      t 取到 10 附近 loss 天然小），它消除了"随机 t"带来的伪震荡。
    注意：VAE 潜变量取 latent_dist.mode()（均值）保证确定性，噪声用固定
    seed 的生成器产生，因此同一时刻的评估值可复现。
    调用方应使用固定参考集并逐图记录。train.py 覆盖整体、脸部与配饰；
    这些仍是训练样本，不是独立验证集，也不等价于生成图片质量。
    """
    dtype = prompt_embeds.dtype
    device = pixel_values.device

    with torch.no_grad():
        pixel_values = pixel_values.to(device=device, dtype=dtype) * 2.0 - 1.0
        latents = vae.encode(pixel_values).latent_dist.mode() * vae.config.scaling_factor
        latents = latents.to(dtype=dtype)

        g = torch.Generator(device=device).manual_seed(1234)
        noise = torch.randn(latents.shape, device=latents.device,
                            dtype=latents.dtype, generator=g)

        losses = []
        ts = [int(t) for t in timesteps]
        for t in ts:
            t_tensor = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
            noisy = noise_scheduler.add_noise(latents, noise, t_tensor)
            pred = unet_lora(
                noisy,
                t_tensor,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids,
                },
                return_dict=False,
            )[0]
            losses.append(F.mse_loss(pred.float(), noise.float()))
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 显存 / 日志 / 曲线
# ---------------------------------------------------------------------------
def get_vram_usage_gb() -> float:
    """当前 CUDA 显存占用（GB）；环境异常时返回 0.0（不阻断训练）。"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    except Exception:
        return 0.0


def log_training(epoch: int, num_epochs, step: int, max_steps: int,
                 loss: float, lr: float, vram_gb: float, eval_loss=None) -> None:
    """训练日志打印，格式严格对齐规格 §25。

    示例: Epoch 2/10  Step 120/800  Loss: 0.1842  LR: 0.0001  VRAM: 7.21 GB
    eval_loss 不为 None 时追加固定时间步评估 loss（体现真实学习进度）。
    """
    num_epochs_str = num_epochs if num_epochs is not None else "?"
    ev = f"  EvalLoss: {eval_loss:.4f}" if eval_loss is not None else ""
    print(f"Epoch {epoch}/{num_epochs_str}  Step {step}/{max_steps}  "
          f"Loss: {loss:.4f}{ev}  LR: {lr:.4f}  VRAM: {vram_gb:.2f} GB")


def record_curve(csv_path: str, step: int, epoch: int, loss: float, lr: float,
                 eval_loss=None) -> None:
    """原始指标追加写入 CSV（表头 step,epoch,loss,lr,eval_loss）。

    原始数据必须落盘（规格 §14 要求结构化文件），曲线图由 save_curve_plots 另行生成。
    eval_loss 为固定时间步评估 loss（可为 None，兼容旧数据/关闭场景）。
    """
    write_header = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "epoch", "loss", "lr", "eval_loss"])
        row = [step, epoch, f"{loss:.6f}", f"{lr:.8f}"]
        row.append(f"{eval_loss:.6f}" if eval_loss is not None else "")
        writer.writerow(row)


def save_curve_plots(csv_path: str, curves_dir: str, folder_prefix: str) -> None:
    """由 CSV 原始数据生成曲线图（matplotlib Agg 后端，文件名带 folder 前缀）。

    - {folder}_loss_by_step.png   : 每步 loss 原始曲线 + EMA 平滑 + 固定 t 评估 loss
    - {folder}_loss_by_epoch.png  : 按 epoch 均值
    - {folder}_learning_rate.png  : lr 随步变化
    """
    import matplotlib
    matplotlib.use("Agg")  # 无 GUI 后端，避免 PyCharm 控制台挂起
    import matplotlib.pyplot as plt

    if not os.path.isfile(csv_path):
        print(f"[train] 警告: 曲线 CSV 不存在，跳过绘图: {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = {
                "step": int(row["step"]),
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "lr": float(row["lr"]),
            }
            ev = row.get("eval_loss", "").strip()
            r["eval_loss"] = float(ev) if ev else None
            rows.append(r)
    if not rows:
        print(f"[train] 警告: 曲线 CSV 无数据，跳过绘图: {csv_path}")
        return

    os.makedirs(curves_dir, exist_ok=True)
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    lrs = [r["lr"] for r in rows]

    # 1) 每步 loss 原始曲线（含 EMA 平滑 + 固定 t 评估 loss）
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, marker=".", linestyle="-", markersize=4, alpha=0.55,
            label="train loss (step)")
    # EMA 平滑（alpha=0.1）
    ema = losses[0]
    ema_vals = []
    for v in losses:
        ema = 0.1 * v + 0.9 * ema
        ema_vals.append(ema)
    ax.plot(steps, ema_vals, linestyle="--", linewidth=1.6,
            label="train loss (EMA 0.1)")
    ev_rows = [(r["step"], r["eval_loss"]) for r in rows if r["eval_loss"] is not None]
    if ev_rows:
        ax.plot([s for s, _ in ev_rows], [v for _, v in ev_rows],
                marker="o", linestyle="-", markersize=4, linewidth=1.6,
                label="eval loss @ fixed t (150/500/850)")
        # eval loss EMA（更平滑地展示真实下降趋势；原始值仍受"评估图难度"抖动）
        ev_ema = ev_rows[0][1]
        ev_ema_vals = []
        for _, v in ev_rows:
            ev_ema = 0.2 * v + 0.8 * ev_ema
            ev_ema_vals.append(ev_ema)
        ax.plot([s for s, _ in ev_rows], ev_ema_vals,
                linestyle=":", linewidth=1.8,
                label="eval loss (EMA 0.2)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{folder_prefix} Loss by Step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(curves_dir, f"{folder_prefix}_loss_by_step.png"), dpi=150)
    plt.close(fig)

    # 2) 按 epoch 均值
    epoch_loss = {}
    for r in rows:
        epoch_loss.setdefault(r["epoch"], []).append(r["loss"])
    epoch_keys = sorted(epoch_loss.keys())
    epoch_means = [sum(epoch_loss[e]) / len(epoch_loss[e]) for e in epoch_keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_keys, epoch_means, marker="o", linestyle="-", markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss")
    ax.set_title(f"{folder_prefix} Loss by Epoch (mean)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(curves_dir, f"{folder_prefix}_loss_by_epoch.png"), dpi=150)
    plt.close(fig)

    # 3) lr 随步变化
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, lrs, marker=".", linestyle="-", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{folder_prefix} Learning Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(curves_dir, f"{folder_prefix}_learning_rate.png"), dpi=150)
    plt.close(fig)

    print(f"[train] 曲线图已保存到 {curves_dir}: "
          f"{folder_prefix}_loss_by_step.png / {folder_prefix}_loss_by_epoch.png / "
          f"{folder_prefix}_learning_rate.png")


# ---------------------------------------------------------------------------
# Checkpoint / OOM 建议 / 步数计算
# ---------------------------------------------------------------------------
def save_checkpoint(pipe, unet_lora, folder_prefix: str, step: int) -> None:
    """保存"最新一次"策略的 LoRA checkpoint（文件名 {folder_prefix}_step_{step}.safetensors）。

    保存路径: config.TRAIN_MODELS_DIR（规格 §14，output/train_models/）。
    """
    save_path = os.path.join(config.TRAIN_MODELS_DIR, f"{folder_prefix}_step_{step}.safetensors")
    model_utils.save_lora(pipe, unet_lora, save_path)


def print_oom_advice(e) -> None:
    """CUDA OOM 时打印规格 §25 推荐动作清单后调用方退出。"""
    print("\nCUDA Out Of Memory.")
    print("Recommended actions:")
    print("1. Reduce RESOLUTION")
    print("2. Increase GRADIENT_ACCUMULATION_STEPS")
    print("3. Disable optional modules")
    print("4. Enable CPU offload")
    print(f"错误详情: {type(e).__name__}: {e}")


def get_steps_per_epoch(dataloader, grad_accum: int) -> int:
    """每个 epoch 的优化步数 = ceil(批次数量 / 梯度累积步数)。"""
    n_batches = len(dataloader)
    return max(1, math.ceil(n_batches / max(1, grad_accum)))
