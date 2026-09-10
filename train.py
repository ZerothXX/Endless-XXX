# -*- coding: utf-8 -*-
"""
train.py —— 训练主入口（PyCharm 直接 Run 本文件即可，规格 §24）

流程（规格 §11 / §12 / §14 / §25 / §26）：
    路径检查 -> 角色选择 -> 数据预处理 -> 加载 SDXL 底模 -> UNet 挂 LoRA
    -> 构建 DataLoader -> 优化器 / 学习率调度 -> 训练循环（混合精度 + 梯度累积
    + 梯度裁剪 + 梯度检查点）-> checkpoint 保存 -> 曲线生成 -> 产物清单打印

说明：
- 所有可调数值一律从 config.py 读取，本文件不硬编码任何超参数
- 不引入 argparse（PyCharm 直接 Run，规格 §24）
- 训练全程不加载 VLM / SAM / ControlNet / IP-Adapter（规格 §11.3 / §28 禁止5）
- dataset/ 目录无 __init__.py（且与根目录 dataset.py 模块同名），
  因此用 sys.path 挂载后按模块名导入 data_preprocessing（不创建 __init__.py，
  避免与根模块 dataset 产生包/模块命名歧义）
"""

import math
import json
import os
import sys

import config

# huggingface_hub 在 import 时读取 HF_ENDPOINT 并固定端点（constants.ENDPOINT / URL 模板），
# 之后不再重读。本文件后续的 import dataset -> torchvision -> transformers 会连带导入
# huggingface_hub，因此必须在此处（任何第三方导入之前）写入环境变量。
# 否则 from_single_file 拉取底模对应 diffusers config 时会直连 huggingface.co，
# 国内网络超时报 LocalEntryNotFoundError（实测问题）。config.HF_ENDPOINT 默认指向 hf-mirror.com。
os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)

import torch

import dataset
import image_utils
import model_utils
import train_utils


# ---------------------------------------------------------------------------
# 角色选择
# ---------------------------------------------------------------------------
def select_character() -> str:
    """确定本次训练的角色文件夹名。

    ① config.CHARACTER_ID 对应的 dataset/<id>/images/ 存在 -> 直接使用并打印；
    ② 否则扫描 dataset.scan_character_folders() 打印编号菜单，input() 选择（非法输入重试）；
    ③ 无候选时打印建目录指引并退出。
    """
    folder = config.CHARACTER_ID
    folder_dir = os.path.join(config.DATASET_DIR, folder)
    if os.path.isdir(os.path.join(folder_dir, "images")):
        print(f"[train] 使用 config.CHARACTER_ID: {folder}（dataset/{folder}/images/ 存在）")
        return folder

    print(f"[train] config.CHARACTER_ID={folder} 对应的 dataset/{folder}/images/ 不存在，"
          "改为从已有角色文件夹中选择。")
    folders = dataset.scan_character_folders(config.DATASET_DIR)
    if not folders:
        print("[train] dataset/ 下没有任何含 images/ 子目录的角色文件夹。")
        print("请先创建数据目录（规格 §2 / §4）：")
        print("    dataset/<角色ID>/images/001.png, 002.png, ...")
        print("    dataset/<角色ID>/captions.txt（可选，缺省用空标签）")
        sys.exit(1)

    print("dataset/ 下检测到的角色文件夹：")
    for i, f in enumerate(folders, start=1):
        print(f"  [{i}] {f}")
    while True:
        raw = input("请输入编号选择角色: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(folders):
            return folders[int(raw) - 1]
        print(f"输入无效，请输入 1~{len(folders)} 之间的编号。")


# ---------------------------------------------------------------------------
# 优化器 / 学习率调度
# ---------------------------------------------------------------------------
def _build_optimizer(unet_lora, extra_modules=None):
    """构建优化器：config.USE_8BIT_ADAM 时优先 bitsandbytes.AdamW8bit（省显存）。

    只把 requires_grad=True 的 LoRA 参数交给优化器（冻结的底模参数不需要
    状态，也不应产生更新），避免 8bit Adam 为 25 亿冻结参数白建状态。
    extra_modules：可选扩展参数；当前标准与24GB入口均仅训练 UNet LoRA。
    若 bitsandbytes 导入失败（环境缺失/版本不兼容），try/except 捕获后
    打印降级告警并回退到标准 torch.optim.AdamW（不阻断训练，规格 §1 依赖降级原则）。
    """
    modules = [unet_lora] + list(extra_modules or [])
    trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
    all_params = sum(sum(p.numel() for p in m.parameters()) for m in modules)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] 优化器参数: 仅可训练 LoRA 参数 {n_trainable:,}"
          f"（占全部 {all_params:,} 的"
          f"{n_trainable / max(1, all_params) * 100:.3f}%）")
    optimizer_cls = torch.optim.AdamW
    if config.USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            print("[train] 使用 bitsandbytes.optim.AdamW8bit（8bit Adam，省显存）")
        except Exception as e:
            print(f"[train] 警告: bitsandbytes 导入失败（{e}），降级为 torch.optim.AdamW（fp32）")
    else:
        print("[train] USE_8BIT_ADAM=False，使用 torch.optim.AdamW")
    return optimizer_cls(trainable, lr=config.LEARNING_RATE)


def _build_lr_scheduler(optimizer, max_steps):
    """构建学习率调度器（三档，按 config.LR_SCHEDULER 选择）。

    - "constant"            : 恒定学习率
    - "constant_with_warmup": 前 LR_WARMUP_STEPS 步线性预热后恒定
    - "cosine"              : 前 LR_WARMUP_STEPS 步预热后余弦衰减到 0（小样本推荐，防过拟合）
    """
    from diffusers.optimization import get_scheduler

    if config.LR_SCHEDULER == "constant":
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_steps,
        )
    elif config.LR_SCHEDULER == "constant_with_warmup":
        lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=config.LR_WARMUP_STEPS,
            num_training_steps=max_steps,
        )
    elif config.LR_SCHEDULER == "cosine":
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=config.LR_WARMUP_STEPS,
            num_training_steps=max_steps,
        )
    else:
        raise ValueError(
            f"config.LR_SCHEDULER 仅支持 'constant' / 'constant_with_warmup' / 'cosine'，"
            f"收到: {config.LR_SCHEDULER!r}"
        )
    print(f"[train] 学习率调度器: {config.LR_SCHEDULER}（warmup_steps={config.LR_WARMUP_STEPS}）")
    return lr_scheduler


# ---------------------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """训练主流程（按规格 §11/§12/§14/§25/§26 组织）。"""
    # ================= 0. 路径检查（fail fast） =================
    image_utils.create_dirs()
    status = model_utils.verify_model_paths()
    print("=" * 60)
    print("[路径检查]")
    for key, info in status.items():
        mark = "OK" if info["exists"] else "缺失"
        note = f"（{info.get('note', '')}）" if info.get("note") else ""
        print(f"  [{mark}] {key}: {info['path']} {note}")
    print("=" * 60)
    if not status["BASE_MODEL_PATH"]["exists"] or not status["BASE_VAE_PATH"]["exists"]:
        print("[train] 基础模型文件缺失，无法训练。请检查 models/ 目录与 config 中模型路径后重试。")
        sys.exit(1)

    # ================= 1. 角色选择 =================
    folder = select_character()

    # ================= 2. 训练前自动数据预处理 =================
    # 背景：dataset/ 目录故意没有 __init__.py（避免与根模块 dataset.py 同名歧义），
    # 因此不能 "from dataset.data_preprocessing import ..."；历史实现是
    # "把 dataset/ 挂到 sys.path 再按顶层模块导入"——运行时正确，但 IDE 静态分析
    # 无法解析动态路径，会报"未解析的引用 data_preprocessing / preprocess_character_folder"
    # （误报，不影响运行）。
    # 现改为"按文件路径加载模块"（importlib.util），行为与旧实现完全一致
    # （模块名 data_preprocessing、顶层模块语义），同时不再污染 sys.path，
    # 并配合 TYPE_CHECKING 分支让 IDE 正常识别。
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:  # 仅 IDE 静态分析用，运行时不执行
        from data_preprocessing import preprocess_character_folder  # noqa: F401

    import importlib.util
    _pp_path = os.path.join(config.DATASET_DIR, "data_preprocessing.py")
    _pp_spec = importlib.util.spec_from_file_location("data_preprocessing", _pp_path)
    _pp_mod = importlib.util.module_from_spec(_pp_spec)
    _pp_spec.loader.exec_module(_pp_mod)
    preprocess_character_folder = _pp_mod.preprocess_character_folder
    preprocess_summary = (preprocess_character_folder(folder)
                          if getattr(config, "AUTO_PREPROCESS_DATASET", True) else {"ok": True})
    if not getattr(config, "AUTO_PREPROCESS_DATASET", True):
        print("[train] 数据预处理写入已关闭：只读原始图片与 captions")
    if not preprocess_summary.get("ok"):
        print("[train] 警告: 数据预处理存在异常（见上方警告），继续训练前请确认数据可用。")

    # ================= 3. 角色配置（预设优先于全局，config.get_character_config 已合并） =================
    char_cfg = config.get_character_config(folder)
    resolution = char_cfg["resolution"]
    max_steps = char_cfg["max_train_steps"]
    trigger = char_cfg["trigger"]
    print(f"[train] 角色配置: folder={folder} trigger={trigger} category={char_cfg['category']} "
          f"resolution={resolution} max_train_steps={max_steps}")

    # ================= 4. 随机种子（规格 §26） =================
    train_utils.set_seed(config.SEED)

    # ================= 5. 模型加载 + LoRA 注入 =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] 警告: 未检测到 CUDA，将使用 CPU 训练（极慢，仅用于调试）")
    if config.MIXED_PRECISION == "fp16":
        weight_dtype = torch.float16
    elif config.MIXED_PRECISION == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    print(f"[train] 设备: {device}  权重精度: {weight_dtype}")

    pipe = model_utils.load_train_pipeline(device=device, dtype=weight_dtype)

    # CPU offload 与训练反向传播不兼容：offload hook 会在前向结束后把权重搬回 CPU，
    # 导致 backward 找不到设备一致的权重。因此训练循环前移除 hooks 并将组件放回 GPU
    # （load_train_pipeline 中的 offload 路径保留，用于验证可加载 / 推理侧复用）。
    if config.USE_CPU_OFFLOAD:
        print("[train] 警告: 训练阶段 CPU offload 与反向传播不兼容（offload hook 前向结束后搬回 CPU），"
              "已移除 offload hooks 并将组件置于 GPU。")
        print("        如需严格省显存，请按 OOM 降级清单调整：降低 RESOLUTION / 增大 "
              "GRADIENT_ACCUMULATION_STEPS / 关闭可选模块。")
        pipe.remove_all_hooks()
        pipe.to(device)

    # G24 仅提高超参数，继续使用此训练核心和冻结的文本编码器。
    unet_lora = model_utils.prepare_lora(pipe, rank=config.LORA_RANK, alpha=config.LORA_ALPHA)

    # 梯度检查点（8GB 显存关键优化，规格 §12）
    if config.USE_GRADIENT_CHECKPOINTING:
        pipe.unet.enable_gradient_checkpointing()
        print("[train] 已启用 UNet 梯度检查点（USE_GRADIENT_CHECKPOINTING=True）")

    # ================= 6. 数据 =================
    # prompt 编码缓存按单样本设计（batch["prompt"][0]），batch>1 时 UNet 前向形状会错，
    # 因此显式限制 batch=1（8GB 单卡场景的默认配置，规格 §12 亦为 batch=1）
    assert config.TRAIN_BATCH_SIZE == 1, (
        "当前训练循环仅支持 TRAIN_BATCH_SIZE=1（prompt 编码缓存按单样本设计），"
        f"收到 {config.TRAIN_BATCH_SIZE}；如需 batch>1 请先改造 prompt 广播逻辑"
    )
    train_ds = dataset.get_train_dataset(
        folder=folder,
        resolution=resolution,
        image_limit=config.TRAIN_IMAGE_LIMIT,
    )
    train_dl = dataset.get_train_dataloader(
        dataset=train_ds,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATALOADER_NUM_WORKERS,
    )
    print(f"[train] 训练数据: {len(train_ds)} 张图片（TRAIN_IMAGE_LIMIT={config.TRAIN_IMAGE_LIMIT}）")

    # ================= 7. 优化器 / 学习率调度 =================
    optimizer = _build_optimizer(unet_lora)
    lr_scheduler = _build_lr_scheduler(optimizer, max_steps)

    # ================= 8. 混合精度 + 梯度累积（accelerate） =================
    # 取舍说明（二选一中的 accelerate 方案）：
    #   accelerate 1.10.1 的 Accelerator(mixed_precision, gradient_accumulation_steps)
    #   负责混合精度上下文与梯度同步时机（accumulate context + sync_gradients），
    #   代码量小、与 diffusers 生态同源；代价是引入一个封装层。
    #   训练中通过 accelerator.autocast() 显式包住前向（1.10.1 的 accumulate
    #   不含 autocast，源码核实），lr_scheduler 不 prepare（单卡场景手动 step）。
    from accelerate import Accelerator

    mixed_precision = config.MIXED_PRECISION if config.MIXED_PRECISION not in ("", "no", None) else None
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    )
    lora_modules = [unet_lora]
    prepared = accelerator.prepare(*lora_modules, optimizer, train_dl)
    unet_lora = prepared[0]
    optimizer = prepared[-2]
    train_dl = prepared[-1]
    print(f"[train] accelerate: mixed_precision={mixed_precision} "
          f"gradient_accumulation_steps={config.GRADIENT_ACCUMULATION_STEPS}")

    # ================= 9. 训练循环 =================
    noise_scheduler = train_utils.make_noise_scheduler()
    steps_per_epoch = train_utils.get_steps_per_epoch(train_dl, config.GRADIENT_ACCUMULATION_STEPS)
    planned_epochs = config.NUM_EPOCHS if config.NUM_EPOCHS is not None else max(1, math.ceil(max_steps / steps_per_epoch))
    csv_path = os.path.join(config.CURVES_DIR, f"{folder}_train_metrics.csv")
    # 新一次训练前清空旧 CSV：record_curve 是追加写入（"a" 模式），不清空会把
    # 上一次训练的指标混叠进来，导致曲线图错误（实测 bug）。
    if os.path.isfile(csv_path):
        os.remove(csv_path)
        print(f"[train] 已清空上次训练曲线 CSV，本次重新记录: {csv_path}")

    print(f"[train] 开始训练: 每 epoch {steps_per_epoch} 步（{len(train_dl)} 批 / "
          f"累积 {config.GRADIENT_ACCUMULATION_STEPS}），计划 {planned_epochs} 个 epoch，"
          f"上限 {max_steps} 步")

    prompt_cache = {}          # prompt -> (prompt_embeds, pooled)，text encoder 冻结所以可缓存
    global_step = 0
    best_loss = float("inf")   # 固定时间步评估 loss 最优跟踪（保存 {folder}_best.safetensors）
    # 说明：历史版本以"单步训练 loss 最低"选 best，但单步 loss 受随机 t 影响
    # 极大（t 小 loss 天然低），实际是在选"运气最好的采样"，并非模型最优。
    # 改为用固定时间步评估 loss（eval_loss_fixed_timesteps）选 best，真正反映进度。
    best_loss_note = "（固定时间步评估 loss）"

    if getattr(config, "TRAIN_CACHE_TEXT_ENCODERS", False):
        with torch.no_grad():
            for idx in range(len(train_ds)):
                prompt = train_ds.get_prompt(idx)
                for tokenizer in (pipe.tokenizer, pipe.tokenizer_2):
                    if len(tokenizer(prompt, truncation=False, verbose=False)["input_ids"]) > tokenizer.model_max_length:
                        raise ValueError(f"Training caption exceeds CLIP budget: {train_ds.image_paths[idx]}")
                if prompt not in prompt_cache:
                    prompt_cache[prompt] = tuple(t.detach().cpu() for t in train_utils.encode_prompt(
                        pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, prompt))
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[train] 已缓存 {len(prompt_cache)} 种文本条件，冻结的 text encoders 移至 CPU")

    # Deterministic whole images; per-reference diagnostics are not visual acceptance.
    eval_samples = [(os.path.basename(train_ds.image_paths[idx]), train_ds.get_item(idx, deterministic=True))
                    for idx in train_ds.evaluation_indices()]
    eval_details_path = os.path.join(config.CURVES_DIR, f"{folder}_reference_eval.jsonl")
    print(f"[train] 固定训练诊断集（非独立验证集）: {[name for name, _ in eval_samples]}")

    for epoch in range(1, planned_epochs + 1):
        # 每 epoch 重置均值统计（否则"平均 Loss"会变成跨 epoch 累计值，日志误导）
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for batch in train_dl:
            with accelerator.accumulate(unet_lora):
                # ---- prompt 编码（缓存命中则跳过 TE 前向；TE 冻结可安全缓存） ----
                prompt = batch["prompt"][0]
                if prompt not in prompt_cache:
                    prompt_cache[prompt] = train_utils.encode_prompt(
                        pipe.tokenizer, pipe.tokenizer_2,
                        pipe.text_encoder, pipe.text_encoder_2,
                        prompt,
                    )
                prompt_embeds, pooled_prompt_embeds = prompt_cache[prompt]
                prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                pixel_values = batch["pixel_values"].to(accelerator.device)

                # ---- SDXL added_cond 时间条件：按实际样本尺寸构造 ----
                # （v3 混合内容裁剪输出非正方形、每样本可不同的 8 倍数尺寸，
                #   original/target 必须与真实 H/W 一致）
                _, _, ph, pw = pixel_values.shape
                time_ids = train_utils.build_add_time_ids(
                    pixel_values.shape[0], (ph, pw), (ph, pw),
                    device=accelerator.device, dtype=weight_dtype,
                )

                # ---- 前向 + 反向（混合精度上下文包裹前向） ----
                with accelerator.autocast():
                    loss = train_utils.train_step(
                        unet_lora, pipe.vae, noise_scheduler, pixel_values,
                        prompt_embeds, pooled_prompt_embeds, time_ids,
                    )
                accelerator.backward(loss)

                # ---- 梯度累积边界：同步梯度时执行一次优化步 ----
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet_lora.parameters(), config.MAX_GRAD_NORM)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    loss_item = loss.detach().item()
                    lr_now = lr_scheduler.get_last_lr()[0]
                    epoch_loss_sum += loss_item
                    epoch_loss_count += 1

                    # Per-reference fixed-noise diagnostics, retaining each accessory score.
                    eval_loss = None
                    if getattr(config, "EVAL_LOSS_INTERVAL", 20) > 0 \
                            and global_step % config.EVAL_LOSS_INTERVAL == 0:
                        reference_losses = {}
                        for name, sample in eval_samples:
                            ep = sample["prompt"]
                            if ep not in prompt_cache:
                                with torch.no_grad():
                                    prompt_cache[ep] = train_utils.encode_prompt(
                                        pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, ep)
                            ee, pooled = prompt_cache[ep]
                            pixel = sample["pixel_values"]
                            _, eh, ew = pixel.shape
                            tids = train_utils.build_add_time_ids(1, (eh, ew), (eh, ew),
                                device=accelerator.device, dtype=weight_dtype)
                            reference_losses[name] = train_utils.eval_loss_fixed_timesteps(
                                unet_lora, pipe.vae, noise_scheduler, pixel.unsqueeze(0).to(accelerator.device),
                                ee.to(device=accelerator.device, dtype=weight_dtype),
                                pooled.to(device=accelerator.device, dtype=weight_dtype), tids).detach().item()
                        eval_loss = sum(reference_losses.values()) / len(reference_losses)
                        with open(eval_details_path, "a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"step": global_step, "losses": reference_losses,
                                                      "mean": eval_loss}) + "\n")

                    # 日志（规格 §25 格式）+ 原始指标落盘（step/epoch 双粒度，规格 §14）
                    if global_step % config.LOG_EVERY_N_STEPS == 0:
                        train_utils.log_training(
                            epoch, planned_epochs, global_step, max_steps,
                            loss_item, lr_now, train_utils.get_vram_usage_gb(),
                            eval_loss=eval_loss,
                        )
                        train_utils.record_curve(csv_path, global_step, epoch, loss_item, lr_now,
                                                 eval_loss=eval_loss)

                    # checkpoint 双策略（工作要求 / 规格 §14）：
                    #   1) 最新一次：每 SAVE_STEPS 步保存 {folder}_step_{N}.safetensors
                    #   2) 最优一次：固定时间步评估 loss 创新低时保存 {folder}_best.safetensors
                    #      （本项目规格无独立验证集，20 张图全部参与训练；评估 loss 在
                    #        固定 t 上计算，比单步训练 loss 稳定可靠。注意只在有
                    #        eval_loss 的步（EVAL_LOSS_INTERVAL 的倍数）更新 best，
                    #        避免"单步 t 运气"污染选择）
                    if config.SAVE_STEPS > 0 and global_step % config.SAVE_STEPS == 0:
                        train_utils.save_checkpoint(pipe, unet_lora, folder, global_step)
                    if eval_loss is not None and eval_loss < best_loss:
                        best_loss = eval_loss
                        best_path = os.path.join(config.TRAIN_MODELS_DIR, f"{folder}_best.safetensors")
                        model_utils.save_lora(pipe, unet_lora, best_path)

            if global_step >= max_steps:
                break
        print(f"[train] Epoch {epoch}/{planned_epochs} 完成，平均 Loss: "
              f"{epoch_loss_sum / max(1, epoch_loss_count):.4f}")
        if global_step >= max_steps:
            break

    # ================= 10. 收尾：最终权重 + 曲线 + 产物清单 =================
    final_path = os.path.join(config.TRAIN_MODELS_DIR, f"{folder}_final.safetensors")
    model_utils.save_lora(pipe, unet_lora, final_path)

    train_utils.save_curve_plots(csv_path, config.CURVES_DIR, folder)

    # 训练摘要写入 logs 目录
    summary_path = os.path.join(config.LOGS_DIR, f"{folder}_train_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== 训练摘要 ==========\n")
        f.write(f"角色 folder          : {folder}\n")
        f.write(f"触发词 trigger       : {trigger}\n")
        f.write(f"分辨率 resolution    : {resolution}\n")
        f.write(f"总步数 max_steps     : {max_steps}（实际完成 {global_step}）\n")
        f.write(f"epochs 计划          : {planned_epochs}\n")
        f.write(f"batch / 累积         : {config.TRAIN_BATCH_SIZE} / {config.GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"混合精度             : {config.MIXED_PRECISION}\n")
        f.write(f"LoRA rank / alpha    : {config.LORA_RANK} / {config.LORA_ALPHA}\n")
        f.write(f"学习率 / 调度器      : {config.LEARNING_RATE} / {config.LR_SCHEDULER}\n")
        f.write(f"最终权重             : {final_path}\n")
        f.write(f"曲线原始数据         : {csv_path}\n")
        f.write(f"曲线图目录           : {config.CURVES_DIR}\n")
        f.write(f"8bit Adam            : {config.USE_8BIT_ADAM}\n")
        f.write(f"梯度检查点           : {config.USE_GRADIENT_CHECKPOINTING}\n")
    print(f"[train] 训练摘要已保存: {summary_path}")

    print("=" * 60)
    print("[train] 训练完成，产物清单:")
    print(f"  LoRA 权重目录 : {config.TRAIN_MODELS_DIR}")
    print(f"     最终权重   : {folder}_final.safetensors")
    print(f"     最优权重   : {folder}_best.safetensors"
          f"（固定时间步评估 loss 最低，best_loss={best_loss:.4f} {best_loss_note.strip()}）")
    print(f"     周期权重   : {folder}_step_*.safetensors（每 {config.SAVE_STEPS} 步）")
    print(f" 曲线目录       : {config.CURVES_DIR}")
    print(f"     {folder}_loss_by_step.png / {folder}_loss_by_epoch.png / {folder}_learning_rate.png")
    print(f" 曲线原始数据   : {csv_path}")
    print(f" 日志目录       : {config.LOGS_DIR}（{folder}_train_summary.txt）")
    print("=" * 60)


def run_training(run_root=None):
    """Public CLI lifecycle; web workers keep calling main with their own job paths."""
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，拒绝意外启动长时间 CPU 训练")
    if getattr(config, "TRAIN_ISOLATED_RUN", True):
        from training_run import run_isolated
        return run_isolated(main, select_character(), run_root=run_root)
    main()


if __name__ == "__main__":
    """PyCharm 直接运行入口（规格 §24）。OOM 时打印降级建议清单后退出（规格 §25）。"""
    try:
        run_training()
    except torch.cuda.OutOfMemoryError as e:
        train_utils.print_oom_advice(e)
        sys.exit(1)
