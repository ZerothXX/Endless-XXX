# -*- coding: utf-8 -*-
"""
test.py —— 推理主入口（PyCharm 直接 Run 本文件即可，规格 §15 / §24）

全链路（规格 §15 / §31）：
    输入解析 -> 角色选择 -> VLM 主体理解 + 配饰迁移规划 -> SAM2 主体分割
    -> 管线装配（SDXL + LoRA + ControlNet / img2img，IP-Adapter 可选）
    -> 逐图生成 -> 可选 VLM 结果评价 -> 产物落盘（png + txt + json）

显存分时策略（RTX 4060 8GB 关键，规格 §3）：
    VLM 段（主体分析 + 规划）-> client.unload() 释放显存
    SAM 段（主体分割）       -> unload_sam 释放显存
    扩散段（生成）           -> 管线独占显存
    （可选）评价段           -> 重新加载 VLM 评价后再次卸载

 降级语义（规格 §27，每步打印实际启用状态）：
    VLM 不可用        -> 主 体分析返回 unknown、配饰规划走规则表
    SAM 不可用        -> 整图 mask（Canny 背景抑制等效不生效）
    ControlNet 缺失   -> img2img（strength=config.IMG2IMG_STRENGTH）
    IP-Adapter 缺失   -> 跳过（LoRA + ControlNet/img2img）
    LoRA 缺失         -> 纯底模推理

说明：
- 所有可调数值一律从 config.py 读取，本文件不硬编码任何超参数
- 不引入 argparse（PyCharm 直接 Run，规格 §24）
- HF_ENDPOINT 必须在任何第三方导入（torch/diffusers/transformers 链）之前写入，
  否则 from_single_file 拉取 SDXL 配置会直连 huggingface.co 超时（实测问题，
  与 train.py 同一接线模式）
"""

import gc
import json
import os
import sys
import time
import secrets
import hashlib

import config

# huggingface_hub 在 import 时读取 HF_ENDPOINT 并固定端点（constants.ENDPOINT），
# 之后不再重读。本文件后续的 import 链（numpy -> PIL -> torch -> diffusers ->
# transformers）会连带导入 huggingface_hub，因此必须在此处（任何第三方导入之前）
# 写入环境变量（与 train.py 同一模式）。config.HF_ENDPOINT 默认指向 hf-mirror.com。
os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)

import numpy as np
from PIL import Image

import torch

import dataset
import image_utils
import model_utils
from character.semantic_memory import AccessoryAdaptationPlanner, load_character_memory, card_from_memory
from character.semantic_memory import style_policy_for_subject
from character.inference_policy import generation_settings
from character.transfer_constraints import composition_constraints
from vision.segmentation import get_subject_mask_or_full, segment_subject, load_sam2, unload_sam
from vision.background_guard import preserve_background, validate_subject_mask
from vision.subject_analyzer import analyze_subject
from vlm.vlm_utils import EVAL_PROMPT, VLMClient


# ---------------------------------------------------------------------------
# 输入 / 角色 / LoRA 路径解析
# ---------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")   # 输入图片扩展名（规格 §32 批量）


def _resolve_input_images() -> list:
    """解析 config.INPUT_IMAGE：文件 -> 单图列表；目录 -> 扫描图片列表（§32 批量）。"""
    path = config.INPUT_IMAGE
    if os.path.isfile(path):
        print(f"[test] 输入为单文件: {path}")
        return [path]
    if os.path.isdir(path):
        names = sorted((n for n in os.listdir(path) if n.lower().endswith(IMAGE_EXTS)),
                       key=dataset._natural_key)   # 自然排序：010.png 排在 002.png 之后
        paths = [os.path.join(path, n) for n in names]
        print(f"[test] 输入为目录，扫描到 {len(paths)} 张图片: {path}")
        return paths
    print(f"[test] 错误: config.INPUT_IMAGE 既不是文件也不是目录: {path}")
    return []


def _resolve_character_folder() -> str:
    """确定推理角色文件夹（与 train 共用 dataset.scan_character_folders）。

    config.CHARACTER_ID 对应的 dataset/<id>/images/ 存在 -> 直接使用；
    否则扫描 dataset/，多文件夹时自动取排序第一个并打印说明
    （无人值守推理不采用训练菜单式的 input() 交互）。
    """
    folder = config.CHARACTER_ID
    folder_dir = os.path.join(config.DATASET_DIR, folder)
    if os.path.isdir(os.path.join(folder_dir, "images")):
        print(f"[test] 使用 config.CHARACTER_ID: {folder}（dataset/{folder}/images/ 存在）")
        return folder

    print(f"[test] config.CHARACTER_ID={folder} 对应的 dataset/{folder}/images/ 不存在，"
          "改为扫描 dataset/ 下的角色文件夹。")
    folders = dataset.scan_character_folders(config.DATASET_DIR)
    if not folders:
        raise RuntimeError("[test] dataset/ 下没有任何含 images/ 子目录的角色文件夹，无法推理。"
                           "请先创建 dataset/<角色ID>/images/（规格 §2 / §4）。")
    print(f"[test] 检测到角色文件夹: {folders}，自动使用第一个: {folders[0]}"
          "（可修改 config.CHARACTER_ID 指定）")
    return folders[0]


def _resolve_lora_path(folder: str):
    """Resolve exactly the configured role/model policy; log the effective path."""
    from weight_selection import resolve_weight
    path = resolve_weight(folder)
    print(f"[test] LoRA 来源: {config.LORA_MODEL_SOURCE}; 选择模式: {config.LORA_SELECTION_MODE}; 实际权重: {path or '纯底模（显式允许）'}")
    return path


def _write_json(path: str, obj) -> None:
    """写结构化 JSON 文件（UTF-8, indent=2），目录不存在自动创建。"""
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[test] 已保存: {path}")


# ---------------------------------------------------------------------------
# 推理主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """推理主流程（规格 §15 全链路 + §27 逐级降级 + §32 批量）。"""
    t_start = time.time()
    image_utils.create_dirs()

    # ================= 0. 模块状态摘要（降级语义可见性，规格 §27） =================
    print("=" * 60)
    print("[test] 模块状态:")
    print(f"  USE_VLM={config.USE_VLM}  USE_SAM={config.USE_SAM}  "
          f"USE_CONTROLNET={config.USE_CONTROLNET}  USE_IP_ADAPTER={config.USE_IP_ADAPTER}")
    print(f"  ENABLE_RESULT_EVALUATION={config.ENABLE_RESULT_EVALUATION}  "
          f"USE_CPU_OFFLOAD={config.USE_CPU_OFFLOAD}")
    print("=" * 60)

    # ================= 1. 输入解析 + 统一推理尺寸 =================
    image_paths = _resolve_input_images()
    if not image_paths:
        print("[test] 没有可处理的输入图片，结束。请检查 config.INPUT_IMAGE。")
        return
    images = {}
    for p in image_paths:
        img = image_utils.load_image(p)
        # 保持原图完整构图与宽高比：等比缩放至**短边 = INFERENCE_RESOLUTION**、
        # 长边 ≤ INFERENCE_MAX_LONG，不裁剪（v5.1 修正：早期"长边=640"使全身/宽幅
        # 画面里的面孔只有 ~40-80px，五官/服饰/姿态被拖垮到扭曲——旧方窗虽及格
        # 但会裁掉头；本方案兼顾"比例完整"与"主体像素量充分"）。
        img = image_utils.resize_fit(img, config.INFERENCE_RESOLUTION,
                                     max_long=int(getattr(config, "INFERENCE_MAX_LONG", 960)))
        images[p] = img
    print(f"[test] 输入 {len(image_paths)} 张图片，等比适配（不裁剪，短边="
          f"{config.INFERENCE_RESOLUTION}）：")
    for p in image_paths:
        print(f"    {os.path.basename(p)} -> {images[p].size[0]}x{images[p].size[1]}")

    # ================= 2. 角色 =================
    folder = _resolve_character_folder()
    trigger = config.get_trigger(folder)
    print(f"[test] 角色 folder={folder}  trigger={trigger}")

    # ================= 3. VLM 段（主体理解 + 配饰迁移规划，用完即卸载） =================
    # 取舍说明（二选一，选择"VLM 段内先算好各图 plan 再卸载"）：
    #   AccessoryAdaptationPlanner.plan 的 VLM 主路径需要 subject_image（PIL 或路径）
    #   与可用 client（semantic_memory 源码核实），VLM 卸载后只能走规则表兜底；
    #   先规划后卸载语义质量更高，且 VLM 显存不会与扩散管线叠加（8GB 分时关键，规格 §3）。
    #
    # 角色身份描述：卡片优先（config.CHARACTER_CARDS，人工校对）——
    # VLM 角色解析（Qwen2.5-VL）实测会把粉发角色幻觉成 "light blue hair /
    # Flower Fairy / light blue dress"，该描述被拼进 prompt 后直接抵消 LoRA
    # 的身份信号。卡片存在时不再加载 VLM 语义记忆（output/semantic/*.json），
    # VLM 只保留两个职责：输入主体理解 + 配饰迁移规划（subject 分析）。
    card = config.get_character_card(folder)
    if card:
        print(f"[test] 使用角色卡片（人工校对）: {folder} "
              f"style_policy={card.get('style_policy', 'preserve')}")
    else:
        print("[test] 角色无卡片，回退 VLM 语义记忆路径（output/semantic/）")

    client = None
    if config.USE_VLM:
        client = VLMClient(
            config.VLM_PATH,
            dtype=config.VLM_DTYPE,
            load_in_4bit=config.VLM_LOAD_IN_4BIT,
            offload_cpu=config.VLM_OFFLOAD_CPU,
        )
        if not getattr(client, "available", False):
            print("[test] 降级: VLM 不可用，主体分析返回 unknown、配饰规划走规则表（规格 §27）")
            client = None
    else:
        print("[test] USE_VLM=False：主体分析返回 unknown、配饰规划走规则表（规格 §27）")

    # 角色语义记忆：卡片路径不需要；无卡片时已有记忆文件直接读取（output/semantic/），
    # 不存在则由 VLM 生成（client 不可用 / USE_VLM=False 时 offline 兜底，规格 §5.2 / §27）
    memory = load_character_memory(folder, client)

    # 配饰语义表示：卡片 accessory 优先；否则取记忆中的主配饰（signature_accessories
    # 优先，其次 accessories，缺省空 dict）
    accessory = {}
    if card and card.get("accessory"):
        accessory = card["accessory"]
    else:
        for key in ("signature_accessories", "accessories"):
            for it in (memory or {}).get(key) or []:
                if isinstance(it, dict) and it.get("name"):
                    accessory = it
                    break
            if accessory:
                break
    print(f"[test] 主配饰: {accessory.get('name', '（无，使用签名符号占位）')}")

    # 逐图：主体分析 + 配饰迁移规划（在 VLM 段内完成，理由见上）
    planner = AccessoryAdaptationPlanner(client=client, memory=memory)
    subject_reprs = {}
    plans = {}
    for p in image_paths:
        subject_reprs[p] = analyze_subject(client, images[p])
        planner.subject_image = images[p]      # VLM 规划路径需要主体图片
        plans[p] = planner.plan(subject_reprs[p], accessory)
        stem = os.path.splitext(os.path.basename(p))[0]
        _write_json(os.path.join(config.RESULT_DIR, f"{stem}_subject.json"), subject_reprs[p])
        _write_json(os.path.join(config.RESULT_DIR, f"{stem}_adaptation.json"), plans[p])
        print(f"[test] {os.path.basename(p)}: subject_type="
              f"{subject_reprs[p].get('subject_type')} plan.source={plans[p].get('source')}")

    if config.USE_VLM and client is not None:
        client.unload()   # 释放 VLM 显存（8GB 分时关键）
        client = None

    # ================= 4. SAM 段（主体分割，用完即卸载） =================
    masks = {}
    sam = None
    protect_background = bool(getattr(config, "PRESERVE_BACKGROUND", False))
    if protect_background and not config.USE_SAM:
        raise ValueError("PRESERVE_BACKGROUND requires USE_SAM; refusing silent full-image edits")
    if config.USE_SAM and (config.SUPPRESS_BACKGROUND_EDGES or protect_background):
        sam = load_sam2()
        if protect_background and sam is None:
            raise RuntimeError("Background protection requested but SAM could not load")
        if sam is None:
            print("[test] 降级: SAM2 加载失败，使用整图 mask（规格 §9/§27）")
    else:
        print("[test] 无区域 mask 消费者：跳过 SAM；Canny 保留整图结构")
    for p in image_paths:
        if protect_background:
            mask = segment_subject(images[p], sam)
            mask, _ = validate_subject_mask(mask, images[p].size)
            masks[p] = mask
            image_utils.save_image(mask, os.path.join(config.RESULT_DIR,
                                    f"{os.path.splitext(os.path.basename(p))[0]}_subject_mask.png"))
        else:
            masks[p] = (get_subject_mask_or_full(images[p], sam)
                        if config.USE_SAM and config.SUPPRESS_BACKGROUND_EDGES else None)
    if sam is not None:
        unload_sam(sam)
        sam = None

    # ================= 5. 管线装配（底模 + VAE + LoRA + ControlNet/img2img） =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[test] 警告: 未检测到 CUDA，将使用 CPU 推理（极慢，仅用于调试）")
    lora_path = _resolve_lora_path(folder)
    pipe, use_controlnet = model_utils.load_inference_pipeline(
        lora_path=lora_path, device=device)

    ip_adapter_loaded = False
    if config.USE_IP_ADAPTER:
        ip_adapter_loaded = model_utils.load_ip_adapter(pipe)
    else:
        print("[test] USE_IP_ADAPTER=False：不加载 IP-Adapter（规格 §3.3）")

    # 生成器：INFERENCE_SEED >= 0 固定种子复现；-1 随机（规格 §26）
    # 每张图新建 generator：固定种子下若复用同一 generator，第 2 张图起的噪声序列
    # 会与前图相关，破坏单图复现语义
    seed = config.INFERENCE_SEED
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed >= 0:
        print(f"[test] INFERENCE_SEED={seed}：每张输入图使用固定种子复现")
    else:
        print("[test] INFERENCE_SEED=-1：本次推理随机种子（不固定）")

    # ================= 6. 逐图生成 + 落盘（规格 §15 / §32） =================
    for i, p in enumerate(image_paths, start=1):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = images[p]
        subject_repr = subject_reprs[p]
        plan = plans[p]
        category = config.get_character_config(folder)["category"]
        if card:
            prompt = planner.build_card_transfer_prompt(
                subject_repr, plan, card, trigger, category=category,
                tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2, max_tokens=76)
        else:
            prompt = planner.build_card_transfer_prompt(
                subject_repr, plan, card_from_memory(memory), trigger, category=category,
                tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2, max_tokens=76)
        actual_seed = seed if seed >= 0 else secrets.randbelow(2**32)
        generator = torch.Generator(device=gen_device).manual_seed(actual_seed)
        print(f"[test] [{i}/{len(image_paths)}] 生成: {os.path.basename(p)}")
        print(f"    prompt: {prompt}")

        # ---- 主体感知负向提示词 + LoRA 强度（规格 §18.2） ----
        # 动物/物品主体必须抑制 LoRA 训练烙进的 "1girl/少女" 先验，否则猫会生成成
        # 猫娘、杯子生成成人物。人物主体保持默认强度与负向词。
        # 角色画风策略（style_policy="character"）时追加写实/照片/油画负向词：
        # preserve 策略不能同时抑制输入的油画/照片/3D 画风。
        st = str(subject_repr.get("subject_type", "unknown")).lower()
        neg_prompt = config.NEGATIVE_PROMPT
        _, coverage_negative = composition_constraints(subject_repr)
        if coverage_negative:
            neg_prompt = ", ".join(coverage_negative) + ", " + neg_prompt
        use_card_style = bool(card) and style_policy_for_subject(card, subject_repr) == "character"
        if use_card_style:
            neg_prompt = f"{neg_prompt}, {config.ANIME_NEGATIVE_PROMPT}"
        settings = generation_settings(st)
        lora_scale = settings["lora_scale"]
        if st in ("animal", "object", "unknown"):
            neg_prompt = f"{neg_prompt}, {config.NON_HUMAN_NEGATIVE_PROMPT}"
        if lora_path is not None and hasattr(pipe, "set_adapters"):
            pipe.set_adapters("default", adapter_weights=lora_scale)
            if lora_scale != 1.0 and st != "human":
                print(f"    主体 {st}：LoRA 融合比例 {lora_scale}（抑制拟人先验），负向词附加拟人抑制")

        # ---- 结构条件：Canny 边缘（规格 §10.2/§10.3） ----
        canny_img = image_utils.extract_canny(img, config.CANNY_LOW, config.CANNY_HIGH)
        # 背景边缘抑制（规格 §9，默认关闭）：SUPPRESS_BACKGROUND_EDGES=True 时用主体
        # mask 把主体外的 Canny 置黑（背景自由重绘、主体结构更突出）；默认 False
        # 保留全图边缘，避免背景（如蒙娜丽莎的风景）被抹掉后失控变成纯色。
        mask = masks[p]
        if mask is not None and config.USE_SAM and config.SUPPRESS_BACKGROUND_EDGES:
            mask_resized = mask.convert("L").resize(canny_img.size, Image.NEAREST)
            canny_arr = np.array(canny_img.convert("L"))
            canny_arr[np.array(mask_resized) == 0] = 0
            canny_img = Image.fromarray(canny_arr).convert("RGB")
            print("    已用主体 mask 抑制背景边缘（主体区域外 Canny 置黑，规格 §9）")
        else:
            print("    背景边缘抑制未启用（SUPPRESS_BACKGROUND_EDGES=False 或 SAM 关闭），保留全图结构")

        # ---- 生成调用（混合 img2img+ControlNet / 纯 ControlNet / img2img 三条路径） ----
        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            generator=generator,
        )
        if ip_adapter_loaded:
            gen_kwargs["ip_adapter_image"] = img
        # 输出尺寸 = 输入等比适配后的实际尺寸（保持原图比例，不再强制正方形）
        gen_kwargs["width"] = img.size[0]
        gen_kwargs["height"] = img.size[1]
        mode = getattr(pipe, "_generation_mode", "controlnet")
        if use_controlnet:
            gen_kwargs["control_guidance_end"] = settings["control_end"]
        if use_controlnet and mode == "hybrid":
            # 混合路径（推荐）：init 图像锚定姿势/背景，canny 低 scale 保清晰，
            # LoRA+prompt 提供身份（详见 config.py 推理参数注释与消融实验 expD~P）。
            # 各主体独立读取参数；人物偏向角色还原，仍须检查手部与姿态。
            strength = settings["strength"]
            cn_scale = settings["control_scale"]
            gen_kwargs["image"] = img
            gen_kwargs["control_image"] = canny_img
            gen_kwargs["strength"] = strength
            gen_kwargs["controlnet_conditioning_scale"] = cn_scale
            print(f"    混合路径[{st}]: image=原图, strength={strength}, "
                  f"control_image=canny, controlnet_conditioning_scale={cn_scale}")
        elif use_controlnet:
            gen_kwargs["image"] = canny_img
            gen_kwargs["controlnet_conditioning_scale"] = settings["control_scale"]
            print(f"    ControlNet 路径: image=canny, "
                  f"controlnet_conditioning_scale={settings['control_scale']}")
        else:
            gen_kwargs["image"] = img
            strength = settings["strength"]
            gen_kwargs["strength"] = strength
            print(f"    img2img 路径: image=原图, strength={strength}")

        result = pipe(**gen_kwargs).images[0]
        background_audit = {"enabled": False}
        if protect_background:
            raw_path = os.path.join(config.RESULT_DIR, f"{stem}_raw_result.png")
            image_utils.save_image(result, raw_path)
            result, background_audit = preserve_background(img, result, masks[p])
            background_audit.update(enabled=True, raw_result=raw_path,
                                    mask_source="sam_automatic_unverified")

        # ---- 落盘：结果图 + 推理记录 txt + 结构化 JSON（规格 §15/§32） ----
        result_path = os.path.join(config.RESULT_DIR, f"{stem}_result.png")
        image_utils.save_image(result, result_path)

        txt_lines = [
            "========== 角色语义迁移推理记录（test.py） ==========",
            f"输入图片                : {p}",
            f"角色 folder             : {folder}",
            f"触发词 trigger          : {trigger}",
            f"LoRA 权重               : {lora_path or '无（纯底模，规格 §27 降级）'}",
            f"管线类型                : {pipe.__class__.__name__}",
            f"use_controlnet          : {use_controlnet}",
            f"IP-Adapter              : {'已加载' if ip_adapter_loaded else '未启用（规格 §3.3）'}",
            "",
            "--- 主体分析 subject_repr（JSON） ---",
            json.dumps(subject_repr, ensure_ascii=False, indent=2),
            "",
            "--- 配饰迁移规划 adaptation plan（JSON） ---",
            json.dumps(plan, ensure_ascii=False, indent=2),
            "",
            "--- 最终 Prompt（与生成调用完全一致） ---",
            prompt,
            "",
            "--- Negative Prompt ---",
            neg_prompt,
            "",
            "--- 超参摘要 ---",
            f"num_inference_steps           = {config.NUM_INFERENCE_STEPS}",
            f"guidance_scale                = {config.GUIDANCE_SCALE}",
            f"controlnet_conditioning_scale = {gen_kwargs.get('controlnet_conditioning_scale')}"
            "（仅 ControlNet 路径生效）",
            f"img2img strength              = {gen_kwargs.get('strength')}（仅 img2img 路径生效）",
            f"inference resolution          = {config.INFERENCE_RESOLUTION}",
            f"seed                          = {actual_seed}（请求种子={seed}）",
            f"lora scale                    = {lora_scale}（subject_type={st}）",
            f"canny 阈值                    = ({config.CANNY_LOW}, {config.CANNY_HIGH})",
            f"SAM 背景边缘抑制              = "
            f"{'是' if (config.USE_SAM and config.SUPPRESS_BACKGROUND_EDGES and mask is not None) else '否（默认关闭，保留全图结构）'}",
            "",
            "说明: subject_repr / adaptation plan 由 VLM 或规则表自动生成，仅供推理参考，"
            "不代表客观事实。",
        ]
        prompt_txt_path = os.path.join(config.RESULT_DIR, f"{stem}_prompt.txt")
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        print(f"    已保存: {result_path}")
        print(f"    已保存: {prompt_txt_path}")
        _write_json(os.path.join(config.RESULT_DIR, f"{stem}_subject.json"), subject_repr)
        _write_json(os.path.join(config.RESULT_DIR, f"{stem}_adaptation.json"), plan)
        with open(p, "rb") as input_file:
            input_hash = hashlib.sha256(input_file.read()).hexdigest()
        _write_json(os.path.join(config.RESULT_DIR, f"{stem}_generation.json"), {
            "seed": actual_seed, "input_sha256": input_hash,
            "lora_path": lora_path, "lora_scale": lora_scale,
            "lora_model_source": config.LORA_MODEL_SOURCE,
            "lora_selection_mode": config.LORA_SELECTION_MODE,
            "pipeline": pipe.__class__.__name__, "width": img.width, "height": img.height,
            "strength": gen_kwargs.get("strength"),
            "controlnet_conditioning_scale": gen_kwargs.get("controlnet_conditioning_scale"),
            "control_guidance_end": gen_kwargs.get("control_guidance_end"),
            "effective_style_policy": style_policy_for_subject(card, subject_repr),
            "steps": config.NUM_INFERENCE_STEPS, "guidance_scale": config.GUIDANCE_SCALE,
            "prompt": prompt, "negative_prompt": neg_prompt,
            "prompt_audit": getattr(planner, "last_prompt_audit", {}),
            "effective_character_memory": memory,
            "background_protection": background_audit,
        })

    # ================= 7. 可选结果评价（规格 §21） =================
    if config.ENABLE_RESULT_EVALUATION:
        print("[test] ENABLE_RESULT_EVALUATION=True：重新加载 VLM 对结果图进行评价")
        # 显存分时（规格 §3 8GB 关键）：扩散管线已不再需要，先释放再加载 VLM，
        # 避免两套模型同时驻留显存（实测不释放时峰值约 14GB，远超 8GB 预算）
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_client = VLMClient(
            config.VLM_PATH,
            dtype=config.VLM_DTYPE,
            load_in_4bit=config.VLM_LOAD_IN_4BIT,
            offload_cpu=config.VLM_OFFLOAD_CPU,
        )
        if getattr(eval_client, "available", False):
            for p in image_paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                result_path = os.path.join(config.RESULT_DIR, f"{stem}_result.png")
                ev = eval_client.analyze_image(image_utils.load_image(result_path), EVAL_PROMPT)
                eval_path = os.path.join(config.RESULT_DIR, f"{stem}_evaluation.json")
                _write_json(eval_path, ev)
                print(f"    {stem}: {json.dumps(ev, ensure_ascii=False)}")
            eval_client.unload()
            # 重要说明：这些分数是 VLM 模型的主观评价（规格 §21），不是严格科学指标，
            # 不能作为论文/报告中的客观 Ground Truth
            print("[test] 注意: 上述分数为 VLM 模型主观评价（规格 §21），非客观指标，"
                  "不可当作 Ground Truth 使用")
        else:
            print("[test] 降级: VLM 不可用，跳过结果评价（规格 §27）")
    else:
        print("[test] ENABLE_RESULT_EVALUATION=False：跳过结果评价（规格 §21 可选）")

    # ================= 8. 结束打印：产物清单 + 总耗时 =================
    print("=" * 60)
    print("[test] 推理完成，产物清单:")
    for p in image_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        print(f"  {os.path.basename(p)} ->")
        print(f"      {stem}_result.png       结果图")
        print(f"      {stem}_prompt.txt       推理记录（prompt/negative/超参/JSON 摘要）")
        print(f"      {stem}_subject.json     主体分析结构化结果")
        print(f"      {stem}_adaptation.json  配饰迁移规划结构化结果")
        if config.ENABLE_RESULT_EVALUATION:
            print(f"      {stem}_evaluation.json 结果评价（VLM 主观分数，规格 §21）")
    print(f"[test] 输出目录: {config.RESULT_DIR}")
    print(f"[test] 总耗时: {time.time() - t_start:.1f} 秒")
    if torch.cuda.is_available():
        print(f"[test] 本进程显存峰值: "
              f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB "
              f"（torch.cuda.max_memory_allocated）")
    print("=" * 60)


if __name__ == "__main__":
    """PyCharm 直接运行入口（规格 §24）。OOM 时打印降级建议清单后退出（规格 §25）。"""
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        print("\n[test] CUDA Out Of Memory.")
        print("Recommended actions:")
        print("  1. 降低 config.INFERENCE_RESOLUTION（1024 -> 768 -> 640 -> 512）")
        print("  2. 关闭可选模块: ENABLE_RESULT_EVALUATION / USE_VLM / USE_SAM（config.py 开关）")
        print("  3. 改用 img2img 路径: USE_CONTROLNET=False（无 ControlNet 时显存占用更低）")
        print("  4. 启用 CPU offload: config.USE_CPU_OFFLOAD=True（慢但稳定，规格 §12.1 第六步）")
        sys.exit(1)
