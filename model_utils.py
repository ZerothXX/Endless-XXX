# -*- coding: utf-8 -*-
"""
model_utils.py —— 模型加载 / LoRA 注入 / 权重保存（训练侧）

职责（规格 §3 / §11 / §22）：
- load_train_pipeline : 加载 SDXL 底模 + 独立 VAE（仅训练所需组件，不加载 VLM/SAM/ControlNet/IP-Adapter，规格 §11.3）
- prepare_lora         : UNet 冻结全部参数后挂 LoRA（peft），两个 text_encoder 冻结不训（省显存，规格 §12）
- save_lora            : 以 diffusers 0.31 格式保存 LoRA 权重（目录 + 单文件 safetensors，可被 pipe.load_lora_weights 重新加载）
- verify_model_paths   : 启动前检查模型路径存在性（train.py 启动时调用并打印）

推理侧函数（load_inference_pipeline / load_ip_adapter，S6 实现，仅供 test.py 调用，训练侧禁止）。
"""

import os

import torch

import config


# ---------------------------------------------------------------------------
# 推理侧管线加载（S6 实现；训练侧禁止调用）
# ---------------------------------------------------------------------------
def load_inference_pipeline(lora_path: str = None, device: str = "cuda",
                            dtype=torch.float16):
    """加载推理侧扩散管线（规格 §3 / §15 / §27），返回 (pipe, use_controlnet: bool)。

    装配顺序（与规格 §15 一致）：
        1. 底模 + VAE：与 load_train_pipeline 完全相同的加载逻辑
           （from_single_file + fp16 修复版 VAE + vae_slicing/tiling）
        2. LoRA（可选项）：lora_path 指向单文件 safetensors（save_lora 产物，
           键带 unet. 前缀的 diffusers 格式）。文件存在 -> load_lora_weights；
           不存在 -> 打印警告继续（纯底模推理，规格 §27）。
           LoRA 融合比例：diffusers 0.31.0 无 pipe.set_lora_scale（该 API 0.32
           才引入，源码核实），等价能力用 set_adapters(adapter_weights=...) 实现
           （0.31 源码核实 set_adapters 支持 adapter_weights，加载端默认 adapter
           名为 "default"，此处显式指定保证名称一致）。
        3. ControlNet（可选项）：config.USE_CONTROLNET 且 CONTROLNET_PATH 存在 ->
           ControlNetModel.from_pretrained(..., torch_dtype=dtype, variant="fp16")
           （0.31 的 from_pretrained 为 **kwargs 签名，variant 在加载链中生效，
           源码核实；本地模型文件为 diffusion_pytorch_model.fp16.safetensors）-> 用
           StableDiffusionXLControlNetPipeline(**pipe.components, controlnet=controlnet)
           装配（0.31 的 components 属性返回全部 9 个模块键，与两类管线 init 签名
           完全对齐，源码核实）。加载失败 / 目录缺失 -> 打印降级（规格 §27）返回
           img2img 管线。
        4. 显存：config.USE_CPU_OFFLOAD -> enable_model_cpu_offload()，否则 .to(device)

    返回：
        (pipe, use_controlnet)
        - use_controlnet=True  : pipe 为 StableDiffusionXLControlNetPipeline，
                                 生成时传 image=canny + controlnet_conditioning_scale
        - use_controlnet=False : pipe 为 StableDiffusionXLImg2ImgPipeline（img2img
                                 路径，生成时传 image=原图 + strength，规格 §27）
    """
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    # ---- 1. 底模（Illustrious XL 单文件，与训练侧同一加载逻辑） ----
    if not os.path.isfile(config.BASE_MODEL_PATH):
        raise FileNotFoundError(
            f"[model] 基础模型文件不存在: {config.BASE_MODEL_PATH}\n"
            "请检查 models/ 目录与 config.BASE_MODEL_PATH 配置。"
        )
    print(f"[model] 加载底模: {config.BASE_MODEL_PATH}（dtype={dtype}）")
    pipe = StableDiffusionXLPipeline.from_single_file(
        config.BASE_MODEL_PATH,
        torch_dtype=dtype,
    )

    # ---- 2. VAE（fp16 修复版单文件，与训练侧同一加载逻辑） ----
    if not os.path.isfile(config.BASE_VAE_PATH):
        raise FileNotFoundError(
            f"[model] VAE 文件不存在: {config.BASE_VAE_PATH}\n"
            "请检查 models/ 目录与 config.BASE_VAE_PATH 配置。"
        )
    print(f"[model] 加载 VAE: {config.BASE_VAE_PATH}（dtype={dtype}）")
    pipe.vae = AutoencoderKL.from_single_file(config.BASE_VAE_PATH, torch_dtype=dtype)
    # G24 参数版复用本加载逻辑，不包含独立的 TE-LoRA 训练。

    # ---- 3. LoRA（可选项；save_lora 单文件产物，diffusers 格式） ----
    if lora_path is not None:
        if os.path.isfile(lora_path):
            # Diffusers 0.31 offline mode requires weight_name even for a local file.
            pipe.load_lora_weights(os.path.dirname(os.path.abspath(lora_path)),
                                   weight_name=os.path.basename(lora_path),
                                   adapter_name="default", local_files_only=True)
            print(f"[model] 已加载 LoRA: {lora_path}")
            # 0.31 无 set_lora_scale（0.32 才引入，源码核实），等价能力用 set_adapters
            if config.LORA_SCALE != 1.0:
                pipe.set_adapters("default", adapter_weights=config.LORA_SCALE)
                print(f"[model] 应用 LoRA 融合比例: config.LORA_SCALE={config.LORA_SCALE}")
        else:
            raise FileNotFoundError(f"已选 LoRA 在加载前消失，拒绝静默使用纯底模: {lora_path}")
    else:
        if not config.ALLOW_BASE_MODEL_WITHOUT_LORA:
            raise ValueError("未提供 LoRA，且 ALLOW_BASE_MODEL_WITHOUT_LORA=False")
        print("[model] 未指定 LoRA 权重（lora_path=None），使用纯底模推理")

    # ---- 4. ControlNet（可选项；缺失 / 加载失败 -> img2img 降级，规格 §27） ----
    use_controlnet = False
    if config.USE_CONTROLNET:
        if os.path.isdir(config.CONTROLNET_PATH):
            try:
                controlnet = ControlNetModel.from_pretrained(
                    config.CONTROLNET_PATH, torch_dtype=dtype, variant="fp16")
                if getattr(config, "USE_HYBRID_IMG2IMG_CONTROLNET", False):
                    # 混合路径：img2img + ControlNet（StableDiffusionXLControlNetImg2ImgPipeline）。
                    # 姿势由 init 图像锚定，canny 用低 scale 保持清晰（详见 config 注释与
                    # 消融实验 expD~P：canney>=0.7 纯 ControlNet 会产出"朦胧空气感"）。
                    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
                        **pipe.components, controlnet=controlnet)
                    pipe._generation_mode = "hybrid"
                    print("[model] 管线已装配为 StableDiffusionXLControlNetImg2ImgPipeline"
                          "（混合路径 img2img+ControlNet）")
                else:
                    pipe = StableDiffusionXLControlNetPipeline(
                        **pipe.components, controlnet=controlnet)
                    pipe._generation_mode = "controlnet"
                    print("[model] 管线已装配为 StableDiffusionXLControlNetPipeline")
                use_controlnet = True
                print(f"[model] 已加载 ControlNet(canny): {config.CONTROLNET_PATH}（variant=fp16）")
            except Exception as e:
                print(f"[model] 警告: ControlNet 加载失败，降级为 img2img（规格 §27）。"
                      f"原因: {type(e).__name__}: {e}")
        else:
            print(f"[model] 警告: ControlNet 目录不存在，降级为 img2img（规格 §27）: "
                  f"{config.CONTROLNET_PATH}")
    else:
        print("[model] USE_CONTROLNET=False：按 img2img 路径装配（规格 §27 最低可运行形态）")

    if not use_controlnet:
        pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        pipe._generation_mode = "img2img"
        print("[model] 管线已装配为 StableDiffusionXLImg2ImgPipeline（img2img 路径）")

    # ---- 5. 显存优化（8GB 显卡关键，与训练侧一致） ----
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    if config.USE_CPU_OFFLOAD:
        print("[model] USE_CPU_OFFLOAD=True：启用模型 CPU offload（推理速度会明显下降）")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # ---- 6. 装配摘要 ----
    print(f"[model] 推理管线装配完成: pipe={pipe.__class__.__name__} "
          f"use_controlnet={use_controlnet} "
          f"device={'cpu-offload' if config.USE_CPU_OFFLOAD else device} dtype={dtype}")
    return pipe, use_controlnet


def load_ip_adapter(pipe) -> bool:
    """加载 IP-Adapter（可选项，规格 §3.3），返回是否加载成功。

    - config.IP_ADAPTER_PATH 目录不存在 -> 打印降级提示并返回 False（自动降级为
      LoRA + ControlNet/img2img，不阻断程序，规格 §3.3）
    - 存在则尝试加载并返回 True；加载失败打印原因并返回 False
    - 说明：diffusers 0.31 的 load_ip_adapter 要求显式 subfolder / weight_name
      （源码核实，两者为无默认值的必填参数），这里按常见本地目录布局
      image_encoder/ + ip_adapter/model.safetensors 尝试；若实际目录结构不同
      会加载失败并打印提示（本机 models/ip-adapter 不存在，此路径不实跑）
    """
    if not os.path.isdir(config.IP_ADAPTER_PATH):
        print(f"[model] 提示: IP-Adapter 模型目录不存在，自动降级为 LoRA + ControlNet/img2img"
              f"（规格 §3.3）: {config.IP_ADAPTER_PATH}")
        return False
    try:
        pipe.load_ip_adapter(
            config.IP_ADAPTER_PATH,
            subfolder="ip_adapter",
            weight_name="model.safetensors",
        )
        print(f"[model] 已加载 IP-Adapter: {config.IP_ADAPTER_PATH}")
        return True
    except Exception as e:
        print(f"[model] 警告: IP-Adapter 加载失败，自动降级为 LoRA + ControlNet/img2img"
              f"（规格 §3.3）。原因: {type(e).__name__}: {e}")
        print("[model] 提示: 请核对 config.IP_ADAPTER_PATH 目录布局是否为 "
              "image_encoder/ + ip_adapter/model.safetensors")
        return False


# ---------------------------------------------------------------------------
# 路径检查
# ---------------------------------------------------------------------------
def verify_model_paths() -> dict:
    """检查 config 中各模型路径的存在性，返回状态 dict（train.py 启动时打印）。

    - BASE_MODEL_PATH / BASE_VAE_PATH 必须是文件，缺失时训练无法进行（fail fast）
    - CONTROLNET_PATH 是目录，仅推理阶段使用，此处只记录存在性不阻断训练
    """
    status = {
        "BASE_MODEL_PATH": {
            "exists": os.path.isfile(config.BASE_MODEL_PATH),
            "path": config.BASE_MODEL_PATH,
        },
        "BASE_VAE_PATH": {
            "exists": os.path.isfile(config.BASE_VAE_PATH),
            "path": config.BASE_VAE_PATH,
        },
        "CONTROLNET_PATH": {
            "exists": os.path.isdir(config.CONTROLNET_PATH),
            "path": config.CONTROLNET_PATH,
            "note": "推理阶段使用（规格 §3.2 / §10），训练不加载，此处仅记录",
        },
    }
    return status


# ---------------------------------------------------------------------------
# 训练管线加载
# ---------------------------------------------------------------------------
def load_train_pipeline(device: str = "cuda", dtype=torch.float16):
    """加载 SDXL 训练管线（Illustrious XL 底模单文件 + 独立 VAE 单文件）。

    说明：
    - diffusers 0.31.0 的 from_single_file 无 use_safetensors 参数，
      load_single_file_checkpoint 按文件扩展名自动选择 safetensors/pickle 加载，
      因此这里不传 use_safetensors（以源码为准）。
    - 训练阶段不加载 ControlNet / SAM / IP-Adapter / VLM（规格 §11.3）。
    - 打印各组件 dtype 与参数数量，便于核对加载结果。
    - 加载失败时抛出带明确路径的异常（fail fast）。
    """
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    # ---- 底模 ----
    if not os.path.isfile(config.BASE_MODEL_PATH):
        raise FileNotFoundError(
            f"[model] 基础模型文件不存在: {config.BASE_MODEL_PATH}\n"
            "请检查 models/ 目录与 config.BASE_MODEL_PATH 配置。"
        )
    print(f"[model] 加载底模: {config.BASE_MODEL_PATH}（dtype={dtype}）")
    pipe = StableDiffusionXLPipeline.from_single_file(
        config.BASE_MODEL_PATH,
        torch_dtype=dtype,
    )

    # ---- 替换 VAE（fp16 修复版，单文件） ----
    if not os.path.isfile(config.BASE_VAE_PATH):
        raise FileNotFoundError(
            f"[model] VAE 文件不存在: {config.BASE_VAE_PATH}\n"
            "请检查 models/ 目录与 config.BASE_VAE_PATH 配置。"
        )
    print(f"[model] 加载 VAE: {config.BASE_VAE_PATH}（dtype={dtype}）")
    pipe.vae = AutoencoderKL.from_single_file(config.BASE_VAE_PATH, torch_dtype=dtype)

    # ---- 显存优化（8GB 显卡友好，规格 §12） ----
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # ---- 设备放置 ----
    if config.USE_CPU_OFFLOAD:
        # CPU offload 是 OOM 降级的最后手段（规格 §12.1 第六步），会显著变慢
        print("[model] USE_CPU_OFFLOAD=True：启用模型 CPU offload（训练速度会明显下降）")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # ---- 打印各组件 dtype / 参数数量 ----
    print("[model] 组件信息:")
    for name in ("unet", "vae", "text_encoder", "text_encoder_2"):
        comp = getattr(pipe, name)
        n_params = sum(p.numel() for p in comp.parameters())
        print(f"  {name:14s} dtype={str(comp.dtype):10s} 参数数量={n_params:,}（{n_params / 1e6:.1f}M）")

    return pipe


# ---------------------------------------------------------------------------
# LoRA 注入
# ---------------------------------------------------------------------------
def _resolve_lora_target_modules(unet) -> list:
    """确定 LoRA 注入的目标模块名列表。

    优先使用 config.LORA_TARGET_MODULES（若用户显式配置）；
    未配置时自动扫描 UNet 线性层，按常见注意力投影层命名规则（to_q/to_k/to_v/to_out.0
    或 q_proj/k_proj/v_proj/out_proj）匹配，并在运行时打印最终实际使用的列表。
    peft 的 target_modules 列表按"模块名以列表项结尾"匹配（peft 0.17.1 源码确认），
    因此这里收集的是与模块名结尾匹配的模式串。
    """
    explicit = getattr(config, "LORA_TARGET_MODULES", None)
    if explicit:
        target = list(explicit)
        print(f"[lora] 使用 config.LORA_TARGET_MODULES: {target}")
        return target

    # 兜底：扫描 UNet 全部线性层，按常见注意力投影命名规则匹配
    patterns = ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "out_proj"]
    found = set()
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            for pattern in patterns:
                if name.endswith(pattern):
                    found.add(pattern)
    if not found:
        # 扫描未命中时退回 SDXL 标准注意力投影集合，保证流程可运行
        fallback = ["to_k", "to_q", "to_v", "to_out.0"]
        print(f"[lora] 警告: 自动扫描未命中任何注意力投影层，回退到默认列表: {fallback}")
        return fallback

    target = sorted(found)
    print(f"[lora] config 未配置 LORA_TARGET_MODULES，自动扫描 UNet 线性层命中: {target}")
    return target


def prepare_lora(pipe, rank: int, alpha: int):
    """在 UNet 上注入 LoRA，返回 peft 模型（PeftModel）。

    - UNet 全部参数冻结（requires_grad_(False)）后挂 LoRA，只有 LoRA 参数可训练
    - 两个 text_encoder 与 VAE 冻结不训：
        角色 LoRA 只学习视觉身份（规格 §0.1），text encoder 全量训练会显著增加
        显存与过拟合风险，且 20 张小样本（规格 §13）不需要；冻结它们可把
        训练显存集中在 UNet，适配 RTX 4060 8GB。
    - G24 参数版使用相同的 UNet LoRA 注入逻辑。
    - 打印可训练参数占比，便于核对注入结果。
    """
    from peft import LoraConfig, get_peft_model

    unet = pipe.unet
    # 冻结 UNet / VAE / 两个 text encoder（只训练 LoRA 参数）
    unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    target_modules = _resolve_lora_target_modules(unet)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,     # 小样本训练不需要 dropout（规格 §13）
        bias="none",
    )
    unet_lora = get_peft_model(unet, lora_config)

    total_params = sum(p.numel() for p in unet_lora.parameters())
    trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
    ratio = trainable_params / total_params * 100 if total_params > 0 else 0.0
    print(f"[lora] LoRA 注入完成: rank={rank} alpha={alpha} target_modules={target_modules}")
    print(f"[lora] 可训练参数 {trainable_params:,}（{trainable_params / 1e6:.2f}M），"
          f"占 UNet 总参数 {total_params:,}（{total_params / 1e6:.2f}M）的 {ratio:.3f}%")
    return unet_lora


# ---------------------------------------------------------------------------
# LoRA 权重保存
# ---------------------------------------------------------------------------
def save_lora(pipe, unet_lora, save_path: str) -> None:
    """把训练好的 UNet LoRA 保存为 diffusers 格式（目录 + 单文件 safetensors）。

    0.31.0 的 StableDiffusionXLLoraLoaderMixin.save_lora_weights 签名（源码核实）：
        save_lora_weights(save_directory, unet_lora_layers=None, text_encoder_lora_layers=None,
                          text_encoder_2_lora_layers=None, is_main_process=True, weight_name=None,
                          save_function=None, safe_serialization=True)
    - save_directory 必须是目录（write_lora_layers 中 os.path.isfile(save_directory) 会报错）
    - weight_name 指定目录内文件名，最终产出 save_directory/weight_name 单文件
    - G24 参数版也使用本保存逻辑，不保存 TE-LoRA。

    键格式说明（关键，diffusers 0.31 + peft 0.17.1 源码核实）：
    - PeftModel.state_dict() 的键带 "base_model.model." 前缀与 adapter 名（如 lora_A.default）
    - 而 load_lora_weights 加载端（unet.py _process_lora -> set_peft_model_state_dict）
      期望键为无前缀、无 adapter 名的形式（如 down_blocks.0.attentions.0....to_q.lora_A.weight）
    - 因此这里用 peft.get_peft_model_state_dict 配合裸模型（get_base_model）的 state_dict
      生成正确键格式后再保存
    产物形态：save_path 指向的单文件 safetensors，可直接传给 pipe.load_lora_weights(save_path)。
    """
    from peft import get_peft_model_state_dict

    # 生成 diffusers 兼容键格式（去 base_model.model. 前缀、去 adapter 名）
    unet_lora_layers = get_peft_model_state_dict(
        unet_lora,
        state_dict=unet_lora.get_base_model().state_dict(),
    )
    # 张量转 CPU（safetensors 保存要求；LoRA 权重保持原 dtype，通常 fp16）
    unet_lora_layers = {k: v.detach().cpu() for k, v in unet_lora_layers.items()}

    out_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(out_dir, exist_ok=True)
    weight_name = os.path.basename(save_path)

    pipe.save_lora_weights(
        out_dir,
        unet_lora_layers=unet_lora_layers,
        weight_name=weight_name,
    )
    print(f"[lora] 权重已保存: {save_path}（共 {len(unet_lora_layers)} 个 LoRA 张量）")
