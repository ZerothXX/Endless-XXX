# -*- coding: utf-8 -*-
"""
image_utils.py —— 图像工具模块

提供图片读取（含 EXIF 方向修正）、等比缩放+中心裁剪、保存、
Canny 边缘提取、张量转 PIL、以及项目目录初始化等基础工具。

只依赖 PIL / opencv-python / numpy，不 import torch，
可在任何环境（包括纯数据处理脚本）中安全使用。
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageOps

import config


def load_image(path: str) -> Image.Image:
    """读取图片，返回 RGB 模式的 PIL.Image。

    - 根据 EXIF 信息自动修正旋转方向（手机/相机照片常见问题）
    - 统一转为 RGB 三通道，避免灰度/带透明通道图片导致后续流程异常
    - 文件不存在或损坏时抛出明确异常（fail fast，指明具体路径）
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[image_utils] 图片文件不存在: {path}")

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # 按 EXIF Orientation 修正旋转
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise IOError(f"[image_utils] 图片读取失败: {path}，原因: {e}") from e


def resize_and_crop(img: Image.Image, size: int) -> Image.Image:
    """等比缩放后中心裁剪，输出 size×size 的正方形图。

    缩放按短边对齐（cover 方式），长边可能超出，因此随后裁剪掉两侧多余部分。
    注意：此函数会丢弃原图边缘信息（全身竖图会裁掉头部/脚部），
    推理链路已改用 resize_fit（保持原图比例），本函数仅保留给实验/旧路径。
    """
    if size <= 0:
        raise ValueError(f"[image_utils] size 必须为正整数，收到: {size}")

    w, h = img.size
    scale = max(size / w, size / h)          # 按短边放大，保证裁剪后不拉伸
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def resize_fit(img: Image.Image, max_side: int, max_long: int = 960,
               min_dim: int = 256) -> Image.Image:
    """等比缩放（**不裁剪、不拉伸**），短边对齐 max_side、长边不超过 max_long。

    三次演进（v5 教训）：
      v5 早期：长边 = max_side（640）——全身/宽幅构图下面孔只有 ~40-80px，
               五官、服饰、姿态被小脸+极端长宽比拖垮到扭曲（v5 人物/猫
               五颜六色且五官变形即此因）；
      v5.1（本版）：**短边 = max_side**（如 640）→ 全身照 640×960、
               宽幅猫 960×528——主体占据的像素量翻倍以上，五官不再扭曲，
               同时原图构图/比例 100% 保留（>1.5:1 的极端图才触顶 max_long
               轻微整体缩放）。面积 ≤ 640×960 ≈ 0.61MP，与已验证可跑的
               768²（0.59MP）同级，8GB 显存安全。
    尺寸取下取整到 8 的倍数（潜空间标准）；小图自动放大。
    """
    if max_side <= 0 or max_long <= 0:
        raise ValueError(f"[image_utils] max_side/max_long 必须为正整数，收到: "
                         f"{max_side}/{max_long}")

    w, h = img.size
    scale = max_side / min(w, h)                 # 短边先对齐 max_side
    if max(w * scale, h * scale) > max_long:     # 长边超限则整体缩小
        scale = max_long / max(w, h)
    new_w = max(min_dim, int(w * scale / 8) * 8)
    new_h = max(min_dim, int(h * scale / 8) * 8)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def save_image(img: Image.Image, path: str) -> None:
    """保存图片为 PNG。目录不存在时自动创建（幂等）。

    保存失败时抛出异常并指明路径，不静默吞掉。
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    try:
        img.save(path, format="PNG")
    except Exception as e:
        raise IOError(f"[image_utils] 图片保存失败: {path}，原因: {e}") from e


def extract_canny(img: Image.Image, low: int = 50, high: int = 150) -> Image.Image:
    """提取 Canny 边缘图，返回 3 通道 RGB 的 PIL.Image（供 ControlNet 使用）。

    cv2.Canny 输入输出均为灰度图，这里转成三通道 RGB 以符合
    ControlNet 图像输入约定（黑底白边）。
    """
    arr = np.array(img.convert("L"))
    edges = cv2.Canny(arr, low, high)
    return Image.fromarray(edges).convert("RGB")


def to_pil(tensor) -> Image.Image:
    """把张量/数组转为 PIL.Image（范围 [0,1] 浮点或 [0,255] 整数均可）。

    兼容 numpy 数组（H,W,C 或 C,H,W）与 torch 张量（通过 duck-typing 处理，
    本模块本身不 import torch）。用于 VAE decode 输出等场景的落盘前转换。
    """
    arr = tensor
    # 兼容 torch.Tensor：不 import torch，仅按接口特征判断
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr)

    # 去掉 batch 维度
    if arr.ndim == 4:
        arr = arr[0]
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[2]:
        arr = arr.transpose(1, 2, 0)

    # 数值范围归一化到 [0, 255]
    if arr.dtype in (np.float16, np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # 单通道灰度 -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def create_dirs() -> None:
    """创建项目运行所需的目录结构（幂等，可重复调用）。

    包括 input/ 与 output/ 下五个子目录（规格 §2），
    以及训练/推理必需的目录。
    """
    dirs = [
        config.INPUT_DIR,
        config.TRAIN_MODELS_DIR,
        config.CURVES_DIR,
        config.SEMANTIC_DIR,
        config.RESULT_DIR,
        config.LOGS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
