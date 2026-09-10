# -*- coding: utf-8 -*-
"""
vision/segmentation.py —— SAM2 主体分割（规格 §9，含降级）

难点说明：本地 models/sam2 是 **视频模型**（config.json: architectures=['Sam2VideoModel']，
model_type='sam2_video'），但本项目只需要单图主体分割。经核实 transformers 4.56.2
源码，本模块的实现依据如下：

1. Sam2VideoModel 的公共 forward 是"流式视频推理"接口，签名只接受
   inference_session / frame_idx / frame / reverse（modeling_sam2_video.py:1696），
   无法直接传单图 pixel_values + 提示点。
2. 单图分割唯一可用入口是 Sam2VideoModel._single_frame_forward
   （modeling_sam2_video.py:1854），签名与 Sam2Model.forward 完全一致
   （pixel_values / input_points / input_labels / input_boxes / multimask_output），
   返回 Sam2VideoImageSegmentationOutput（modeling_sam2_video.py:603）：
   - iou_scores       (B, point_batch, num_masks=3)   每个 mask 的 IoU 预测分数
   - pred_masks       (B, point_batch, H_low, W_low)  低分辨率最佳 mask
   - high_res_masks   (B, point_batch, 1024, 1024)    高分辨率最佳 mask（1024=image_size）
   - object_score_logits (B, point_batch, 1)          对象出现分数（>0 表示该点处有对象）
   该方法在模型内部被 _use_mask_as_output 正常调用（modeling_sam2_video.py:2078），
   是活跃代码路径而非废弃接口。
3. 提示点坐标链路（无需手工归一化）：
   用户给原图像素坐标 -> Sam2VideoProcessor 等比缩放至 1024 空间
   （processing_sam2_video.py:199-227）-> prompt encoder 内除以 1024 归一化到 [0,1]
   （modeling_sam2_video.py:1139-1154）。
4. 后处理：processor.post_process_masks(masks, original_sizes, ...) 把 1024 空间
   的 mask 去 pad 并 resize 回原图尺寸（image_processing_sam2_fast.py:647-698）。
   注意其内部按 batch 索引 masks[i] 后直接 F.interpolate，因此 masks 必须以
   list（每元素 4D (B, N, H, W)）形式传入；实测 4D 张量直接传会崩（3D 切片
   无法 bilinear 插值到 2 维 size）。
5. torch<2.3 兼容：transformers 4.56 的 fast image processor 在 resize 时调用
   torch.compiler.is_compiling()（image_processing_utils_fast.py:292），该 API
   torch 2.3 才引入；本环境 torch 2.2.1 会 AttributeError。SAM2 只有 fast
   processor（Sam2VideoProcessor.attributes 固定 image_processor_class=
   "Sam2ImageProcessorFast"，processing_sam2_video.py:56-58，无 slow 版本），
   因此 load_sam2 时做幂等 monkey patch（注入 is_compiling = lambda: False）。

自动主体提示（无人工点选）：3×3 网格（含中心点）共 9 个候选点。实测发现
transformers 4.56 的 Sam2VideoModel._single_frame_forward 在 num_objects>1（即
input_points 第二维 >1，本项目 9 个候选点即此形态）时存在内部形状 bug
（object_score_logits 返回 (B,1,N,1) 而 low_res_multimasks 为 (B,N,3,H,W)，
torch.where 广播在第 3 维冲突：'size of tensor a (9) must match the size of
tensor b (256)'），因此不能一次前向传 9 个对象。修复：视觉特征只编码一次
（get_image_embeddings，含 no_memory_embedding 注入，与 _single_frame_forward
内部一致），9 个候选点各自作为单对象（num_objects=1）逐点前向（每点仅
prompt decoder 开销，实测 9 点共约 0.3 秒），语义与"一次前向 9 点"等价。
过滤规则：
- object_score_logits > 0（模型认为该点处存在对象，否则 mask 被强制为全背景）
- mask 面积占比 ∈ [0.05, 0.9]（过滤全图 / 极小碎片，规格 §9）
取剩余候选中 iou_scores 最高者；全部被过滤则返回 None，由上层降级整图 mask（§27）。

显存预算：Sam2VideoModel 约 176M 参数，fp16 权重约 0.35GB；单帧 1024 推理峰值
激活约 1~2GB（RTX 4060 8GB 无压力）。
"""

import gc
import os
import sys

import numpy as np
from PIL import Image
import torch

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

# =========================
# 分割参数（规格 §9：过滤全图 / 极小碎片）
# =========================
MIN_MASK_AREA_RATIO = 0.05   # mask 面积占比下限（低于视为碎片）
MAX_MASK_AREA_RATIO = 0.9    # mask 面积占比上限（高于视为全图误分割）

# 自动主体提示：3×3 网格相对坐标（相对原图宽高的比例，含中心点 (0.5, 0.5)）
GRID_REL_POINTS = [
    (0.25, 0.25), (0.50, 0.25), (0.75, 0.25),
    (0.25, 0.50), (0.50, 0.50), (0.75, 0.50),
    (0.25, 0.75), (0.50, 0.75), (0.75, 0.75),
]


def _patch_torch_compiler_for_fast_processor() -> None:
    """幂等兼容补丁：torch<2.3 无 torch.compiler.is_compiling。

    transformers 4.56 的 fast image processor（SAM2 唯一 processor 类型）在 resize
    时调用 torch.compiler.is_compiling()（image_processing_utils_fast.py:292），该
    API 于 torch 2.3 引入；torch 2.2.1 下直接调用会 AttributeError。
    注入返回 False 的占位函数（默认语义：不在编译中），对运行无任何副作用。
    """
    import torch.compiler  # torch 2.2 起 torch.compiler 模块已存在
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]


def load_sam2(path=None):
    """加载本地 SAM2 视频模型（fp16 GPU）及其 processor。

    参数：
        path: 模型目录（默认取 config.SAM_PATH，即 models/sam2）

    返回：
        (model, processor) 元组；任何加载异常（路径不存在 / 权重损坏 /
        无 CUDA / ImportError 等）打印原因并返回 None，调用方据此降级
        （config.USE_SAM=False -> 整图 mask，规格 §9/§27）。
    """
    if path is None:
        path = config.SAM_PATH

    try:
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        # torch<2.3 兼容补丁（见文件头说明第 5 条），必须在 processor 预处理前生效
        _patch_torch_compiler_for_fast_processor()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用（SAM2 fp16 需 GPU 推理，本环境不允许 CPU 大模型推理）")

        processor = Sam2VideoProcessor.from_pretrained(path)
        model = Sam2VideoModel.from_pretrained(path, torch_dtype=torch.float16)
        model.to("cuda")
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        vram_gb = num_params * 2 / (1024 ** 3)  # fp16 = 2 字节/参数
        print(f"[SAM2] 已加载: {path}")
        print(f"[SAM2] 参数 {num_params / 1e6:.1f}M，fp16 权重显存约 {vram_gb:.2f}GB，单帧推理峰值约 1~2GB")
        return model, processor
    except Exception as e:
        print(f"[SAM2] 加载失败（返回 None），原因: {type(e).__name__}: {e}")
        print("[SAM2] 降级建议: 在 config.py 中设置 USE_SAM=False，"
              "主体分割退化为整图 mask（规格 §9/§27，最小模式不受影响）")
        return None


def segment_subject(image, sam):
    """SAM2 自动主体分割（无人工点选），返回单通道 255/0 的 PIL.Image 或 None。

    参数：
        image: PIL.Image（RGB，原图尺寸；坐标以原图为准，processor 负责归一化）
        sam  : load_sam2 返回的 (model, processor) 元组，或 None（直接返回 None）

    流程：
        3×3 网格 9 个候选提示点 -> 视觉特征只编码一次（get_image_embeddings）
        -> 每点作为单对象逐点前向（num_objects=1，规避 transformers 4.56 的
        num_objects>1 形状 bug，见文件头）-> 后处理到原图尺寸
        -> 过滤（对象分数 > 0 且面积占比 ∈ [0.05, 0.9]）-> 取 IoU 分数最高者
        -> 255/0 单通道 mask。

    返回：
        成功返回与 image 同尺寸的单通道 "L" 模式 PIL.Image（前景 255，背景 0）；
        任何失败（sam 为 None / 模型异常 / 全部候选被过滤）打印原因并返回 None，
        由 get_subject_mask_or_full 降级为整图 mask（规格 §27）。
    """
    if sam is None:
        print("[SAM2] sam 为 None，无法分割")
        return None
    model, processor = sam
    if not isinstance(image, Image.Image):
        raise TypeError(f"[SAM2] image 必须是 PIL.Image，收到: {type(image)}")

    try:
        w, h = image.size
        # 1. 候选提示点：3×3 网格（含中心点），原图像素坐标（x=列, y=行）
        #    格式 [image][object][point][x, y]，每个 object 一个前景点（label=1）
        points = [[[[rx * (w - 1), ry * (h - 1)]] for rx, ry in GRID_REL_POINTS]]
        labels = [[[1] for _ in GRID_REL_POINTS]]

        # 2. 预处理：缩放到 1024 空间 + 坐标归一化（processing_sam2_video.py:67-195）
        inputs = processor(images=image, input_points=points, input_labels=labels, return_tensors="pt")
        try:
            device = next(model.parameters()).device  # 正常取模型所在设备（cuda）
        except (StopIteration, AttributeError):
            # 防御：异常时回退到 CUDA（模型加载时就已 .to("cuda")）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)

        # 3. 视觉特征只编码一次（get_image_embeddings 内部已加 no_memory_embedding，
        #    与 _single_frame_forward 传 pixel_values 的路径一致）
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(pixel_values)

        # 4. 逐点前向：每个网格点作为单对象（num_objects=1）调用
        #    _single_frame_forward（modeling_sam2_video.py:1854），规避 transformers
        #    4.56 在 num_objects>1 时的 torch.where 广播 bug（见文件头说明）。
        #    multimask_output=True：每点预测 3 个 mask，模型内部自动取 IoU 最高者。
        #    inputs["input_points"] 形状 (1, 9, 1, 2)，按点切片复用坐标归一化结果。
        all_pts = inputs["input_points"].to(device)
        all_lbs = inputs["input_labels"].to(device)
        high_res_list = []
        iou_list = []
        obj_list = []
        with torch.no_grad():
            for i in range(all_pts.shape[1]):
                outputs = model._single_frame_forward(
                    pixel_values=None,
                    image_embeddings=image_embeddings,
                    input_points=all_pts[:, i:i + 1],
                    input_labels=all_lbs[:, i:i + 1],
                    multimask_output=True,
                )
                high_res_list.append(outputs.high_res_masks.float())
                iou_list.append(outputs.iou_scores[0])            # (1, 3)
                obj_list.append(outputs.object_score_logits[0, :, 0])  # (1,)

        # 5. 后处理：逐点 high_res_masks (1, 1, 1024, 1024) -> 原图尺寸 (1, 1, H, W)
        #    必须包一层 list（fast image processor 按 batch 索引后直接 interpolate）
        #    .float() 避免 fp16 interpolate 的精度/兼容问题
        # 注意：original_sizes 必须转为普通 list（元素为 [H, W]），
        # 传 tensor 会原样进入 F.interpolate 的 size 参数导致类型错误
        orig_sizes = inputs["original_sizes"].tolist()
        post = processor.post_process_masks(
            high_res_list, [orig_sizes[0]] * len(high_res_list),
            mask_threshold=0.0, binarize=False,
        )
        masks = torch.cat([p[0] for p in post], dim=0)   # (N, H, W) float logits
        iou_scores = torch.cat(iou_list, dim=0)          # (N, 3) 每点 3 个 mask 的 IoU
        obj_scores = torch.cat(obj_list, dim=0)          # (N,) 对象出现分数
        best_iou = iou_scores.max(dim=-1).values         # (N,) 每点最佳 mask 的 IoU

        # 6. 过滤 + 选择：对象分数 > 0（模型认为该处有对象）且面积占比在 [0.05, 0.9]
        candidates = []
        for i in range(masks.shape[0]):
            if obj_scores[i].item() <= 0:
                continue
            frac = (masks[i] > 0).float().mean().item()
            if MIN_MASK_AREA_RATIO <= frac <= MAX_MASK_AREA_RATIO:
                candidates.append((best_iou[i].item(), i, frac))

        if not candidates:
            print("[SAM2] 全部候选提示点未通过过滤（无对象或面积占比异常），"
                  "返回 None，上层将降级为整图 mask")
            return None

        best_score, best_idx, best_frac = max(candidates, key=lambda t: t[0])
        mask_np = (masks[best_idx] > 0).float().cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        print(f"[SAM2] 主体 mask 已生成: 网格点 #{best_idx}（{GRID_REL_POINTS[best_idx]}），"
              f"IoU={best_score:.3f}，面积占比={best_frac:.2f}，尺寸={mask_img.size}")
        return mask_img
    except Exception as e:
        print(f"[SAM2] 分割失败，返回 None，上层将降级为整图 mask。"
              f"原因: {type(e).__name__}: {e}")
        return None


def get_subject_mask_or_full(image, sam):
    """主体 mask 获取入口（规格 §9/§27 降级语义）。

    segment_subject 成功 -> 返回分割 mask；
    失败 / SAM 不可用 -> 返回与原图同尺寸的全白整图 mask（255），
    并打印“已降级为整图 mask”（整图 ControlNet 条件，保证流程不断）。
    """
    mask = segment_subject(image, sam)
    if mask is None:
        w, h = image.size
        full = Image.new("L", (w, h), 255)
        print("[SAM2] 已降级为整图 mask（规格 §9/§27：SAM 不可用或分割失败 -> 整图 ControlNet）")
        return full
    return mask


def unload_sam(sam, processor=None):
    """释放 SAM2 模型与 processor，清空 CUDA 缓存（显存腾挪用）。

    兼容两种调用：unload_sam(sam)（sam 为 load_sam2 返回的元组）
    或 unload_sam(model, processor)。
    """
    if processor is not None:
        model = sam  # 分开传参形式
    elif isinstance(sam, tuple) and len(sam) == 2:
        model, processor = sam
    else:
        model = sam
        processor = None

    for obj in (model, processor):
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SAM2] 已卸载并清空 CUDA 缓存")


if __name__ == "__main__":
    """自检块（PyCharm 直接运行，不加载任何模型，纯 CPU）。"""
    print("=" * 60)
    print("vision/segmentation.py 自检（不加载模型）")
    print("=" * 60)

    from PIL import Image as _PILImage

    # 1. 降级路径：sam=None -> get_subject_mask_or_full 返回全白整图 mask（§27）
    img = _PILImage.new("RGB", (80, 60), (100, 100, 100))
    full = get_subject_mask_or_full(img, None)
    arr = np.array(full)
    ok1 = (full.size == img.size and full.mode == "L"
           and arr.shape == (60, 80) and arr.min() == 255 and arr.max() == 255)
    print(f"[1] sam=None 降级整图 mask: size={full.size} mode={full.mode} 全白={arr.min()==255}  {'OK' if ok1 else 'FAIL'}")

    # 2. segment_subject(img, None) 直接返回 None
    r = segment_subject(img, None)
    ok2 = (r is None)
    print(f"[2] segment_subject(img, None) -> {r}  {'OK' if ok2 else 'FAIL'}")

    # 3. 模型异常路径：假 sam 对象（_single_frame_forward 抛异常）-> 返回 None
    class _FakeBrokenModel:
        def _single_frame_forward(self, **kwargs):
            raise RuntimeError("模拟模型推理崩溃")

    class _FakeProcessor:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("不该走到预处理")

    fake_sam = (_FakeBrokenModel(), _FakeProcessor())
    r3 = segment_subject(img, fake_sam)
    ok3 = (r3 is None)
    print(f"[3] 模型异常 -> segment_subject 返回 {r3}（不崩溃）  {'OK' if ok3 else 'FAIL'}")

    # 4. unload_sam 兼容两种调用形式（不触发 CUDA 清理逻辑的报错）
    unload_sam(None)
    unload_sam(fake_sam)
    unload_sam(_FakeBrokenModel(), _FakeProcessor())
    ok4 = True
    print(f"[4] unload_sam 三种调用形式均不抛异常  {'OK' if ok4 else 'FAIL'}")

    # 5. 配置联动提示
    print(f"[5] config.USE_SAM={config.USE_SAM} config.SAM_PATH={config.SAM_PATH}")
    print(f"    SAM2 fp16 权重显存预算: 约 {176e6 * 2 / 1024**3:.2f}GB（model.safetensors 约 176M 参数）")

    print("=" * 60)
    print("全部通过" if (ok1 and ok2 and ok3 and ok4) else "存在 FAIL")
