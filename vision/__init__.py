# -*- coding: utf-8 -*-
"""
vision/ —— 视觉模块（规格 §2 目录结构）

对外导出（供 test.py 等上层调用）：
- subject_analyzer.analyze_subject / get_style_token   输入主体理解（§7 / §17）
- segmentation.load_sam2 / segment_subject / get_subject_mask_or_full / unload_sam
                                                       SAM2 主体分割（§9 / §27）
"""

from vision.subject_analyzer import analyze_subject, get_style_token
from vision.segmentation import (
    load_sam2,
    segment_subject,
    get_subject_mask_or_full,
    unload_sam,
)

__all__ = [
    "analyze_subject",
    "get_style_token",
    "load_sam2",
    "segment_subject",
    "get_subject_mask_or_full",
    "unload_sam",
]
