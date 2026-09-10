# -*- coding: utf-8 -*-
"""
character/character_parser.py —— 角色图片语义解析

- pick_representative_images : 整体和脸部建立身份，再覆盖所有配饰特写
  （无可用标签时取 001/中间/最后）
- parse_character_with_vlm    : 代表图逐张过 VLM，合并为规格 §6.2 结构
- parse_character_offline     : 无 VLM 时的兜底结构（字段齐全、值保守）

依赖：config / dataset.load_captions, scan_images / image_utils / vlm.vlm_utils。
只读数据集，不修改原始数据（规格 §5.2 / 禁止 9）。
"""

import os
import sys

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from dataset import load_captions, scan_images  # noqa: E402
import image_utils  # noqa: E402
from PIL import Image  # noqa: E402  用于小尺寸特写放大（四叶发饰等，见 parse_character_with_vlm）
from vlm.vlm_utils import CHARACTER_PROMPT  # noqa: E402


def pick_representative_images(image_files: list, captions: dict) -> list:
    """整体图优先建立身份，其次脸部，再覆盖所有标志/配饰参考。

    不限制最多三张，避免遗漏第二个配饰；独立配饰不作为人物身份基底。
    captions 为空 / 没有可用标签时：取第 1 张 / 中间 / 最后 1 张，保证代表性。
    结果去重、按挑选优先级返回。
    """
    if not image_files:
        return []

    def _has_tag(path: str, keyword: str) -> bool:
        tags = [t.lower() for t in captions.get(os.path.basename(path), [])]
        return any(keyword in t for t in tags)

    face_files = [p for p in image_files if _has_tag(p, "face close-up")]
    # 标志配饰：发饰/四叶/标志关键词优先（四叶发饰是角色标志，规格 §5.1/§6.2）
    sig_acc = [p for p in image_files
               if (_has_tag(p, "hair") or _has_tag(p, "four")
                   or _has_tag(p, "leaf") or _has_tag(p, "signature"))]
    acc_files = [p for p in image_files if _has_tag(p, "accessory")]

    picks = []
    # 整体/人脸提供身份；所有配饰特写逐一解析，不能只取首张并遗漏 013。
    whole = [p for p in image_files if _has_tag(p, "full body")]
    if whole:
        picks.append(whole[0])
    # 2. 人脸特写
    if face_files and face_files[0] not in picks:
        picks.append(face_files[0])
    for p in sig_acc + acc_files:
        if p not in picks:
            picks.append(p)
    # 3. 无标注时补足整体参考
    rest = [p for p in image_files if p not in picks]
    if rest and len(picks) < 3:
        picks.append(rest[0])
    # 若标签缺失（含 captions 为空）导致不足 3 张，补 001/中间/最后
    if len(picks) < 3:
        for idx in (0, len(image_files) // 2, len(image_files) - 1):
            if 0 <= idx < len(image_files) and image_files[idx] not in picks:
                picks.append(image_files[idx])
            if len(picks) >= 3:
                break

    print(f"[character_parser] 代表图挑选: {[os.path.basename(p) for p in picks]}")
    return picks


def _merge_character(base: dict, extra: dict) -> None:
    """把 extra 的 §6.2 字段合并进 base（首张为主，extra 只做补充）。

    - 标量/字符串字段：base 为空时由 extra 填充
    - 列表字段（color_palette / dominant_colors）：按值去重追加
    - signature_accessories：按 name 去重追加（补充首张缺失的配饰）
    """
    if not isinstance(extra, dict):
        return
    for key, val in extra.items():
        if isinstance(val, dict):
            bd = base.setdefault(key, {})
            if not isinstance(bd, dict):
                bd = {}
                base[key] = bd
            for k2, v2 in val.items():
                if (not bd.get(k2) or bd.get(k2) == "unknown") and v2:
                    bd[k2] = v2
        elif isinstance(val, list):
            if key == "signature_accessories":
                # 按 name 去重，只追加 base 中不存在的配饰
                names = {a.get("name") for a in base.get(key, []) if isinstance(a, dict)}
                for a in val:
                    if isinstance(a, dict) and a.get("name") and a["name"] not in names:
                        base.setdefault(key, []).append(a)
                        names.add(a["name"])
            else:
                base_list = base.setdefault(key, [])
                for item in val:
                    if item not in base_list:
                        base_list.append(item)
        else:
            if (not base.get(key) or base.get(key) == "unknown") and val:
                base[key] = val


def parse_character_with_vlm(client, folder_dir: str, captions: dict) -> dict:
    """对代表图逐张调用 client.analyze_image(img, CHARACTER_PROMPT)，合并返回 §6.2 结构。

    - 代表图由 pick_representative_images 挑选（包含全部配饰特写）
    - 单张图片分析失败只打印警告并跳过（不中断整体流程）
    - 全部失败 / client 不可用：回退 parse_character_offline 兜底结构
    """
    if client is None or not getattr(client, "available", False):
        print("[character_parser] 告警: VLM 不可用，回退 offline 解析")
        return parse_character_offline(captions)

    image_files = scan_images(folder_dir)  # 目录缺失/为空会抛异常（fail fast）
    picks = pick_representative_images(image_files, captions)

    results = []
    for path in picks:
        try:
            img = image_utils.load_image(path)
            # 特写图可能很小（如 013 发饰 114x116），直接送 VLM 细节不可辨，
            # 放大到短边 512 再分析（仅内存副本，不落盘、不改数据集，规格 §禁止9）
            if max(img.size) < 512:
                scale = 512 / max(img.size)
                img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                                 Image.LANCZOS)
            tags = captions.get(os.path.basename(path), [])
            detail = any("accessory" in t.lower() for t in tags)
            prompt = CHARACTER_PROMPT
            if detail:
                prompt += ("\nThis is an isolated accessory close-up. Describe its actual object "
                           "type, geometry, number of lobes, outer-to-inner color layers and center. "
                           "Do not invent a person, hair, eyes or clothing. Output those fields as "
                           "unknown and put observations in signature_accessories.")
            r = client.analyze_image(img, prompt, temperature=0)
            if detail and r:
                r = {"signature_accessories": r.get("signature_accessories", [])}
                for accessory in r["signature_accessories"]:
                    if isinstance(accessory, dict):
                        accessory["source_images"] = [os.path.basename(path)]
        except Exception as e:
            print(f"[character_parser] 告警: {path} 分析失败，跳过: {type(e).__name__}: {e}")
            r = {}
        if r:
            results.append(r)
        else:
            print(f"[character_parser] 告警: {os.path.basename(path)} 未返回有效语义，跳过")

    if not results:
        print("[character_parser] 告警: 所有代表图分析均失败，返回 offline 兜底结构")
        return parse_character_offline(captions)

    # 首张为主，其余只补充缺失字段/配饰
    base = dict(results[0])
    for extra in results[1:]:
        _merge_character(base, extra)
    print(f"[character_parser] VLM 角色解析完成（融合 {len(results)} 张代表图）")
    return base


def parse_character_offline(captions: dict) -> dict:
    """无 VLM 时的 §6.2 兜底结构：字段齐全、值保守（unknown / 空列表）。"""
    return {
        "character_identity": "unknown",
        "visual_style": "unknown",
        "color_palette": [],
        "hair": {"color": "unknown", "style": "unknown"},
        "face": {"eye_color": "unknown", "face_features": "unknown"},
        "clothing": {"description": "unknown", "dominant_colors": []},
        "signature_accessories": [],
        "note": "offline fallback (no VLM)",
    }


if __name__ == "__main__":
    """自检块（PyCharm 直接运行）：不加载 VLM，仅验证代表图挑选与 offline 兜底结构。"""
    print("=" * 60)
    print("character/character_parser.py 自检")
    print("=" * 60)

    folder_dir = os.path.join(config.DATASET_DIR, config.CHARACTER_ID)
    cap_file = os.path.join(folder_dir, "captions.txt")

    # 1. 代表图挑选（VLM 路径的前置步骤）
    imgs = scan_images(folder_dir)
    caps = load_captions(cap_file)
    print(f"[1] 图片共 {len(imgs)} 张，captions 共 {len(caps)} 条")
    picks = pick_representative_images(imgs, caps)
    print(f"    挑选结果: {[os.path.basename(p) for p in picks]}")

    # 2. 空 captions 场景（应回退 001/中间/最后）
    picks_no_cap = pick_representative_images(imgs, {})
    print(f"[2] 空 captions 挑选: {[os.path.basename(p) for p in picks_no_cap]}")

    # 3. offline 兜底结构
    offline = parse_character_offline(caps)
    print(f"[3] offline 兜底结构字段: {sorted(offline.keys())}")
    print(f"    note = {offline['note']!r}")

    print("=" * 60)
