# -*- coding: utf-8 -*-
"""
vision/subject_analyzer.py —— 输入主体理解（规格 §7）

职责：
- SUBJECT_PROMPT 不在此重复定义：vlm/vlm_utils.py 中已有唯一版本（§7 JSON 结构），
  本文件直接引用，保证训练/推理侧提示词永远一致（单一定义原则）。
- analyze_subject(client, image) -> dict：调用 VLM 分析输入图片的主体并补全字段。
  任何失败（client 为 None / analyze_image 抛异常 / 返回空 dict / subject_type 非法）
  都返回 {"subject_type": "unknown"} 兜底（规格 §27 失败回退 / §32 unknown 不崩溃）。
- get_style_token(style) -> str：规格 §17“原画风保持”的 style -> prompt 片段映射，
  供下游 prompt 构建使用；无匹配返回空串（不注入任何风格约束）。

接口约定（与 vlm/vlm_utils.py 的 VLMClient 对齐）：
    client.analyze_image(image, system_prompt, max_new_tokens=None, temperature=None) -> dict
"""

import os
import sys

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vlm.vlm_utils import SUBJECT_PROMPT  # noqa: E402  引用唯一提示词定义（§7）

# =========================
# §7 默认主体语义（字段与 SUBJECT_PROMPT 完全对齐）
# =========================
# analyze_subject 返回前用本结构补全：VLM 未给出的键填 None / []，
# 保证下游（迁移策略 / prompt 构建）读取任何键都不会 KeyError。
DEFAULT_SUBJECT = {
    "subject_type": "unknown",     # human | animal | object | unknown（§7 四选一）
    "pose": None,                  # 姿态，如 standing / sitting
    "framing": "unknown",
    "hands": "unknown",
    "coverage": {},
    "species": None,               # 动物种类，如 cat（仅 animal 时有意义）
    "object_category": None,       # 物品类别，如 cup / bag（仅 object 时有意义）
    "material": None,              # 材质，如 fur / ceramic / fabric
    "style": None,                 # 画风，如 pixel art / realistic photo / 3D render / anime illustration
    "structure": [],               # 主体主要结构部件列表
    "visible_regions": [],         # 可见区域列表
    "attachment_regions": [],      # 可佩戴/悬挂配饰的区域（§7 示例：head / ear / neck / bag surface）
    "surface_regions": [],         # 适合表面纹样的平坦区域（§7 示例：cup body / handle）
}

# 合法的 subject_type 取值（§7）
VALID_SUBJECT_TYPES = ("human", "animal", "object", "unknown")


def analyze_subject(client, image) -> dict:
    """分析输入图片的主体，返回补全字段后的 dict（绝不抛异常）。

    参数：
        client: VLMClient 实例或 None（None 表示 VLM 不可用，直接走 unknown 兜底）
        image : PIL.Image（RGB）或图片路径字符串

    返回：
        补全后的 §7 结构 dict；任何失败路径返回 DEFAULT_SUBJECT 副本
        （subject_type="unknown"，规格 §27/§32）。
    """
    # client 不可用（VLM 关闭 / 加载失败）：直接兜底，不打印噪音
    if client is None:
        return dict(DEFAULT_SUBJECT)

    try:
        raw = client.analyze_image(image, SUBJECT_PROMPT, temperature=0)
    except Exception as e:
        # analyze_image 本身承诺不抛异常，但防御一层，绝不崩溃（§32）
        print(f"[subject_analyzer] 告警: 主体分析调用异常，回退 unknown。原因: {type(e).__name__}: {e}")
        return dict(DEFAULT_SUBJECT)

    # analyze_image 返回 {}（推理失败 / 输出无法解析为 JSON）→ unknown 兜底
    if not isinstance(raw, dict) or not raw:
        return dict(DEFAULT_SUBJECT)

    # 只保留已知字段（VLM 偶发的多余键不进入结果），缺失键由 DEFAULT_SUBJECT 补全
    result = dict(DEFAULT_SUBJECT)
    for key in DEFAULT_SUBJECT:
        if key in raw and raw[key] is not None:
            result[key] = raw[key]

    # subject_type 合法性校验：不在四选一内一律视为 unknown（§7/§32）
    if result["subject_type"] not in VALID_SUBJECT_TYPES:
        result["subject_type"] = "unknown"
    for key in ("visible_regions", "structure", "attachment_regions", "surface_regions"):
        value = result[key]
        result[key] = [str(v).lower() for v in value] if isinstance(value, list) else []
    coverage = result.get("coverage")
    result["coverage"] = ({k: v for k, v in coverage.items()
                           if k in {"legs", "chest"} and v in
                           {"bare", "covered", "out_of_frame", "exposed", "unknown"}}
                          if isinstance(coverage, dict) else {})
    return result


def get_style_token(style) -> str:
    """规格 §17：把 VLM 分析的画风（style）映射为 prompt 风格片段。

    匹配规则（不区分大小写，子串匹配）：
        "pixel"              -> "pixel art, preserve pixelated visual style"
        "photo"/"realistic"  -> "photorealistic, preserve photographic appearance"
        "3d"/"render"        -> "3D rendered appearance"
        "anime"/"illustration"/"2d" -> "anime style"
        无匹配               -> ""（不强加任何风格约束）

    注意顺序：pixel -> photo/realistic -> 3d/render -> anime/illustration/2d，
    各分支关键词互不包含（如 "3d render" 命中 3d 分支而非 2d 分支），
    因此顺序不影响结果，按优先级排列仅为可读性。
    """
    if not style or not isinstance(style, str):
        return ""
    s = style.strip().lower()
    if "pixel" in s:
        return "pixel art, preserve pixelated visual style"
    if "painting" in s:
        return "oil painting" if "oil" in s else "painting"
    if "3d" in s or "render" in s:
        return "3D rendered appearance"
    if "photo" in s or "realistic" in s:
        return "photorealistic, preserve photographic appearance"
    if "anime" in s or "illustration" in s or "2d" in s:
        return "anime style"
    return ""


if __name__ == "__main__":
    """自检块（PyCharm 直接运行，不加载任何模型，纯 CPU）。"""
    print("=" * 60)
    print("vision/subject_analyzer.py 自检")
    print("=" * 60)

    # 1. get_style_token 全分支断言（§17）
    cases = [
        ("pixel art", "pixel art, preserve pixelated visual style"),
        ("PIXEL ART", "pixel art, preserve pixelated visual style"),
        ("realistic photo", "photorealistic, preserve photographic appearance"),
        ("photorealistic portrait", "photorealistic, preserve photographic appearance"),
        ("3D render", "3D rendered appearance"),
        ("3d", "3D rendered appearance"),
        ("anime illustration", "anime style"),
        ("2D flat illustration", "anime style"),
        ("", ""),
        (None, ""),
        ("watercolor sketch", ""),
    ]
    all_ok = True
    for style, expect in cases:
        got = get_style_token(style)
        ok = (got == expect)
        all_ok = all_ok and ok
        print(f"[1] style={style!r:32s} -> {got!r}  {'OK' if ok else 'FAIL(期望 ' + expect + ')'}")

    # 2. analyze_subject(None, img) 兜底：unknown 且字段补全（§32）
    from PIL import Image
    dummy_img = Image.new("RGB", (64, 64), (0, 0, 0))
    r = analyze_subject(None, dummy_img)
    ok = isinstance(r, dict) and r["subject_type"] == "unknown"
    ok = ok and all(k in r for k in ("pose", "species", "material", "style",
                                     "structure", "attachment_regions", "surface_regions"))
    all_ok = all_ok and ok
    print(f"[2] analyze_subject(None, img) -> {r}  {'OK' if ok else 'FAIL'}")

    # 3. analyze_image 返回 {} 的假 client 兜底（§27）
    class _FakeClient:
        def analyze_image(self, image, system_prompt, max_new_tokens=None, temperature=None):
            return {}
    r2 = analyze_subject(_FakeClient(), dummy_img)
    ok = r2["subject_type"] == "unknown"
    all_ok = all_ok and ok
    print(f"[3] analyze_image 返回 {{}} 的假 client -> {r2}  {'OK' if ok else 'FAIL'}")

    # 4. 正常结果：字段透传 + 缺失键补全 + 非法 subject_type 归 unknown
    class _GoodClient:
        def analyze_image(self, image, system_prompt, max_new_tokens=None, temperature=None):
            return {"subject_type": "animal", "species": "cat", "pose": "sitting"}
    r3 = analyze_subject(_GoodClient(), dummy_img)
    ok = (r3["subject_type"] == "animal" and r3["species"] == "cat"
          and r3["material"] is None and r3["structure"] == [])
    all_ok = all_ok and ok
    print(f"[4] 部分字段 client -> {r3}  {'OK' if ok else 'FAIL'}")

    class _BadTypeClient:
        def analyze_image(self, image, system_prompt, max_new_tokens=None, temperature=None):
            return {"subject_type": "robot", "style": "anime"}
    r4 = analyze_subject(_BadTypeClient(), dummy_img)
    ok = r4["subject_type"] == "unknown" and r4["style"] == "anime"
    all_ok = all_ok and ok
    print(f"[5] 非法 subject_type -> {r4}  {'OK' if ok else 'FAIL'}")

    print(f"SUBJECT_PROMPT 引用自 vlm/vlm_utils.py，长度={len(SUBJECT_PROMPT)}")
    print("=" * 60)
    print("全部通过" if all_ok else "存在 FAIL")
