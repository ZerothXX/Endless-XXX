# -*- coding: utf-8 -*-
"""
dataset/data_preprocessing.py —— 训练前数据自动检查与修复

针对单个角色文件夹（dataset/<角色名>/）执行：
1. check_and_convert_images : 非 .png 图片统一转为同名 .png（成功后删除原件）
2. renumber_images          : 检查是否 001 起连续零填充编号，不是则两阶段重命名
3. remap_captions           : 重编号后按映射重写 captions.txt 首列文件名
4. generate_captions_with_vlm : 无 captions.txt 时用 VLM 自动生成短标签（VLM 模块未就绪时优雅降级）

可独立运行（PyCharm 直接运行本文件即可），也可被其他脚本 import 调用。
对已合规的数据（如 dataset/37/）执行应完全幂等、无副作用。

本文件不依赖 torch，仅使用标准库 + PIL + opencv（经 image_utils）。
"""

import os
import re
import sys

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
import image_utils


def _natural_key(name: str):
    """文件名自然排序键（001.png < 002.png < ... < 010.png）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _images_dir(folder_dir: str) -> str:
    """返回角色文件夹的 images/ 路径，不存在则抛异常。"""
    images_dir = os.path.join(folder_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"[preprocess] 图片目录不存在: {images_dir}")
    return images_dir


def check_and_convert_images(folder_dir: str) -> dict:
    """扫描 images/，把非 .png 后缀的图片转为同名 .png。

    - 转换成功：打印变更，删除原文件
    - 转换失败：保留原文件，打印告警（不中断整个流程）
    返回 {"converted": [旧文件], "failed": [旧文件]}。
    """
    images_dir = _images_dir(folder_dir)
    result = {"converted": [], "failed": []}

    entries = sorted(os.listdir(images_dir), key=_natural_key)
    for name in entries:
        src = os.path.join(images_dir, name)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext == ".png":
            continue  # 已是 PNG，无需处理（幂等）

        dst = os.path.join(images_dir, os.path.splitext(name)[0] + ".png")
        try:
            img = image_utils.load_image(src)   # 含 EXIF 方向修正 + 转 RGB
            image_utils.save_image(img, dst)
            os.remove(src)                      # 转换成功后删除原文件
            result["converted"].append(name)
            print(f"[preprocess] 转换: {name} -> {os.path.basename(dst)}")
        except Exception as e:
            result["failed"].append(name)
            print(f"[preprocess] 警告: 转换失败，保留原文件 {src}，原因: {e}")

    return result


def _is_compliant(names: list, n: int) -> bool:
    """检查文件名列表是否满足 `001 起连续零填充编号` 规范。"""
    width = max(3, len(str(n)))
    for i, name in enumerate(names, start=1):
        if name != f"{i:0{width}d}.png":
            return False
    return True


def renumber_images(folder_dir: str) -> dict:
    """检查 images/ 编号规范；不合规时两阶段重命名为连续零填充编号。

    两阶段策略（避免改名冲突）：
        阶段一: 所有文件改为 __tmp_{i:0{width}d}.png
        阶段二: 所有文件改为 {i:0{width}d}.png
    编号位数 = max(3, len(str(图片总数)))，即 99 张内用 3 位，100 张用 4 位……

    返回 old->new 文件名映射 dict；已合规返回空 dict {}。
    """
    images_dir = _images_dir(folder_dir)
    # 只处理 PNG（非 PNG 文件跳过并告警，不阻塞流程）
    png_names = [n for n in os.listdir(images_dir)
                 if os.path.isfile(os.path.join(images_dir, n))
                 and n.lower().endswith(".png")]
    png_names.sort(key=_natural_key)
    n = len(png_names)

    others = [n for n in os.listdir(images_dir) if n not in png_names and os.path.isfile(os.path.join(images_dir, n))]
    if others:
        print(f"[preprocess] 警告: {images_dir} 中存在非 PNG 文件（未参与重编号）: {others}")

    if n == 0:
        raise RuntimeError(f"[preprocess] {images_dir} 中没有 PNG 图片，无法重编号")

    if _is_compliant(png_names, n):
        print(f"[preprocess] 重编号: 已合规（{n} 张，001 起连续零填充），无需修改")
        return {}

    width = max(3, len(str(n)))
    mapping = {}
    tmp_names = []
    # 阶段一：全部改为 __tmp_ 前缀，杜绝与目标名冲突
    for i, old in enumerate(png_names, start=1):
        tmp = f"__tmp_{i:0{width}d}.png"
        os.rename(os.path.join(images_dir, old), os.path.join(images_dir, tmp))
        tmp_names.append(tmp)
    # 阶段二：__tmp_ 改回正式连续编号
    for i, tmp in enumerate(tmp_names, start=1):
        new = f"{i:0{width}d}.png"
        os.rename(os.path.join(images_dir, tmp), os.path.join(images_dir, new))
        mapping[png_names[i - 1]] = new
        print(f"[preprocess] 重编号: {png_names[i - 1]} -> {new}")

    return mapping


def remap_captions(caption_file: str, mapping: dict) -> bool:
    """按重编号映射重写 captions.txt 首列文件名，标签原样保留。

    - mapping 为空：直接返回 False，不触碰文件（幂等）
    - 文件中未出现在 mapping 里的文件名行保持原样
    - 返回是否有行被改写
    """
    if not mapping:
        return False
    if not os.path.isfile(caption_file):
        print(f"[preprocess] 提示: captions.txt 不存在，跳过重映射: {caption_file}")
        return False

    lines = []
    changed = False
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if "," in stripped:
                old_name, _, rest = stripped.partition(",")
                if old_name.strip() in mapping:
                    new_name = mapping[old_name.strip()]
                    stripped = f"{new_name},{rest}"
                    changed = True
            lines.append(stripped + "\n")

    if changed:
        with open(caption_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[preprocess] captions 重映射完成: {caption_file}")
    else:
        print(f"[preprocess] captions 重映射: 无匹配条目，内容不变: {caption_file}")
    return changed


def generate_captions_with_vlm(folder_dir: str, vlm_path: str) -> bool:
    """无 captions.txt 时，用 VLM 逐图生成 `文件名.png, 短标签1, 短标签2` 标注。

    预期接口（vlm/vlm_utils.py 后续阶段实现，本阶段仅预留调用约定）：
        from vlm.vlm_utils import VLMClient
        client = VLMClient(model_path=vlm_path)      # 加载本地 VLM
        tags = client.generate_tags(image_path)      # 返回英文短标签 list[str]

    系统提示词约束：
        - 只输出 2~4 个视角/构图类英文短标签（front view / face close-up /
          accessory close-up / full body shot 等，对齐规格 §4.2 短标签体系）
        - 禁止输出身份属性（发色、瞳色、服装、配饰描述等，角色身份由触发词学习）

    VLM 不可用（vlm_utils 未实现 / 加载失败）时打印提示并返回 False，
    不阻塞数据预处理主流程。
    """
    images_dir = _images_dir(folder_dir)
    caption_file = os.path.join(folder_dir, "captions.txt")

    if os.path.isfile(caption_file):
        print(f"[preprocess] VLM 标注: captions.txt 已存在，跳过: {caption_file}")
        return True

    try:
        # 提示词常量统一在 vlm/vlm_utils.py 顶部定义（本文件不再内联，消除重复）
        from vlm.vlm_utils import CAPTION_PROMPT, VLMClient  # noqa: F401
    except ImportError:
        print("[preprocess] 跳过自动标注: VLM 模块 (vlm/vlm_utils.py) 尚未实现或不可用，"
              "请稍后手动补充 captions.txt")
        return False

    # ---- 以下代码在 vlm.vlm_utils 就绪后生效 ----
    client = None
    try:
        # 显存配置透传 config（8GB 卡上若默认 fp16 紧张，可开 4bit / CPU offload）；
        # 标注完成后必须在 finally 中 unload，避免与后续训练管线争抢显存
        client = VLMClient(
            model_path=vlm_path,
            dtype=config.VLM_DTYPE,
            load_in_4bit=config.VLM_LOAD_IN_4BIT,
            offload_cpu=config.VLM_OFFLOAD_CPU,
        )
        png_names = sorted([n for n in os.listdir(images_dir)
                            if n.lower().endswith(".png")], key=_natural_key)
        system_prompt = CAPTION_PROMPT
        lines = []
        for name in png_names:
            tags = client.generate_tags(os.path.join(images_dir, name),
                                        system_prompt=system_prompt)
            if not tags or not isinstance(tags, list):
                print(f"[preprocess] 警告: {name} 的 VLM 标签为空，跳过该图标注")
                continue
            tags = [t.strip() for t in tags if t.strip()][:4]
            lines.append(f"{name}, {', '.join(tags)}")
        if not lines:
            print("[preprocess] 警告: VLM 未生成任何标签，未写入 captions.txt")
            return False
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[preprocess] VLM 标注完成，已写入 {len(lines)} 条: {caption_file}")
        return True
    except Exception as e:
        print(f"[preprocess] 警告: VLM 自动标注失败（{e}），请稍后手动补充 captions.txt")
        return False
    finally:
        if client is not None:
            try:
                client.unload()
                print("[preprocess] VLM 已卸载（自动标注仅临时加载）")
            except Exception as e:
                print(f"[preprocess] 警告: VLM 卸载失败（{e}）")


def preprocess_character_folder(folder_name: str) -> dict:
    """编排角色文件夹的预处理流程（转换 -> 重编号 -> captions 重映射 -> VLM 标注）。

    任何一步异常只打印告警、记录到摘要 dict，不中断后续步骤。
    返回摘要 dict，供调用方/自检查看。
    """
    folder_dir = os.path.join(config.DATASET_DIR, folder_name)
    summary = {"folder": folder_name, "ok": True}

    if not os.path.isdir(folder_dir):
        summary["ok"] = False
        summary["error"] = f"角色文件夹不存在: {folder_dir}"
        print(f"[preprocess] 错误: {summary['error']}")
        return summary

    print(f"\n========== 预处理角色: {folder_name} ==========")

    # 步骤 1: 非 PNG -> PNG
    try:
        conv = check_and_convert_images(folder_dir)
        summary["converted"] = conv["converted"]
        summary["convert_failed"] = conv["failed"]
    except Exception as e:
        summary["ok"] = False
        summary["convert_error"] = str(e)
        print(f"[preprocess] 警告: 图片转换步骤异常: {e}")

    # 步骤 2: 连续编号检查/重命名
    try:
        mapping = renumber_images(folder_dir)
        summary["renumbered"] = len(mapping) > 0
        summary["rename_mapping"] = mapping
    except Exception as e:
        summary["ok"] = False
        summary["renumber_error"] = str(e)
        print(f"[preprocess] 警告: 重编号步骤异常: {e}")
        mapping = {}

    # 步骤 3: 按重编号映射改写 captions.txt
    try:
        caption_file = os.path.join(folder_dir, "captions.txt")
        summary["captions_remapped"] = remap_captions(caption_file, mapping)
    except Exception as e:
        summary["ok"] = False
        summary["remap_error"] = str(e)
        print(f"[preprocess] 警告: captions 重映射步骤异常: {e}")

    # 步骤 4: 无 captions.txt 时 VLM 自动标注
    try:
        summary["vlm_generated"] = generate_captions_with_vlm(folder_dir, config.VLM_PATH)
    except Exception as e:
        summary["ok"] = False
        summary["vlm_error"] = str(e)
        print(f"[preprocess] 警告: VLM 标注步骤异常: {e}")

    # 汇总打印
    print("---------- 预处理总结 ----------")
    print(f"  图片转换        : {len(summary.get('converted', []))} 张"
          f"（失败 {len(summary.get('convert_failed', []))} 张）")
    if summary.get("renumbered"):
        print(f"  重编号          : {len(summary['rename_mapping'])} 张文件被重命名")
    else:
        print("  重编号          : 已合规、无需修改")
    print(f"  captions 重映射 : {'是' if summary.get('captions_remapped') else '否（无需修改或文件不存在）'}")
    print(f"  VLM 自动标注    : {summary.get('vlm_generated')}")
    print("================================")
    return summary


if __name__ == "__main__":
    """独立运行入口（PyCharm 可直接运行）。

    用法:
        python dataset/data_preprocessing.py            # 默认处理 config.CHARACTER_ID
        python dataset/data_preprocessing.py 37         # 指定角色文件夹
    """
    target = sys.argv[1] if len(sys.argv) > 1 else config.CHARACTER_ID
    result = preprocess_character_folder(target)
    if not result.get("ok"):
        print(f"\n[preprocess] 预处理完成，但存在异常（见上方警告），文件夹: {target}")
    else:
        print(f"\n[preprocess] 预处理完成，文件夹: {target}")
