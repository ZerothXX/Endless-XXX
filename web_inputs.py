# -*- coding: utf-8 -*-
"""
web_inputs.py —— 网页输入资料的读写（不含 torch）

网页（web/）在 dataset/<角色名>/ 下写逐图标注：用户每提交一次就是"一张图 + 一段标注"，
服务端把它写成一行 `images/<文件名>\\t<描述>`：

    mark_refs.txt  网页逐图标注（当前网页界面的统一输入方式，一张图一段标注）
    marks.txt      只有文字、没有对应图片的角色描述；网页界面已不再产生，
                   服务器仍保留该写入分支，这里同样兼容读取

它们不像 captions.txt 那样是"文件名, 标签"的短标签格式，因此本模块把解析、
重映射与合并集中在一处，供训练侧（dataset.py）与预处理侧
（dataset/data_preprocessing.py）共用。

本模块只依赖标准库与 config：data_preprocessing.py 明确不依赖 torch，
不能让"读两行文本"把 torch 拖进纯数据处理流程，因此解析逻辑放在这里而不是
dataset.py（dataset.py 需要 torch 构建 Dataset）。
"""

import os

import config


# "配饰特写"是既有标注体系的约定标签：用户在自己的标注里写它，
# dataset.build_text 就会把类别切到 accessory、并对该图使用细节裁剪。
# 网页解析不会替用户追加这个标签。
ACCESSORY_CLOSEUP_TAG = "accessory close-up"


def marks_path(folder_dir: str) -> str:
    return os.path.join(folder_dir, getattr(config, "MARKS_FILE", "marks.txt"))


def mark_refs_path(folder_dir: str) -> str:
    return os.path.join(folder_dir, getattr(config, "MARK_REFS_FILE", "mark_refs.txt"))


def load_marks(folder_dir: str) -> list:
    """解析 marks.txt，返回按写入顺序的规范描述行（文件不存在返回空列表）。

    只做空白归一与空行过滤，不在这里去重：网页每提交一次就追加一行，
    保留原始行数才能如实反映用户输入；去重由 merged_marks 在合并进 prompt 时完成。
    """
    path = marks_path(folder_dir)
    if not os.path.isfile(path):
        return []
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                text = " ".join(line.split())
                if text:
                    lines.append(text)
    except OSError as exc:
        print(f"[web_inputs] 警告: 角色描述读取失败（{exc}），按无描述处理: {path}")
        return []
    return lines


def load_mark_refs(folder_dir: str) -> dict:
    """解析 mark_refs.txt，返回 {图片文件名: [标签]}（一图一标注）。

    行格式：`images/<文件名>\\t<描述>`（分隔符优先取 TAB；没有 TAB 时退回首个逗号，
    兼容手工编辑过的文件）。文件名只取 basename，避免写入路径参与匹配；
    描述按逗号拆成标签后**原样使用**：网页现在是"一张图一段标注"的统一输入，
    标注就是用户对这张图的说明，既可能是配饰特写也可能是全身参考，
    因此不在这里替用户追加任何标签（需要配饰特写语义时由用户在标注里写明，
    dataset.build_text 仍会按既有规则识别 "accessory close-up"）。
    """
    path = mark_refs_path(folder_dir)
    refs = {}
    if not os.path.isfile(path):
        return refs
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if "\t" in stripped:
                    raw_name, _, description = stripped.partition("\t")
                elif "," in stripped:
                    raw_name, _, description = stripped.partition(",")
                else:
                    print(f"[web_inputs] 警告: {path} 第 {line_no} 行缺少分隔符，已跳过: {stripped!r}")
                    continue
                name = os.path.basename(raw_name.strip().replace("\\", "/"))
                tags = [t.strip() for t in description.split(",") if t.strip()]
                if not name or not tags:
                    print(f"[web_inputs] 警告: {path} 第 {line_no} 行缺少文件名或描述，已跳过: {stripped!r}")
                    continue
                refs[name] = tags
    except OSError as exc:
        print(f"[web_inputs] 警告: 标志图像描述读取失败（{exc}），按无描述处理: {path}")
        return {}
    return refs


def merged_marks(lines: list, budget: int = None) -> list:
    """把 marks.txt 的多行描述合并为标签列表：保序去重，并受字符预算约束。

    训练 prompt 受 CLIP 双 tokenizer 77 token 限制（train.py 会按 tokenizer 复核），
    config.WEB_DESCRIPTION_MAX_CHARS 是这里的字符预算；超预算的描述被丢弃并提示，
    避免用户随手写的一长段中文把整个训练任务卡死。
    """
    if budget is None:
        budget = int(getattr(config, "WEB_DESCRIPTION_MAX_CHARS", 240) or 0)
    tags, used, dropped = [], 0, 0
    seen = set()
    for line in lines:
        text = " ".join(str(line).split())
        if not text or text in seen:
            continue
        seen.add(text)
        if budget > 0 and used + len(text) > budget:
            if not tags:               # 单条就超预算：截断保留，而不是整条丢弃
                tags.append(text[:budget])
                used = budget
                break
            dropped += 1
            continue
        tags.append(text)
        used += len(text)
    if dropped:
        print(f"[web_inputs] 提示: {dropped} 条角色描述超出 {budget} 字预算，未参与训练 prompt"
              f"（config.WEB_DESCRIPTION_MAX_CHARS 可调）")
    return tags


def user_descriptions(folder_dir: str) -> dict:
    """汇总网页输入的资料规模，供预处理判断"用户是否已提供描述"。"""
    marks = load_marks(folder_dir)
    refs = load_mark_refs(folder_dir)
    return {"marks": len(marks), "mark_refs": len(refs),
            "described": bool(marks or refs)}


def remap_mark_refs(folder_dir: str, mapping: dict) -> bool:
    """按重编号映射改写 mark_refs.txt 的文件名列，描述原样保留。

    与 data_preprocessing.remap_captions 同一目的：图片被重编号后，
    指向旧文件名的标志图像描述必须同步改名，否则描述会全部失配。
    mapping 为空或文件不存在时返回 False，不触碰文件（幂等）。
    """
    if not mapping:
        return False
    path = mark_refs_path(folder_dir)
    if not os.path.isfile(path):
        return False
    lines, changed = [], False
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                stripped = line.rstrip("\n")
                body = stripped.strip()
                if body:
                    separator = "\t" if "\t" in body else ("," if "," in body else "")
                    if separator:
                        raw_name, _, rest = body.partition(separator)
                        name = os.path.basename(raw_name.strip().replace("\\", "/"))
                        if name in mapping:
                            stripped = f"images/{mapping[name]}{separator}{rest}"
                            changed = True
                lines.append(stripped + "\n")
    except OSError as exc:
        print(f"[web_inputs] 警告: 标志图像描述重映射失败（{exc}）: {path}")
        return False
    if changed:
        with open(path, "w", encoding="utf-8") as stream:
            stream.writelines(lines)
        print(f"[web_inputs] 标志图像描述重映射完成: {path}")
    return changed
