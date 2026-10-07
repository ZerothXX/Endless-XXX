# -*- coding: utf-8 -*-
"""
dataset.py —— 训练数据入口（根目录模块，与 dataset/ 数据目录同名不冲突）

职责：
- 解析 dataset/<角色>/captions.txt（"文件名, 短标签1, 短标签2" 格式，规格 §4.2）
- 解析网页（web/）输入的逐图标注：mark_refs.txt（一张图一段标注，网页的统一输入方式）
  与 marks.txt（只有文字的角色描述，兼容保留）
- 扫描角色文件夹与图片（自动检测实际数量，规格 §4.1）
- 构建训练 prompt：`ch{folder}, {category}, {quality}, {tags}`（规格 §4.4）
- validate_dataset：网页 /api/train 的训练前准入检查（图片数量下限）
- 提供 torch Dataset / DataLoader，供 train.py 使用

注意：prompt 中类别词 category 来自 config.CHARACTER_PRESETS 预设，
不要在模块内硬编码角色类别。
"""

import os
import random
import re

from PIL import Image, ImageOps

import config

# torch.utils.data 为本模块的核心依赖（仅构造 Dataset，不执行训练）
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import image_utils
from web_inputs import (ACCESSORY_CLOSEUP_TAG, load_mark_refs, load_marks,  # noqa: F401
                        merged_marks, remap_mark_refs)


def load_captions(caption_file: str) -> dict:
    """解析 captions.txt，返回 {文件名: [标签1, 标签2, ...]}。

    格式（规格 §4.2）：每行 `文件名, 短标签1, 短标签2`
    - 首个逗号前为文件名（原样保留，不去空白）
    - 其余字段各自 strip 后作为标签列表
    - 空行 / 无逗号的行：跳过并打印警告（不静默吞掉）
    - 文件不存在：返回空 dict（由调用方决定如何处理）
    """
    captions = {}
    if not os.path.isfile(caption_file):
        return captions

    with open(caption_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                print(f"[dataset] 警告: {caption_file} 第 {line_no} 行缺少逗号，已跳过: {line!r}")
                continue
            filename, _, rest = line.partition(",")
            tags = [t.strip() for t in rest.split(",") if t.strip()]
            captions[filename] = tags
    return captions


def build_text(trigger: str, category: str, tags) -> str:
    """合成训练 prompt：`{trigger}, {category}, {质量词}, {tags...}`。

    标签为空时省略标签部分，避免出现多余的尾逗号。
    质量词来自 config.TRAIN_QUALITY_TAGS（训练/推理两侧对齐，
    避免推理端新增 "masterpiece, best quality" 造成分布偏差）。
    """
    # 独立配饰图不是人物；类别污染会把相机/徽章和人体共同绑定。
    if any("accessory close-up" in str(tag).lower() for tag in tags):
        category = "accessory"
    parts = [trigger, category]
    quality = str(getattr(config, "TRAIN_QUALITY_TAGS", "") or "").strip()
    if quality:
        parts.append(quality)
    tag_str = ", ".join(tags)
    if tag_str:
        parts.append(tag_str)
    return ", ".join(parts)


def scan_character_folders(dataset_dir: str) -> list:
    """扫描 dataset/ 下所有含 images/ 子目录的角色文件夹，返回文件夹名列表。

    供角色选择菜单使用；按名称排序保证菜单顺序稳定。
    """
    folders = []
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"[dataset] 数据集目录不存在: {dataset_dir}")
    for name in sorted(os.listdir(dataset_dir)):
        images_dir = os.path.join(dataset_dir, name, "images")
        if os.path.isdir(images_dir):
            folders.append(name)
    return folders


def _natural_key(name: str):
    """文件名自然排序键（001.png < 002.png < ... < 010.png）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def scan_images(folder_dir: str) -> list:
    """扫描角色文件夹 images/ 下的图片文件，按文件名自然排序返回完整路径列表。

    图片目录不存在或为空时抛出明确异常（fail fast，训练数据缺失不能静默）。
    """
    images_dir = os.path.join(folder_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"[dataset] 角色图片目录不存在: {images_dir}")

    names = [n for n in os.listdir(images_dir)
             if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
    names.sort(key=_natural_key)
    paths = [os.path.join(images_dir, n) for n in names]

    if not paths:
        raise RuntimeError(f"[dataset] 角色图片目录为空: {images_dir}，无法训练")
    return paths


# ===========================================================================
# 网页（web/）输入资料的读取
# ---------------------------------------------------------------------------
# 文件格式与解析集中在 web_inputs.py（不依赖 torch，供预处理流程共用）；
# 这里直接复用，dataset 侧把它们折算成与 captions.txt 相同的 {文件名: [标签]} 语义。
# ===========================================================================


def resolve_captions(folder_dir: str, folder: str = None) -> tuple:
    """汇总一个角色的全部逐图标签，返回 (标签表, 自由文本文件名集合)。

    标签优先级（后者覆盖前者）：
        1. captions.txt —— 既有标注体系，命令行路径与网页路径共用
        2. mark_refs.txt —— 网页"标志图像输入"：一张图一段标注，按文件名精确对应
        3. config.TRAIN_CAPTION_OVERRIDES —— 人工校对覆盖，最高优先
    返回的第二个值是"标注由用户自由文本写成"的文件名集合：这类文本长度不可控，
    训练前需要按 CLIP token 预算裁剪（见 CharacterDataset.fit_prompts）；
    人工校对覆盖属于受控标注，不计入该集合，超预算仍按错误处理。
    全局角色描述（marks.txt）不在这里合并：它没有文件名，由调用方作为
    "无逐图标注图片"的兜底，见 CharacterDataset。
    """
    captions = load_captions(os.path.join(folder_dir, "captions.txt"))
    web_captions = load_mark_refs(folder_dir)
    captions.update(web_captions)
    from_web = set(web_captions)
    if folder:
        overrides = getattr(config, "TRAIN_CAPTION_OVERRIDES", {}).get(folder, {})
        captions.update(overrides)
        from_web -= set(overrides)      # 人工校对过的条目按受控标注对待
    return captions, from_web


def validate_dataset(character: str, minimum: int = None) -> dict:
    """校验角色是否具备训练条件（web/server.py 的 /api/train 准入检查）。

    - 角色文件夹或 images/ 缺失 -> FileNotFoundError
    - 图片不足 minimum 张（默认 config.WEB_MIN_TRAIN_IMAGES）-> RuntimeError
    返回摘要 dict（图片数、标签来源计数），供调用方/网页提示使用。
    """
    if minimum is None:
        minimum = int(getattr(config, "WEB_MIN_TRAIN_IMAGES", 10) or 1)
    folder_dir = os.path.join(config.DATASET_DIR, str(character))
    if not os.path.isdir(folder_dir):
        raise FileNotFoundError(f"[dataset] 角色文件夹不存在: {folder_dir}")
    paths = scan_images(folder_dir)          # images/ 缺失或为空在此抛异常
    if len(paths) < minimum:
        raise RuntimeError(f"[dataset] {character} 只有 {len(paths)} 张图片，"
                           f"少于训练所需的 {minimum} 张")
    names = [os.path.basename(path) for path in paths]
    captions = load_captions(os.path.join(folder_dir, "captions.txt"))
    refs = load_mark_refs(folder_dir)
    marks = merged_marks(load_marks(folder_dir))
    return {
        "ok": True,
        "character": str(character),
        "folder": folder_dir,
        "minimum": minimum,
        "images": len(paths),
        "captions": sum(1 for name in names if name in captions),
        "mark_refs": sum(1 for name in names if name in refs),
        "marks": len(marks),
        "unlabeled": sum(1 for name in names if name not in captions and name not in refs),
    }


class MixedContentCrop:
    """混合内容训练裁剪（v3 方案：尽量利用原图全部信息 + 保留身份细节）。

    每个样本随机选择一种窗口策略：
      1) full（30%）：整图保留——按原图宽高比吸附到**预算桶**（长边 ≤768、
         面积 ≤640²、尺寸为 64 的倍数），完全不裁剪
         → 全身比例/整体造型信息完整；
      2) top（50%）：短边对齐 size(640) 后从画面顶部 40% 高度内随机取窗
         → 头/脸/发饰/上半身主导（身份细节的来源）；
      3) rand（20%）：全图均匀随机方形窗 → 少量腿部/构图多样性。
    输出尺寸固定吸附到少量"桶"（64 的倍数），避免任意尺寸导致 cuDNN 内核
    反复重调优（实测全随机尺寸使训练慢 3~5 倍）；
    train.py 按实际尺寸构造 added_cond time_ids（任意长宽比训练）。

    背景：v1 方案 Resize(短边)+CenterCrop 把 448×1088 全身图裁成"腰部中带"，
    v2 方案 TopBiasedRandomCrop 已修复脸部但丢掉了整图信息；v3 混合策略
    二者兼顾（详见 docs/问题诊断与修复报告.md「方案演进史」）。
    """

    # 预算桶（w, h，均为 64 的倍数；面积 ≤ max_area、长边 ≤ max_long）
    # 默认取 config.TRAIN_BUCKETS（8GB 版）；G24 版本注入更高分辨率桶
    BUCKETS = [
        (640, 640), (704, 576), (576, 704), (768, 512), (512, 768),
        (704, 512), (512, 704), (640, 512), (512, 640),
    ]

    def __init__(self, size: int = 640, max_long: int = 768,
                 max_area: int = 640 * 640,
                 full_prob: float = 0.30, top_prob: float = 0.50,
                 top_bias: float = 0.40, buckets=None):
        if size <= 0:
            raise ValueError(f"MixedContentCrop size 必须为正整数: {size}")
        self.size = size
        self.max_long = max_long
        self.max_area = max_area
        self.full_prob = full_prob
        self.top_prob = top_prob
        self.top_bias = max(0.0, min(1.0, top_bias))
        self.buckets = list(buckets) if buckets else list(
            getattr(config, "TRAIN_BUCKETS", None) or self.BUCKETS)

    def _pick_bucket(self, w: int, h: int):
        """按宽高比选择最接近的预算桶（保持比例、面积/长边受约束）。"""
        best, best_ratio = None, float("inf")
        for bw, bh in self.buckets:
            if bw > self.max_long or bh > self.max_long:
                continue
            if bw * bh > self.max_area:
                continue
            ratio = abs((w / h) - (bw / bh))
            if ratio < best_ratio:
                best, best_ratio = (bw, bh), ratio
        return best or (self.size, self.size)

    def __call__(self, img):
        w, h = img.size
        mode = random.random()
        if mode < self.full_prob:
            # ---- 整图保留：吸附到最接近的预算桶，不裁剪 ----
            bw, bh = self._pick_bucket(w, h)
            img = ImageOps.pad(img, (bw, bh), method=Image.Resampling.LANCZOS,
                               color=(255, 255, 255))
            return img

        # ---- 方形窗口：短边对齐 size 后取窗 ----
        s = self.size
        scale = s / min(w, h)
        nw, nh = max(s, int(round(w * scale))), max(s, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (w, h) else img
        max_left, max_top = nw - s, nh - s
        left = random.randint(0, max_left) if max_left > 0 else 0
        if mode < self.full_prob + self.top_prob:
            # 顶部偏置：80% 从顶部 top_bias 区域取窗，20% 全随机
            if max_top > 0:
                if random.random() < 0.80:
                    top = random.randint(0, int(max_top * self.top_bias))
                else:
                    top = random.randint(0, max_top)
            else:
                top = 0
        else:
            top = random.randint(0, max_top) if max_top > 0 else 0
        return resized.crop((left, top, left + s, top + s))


def make_transforms(resolution: int, enable_hflip: bool = False,
                    enable_color_jitter: bool = False):
    """构造训练图像变换（v3 混合内容策略，见 MixedContentCrop）。

    返回 torchvision.transforms.Compose：
        MixedContentCrop(resolution) + 可选增强（默认关） + ToTensor。
    注意：输出为 8 的倍数的任意尺寸（不强制正方形），train.py 按实际尺寸
    构造 time_ids（任意长宽比训练）。整图桶的尺寸上限（max_long/max_area/
    TRAIN_BUCKETS）取自 config，G24 版本通过 config 覆盖实现高分辨率训练。
    """
    ops = [
        MixedContentCrop(
            size=resolution,
            max_long=int(getattr(config, "TRAIN_MAX_LONG", 768)),
            max_area=int(getattr(config, "TRAIN_MAX_AREA", 640 * 640)),
        ),
    ]
    if enable_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if enable_color_jitter:
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class CharacterDataset(Dataset):
    """角色 LoRA 训练数据集。

    数据来源：dataset/<folder>/images/*.png + 逐图标签 + 网页输入的角色描述
    每个样本返回:
        pixel_values: 图像张量 [3, H, W]（0~1 范围，训练管线自行决定归一化）
        prompt:       训练文本 `ch{folder}, {category}, {quality}, {tags}`（规格 §4.4）

    逐图标签优先级见 resolve_captions（mark_refs.txt > captions.txt > 人工校对覆盖）；
    都没有的图片回退到 marks.txt 的全局角色描述（网页"角色图像输入"的文字），
    仍然没有时用空标签——角色身份由触发词学习（规格 §4.4）。
    """

    def __init__(self, folder: str, resolution: int, image_limit: int = 0, transform=None):
        # folder 为角色文件夹名（如 "37"），路径一律通过 config 推导
        self.folder_dir = os.path.join(config.DATASET_DIR, folder)
        if not os.path.isdir(self.folder_dir):
            raise FileNotFoundError(f"[dataset] 角色文件夹不存在: {self.folder_dir}")

        self.image_paths = scan_images(self.folder_dir)
        # image_limit > 0 时仅取前 N 张（冒烟测试用）；0 = 全部
        if image_limit and image_limit > 0:
            self.image_paths = self.image_paths[:image_limit]

        self.captions, self.free_text = resolve_captions(self.folder_dir, folder)
        # 网页输入的全局角色描述：只用于没有逐图标注的图片（不覆盖任何精确标注）
        self.marks = merged_marks(load_marks(self.folder_dir))
        if self.marks:
            print(f"[dataset] 使用网页输入的角色描述 {len(self.marks)} 条，"
                  f"覆盖 {sum(1 for p in self.image_paths if os.path.basename(p) not in self.captions)}"
                  f" 张无逐图标注的图片")
        # 提示哪些图片完全没有标注（只提示，不报错：用空标签）
        missing = [os.path.basename(p) for p in self.image_paths
                   if os.path.basename(p) not in self.captions and not self.marks]
        if missing:
            print(f"[dataset] 提示: {len(missing)} 张图片无任何标注，将使用空标签"
                  f"（角色身份由触发词学习）: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        if self.free_text:
            print(f"[dataset] 网页逐图标注 {len(self.free_text)} 条（自由文本，"
                  f"超 CLIP 预算时按 token 裁剪，不中断训练）")

        # 类别词 / 触发词来自 config 预设，保证唯一参数来源
        char_cfg = config.get_character_config(folder)
        self.trigger = char_cfg["trigger"]
        self.category = char_cfg["category"]

        self.transform = transform or make_transforms(resolution)
        self.detail_transform = transforms.Compose([
            MixedContentCrop(size=resolution, max_long=resolution,
                             max_area=resolution * resolution, full_prob=1,
                             buckets=[(resolution, resolution)]),
            transforms.ToTensor(),
        ])
        self.eval_transform = transforms.Compose([
            MixedContentCrop(size=resolution, full_prob=1,
                             max_long=config.TRAIN_MAX_LONG, max_area=config.TRAIN_MAX_AREA),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        return self.get_item(idx)

    def tags_for(self, filename: str) -> list:
        """返回某张图片的标签：逐图标注优先，其次网页输入的全局角色描述。"""
        tags = self.captions.get(filename)
        if tags:
            return list(tags)
        return list(self.marks)

    def caption_source(self, filename: str) -> str:
        """标签来源："web"（网页逐图标注/角色描述，自由文本）| "captions"（受控标注）| "none"。

        train.py 用它决定超长 prompt 的处理方式：网页自由文本可以按 token 预算裁剪，
        而 captions.txt / 人工校对属于受控数据，超预算仍按错误处理（保持原有严格性）。
        """
        if filename in self.free_text:
            return "web"
        if self.captions.get(filename):
            return "captions"
        return "web" if self.marks else "none"

    def fit_prompts(self, measure, budget: int) -> int:
        """按 token 预算裁剪"网页自由文本"来源的 prompt，返回被裁剪的样本数。

        网页的标注是用户随手写的一句话（常见中文长句），token 数不受控；
        CLIP 双 tokenizer 的 77 token 上限若被突破，train.py 会直接报错终止训练。
        这里只裁剪 caption_source=="web" 的样本（captions.txt / 人工校对等受控标注
        仍走原来的严格报错），截断结果记录在 _fitted 中，get_prompt 与缓存/训练循环
        读取同一份文本，不会出现"缓存用短文本、训练用长文本"的不一致。
        """
        self._fitted = {}
        for idx in range(len(self.image_paths)):
            tags = self.tags_for(os.path.basename(self.image_paths[idx]))
            if not tags or self.caption_source(os.path.basename(self.image_paths[idx])) != "web":
                continue
            if measure(build_text(self.trigger, self.category, tags)) <= budget:
                continue
            text = ", ".join(str(tag) for tag in tags)
            while text and measure(build_text(self.trigger, self.category, [text])) > budget:
                text = text[: max(1, int(len(text) * 0.8))]
            self._fitted[idx] = build_text(self.trigger, self.category, [text] if text else [])
        if self._fitted:
            print(f"[dataset] 提示: {len(self._fitted)} 张图片的网页标注超出 "
                  f"{budget} token 预算，已按预算截断（其余图片的标注未改动）")
        return len(self._fitted)

    def get_prompt(self, idx: int) -> str:
        fitted = getattr(self, "_fitted", None)
        if fitted and idx in fitted:
            return fitted[idx]
        return build_text(self.trigger, self.category,
                          self.tags_for(os.path.basename(self.image_paths[idx])))

    def evaluation_indices(self):
        """Fixed training diagnostics, not a held-out validation set."""
        indices = []
        for tag in ("full body", "face close-up"):
            match = next((i for i, p in enumerate(self.image_paths)
                          if any(tag in t.lower()
                                 for t in self.tags_for(os.path.basename(p)))), None)
            if match is not None and match not in indices:
                indices.append(match)
        for i, p in enumerate(self.image_paths):
            if any(ACCESSORY_CLOSEUP_TAG in t.lower()
                   for t in self.tags_for(os.path.basename(p))):
                if i not in indices:
                    indices.append(i)
        return indices or [0]

    def get_item(self, idx: int, deterministic=False) -> dict:
        img_path = self.image_paths[idx]
        img = image_utils.load_image(img_path)          # 读取失败会抛异常并指明路径
        filename = os.path.basename(img_path)
        tags = self.tags_for(filename)
        is_accessory = any(ACCESSORY_CLOSEUP_TAG in t.lower() for t in tags)
        transform = self.eval_transform if deterministic else self.transform
        pixel_values = (self.detail_transform if is_accessory else transform)(img)
        prompt = self.get_prompt(idx)
        return {"pixel_values": pixel_values, "prompt": prompt}


def get_train_dataset(folder: str, resolution: int, image_limit: int = 0,
                      transform=None) -> CharacterDataset:
    """构建训练数据集（fail fast：文件夹缺失/图片为空直接抛异常）。"""
    return CharacterDataset(folder=folder, resolution=resolution,
                            image_limit=image_limit, transform=transform)


def get_train_dataloader(dataset: Dataset, batch_size: int = None, shuffle: bool = True,
                         num_workers: int = None) -> DataLoader:
    """构建训练 DataLoader。

    shuffle=True 时使用 config.SEED 固定的 Generator，保证多次运行
    数据洗牌顺序一致（可复现性，规格 §26）。
    """
    if batch_size is None:
        batch_size = config.TRAIN_BATCH_SIZE
    if num_workers is None:
        num_workers = config.DATALOADER_NUM_WORKERS

    generator = None
    if shuffle:
        import torch
        generator = torch.Generator().manual_seed(config.SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )


if __name__ == "__main__":
    """自检块（PyCharm 直接运行）：验证 captions 解析、图片扫描、prompt 合成与 Dataset 构造。

    仅 CPU 构造，不执行任何训练。
    """
    print("=" * 60)
    print("dataset.py 自检")
    print("=" * 60)

    # 1. captions / 网页描述解析
    folder_dir = os.path.join(config.DATASET_DIR, config.CHARACTER_ID)
    cap_file = os.path.join(folder_dir, "captions.txt")
    caps = load_captions(cap_file)
    print(f"[1] captions 解析: {cap_file}")
    print(f"    共 {len(caps)} 条标注，示例: {list(caps.items())[0] if caps else '（无）'}")
    print(f"[1b] 网页资料: 角色描述 {len(load_marks(folder_dir))} 条 / "
          f"标志图像描述 {len(load_mark_refs(folder_dir))} 条")
    try:
        print(f"[1c] 训练准入检查 validate_dataset: "
              f"{validate_dataset(config.CHARACTER_ID)}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[1c] 训练准入检查未通过: {type(exc).__name__}: {exc}")

    # 2. 图片扫描
    imgs = scan_images(folder_dir)
    print(f"[2] 图片扫描: {folder_dir}")
    print(f"    共 {len(imgs)} 张图片，前 3 张: {[os.path.basename(p) for p in imgs[:3]]}")

    # 3. prompt 合成样例（无逐图标注时回退网页描述，最后才是空标签）
    first = os.path.basename(imgs[0])
    sample = build_text(config.get_trigger(config.CHARACTER_ID),
                        config.get_character_config(config.CHARACTER_ID)["category"],
                        caps.get(first) or merged_marks(load_marks(folder_dir)))
    print(f"[3] prompt 样例: {sample!r}")

    # 4. Dataset 构造（需 torch，仅 CPU）
    try:
        ds = get_train_dataset(folder=config.CHARACTER_ID, resolution=config.RESOLUTION,
                               image_limit=min(2, len(imgs)))
        item = ds[0]
        print(f"[4] Dataset: 共 {len(ds)} 个样本（image_limit=2 截断）")
        print(f"    pixel_values.shape = {tuple(item['pixel_values'].shape)}")
        print(f"    prompt = {item['prompt']!r}")
    except Exception as e:
        print(f"[4] Dataset 构造失败（可能 torch 正在重装或缺失）: {type(e).__name__}: {e}")

    print("=" * 60)
