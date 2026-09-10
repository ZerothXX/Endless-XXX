# -*- coding: utf-8 -*-
"""
vlm/vlm_utils.py —— VLM 统一封装（Qwen2.5-VL 本地推理）

被 5 个模块依赖的核心：
- character/character_parser.py   角色图片语义解析（CHARACTER_PROMPT）
- vision/subject_analyzer.py      输入主体理解（SUBJECT_PROMPT，S5 使用）
- character/semantic_memory.py    配饰迁移规划（ACCESSORY_PROMPT）
- test.py                         结果评价（EVAL_PROMPT）
- dataset/data_preprocessing.py   captions 自动标注（CAPTION_PROMPT + generate_tags）

职责：
- 加载链（按序尝试并打印降级原因）：fp16 GPU -> 4bit 量化 -> CPU(float32) ->
  全部失败置 self.available=False 并打印告警（对应 config.USE_VLM 全局开关语义）
- analyze_image : 单图 + 系统提示词 -> 结构化 dict（任何异常返回 {} 并打印原因，绝不崩溃）
- generate_tags : 单图 -> 2~4 个英文短标签 list[str]（data_preprocessing 的调用约定）
- parse_json / repair_json : 容错 JSON 解析（规格 §6.2 的"解析->清理->修复->默认"链路）
- get_default_memory : 无 VLM 时的最小语义记忆兜底（规格 §27 VLM 失败回退）

系统提示词常量集中在本文件顶部（config.py 中无对应项，按约定在本文件定义）。

兼容性说明：
- transformers 4.56 的 fast image processor 在 torch<2.3 下会调用不存在的
  torch.compiler.is_compiling() 而崩溃，因此 torch<2.3 时强制 use_fast=False
  走 slow 路径（实测通过）。
- 本模块只做推理，绝不进入 LoRA 训练计算图（规格 §3.4 / §11.3）。
"""

from __future__ import annotations  # Python 3.9 兼容：允许 dict | None 等注解写法

import gc
import json
import os
import re
import sys

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402

import config  # noqa: E402
import image_utils  # noqa: E402
from dataset import load_captions  # noqa: E402

# =========================
# 系统提示词常量（规格 §6.2 / §7 / §19 / §21 / §4.2）
# =========================

# 角色图片语义解析：输出 §6.2 JSON 结构
CHARACTER_PROMPT = (
    "You are an anime character analyst. Analyze the character in this image and "
    "output ONLY one valid JSON object with exactly this structure:\n"
    "{\n"
    '  "character_identity": "short name or description of who this character is",\n'
    '  "visual_style": "art style of the image, e.g. anime illustration",\n'
    '  "color_palette": ["dominant color 1", "dominant color 2", "dominant color 3"],\n'
    '  "hair": {"color": "...", "style": "..."},\n'
    '  "face": {"eye_color": "...", "face_features": "..."},\n'
    '  "clothing": {"description": "...", "dominant_colors": ["...", "..."]},\n'
    '  "signature_accessories": [{"name": "...", "shape": "...", "color": "...", '
    '"location": "...", "semantic_importance": "high"}]\n'
    "}\n"
    "Rules:\n"
    "- Output ONLY the JSON object, no extra text, no markdown fences.\n"
    "- IGNORE any UI elements, HUD overlays, on-screen icons, logos, watermark, and "
    "text. They are NOT part of the character design and must NOT be listed as "
    "accessories.\n"
    "- The image may be a close-up crop showing only part of the character "
    "(e.g. a hair ornament, accessory, or face).\n"
    "- signature_accessories: list the character's WORN mark accessories (distinctive "
    "hair ornaments, hairpins, four-leaf hair clips, badges, brooches, bows). "
    "Use semantic_importance one of high/medium/low. NEVER list UI icons, cameras, "
    "bluetooth symbols, or on-screen overlays as accessories.\n"
    "- color_palette: 3-5 dominant colors of the character design."
)

# 输入主体理解：输出 §7 JSON 结构（subject_type ∈ human|animal|object|unknown）
SUBJECT_PROMPT = (
    "You are a subject analyst for image-to-image character transfer. Analyze the "
    "MAIN SUBJECT of this image and output ONLY one valid JSON object with exactly "
    "this structure:\n"
    "{\n"
    '  "subject_type": "human" or "animal" or "object" or "unknown",\n'
    '  "pose": "...",\n'
    '  "framing": "full body / upper body / bust / close-up / unknown",\n'
    '  "hands": "overlapping hands / folded hands / separate hands / hidden / unknown",\n'
    '  "coverage": {"legs": "bare / covered / out_of_frame / unknown", '
    '"chest": "covered / exposed / unknown"},\n'
    '  "species": "e.g. cat, only for animal",\n'
    '  "object_category": "e.g. cup, bag, only for object",\n'
    '  "material": "e.g. fur, ceramic, fabric",\n'
    '  "style": "visual style of the image, e.g. pixel art / realistic photo / '
    '3D render / anime illustration",\n'
    '  "structure": ["main structural parts"],\n'
    '  "visible_regions": ["which parts are visible"],\n'
    '  "attachment_regions": ["regions where an accessory could be attached, '
    'e.g. head, chest, ear, neck, bag surface"],\n'
    '  "surface_regions": ["visible decorative exterior surfaces, '
    'excluding screens, lenses, openings, controls and other functional areas"]\n'
    "}\n"
    "Rules:\n"
    "- subject_type must be EXACTLY one of: human, animal, object, unknown.\n"
    "- For objects, structure and surface_regions are required.\n"
    "- Objects are not limited to cups and bags. Identify their actual category, material, "
    "visible components and usable decorative surfaces. Preserve existing faces on toys, "
    "but do not infer human anatomy. attachment_regions must be visible usable attachment "
    "points such as a zipper pull or a hanging loop, not an invented body part.\n"
    "- Describe ONLY visible evidence. An upper-body portrait is NOT evidence of standing. "
    "If pose is uncertain use unknown. Do not infer legs or feet outside the frame.\n"
    "- For humans, visible_regions lists visible anatomy only; distinguish covered legs from bare legs. "
    "Preserve overlapping hands and clothing coverage.\n"
    "- Distinguish oil painting from a photograph; realistic painting is still painting.\n"
    "- Output ONLY the JSON object, no extra text, no markdown fences."
)

# 配饰迁移规划：输入主体语义 + 配饰语义，输出 §19 JSON 结构
# 占位符 {{accessory}} / {{subject}} 在调用时替换为 JSON 字符串
ACCESSORY_PROMPT = (
    "You are an accessory adaptation planner for character semantic transfer. "
    "The character has a signature accessory with this semantic representation:\n"
    "{{accessory}}\n"
    "The target subject has this semantic representation:\n"
    "{{subject}}\n"
    "Decide how the character's signature accessory should be RE-EXPRESSED on the "
    "target subject (NOT pasted, NOT cropped, NOT overlaid). Output ONLY one valid "
    "JSON object with exactly this structure:\n"
    "{\n"
    '  "adaptation_type": "...",\n'
    '  "target_region": "...",\n'
    '  "material": "...",\n'
    '  "scale": "small" or "medium" or "large",\n'
    '  "reason": "short explanation"\n'
    "}\n"
    "Guidelines:\n"
    "- human subject: hair accessory / hair clip / brooch / bag charm on head/chest, "
    "glossy plastic material.\n"
    "- Prefer the accessory's normal_usage when that region is visible. A shoulder brooch "
    "should stay a brooch on a visible shoulder, not become a hair clip by default.\n"
    "- animal subject: ear decoration / collar charm / neck accessory on ear/neck, "
    "soft fabric material, keep animal anatomy.\n"
    "- Any object category: adapt to its OBSERVED material, not a cup/bag template. "
    "Textile: embroidered patch or surface pattern. Ceramic: glaze pattern, painted decoration "
    "or surface pattern. Metal/leather/rubber: embossed ornament, painted decoration or surface "
    "pattern. Wood/plastic/glass/paper: painted decoration or surface pattern. "
    "Mixed or unknown material: surface pattern. Preserve existing texture, shape and parts.\n"
    "- For surface decoration, target_region must EXACTLY copy one of subject.surface_regions. "
    "For a bag charm, zipper ornament or small hanging ornament, copy an observed zipper/loop/" 
    "ring/hook from attachment_regions. Never invent a fastener. Preserve function and do not "
    "cover lenses, screens, controls or openings.\n"
    "- The material must suit the subject, e.g. glossy plastic for human hair "
    "accessory, soft fabric for animal collar charm, ceramic for cup surface.\n"
    "- adaptation_type MUST be a concrete NOUN phrase from the list above for the "
    "given subject type. NEVER output vague words like 'attached', 'applied', "
    "'placed', or 'put on'.\n"
    "- Do not turn objects into people or add human garments/anatomy. The original "
    "object remains identifiable; character design is expressed through palette and motifs.\n"
    "- Output ONLY the JSON object, no extra text, no markdown fences."
)

# 结果评价：输出 §21 JSON 结构（0~1 分数）
EVAL_PROMPT = (
    "You are an image quality evaluator for character semantic transfer. Compare "
    "the generated image with the character reference and output ONLY one valid "
    "JSON object with exactly this structure:\n"
    "{\n"
    '  "character_consistency": 0.0,\n'
    '  "subject_preservation": 0.0,\n'
    '  "accessory_adaptation": 0.0,\n'
    '  "overall_quality": 0.0\n'
    "}\n"
    "Rules:\n"
    "- Each score is a float between 0.0 and 1.0.\n"
    "- character_consistency: how well the generated image matches the character "
    "identity (hair, eyes, palette, clothing).\n"
    "- subject_preservation: how well the original subject structure/pose is kept.\n"
    "- accessory_adaptation: how naturally the signature accessory is re-expressed "
    "on the subject.\n"
    "- Output ONLY the JSON object, no extra text, no markdown fences."
)

# captions 自动标注：只输出 2~4 个视角/构图类英文短标签，禁止身份属性（规格 §4.2/§4.3）
# 与 dataset/data_preprocessing.py 的调用约定一致（generate_tags 的默认提示词）
CAPTION_PROMPT = (
    "You are an image captioner for anime character training data. "
    "Output ONLY 2-4 short English tags describing the view angle and "
    "composition of the image, separated by commas. Allowed tags are like "
    "'front view', 'side view', 'back view', 'three-quarter view', "
    "'full body shot', 'upper body', 'face close-up', 'eyes close-up', "
    "'accessory close-up'. NEVER output identity attributes such as hair "
    "color, eye color, clothing, or accessory descriptions."
)


def _torch_major_minor() -> tuple:
    """返回 torch 主/次版本号（如 (2, 2)），解析失败时返回 (0, 0)。"""
    try:
        parts = torch.__version__.split("+")[0].split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


# =========================
# 容错 JSON 解析（规格 §6.2：解析 -> 清理 markdown 围栏 -> 修复 -> 默认字段）
# =========================

def parse_json(text) -> dict:
    """把 VLM 输出文本解析为 dict，按序尝试：
    1. json.loads 直接解析
    2. 去掉 ```json 围栏后再 loads
    3. 调用 repair_json 启发式修复
    全部失败返回 {}（绝不抛异常，调用方由此走默认字段兜底）。
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    text = text.strip()

    # 1. 直接解析（要求结果是 dict）
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2. 去掉 markdown 围栏（```json / ``` 出现在任意位置）后再解析
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    if cleaned != text:
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3. 启发式修复（截取 {..} 区间、去尾逗号、单引号转双引号、补键引号）
    fixed = repair_json(cleaned if cleaned != text else text)
    if isinstance(fixed, dict):
        return fixed

    # 4. 全部失败：返回空 dict（调用方回退默认字段）
    return {}


def repair_json(text) -> dict | None:
    """启发式修复损坏 JSON（尽力而为）：
    1. 截取首个 { 到配平 } 的区间（跳过字符串内的括号）
    2. 去掉尾逗号（逗号后紧跟 } 或 ]）
    3. 单引号转双引号
    4. 无引号键名补全引号（key: -> "key":）
    5. json.loads；仍失败返回 None。
    注意：单引号转双引号会误伤英文缩写（don't -> don"t），属可接受的尽力修复。
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # 1. 定位配平的 { ... } 区间
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    s = text[start:end + 1]

    # 2. 去尾逗号
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # 3. 单引号 -> 双引号
    s = s.replace("'", '"')
    # 4. 补键引号：{key: -> {"key":  （已有引号的键不受影响）
    s = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)

    # 5. 最终解析
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# =========================
# 无 VLM 兜底：最小语义记忆（规格 §27 VLM 失败回退）
# =========================

def get_default_memory(folder: str) -> dict:
    """基于 captions 的最小语义记忆兜底。

    不含 VLM 才能给的身份信息；字段与 §5.1 完全一致，
    值为 "unknown"/空列表，note 字段注明 "default fallback"。
    同时附上 captions 中出现的全部短标签（caption_tags），供下游参考。
    """
    caption_tags = []
    caption_file = os.path.join(config.DATASET_DIR, str(folder), "captions.txt")
    if os.path.isfile(caption_file):
        caps = load_captions(caption_file)
        seen = set()
        for tags in caps.values():
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    caption_tags.append(t)
    return {
        "folder": str(folder),
        "identity": "unknown",
        "appearance": {"hair": "unknown", "eyes": "unknown"},
        "color_palette": {"main": "unknown", "secondary": "unknown", "accent": "unknown"},
        "clothing": {"style": "unknown"},
        "accessories": [],
        "signature_accessories": [],
        "visual_style": "unknown",
        "caption_tags": caption_tags,
        "note": "default fallback",
    }


# =========================
# VLMClient：Qwen2.5-VL 统一封装
# =========================

class VLMClient:
    """Qwen2.5-VL 本地模型封装（只推理，不参与训练）。

    加载链（按序尝试，每步失败打印降级原因）：
        fp16 GPU -> 4bit 量化(BitsAndBytesConfig) -> CPU(float32)
    全部失败 -> self.available=False 并打印告警（对应 config.USE_VLM 开关语义，
    调用方据此走 offline 兜底路径）。
    """

    def __init__(self, model_path: str, dtype: str = "fp16",
                 load_in_4bit: bool = False, offload_cpu: bool = False):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self.available = False
        self.load_error = None

        if dtype == "bf16":
            self._dtype = torch.bfloat16
        elif dtype == "fp16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        self._load(load_in_4bit=load_in_4bit, offload_cpu=offload_cpu)

    # ---------- 加载 ----------

    def _load(self, load_in_4bit: bool, offload_cpu: bool) -> None:
        if not os.path.isdir(self.model_path):
            print(f"[VLM] 告警: 模型目录不存在，VLM 不可用 (available=False): {self.model_path}")
            self.available = False
            return
        if not torch.cuda.is_available():
            print("[VLM] 提示: CUDA 不可用，直接尝试 CPU(float32) 加载")
            offload_cpu = True

        if offload_cpu:
            # 用户显式要求 CPU offload：跳过 GPU 尝试
            self._try_cpu()
            return
        if load_in_4bit:
            # 用户显式要求 4bit：跳过 fp16，先试 4bit 再试 CPU
            if not self._try_4bit():
                self._try_cpu()
            return
        # 默认链：fp16 GPU -> 4bit -> CPU
        if not self._try_fp16_gpu():
            if not self._try_4bit():
                self._try_cpu()

        if not self.available:
            print(f"[VLM] 告警: 全部加载方式失败，VLM 不可用 (available=False)。"
                  f"最后一次错误: {self.load_error}")

    def _load_processor(self, use_fast: bool) -> bool:
        """加载 processor。torch<2.3 时 fast image processor 会崩溃，强制 use_fast=False。"""
        try:
            from transformers import AutoProcessor
            # transformers 4.56 的 fast processor 调用 torch.compiler.is_compiling()，
            # 该 API 在 torch 2.3 才引入；旧 torch 下必须走 slow 路径
            if _torch_major_minor() < (2, 3):
                use_fast = False
            self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=use_fast)
            return True
        except Exception as e:
            print(f"[VLM] 降级原因: processor 加载失败（{type(e).__name__}: {e}）")
            return False

    def _try_fp16_gpu(self) -> bool:
        """尝试 fp16 GPU 加载。"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            if not self._load_processor(use_fast=True):
                return False
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=self._dtype, device_map="cuda:0")
            self.device = torch.device("cuda:0")
            self.model.eval()
            self.available = True
            print(f"[VLM] 已加载 fp16 GPU: {self.model_path}")
            return True
        except Exception as e:
            self._cleanup_after_failed_attempt()
            self.load_error = f"fp16 GPU 加载失败: {type(e).__name__}: {e}"
            print(f"[VLM] 降级原因: {self.load_error}")
            return False

    def _try_4bit(self) -> bool:
        """尝试 4bit 量化 GPU 加载。"""
        try:
            from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
            if not self._load_processor(use_fast=True):
                return False
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, quantization_config=bnb_config, device_map="cuda:0")
            self.device = torch.device("cuda:0")
            self.model.eval()
            self.available = True
            print(f"[VLM] 已加载 4bit 量化 GPU: {self.model_path}")
            return True
        except Exception as e:
            self._cleanup_after_failed_attempt()
            self.load_error = f"4bit 加载失败: {type(e).__name__}: {e}"
            print(f"[VLM] 降级原因: {self.load_error}")
            return False

    def _try_cpu(self) -> bool:
        """尝试 CPU(float32) 加载（最慢但最省显存）。"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            if not self._load_processor(use_fast=True):
                return False
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=torch.float32, device_map="cpu")
            self.device = torch.device("cpu")
            self.model.eval()
            self.available = True
            print(f"[VLM] 已加载 CPU(float32): {self.model_path}（推理会较慢）")
            return True
        except Exception as e:
            self._cleanup_after_failed_attempt()
            self.load_error = f"CPU 加载失败: {type(e).__name__}: {e}"
            print(f"[VLM] 降级原因: {self.load_error}")
            return False

    def _cleanup_after_failed_attempt(self) -> None:
        """一次加载尝试失败后清理残留对象（不报错）。"""
        self.model = None
        self.processor = None
        self.device = None
        self.available = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 推理 ----------

    def _generate_text(self, image, system_prompt: str,
                       max_new_tokens=None, temperature=None) -> str:
        """核心生成：图片 + 系统提示词 -> 模型原始输出文本。

        - image 可以是 PIL.Image 或图片路径（路径会自动读取）
        - 任何异常打印原因并返回 ""（绝不崩溃）
        """
        if not self.available or self.model is None or self.processor is None:
            print("[VLM] 告警: VLM 不可用，无法生成文本")
            return ""
        if max_new_tokens is None:
            max_new_tokens = config.VLM_MAX_NEW_TOKENS
        if temperature is None:
            temperature = config.VLM_TEMPERATURE

        try:
            from PIL import Image as PILImage
            if isinstance(image, str):
                image = image_utils.load_image(image)  # 路径 -> PIL（读取失败抛异常，由外层捕获）
            if not isinstance(image, PILImage.Image):
                raise ValueError(f"image 参数必须是 PIL.Image 或图片路径，收到: {type(image)}")

            # Qwen2.5-VL 对话格式（规格要求）：content=[{type:image},{type:text}]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": system_prompt},
                ],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # 手动提取图像列表（本环境 transformers 4.56 无 process_vision_info，qwen_vl_utils 未安装）
            images = [c["image"] for c in messages[0]["content"] if c.get("type") == "image"]
            inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
            inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.no_grad():
                sampling = {"temperature": float(temperature), "top_p": 0.9} if float(temperature) > 0 else {}
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=float(temperature) > 0,
                    **sampling,
                    use_cache=True,
                )
            # 去掉输入部分，只保留新生成的 token
            generated = generated[:, inputs["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            return raw
        except Exception as e:
            print(f"[VLM] 告警: 图像推理失败，返回空文本。原因: {type(e).__name__}: {e}")
            return ""

    def analyze_image(self, image, system_prompt: str,
                      max_new_tokens=None, temperature=None) -> dict:
        """对单张图片执行系统提示词分析，返回结构化 dict（规格 §6.2 等）。

        输出解析失败/推理异常时打印原因并返回 {}，绝不崩溃（调用方走默认字段兜底）。
        """
        raw = self._generate_text(image, system_prompt, max_new_tokens, temperature)
        if not raw:
            return {}
        result = parse_json(raw)
        if not result:
            print(f"[VLM] 告警: 输出无法解析为 JSON，返回空 dict。原文片段: {raw[:150]!r}")
        return result

    def generate_tags(self, image_path: str, system_prompt: str = None) -> list:
        """生成 2~4 个视角/构图类英文短标签 list[str]。

        dataset/data_preprocessing.py 的调用约定：
            client.generate_tags(image_path, system_prompt=...) -> list[str]
        未指定提示词时使用 CAPTION_PROMPT（只输出视角/构图标签，禁止身份属性）。
        """
        if system_prompt is None:
            system_prompt = CAPTION_PROMPT
        raw = self._generate_text(image_path, system_prompt)
        if not raw:
            return []
        tags = []
        for part in re.split(r"[,;]|\n+", raw):
            t = part.strip().strip(".'\" ").lower()
            if t and t not in tags:
                tags.append(t)
        return tags[:4]

    # ---------- 卸载 ----------

    def unload(self) -> None:
        """释放模型与 processor，清空 CUDA 缓存（供后续冒烟训练腾出显存）。"""
        self.model = None
        self.processor = None
        self.device = None
        self.available = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VLM] 已卸载模型并清空 CUDA 缓存")


if __name__ == "__main__":
    """自检块（PyCharm 直接运行）：不加载模型，仅验证解析器与提示词常量。"""
    print("=" * 60)
    print("vlm/vlm_utils.py 自检")
    print("=" * 60)

    # 1. repair_json / parse_json 单元测试
    cases = [
        ("markdown 围栏", "```json\n{\"a\": 1}\n```", {"a": 1}),
        ("围栏+前后杂文本", "Here is the result: ```json {\"name\": \"four leaf\", \"shape\": \"four leaf\"} ``` done", {"name": "four leaf", "shape": "four leaf"}),
        ("尾逗号+单引号", "{'color_palette': ['pink', 'blue',],}", {"color_palette": ["pink", "blue"]}),
        # 注意：repair_json 只补键引号，不补值引号（值必须是合法 JSON 字面量）
        ("无引号键名", "{hair: \"pink\", eyes: \"blue\"}", {"hair": "pink", "eyes": "blue"}),
        ("彻底坏输入", "not json at all {{{", {}),
    ]
    for name, text, expect in cases:
        r = repair_json(text) if name != "彻底坏输入" else None
        p = parse_json(text)
        ok = (p == expect)
        print(f"[1] {name}: repair_json={'None' if r is None else r} | parse_json={p} | {'OK' if ok else 'FAIL'}")

    # 2. 提示词常量检查
    print(f"[2] CHARACTER_PROMPT 长度={len(CHARACTER_PROMPT)} 含 signature_accessories={'signature_accessories' in CHARACTER_PROMPT}")
    print(f"    SUBJECT_PROMPT 长度={len(SUBJECT_PROMPT)} 含 subject_type={'subject_type' in SUBJECT_PROMPT}")
    print(f"    ACCESSORY_PROMPT 长度={len(ACCESSORY_PROMPT)} 占位符={'{{accessory}}' in ACCESSORY_PROMPT and '{{subject}}' in ACCESSORY_PROMPT}")
    print(f"    EVAL_PROMPT 长度={len(EVAL_PROMPT)} 含 overall_quality={'overall_quality' in EVAL_PROMPT}")
    print(f"    CAPTION_PROMPT 长度={len(CAPTION_PROMPT)} 含 face close-up={'face close-up' in CAPTION_PROMPT}")

    print("=" * 60)
