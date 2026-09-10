# -*- coding: utf-8 -*-
"""
config.py —— 全项目唯一参数来源（合同文件）

所有路径、模型开关、训练参数、推理参数、VLM 参数、Prompt 模板均集中于此。
后续所有模块（train.py / test.py / dataset.py / image_utils.py 等）一律通过
import config 读取参数，禁止在业务代码中硬编码路径或超参数。

路径全部基于本文件所在目录（PROJECT_ROOT）推导，保证在 PyCharm / 命令行
任意工作目录下运行结果一致。
"""

import os

# =========================
# 基础路径（基于本文件位置推导）
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")      # 数据集根目录（含各角色子目录）
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")       # 预训练模型根目录
INPUT_DIR   = os.path.join(PROJECT_ROOT, "input")        # 推理输入目录（单数命名，见规格 §2）
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "output")       # 所有输出根目录（单数命名，见规格 §2）

# output/ 下五个子目录（规格 §2）
TRAIN_MODELS_DIR = os.path.join(OUTPUT_DIR, "train_models")   # LoRA 权重
CURVES_DIR       = os.path.join(OUTPUT_DIR, "curves")         # 训练曲线图
SEMANTIC_DIR     = os.path.join(OUTPUT_DIR, "semantic")       # Character Semantic Memory
RESULT_DIR       = os.path.join(OUTPUT_DIR, "result")         # 推理结果
LOGS_DIR         = os.path.join(OUTPUT_DIR, "logs")           # 日志
OUTPUT_24_DIR = os.path.join(OUTPUT_DIR, "24")              # 24GB训练独立输出根目录
TRAIN_MODELS_24_DIR = os.path.join(OUTPUT_24_DIR, "train_models")

# =========================
# 预训练模型路径（均为相对 PROJECT_ROOT 的路径字符串）
# =========================
# 基础生成模型：Illustrious XL（SDXL 架构），单文件 safetensors
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors")
# VAE：fp16 修复版单文件
BASE_VAE_PATH = os.path.join(MODEL_DIR, "sdxl-vae-fp16-fix", "sdxl.vae.safetensors")
# ControlNet：canny 版 SDXL，目录形式
CONTROLNET_PATH = os.path.join(MODEL_DIR, "controlnet-canny-sdxl-1.0")
# VLM：Qwen2.5-VL 3B，本地目录
VLM_PATH = os.path.join(MODEL_DIR, "Qwen2.5-VL-3B-Instruct")
# SAM2 本地目录
SAM_PATH = os.path.join(MODEL_DIR, "sam2")
# IP-Adapter 本地目录（当前 models/ 下不存在，默认关闭，见 USE_IP_ADAPTER）
IP_ADAPTER_PATH = os.path.join(MODEL_DIR, "ip-adapter")

# =========================
# 多角色支持（用户要求）
# =========================
# 角色预设：key 为角色文件夹名（如 "37"），value 为该角色的覆盖参数
#   category        角色类别词（如 1girl / cat / object），参与训练 prompt 合成
#   resolution      该角色训练分辨率（覆盖全局 RESOLUTION）
#   max_train_steps 该角色最大训练步数（覆盖全局 MAX_TRAIN_STEPS）
CHARACTER_PRESETS = {
    "37": {
        "category": "1girl",
        "resolution": 640,   # 768 实测 40~50s/步且 WDDM 共享显存溢出致进程被杀，按 OOM 降级阶梯降至 640
        "max_train_steps": 600,  # 可调训练起点；训练 loss 不能证明角色还原达标。
    },
}

# 角色卡片（推理端身份/画风/配饰的"金标准"描述，优先于 VLM 生成的语义记忆）
# ---------------------------------------------------------------------------
# 背景：项目初版用 VLM 从角色图解析记忆（output/semantic/*.json），实测不可靠——
# Qwen2.5-VL 把 37 号角色的粉发/蓝白衣/四叶发饰解析成了 "light blue hair,
# Flower Fairy, light blue dress with pink flower pattern"（严重幻觉），
# 该错误描述被拼进生成 prompt（如 "light blue" 发色），直接抵消了 LoRA 的身份信号。
# 角色身份应来自：触发词(绑定 LoRA) + 本卡片（人工校正的稳定描述）。
# VLM 保留用于：输入主体理解（human/animal/object）与可选结果评价，不再参与角色身份描述。
# 字段：
#   identity_tags   Danbooru 风格身份标签（底模熟知的自然语言，非 "character identity" 空词）
#   quality_tags    质量词（与训练端 TRAIN_QUALITY_TAGS 对齐）
#   style           目标输出画风（character 画风 = 角色所在动漫画风）
#   style_policy    "character"：一律转成角色画风（示例图 1/3：油画/涂鸦 -> 动漫）；
#                   "preserve"：保留输入画风（示例图 2：照片猫保持照片感，只换配色/配饰）
#   accessory       标志配饰（名称/颜色/形状），用于配饰迁移短语
CHARACTER_CARDS = {
    "37": {
        # 人物先保留可见的发型、服饰与结构；使用简洁配饰描述。
        # 非人物保留完整配饰几何；实际长度由两个 SDXL tokenizer 检查。
        "identity_tags": (
            "pale pink hair, pink tips, bangs, gradient hair, short hair, pink eyes, "
            "white blouse, blue jacket, detached sleeves, blue and black corset, "
            "blue ribbon, gold button, choker"
        ),
        "regional_identity_tags": {
            "waist": "cyan camera on orange strap",
            "legs": "blue pleated skirt",
            "feet": "blue ankle boots",
        },
        "core_identity_tags": "pale pink hair, pink tips, bangs, pink eyes",
        "human_required_regions": {
            "torso": "white blouse, blue jacket, blue bow, gold buttons, black choker",
            "arms": "detached sleeves",
        },
        "quality_tags": "masterpiece, best quality, highly detailed",
        "style": "anime style",
        "style_policy": "preserve",
        "human_style_policy": "character",  # 人物以例图 37_1 为准；非人物不改变画风策略。
        "palette": ["blue", "pink", "white"],   # 动物/物品主体用（不注入人际身份标签，避免拟人）
        "accessory": {
            "name": "four-petal emblem",
            "color": "cyan, white and pink",
            "shape": "four rounded petals around a white circular center",
            "description": "four-petal emblem, cyan border, white petals, pink inner petals, cyan teardrops, white center",
            "human_description": "cyan pink four-petal brooch",
            "normal_usage": "shoulder brooch",
            "source_images": ["013.png", "001.png"],
            "source": "visually_reviewed",
        },
        "secondary_accessories": [{
            "name": "cyan compact camera", "color": "cyan with white lens rim",
            "shape": "rectangular camera body with circular dark lens",
            "normal_usage": "waist on orange strap", "source_images": ["012.png", "001.png"],
            "source": "visually_reviewed",
        }],
        # 旧调用方兼容字段；新路径使用 palette + accessory + 主体材质规划。
        "object_extra": (
            "fine detail"
        ),
        "animal_extra": (
            "clear sharp four-leaf flower ornament, detailed fur, "
            "well-defined ornament, fine detail"
        ),
    },
}

# 当前角色 ID：若 dataset/<CHARACTER_ID>/ 文件夹存在则直接使用；
# 否则运行时由 input() 菜单让用户从已有角色文件夹中选择。
CHARACTER_ID = "37"

# 触发词：角色身份由触发词学习（规格 §4.4），此处为函数式模板所需的基础变量
TRIGGER_WORD = f"ch{CHARACTER_ID}"

# 训练 prompt 模板：{trigger} 触发词 / {category} 类别词 / {caption} 短标签
TRAIN_PROMPT_TEMPLATE = "{trigger}, {category}, {caption}"
# 训练质量词（与推理端 quality_tags 对齐，避免"训练从不出现、推理突然出现"的
# 分布偏差；Illustrious 系底模对 masterpiece/best quality 响应显著）
TRAIN_QUALITY_TAGS = "masterpiece, best quality"
# Reviewed sample descriptions supplement training without rewriting source captions.
TRAIN_CAPTION_OVERRIDES = {
    "37": {
        "012.png": ["accessory close-up", "cyan compact camera", "dark circular lens",
                    "white lens rim", "isolated on white background"],
        "013.png": ["accessory close-up", CHARACTER_CARDS["37"]["accessory"]["description"],
                    "isolated on white background"],
    }
}
AUTO_PREPROCESS_DATASET = True
TRAIN_ISOLATED_RUN = True  # 仅隔离预处理数据和审核日志；权重/曲线仍使用统一输出目录。
TRAIN_CACHE_TEXT_ENCODERS = True  # Frozen TE embeddings cached, encoders moved off GPU.
WEB_THEME_VLM_FOR_UNLABELED = True  # 仅参考图缺少标注时调用现有 VLM，不训练额外模型。


def get_trigger(folder: str) -> str:
    """根据角色文件夹名生成触发词（如 folder="37" -> "ch37"）。"""
    return f"ch{folder}"


def get_character_config(folder: str) -> dict:
    """合并全局默认与角色预设，返回该角色的训练配置 dict。

    返回字段：category / resolution / max_train_steps / trigger
    角色预设中缺失的字段回退到全局默认值。
    """
    preset = CHARACTER_PRESETS.get(folder, {})
    return {
        "category": preset.get("category", "1girl"),
        "resolution": preset.get("resolution", RESOLUTION),
        "max_train_steps": preset.get("max_train_steps", MAX_TRAIN_STEPS),
        "trigger": get_trigger(folder),
    }


def get_character_card(folder: str) -> dict:
    """返回角色的"金标准"卡片描述（CHARACTER_CARDS[folder]），不存在时返回空 dict。

    空 dict 表示该角色没有人工校对卡片，调用方回退到 VLM 语义记忆路径。
    """
    return CHARACTER_CARDS.get(folder) or {}


# =========================
# 模块开关（全部可关闭以适配 8GB 显存，规格 §3）
# =========================
USE_VLM = True                 # 视觉语言模型（角色解析 / 主体理解 / 自动标注）
USE_SAM = True                 # SAM2 主体分割
USE_CONTROLNET = True          # ControlNet 结构保持
USE_IP_ADAPTER = False         # IP-Adapter 增强（models/ 下暂无该模型，默认关闭）
ENABLE_RESULT_EVALUATION = False  # 推理结果 VLM 评价
USE_8BIT_ADAM = True           # 8bit Adam 省显存；若 bitsandbytes 不可用，运行时自动降级为普通 AdamW
USE_XFORMERS = False           # 已确认 Windows 下 xformers 与 CUDA wheel 冲突，放弃；
                               # 统一使用 PyTorch 原生 SDPA（scaled_dot_product_attention）
USE_GRADIENT_CHECKPOINTING = True  # 训练时梯度检查点，显著降低显存
USE_CPU_OFFLOAD = False        # 启用 CPU offload（OOM 时的最后降级手段）
ENABLE_COLOR_JITTER = False    # 颜色扰动增强（默认关：角色配色本身就是学习目标，规格 §13.3）
ENABLE_HFLIP = False           # 水平翻转增强（默认关：角色配饰可能有左右不对称语义，规格 §13.3）

# =========================
# 训练参数（规格 §12，面向 RTX 4060 8GB）
# =========================
RESOLUTION = 768                    # 训练分辨率（OOM 时按 768->640->512 逐级下调；37 预设为 640）
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1     # 等效 batch = 1；accum=4 时 768 实测 40~50s/步（8GB 卡溢出共享内存），降为 1 提速
MIXED_PRECISION = "fp16"            # 混合精度
LORA_RANK = 32                      # 身份画得更细：rank 16 -> 32（参数量 23M -> 46M，
                                    # 显存仅 +~90MB，训练时间几乎不变；35 号角色等小型角色可回退 16）
LORA_ALPHA = 32
# G24/config_24.py 仅覆盖高显存超参数，复用同一 UNet LoRA 训练核心。
LEARNING_RATE = 1e-4
LR_SCHEDULER = "constant"           # 身份绑定用恒定 LR（kohya 角色 LoRA 惯例）；
                                    # cosine 在 400 步就衰减到 0，等于后半段没在学（此前欠拟合原因之一）
LR_WARMUP_STEPS = 50
MAX_TRAIN_STEPS = 800               # 步数控制（与 NUM_EPOCHS 二选一，默认用步数；37 预设为 1000）
NUM_EPOCHS = None                   # None 表示不按 epoch 控制，仅由 MAX_TRAIN_STEPS 控制
SAVE_STEPS = 200                    # 每 N 步保存一次 LoRA checkpoint
LOG_EVERY_N_STEPS = 10              # 每 N 步打印一次日志
EVAL_LOSS_INTERVAL = 20             # 每 N 步计算一次"固定时间步评估 loss"（监控真实梯度下降，
                                    # 0=关闭；见 train_utils.eval_loss_fixed_timesteps）
MAX_GRAD_NORM = 1.0                 # 梯度裁剪
SEED = 42                           # 全局随机种子（规格 §26），保证 train/val 划分与训练可复现
TRAIN_IMAGE_LIMIT = 0               # 0=使用全部训练图片；>0 时仅取前 N 张（冒烟测试用）
DATALOADER_NUM_WORKERS = 0          # Windows 下 DataLoader 子进程易出问题，默认 0（主进程加载）
# 训练分辨率桶（整图模式用，宽高比为 64 的倍数；长边 ≤ TRAIN_MAX_LONG、面积 ≤ TRAIN_MAX_AREA）
# 8GB 默认值；G24 版本按 1024 上限覆盖（见 G24/train_24.py）
TRAIN_MAX_LONG = 768
TRAIN_MAX_AREA = 640 * 640
TRAIN_BUCKETS = [
    (640, 640), (704, 576), (576, 704), (768, 512), (512, 768),
    (704, 512), (512, 704), (640, 512), (512, 640),
]

# =========================
# 推理参数
# =========================
INPUT_IMAGE = INPUT_DIR                              # 推理输入，可为单文件或目录（交付默认：批量处理 input/ 全部图片）
NUM_INFERENCE_STEPS = 36
GUIDANCE_SCALE = 7.0
# ---- 生成路径：混合 img2img + ControlNet（推荐，实测效果最好） ----
# 实测结论（消融实验 expD-N/O2/P）：
#   * 纯 ControlNet-canny scale>=0.7：姿势保持 ✓，但输出"朦胧空气感"（ControlNet
#     高 scale 的特征主导把 UNet 细节压平），且身份与细节互相打架；
#   * 纯 ControlNet-canny 0.35~0.45：清晰 ✓ 身份完整 ✓，但姿势被重新演绎；
#   * 纯 img2img strength 0.6~0.68：清晰 ✓ 姿势保留 ✓，但身份只部分迁移
#     （原画的深发/深衣很难被换成角色设计）；
#   * img2img(0.68) + canny(0.35) 混合（StableDiffusionXLControlNetImg2ImgPipeline）：
#     姿势由 init 图像锚定（不再依赖 canny 高 scale），canny 低 scale 保持清晰，
#     LoRA+prompt 提供身份 —— 三赢组合。
USE_HYBRID_IMG2IMG_CONTROLNET = True
IMG2IMG_STRENGTH = 0.64               # 保守重绘起点，须固定种子验证结构与身份的取舍
HUMAN_IMG2IMG_STRENGTH = 0.80         # 角色化起点，尚未完成视觉验收；越高越可能改变双手/姿态。
HUMAN_CN_SCALE = 0.30
HUMAN_CONTROL_GUIDANCE_END = 0.65     # 后段释放原发型/服装边缘，不保证人体结构正确。
HUMAN_LORA_SCALE = 1.0
                                      # 越高身份迁移越强（粉发/蓝白服替换原画），
                                      # 越低原图保留越多。实测 expP: 0.72 时身份最佳
                                      # 且姿势/背景仍完整保留；0.75+ 构图会漂移
                                      # （头部出画，expR1/R3a），不可用）
CONTROLNET_CONDITIONING_SCALE = 0.45  # 结构保留起点，不能替代姿态/手部验收
                                      # +身份自由；结构由 init 图像承担）
# 物品/动物主体的独立参数（实测优化：杯子配饰清晰版 R2b：
#   strength 0.68 + canny 0.40 + object_extra 精确纹样描述；
#   与人物 0.72/0.35 不同——物品不需要"重换装"，偏低强度+偏强边缘
#   让纹样放得稳、画得清）
OBJECT_IMG2IMG_STRENGTH = 0.68
OBJECT_CN_SCALE = 0.40
ANIMAL_IMG2IMG_STRENGTH = 0.70
ANIMAL_CN_SCALE = 0.38
# 纯 ControlNet 路径时的建议值 0.45（保留姿势需 0.7+，但会牺牲清晰度）
SUPPRESS_BACKGROUND_EDGES = False      # 是否用主体 mask 抑制背景 Canny 边缘（False=保留全图结构，背景不丢；
                                       # True=背景边缘置黑自由重绘，主体结构更突出，规格 §9）
PRESERVE_BACKGROUND = False            # Optional SAM mask guard; validate mask selection before enabling.
INFERENCE_RESOLUTION = 640            # 推理分辨率（**短边基准**）：输入等比缩放至短边=640，
                                      # 长边上限 INFERENCE_MAX_LONG（960）——保持原图比例的同时
                                      # 保证主体(脸/配饰)像素量充分（v5 早期"长边=640"导致
                                      # 全身/宽幅画面里面孔只有几十像素、五官扭曲，已弃用）
                                      # G24 用途：24G 机器用 G24/train_24.py 训练后，
                                      # 把本值改为 1024 再运行 test.py 即可（模型通用）
INFERENCE_MAX_LONG = 960              # 推理长边上限（面积 ≈ 640×960 ≈ 0.61MP，
                                      # 与实测可跑的 768² 同级，8GB 安全；
                                      # 24G 建议随 INFERENCE_RESOLUTION=1024 一起
                                      # 改为 1536，即 1024×1536 上限）
INFERENCE_SEED = -1                   # -1=随机；>0 时固定种子复现
LORA_SCALE = 0.9                       # 降低全身训练先验对输入构图的覆盖
NON_HUMAN_LORA_SCALE = 0.7            # 动物/物品主体的 LoRA 融合比例（压低以抑制训练烙进的 "1girl/少女" 先验，规格 §18.2）
NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, blurry, watermark, text"
# 仅显式 character 画风策略使用；preserve 不抑制输入原有画风。
ANIME_NEGATIVE_PROMPT = "photorealistic, realistic, 3d, oil painting, painting, photograph, grayscale, monochrome"
NON_HUMAN_NEGATIVE_PROMPT = "girl, 1girl, human, person, woman, anime girl"   # 动物/物品主体附加负向词，防止生成猫娘/拟人（规格 §18.2）
CANNY_LOW = 50                        # Canny 低阈值
CANNY_HIGH = 150                      # Canny 高阈值
LORA_SELECTION_MODE = "final"  # final / best / auto(final->best) / explicit；CLI 与网页共用。
LORA_MODEL_SOURCE = "standard"  # standard: output/train_models；24: output/24/train_models。
LORA_WEIGHT_PATH = ""  # 仅 explicit 使用；支持项目相对路径及 {character} 占位符。
ALLOW_BASE_MODEL_WITHOUT_LORA = False  # 默认缺权重报错，不随机扫描或静默使用纯底模。
# final 不保证优于 best，需用图像级对照判断；auto 也不会扫描 checkpoint。

# =========================
# VLM 参数
# =========================
VLM_DTYPE = "fp16"              # VLM 加载精度
VLM_LOAD_IN_4BIT = False        # 4bit 量化加载（8GB 显存紧张时可开）
VLM_OFFLOAD_CPU = False         # CPU offload
VLM_MAX_NEW_TOKENS = 512
VLM_TEMPERATURE = 0.2

# =========================
# 环境
# =========================
HF_ENDPOINT = "https://hf-mirror.com"   # HuggingFace 国内镜像（下载模型时使用）


if __name__ == "__main__":
    """PyCharm 直接运行本文件，打印全部配置摘要，便于快速检查参数。"""
    print("=" * 60)
    print("config.py 配置摘要")
    print("=" * 60)
    print(f"[路径] PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"[路径] DATASET_DIR       = {DATASET_DIR}")
    print(f"[路径] MODEL_DIR         = {MODEL_DIR}")
    print(f"[路径] INPUT_DIR         = {INPUT_DIR}")
    print(f"[路径] OUTPUT_DIR        = {OUTPUT_DIR}")
    print(f"[路径] TRAIN_MODELS_DIR  = {TRAIN_MODELS_DIR}")
    print(f"[路径] CURVES_DIR        = {CURVES_DIR}")
    print(f"[路径] SEMANTIC_DIR      = {SEMANTIC_DIR}")
    print(f"[路径] RESULT_DIR        = {RESULT_DIR}")
    print(f"[路径] LOGS_DIR          = {LOGS_DIR}")
    print(f"[模型] BASE_MODEL_PATH   = {BASE_MODEL_PATH}")
    print(f"[模型] BASE_VAE_PATH     = {BASE_VAE_PATH}")
    print(f"[模型] CONTROLNET_PATH   = {CONTROLNET_PATH}")
    print(f"[模型] VLM_PATH          = {VLM_PATH}")
    print(f"[模型] SAM_PATH          = {SAM_PATH}")
    print(f"[模型] IP_ADAPTER_PATH   = {IP_ADAPTER_PATH}")
    print(f"[角色] CHARACTER_ID      = {CHARACTER_ID}  TRIGGER_WORD = {TRIGGER_WORD}")
    print(f"[角色] 预设 = {CHARACTER_PRESETS}")
    print(f"[角色] get_character_config('37') = {get_character_config('37')}")
    print(f"[角色] CHARACTER_CARDS 键 = {sorted(CHARACTER_CARDS.keys())}  "
          f"37 卡片 style_policy={CHARACTER_CARDS.get('37', {}).get('style_policy')}")
    print(f"[开关] USE_VLM={USE_VLM} USE_SAM={USE_SAM} USE_CONTROLNET={USE_CONTROLNET} "
          f"USE_IP_ADAPTER={USE_IP_ADAPTER} USE_8BIT_ADAM={USE_8BIT_ADAM} USE_XFORMERS={USE_XFORMERS}")
    print(f"[开关] GRADIENT_CHECKPOINTING={USE_GRADIENT_CHECKPOINTING} CPU_OFFLOAD={USE_CPU_OFFLOAD} "
          f"COLOR_JITTER={ENABLE_COLOR_JITTER} HFLIP={ENABLE_HFLIP}")
    print(f"[训练] RESOLUTION={RESOLUTION} BATCH={TRAIN_BATCH_SIZE} ACCUM={GRADIENT_ACCUMULATION_STEPS} "
          f"PRECISION={MIXED_PRECISION} RANK={LORA_RANK} ALPHA={LORA_ALPHA}")
    print(f"[训练] LR={LEARNING_RATE} SCHEDULER={LR_SCHEDULER} WARMUP={LR_WARMUP_STEPS} "
          f"STEPS={MAX_TRAIN_STEPS} EPOCHS={NUM_EPOCHS} SEED={SEED} IMG_LIMIT={TRAIN_IMAGE_LIMIT}")
    print(f"[训练] EVAL_LOSS_INTERVAL={EVAL_LOSS_INTERVAL} QUALITY_TAGS={TRAIN_QUALITY_TAGS!r}")
    print(f"[推理] INPUT={INPUT_IMAGE} STEPS={NUM_INFERENCE_STEPS} CFG={GUIDANCE_SCALE} "
          f"CN_SCALE={CONTROLNET_CONDITIONING_SCALE} RES={INFERENCE_RESOLUTION} SEED={INFERENCE_SEED}")
    print(f"[推理] NEG_PROMPT={NEGATIVE_PROMPT[:40]}... CANNY=({CANNY_LOW},{CANNY_HIGH}) "
          f"IMG2IMG_STRENGTH={IMG2IMG_STRENGTH} ANIME_NEG={ANIME_NEGATIVE_PROMPT[:30]}...")
    print(f"[推理] LORA_WEIGHT_PATH={LORA_WEIGHT_PATH} LORA_SCALE={LORA_SCALE}")
    print(f"[VLM]  DTYPE={VLM_DTYPE} 4BIT={VLM_LOAD_IN_4BIT} OFFLOAD={VLM_OFFLOAD_CPU} "
          f"MAX_TOKENS={VLM_MAX_NEW_TOKENS} TEMP={VLM_TEMPERATURE}")
    print(f"[环境] HF_ENDPOINT={HF_ENDPOINT}")
    print("=" * 60)
