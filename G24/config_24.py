"""24GB hyperparameters and separate output paths; shared training loop."""
from pathlib import Path
RESOLUTION = 1024
MAX_TRAIN_STEPS = 1500
LORA_RANK = 64
LORA_ALPHA = 64
LEARNING_RATE = 1.2e-4
SAVE_STEPS = 300
EVAL_LOSS_INTERVAL = 50
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16"
TRAIN_CACHE_TEXT_ENCODERS = True
USE_GRADIENT_CHECKPOINTING = True
USE_8BIT_ADAM = True
USE_CPU_OFFLOAD = False
TRAIN_ISOLATED_RUN = True
TRAIN_MAX_LONG = 1088
TRAIN_MAX_AREA = 1024 * 1024
TRAIN_BUCKETS = [(1024, 1024), (1088, 960), (960, 1088), (1024, 896),
                 (896, 1024), (1088, 832), (832, 1088), (1024, 768), (768, 1024)]
MIN_VRAM_GIB = 22  # Nominal 24GB devices; refuse the local 8GB profile.


def apply(target, character):
    """Override the selected character, not a hardcoded role37 preset."""
    for name, value in globals().items():
        if name.isupper() and name != "MIN_VRAM_GIB":
            setattr(target, name, list(value) if isinstance(value, list) else value)
    preset = dict(target.CHARACTER_PRESETS.get(character, {}))
    preset.update(resolution=RESOLUTION, max_train_steps=MAX_TRAIN_STEPS)
    target.CHARACTER_PRESETS = {**target.CHARACTER_PRESETS, character: preset}
    target.CHARACTER_ID = character
    output = Path(target.OUTPUT_24_DIR)
    target.OUTPUT_DIR = str(output)
    target.TRAIN_MODELS_DIR = str(target.TRAIN_MODELS_24_DIR)
    for name, sub in (("CURVES_DIR", "curves"), ("LOGS_DIR", "logs"),
                      ("SEMANTIC_DIR", "semantic"), ("RESULT_DIR", "result")):
        setattr(target, name, str(output / sub))
