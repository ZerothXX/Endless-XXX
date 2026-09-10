"""Single, deterministic weight-selection policy for CLI and web inference."""
from pathlib import Path
import config


def resolve_weight(character, mode=None, explicit_path=None, allow_missing=None):
    mode = mode or config.LORA_SELECTION_MODE
    source = config.LORA_MODEL_SOURCE
    if source not in {"standard", "24"}:
        raise ValueError(f"无效 LORA_MODEL_SOURCE: {source}，请选择 standard 或 24")
    directory = Path(config.TRAIN_MODELS_24_DIR if source == "24" else config.TRAIN_MODELS_DIR)
    if mode in {"final", "best"}:
        candidates = [directory / f"{character}_{mode}.safetensors"]
    elif mode == "auto":
        candidates = [directory / f"{character}_{suffix}.safetensors" for suffix in ("final", "best")]
    elif mode == "explicit":
        value = explicit_path if explicit_path is not None else config.LORA_WEIGHT_PATH
        if not value:
            raise ValueError("explicit 模式需要设置 LORA_WEIGHT_PATH")
        path = Path(str(value).replace("{character}", character))
        if not path.is_absolute():
            path = Path(config.PROJECT_ROOT) / path
        if not path.name.startswith(character + "_") or path.suffix != ".safetensors":
            raise ValueError(f"权重必须使用当前角色前缀 {character}_ 和 .safetensors 后缀：{path}")
        candidates = [path]
    else:
        raise ValueError(f"无效 LORA_SELECTION_MODE: {mode}")
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    if (allow_missing if allow_missing is not None else config.ALLOW_BASE_MODEL_WITHOUT_LORA):
        return None
    raise FileNotFoundError("未找到指定角色权重；不会静默换成其他模型：" + ", ".join(map(str, candidates)))
