"""Run the current root training implementation with the local 24GB profile."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from G24 import config_24


def check_device(cuda):
    if not cuda.is_available():
        raise RuntimeError("24GB 训练需要 CUDA")
    gib = cuda.get_device_properties(cuda.current_device()).total_memory / 1024 ** 3
    if gib < config_24.MIN_VRAM_GIB:
        raise RuntimeError(f"显存 {gib:.1f} GiB，不满足24GB配置的 {config_24.MIN_VRAM_GIB} GiB 门槛；请运行根目录 train.py")


def main():
    import torch
    check_device(torch.cuda)
    import train
    character = train.select_character()
    config_24.apply(config, character)
    out = Path(config.LOGS_DIR) / "training"
    print(f"[24GB] 角色={character}, resolution={config_24.RESOLUTION}, rank={config_24.LORA_RANK}, steps={config_24.MAX_TRAIN_STEPS}")
    return train.run_training(run_root=out)


if __name__ == "__main__":
    main()
