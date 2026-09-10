"""Optional no-training theme precomputation: python build_theme.py CHARACTER_NAME."""
import json
import os
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
from webapp.server import character_name, characters
from webapp.theme import build_theme, mascot_assets


if __name__ == "__main__":
    name = character_name(sys.argv[1]) if len(sys.argv) == 2 else None
    weights = characters()
    if name not in weights:
        raise SystemExit("用法：python build_theme.py 角色名（角色需已有 final 权重）")
    mascot_assets()
    result = build_theme(name, weights[name], lambda p, phase: print(f"{p:.0f}% {phase}", flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
