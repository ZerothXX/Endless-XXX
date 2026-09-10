"""Fixed-seed portrait tuning, reusing the audited subject observation (no VLM rerun)."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
import config
os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)


def main():
    source = ROOT / "output/experiments/transfer_revision_20260909_194643/revised"
    out = ROOT / "output/experiments" / ("identity_balance_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"AUDIT_DIR={out}", flush=True)
    config.RESULT_DIR = str(out)
    config.IMG2IMG_STRENGTH = 0.70
    config.LORA_SCALE = 1.0
    config.INFERENCE_SEED = 42
    config.USE_CPU_OFFLOAD = True
    config.USE_VLM = False
    import test as inference
    subject = json.loads((source / "human_subject.json").read_text(encoding="utf-8"))
    inference._resolve_input_images = lambda: [str(ROOT / "input/human.png")]
    inference.analyze_subject = lambda client, image: subject
    with open(out / "run.log", "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            inference.main()
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
