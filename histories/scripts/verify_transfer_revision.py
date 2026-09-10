"""Reproducible real-model A/B: same input, weights, seed and resolution.

Runs two representative inputs; outputs to a fresh directory, never result/.
The old arm recreates the recorded pre-fix settings, the new arm uses test.py.
No training or source-image edits. Run with the project's DL Python environment.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import config


def main():
    out = ROOT / "output" / "experiments" / ("transfer_revision_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"AUDIT_DIR={out}", flush=True)
    config.RESULT_DIR = str(out / "revised")
    config.INFERENCE_SEED = 42
    # Explicit final weight: isolate inference changes from checkpoint selection.
    config.LORA_WEIGHT_PATH = str(ROOT / "output/train_models/37_final.safetensors")
    config.USE_CPU_OFFLOAD = True
    config.ENABLE_RESULT_EVALUATION = False
    import test as inference
    import torch
    inputs = [str(ROOT / "input/human.png"), str(ROOT / "input/cup.jpg")]
    inference._resolve_input_images = lambda: inputs

    # Reuse the real pipeline loaded by main for the baseline after revised outputs.
    original_loader = inference.model_utils.load_inference_pipeline
    holder = {}
    def capture(*args, **kwargs):
        pipe, cn = original_loader(*args, **kwargs)
        holder["pipe"] = pipe
        if not cn:
            raise RuntimeError("A/B requires the local ControlNet; refusing an unlabelled fallback")
        return pipe, cn
    inference.model_utils.load_inference_pipeline = capture
    with open(out / "run.log", "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            inference.main()
            pipe = holder["pipe"]
            old_dir = out / "baseline"
            old_dir.mkdir()
            for path in inputs:
                stem = Path(path).stem
                record = (ROOT / "output/result" / f"{stem}_prompt.txt").read_text(encoding="utf-8")
                prompt = record.split("--- 最终 Prompt（与生成调用完全一致） ---")[1].split("--- Negative Prompt ---")[0].strip()
                negative = record.split("--- Negative Prompt ---")[1].split("--- 超参摘要 ---")[0].strip()
                img = inference.image_utils.resize_fit(inference.image_utils.load_image(path),
                    config.INFERENCE_RESOLUTION, max_long=config.INFERENCE_MAX_LONG)
                strength, cn, scale = (0.75, 0.35, 1.15) if stem == "human" else (0.68, 0.40, 0.7)
                pipe.set_adapters("default", adapter_weights=scale)
                result = pipe(prompt=prompt, negative_prompt=negative, image=img,
                    control_image=inference.image_utils.extract_canny(img, config.CANNY_LOW, config.CANNY_HIGH),
                    width=img.width, height=img.height, strength=strength,
                    controlnet_conditioning_scale=cn, num_inference_steps=config.NUM_INFERENCE_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE, generator=torch.Generator("cuda").manual_seed(42)).images[0]
                result.save(old_dir / f"{stem}_result.png")
                (old_dir / f"{stem}_generation.json").write_text(json.dumps({
                    "prompt": prompt, "negative_prompt": negative, "seed": 42,
                    "strength": strength, "controlnet_conditioning_scale": cn, "lora_scale": scale,
                    "lora_path": config.LORA_WEIGHT_PATH, "input": path,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest = {"seed": 42, "weights_sha256": hashlib.sha256(
                Path(config.LORA_WEIGHT_PATH).read_bytes()).hexdigest(),
                "torch": torch.__version__, "inputs": inputs, "note":
                "New arm includes corrected style preservation; compare structure and symbol, not just style."}
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
