"""Bounded local-refinement experiment with visually checked cup surface placement."""
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
import torch
import image_utils
import model_utils
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from vision.accessory_refiner import refine_accessory


def main():
    out = ROOT / "output/experiments" / ("accessory_refine_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"AUDIT_DIR={out}", flush=True)
    with open(out / "run.log", "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            config.USE_CPU_OFFLOAD = True
            config.LORA_SCALE = 0.5
            base, cn = model_utils.load_inference_pipeline(lora_path=config.LORA_WEIGHT_PATH)
            if not cn:
                raise RuntimeError("Local shape test requires ControlNet")
            base.remove_all_hooks()
            pipe = StableDiffusionXLControlNetInpaintPipeline(**base.components)
            del base
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            scene = image_utils.resize_fit(image_utils.load_image(str(ROOT / "input/cup.jpg")), 640)
            reference = image_utils.load_image(str(ROOT / "dataset/37/images/013.png"))
            box = [0.43, 0.49, 0.64, 0.72]
            prompt = ("ch37, ceramic mug, painted four-petal flower emblem, cyan outer border, "
                      "white petals, pink inner petals, cyan teardrops, white circular center, "
                      "glossy ceramic glaze, realistic photograph, fine detail")
            negative = ("girl, person, text, letters, heart, green, yellow, orange, red, rainbow, "
                        "gold, chromatic aberration, blurry, deformed, rectangular patch")
            result, debug = refine_accessory(pipe, scene, reference, box, prompt, negative,
                    torch.Generator("cuda").manual_seed(42))
            result.save(out / "cup_result.png")
            for key, img in debug.items():
                img.save(out / f"{key}.png")
            (out / "parameters.json").write_text(json.dumps({
                "box": box, "box_source": "visually_checked_experiment_not_auto_localization",
                "reference": "dataset/37/images/013.png", "reference_use": "canny_contours_only",
                "prompt": prompt, "negative": negative, "lora": config.LORA_WEIGHT_PATH,
                "lora_scale": 0.5, "seed": 42, "mask": "reference_contour_silhouette",
                "strength": 0.95, "control_scale": 0.85, "resolution": 512, "steps": 30,
            }, indent=2), encoding="utf-8")
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
