"""Human-only ablation. Never changes the default weight or style policy."""
import contextlib
import copy
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
os.environ.setdefault("USE_TF", "0")
import config


def main():
    import torch
    import image_utils
    import model_utils
    from character.semantic_memory import AccessoryAdaptationPlanner

    out = ROOT / "output/experiments" / ("human_recovery_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"AUDIT_DIR={out}", flush=True)
    weight = ROOT / "备份/output/train_models/37_final.safetensors"
    previous = json.loads((ROOT / "output/experiments/weight_comparison_20260910_013901/old/human.json").read_text(encoding="utf-8"))
    manifest = {"status": "running", "weight": str(weight),
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
                "seed": 42, "steps": 30, "cfg": 7, "lora_scale": 1.0,
                "note": "Human-only diagnostic. Anime arm deliberately relaxes style preservation; not a default change.",
                "records": []}
    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    config.USE_CPU_OFFLOAD = True
    with (out / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            pipe, cn = model_utils.load_inference_pipeline(lora_path=str(weight))
            if not cn:
                raise RuntimeError("ControlNet unavailable; refuse an invalid comparison")
            pipe.set_adapters("default", adapter_weights=1.0)
            scene = image_utils.resize_fit(image_utils.load_image(str(ROOT / "input/human.png")), 640, max_long=1024)
            edge = image_utils.extract_canny(scene, 50, 150)
            planner = AccessoryAdaptationPlanner()
            card = copy.deepcopy(config.get_character_card("37"))
            card["human_style_policy"] = "preserve"
            subject = previous["subject"]
            plan = planner.plan(subject, card["accessory"])
            def prompt():
                return planner.build_card_transfer_prompt(subject, plan, card, "ch37", "1girl",
                            tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2)
            corrected = prompt()
            card["style_policy"] = "character"
            card["human_style_policy"] = "character"
            anime = prompt()
            cases = [("01_previous_prompt", previous["prompt"], .70, .45, 1.0),
                     ("02_identity_first", corrected, .70, .45, 1.0),
                     ("03_more_redraw", corrected, .82, .25, .65),
                     ("04_anime_diagnostic", anime, .82, .25, .65)]
            for name, positive, strength, control, end in cases:
                print(f"START {name}: {positive}", flush=True)
                record = {"name": name, "prompt": positive, "negative": previous["negative"],
                          "strength": strength, "control_scale": control, "control_end": end,
                          "width": scene.width, "height": scene.height,
                          "token_counts": [len(t(positive)["input_ids"]) for t in (pipe.tokenizer, pipe.tokenizer_2)]}
                result = pipe(prompt=positive, negative_prompt=previous["negative"], image=scene,
                              control_image=edge, strength=strength, controlnet_conditioning_scale=control,
                              control_guidance_end=end, width=scene.width, height=scene.height,
                              num_inference_steps=30, guidance_scale=7.0,
                              generator=torch.Generator("cuda").manual_seed(42)).images[0]
                result.save(out / f"{name}.png")
                manifest["records"].append(record)
                save()
                print(f"COMPLETE {name}", flush=True)
    manifest["status"] = "complete"
    save()
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
