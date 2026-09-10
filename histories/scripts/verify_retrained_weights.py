"""Compare explicit old/new weights under identical prompts, seeds and settings.

Usage: python histories/scripts/verify_retrained_weights.py NEW_WEIGHT
Text-only probes are labeled separately; they do not test input-shape preservation.
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
os.environ.setdefault("USE_TF", "0")
import config
os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Provide the explicit new .safetensors weight path")
    new = Path(sys.argv[1]).resolve(strict=True)
    old = ROOT / "output/train_models/37_final.safetensors"
    out = ROOT / "output/experiments" / ("weight_comparison_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"AUDIT_DIR={out}", flush=True)
    import torch
    from diffusers import StableDiffusionXLPipeline
    import image_utils
    import model_utils
    from character.semantic_memory import AccessoryAdaptationPlanner
    from character.transfer_constraints import composition_constraints
    config.USE_CPU_OFFLOAD = True
    card = config.get_character_card("37")
    planner = AccessoryAdaptationPlanner()
    records = []
    manifest = {"status": "running", "seed": 42, "steps": 30, "guidance_scale": 7.0,
                "weights": {label: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for label, path in (("old", old), ("new", new))},
                "note": "512px screening, not full-resolution acceptance. Same cached subject observations for both arms; semantics are not re-evaluated by VLM. Text-only probes do not validate image preservation."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with open(out / "run.log", "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            hybrid, cn = model_utils.load_inference_pipeline(lora_path=str(old))
            if not cn:
                raise RuntimeError("Weight comparison requires ControlNet, refusing fallback")
            for arm, weight in (("old", old), ("new", new)):
                target = out / arm
                target.mkdir()
                if arm == "new":
                    hybrid.unload_lora_weights()
                    hybrid.load_lora_weights(str(weight.parent), weight_name=weight.name,
                                             adapter_name="default", local_files_only=True)
                for stem, filename, cache in (
                    ("human", "human.png", ROOT / "output/experiments/transfer_revision_20260909_194643/revised/human_subject.json"),
                    ("cup", "cup.jpg", ROOT / "output/experiments/transfer_revision_20260909_194643/revised/cup_subject.json"),
                    ("cat", "cat.png", ROOT / "output/result/cat_subject.json"),
                ):
                    subject = json.loads(cache.read_text(encoding="utf-8"))
                    plan = planner.plan(subject, card["accessory"])
                    prompt = planner.build_card_transfer_prompt(subject, plan, card, "ch37", "1girl",
                                    tokenizer=hybrid.tokenizer, tokenizer_2=hybrid.tokenizer_2)
                    _, negative_parts = composition_constraints(subject)
                    negative_parts.append(config.NEGATIVE_PROMPT)
                    if stem != "human":
                        negative_parts.append(config.NON_HUMAN_NEGATIVE_PROMPT)
                    negative = ", ".join(negative_parts)
                    strength, control, scale = {"human": (0.70, 0.45, 1.0),
                                               "cup": (0.68, 0.4, 0.7),
                                               "cat": (0.7, 0.38, 0.7)}[stem]
                    hybrid.set_adapters("default", adapter_weights=scale)
                    scene = image_utils.resize_fit(image_utils.load_image(str(ROOT / "input" / filename)),
                                                    512, max_long=768)
                    result = hybrid(prompt=prompt, negative_prompt=negative, image=scene,
                        control_image=image_utils.extract_canny(scene, 50, 150), strength=strength,
                        controlnet_conditioning_scale=control, width=scene.width, height=scene.height,
                        num_inference_steps=30, guidance_scale=7.0,
                        generator=torch.Generator("cuda").manual_seed(42)).images[0]
                    result.save(target / f"{stem}.png")
                    record = {"kind": "image_transfer", "arm": arm, "sample": stem,
                              "input": str(ROOT / "input" / filename), "subject_source": str(cache),
                              "input_sha256": hashlib.sha256((ROOT / "input" / filename).read_bytes()).hexdigest(),
                              "subject": subject, "plan": plan, "prompt": prompt, "negative": negative,
                              "strength": strength, "control_scale": control, "lora_scale": scale,
                              "width": scene.width, "height": scene.height}
                    records.append(record)
                    (target / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    print(f"TRANSFER_COMPLETE {arm}/{stem}", flush=True)

            # Reuse components only after removing hooks; no two active offload owners.
            hybrid.remove_all_hooks()
            components = {k: v for k, v in hybrid.components.items() if k != "controlnet"}
            del hybrid
            pipe = StableDiffusionXLPipeline(**components)
            del components
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            for arm, weight in (("old", old), ("new", new)):
                pipe.unload_lora_weights()
                pipe.load_lora_weights(str(weight.parent), weight_name=weight.name,
                                       adapter_name="default", local_files_only=True)
                probes = {
                    "camera_012": "ch37, accessory, cyan compact camera, dark circular lens, white lens rim, isolated on white background",
                    "emblem_013": "ch37, accessory, " + card["accessory"]["description"] + ", isolated on white background",
                }
                for name, material, surface in (("backpack", "canvas", "front panel"),
                                                  ("headphones", "plastic", "outer earcup")):
                    subject = {"subject_type": "object", "object_category": name, "material": material,
                               "surface_regions": [surface], "style": "photo"}
                    probes[name] = planner.build_card_transfer_prompt(subject, planner.plan(subject, {}),
                            card, "ch37", tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2).replace(
                                "original shape and parts, ", "")
                for name, prompt in probes.items():
                    scale = 1.0 if name in {"camera_012", "emblem_013"} else 0.7
                    pipe.set_adapters("default", adapter_weights=scale)
                    negative = config.NEGATIVE_PROMPT + ", " + config.NON_HUMAN_NEGATIVE_PROMPT
                    result = pipe(prompt=prompt, negative_prompt=negative, width=512, height=512,
                                  num_inference_steps=30, guidance_scale=7.0,
                                  generator=torch.Generator("cuda").manual_seed(42)).images[0]
                    result.save(out / arm / f"probe_{name}.png")
                    record = {"kind": "text_only_probe_NOT_input_preservation", "arm": arm,
                              "sample": name, "prompt": prompt, "negative": negative, "lora_scale": scale,
                              "width": 512, "height": 512}
                    records.append(record)
                    (out / arm / f"probe_{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    print(f"PROBE_COMPLETE {arm}/{name}", flush=True)
    manifest.update(status="complete", records=records)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
