"""One GPU task per child process; process exit releases all model memory."""
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
import config


def event(percent, phase, **extra):
    print("@EVENT " + json.dumps({"percent": percent, "phase": phase, **extra}, ensure_ascii=False), flush=True)


def run(spec):
    kind, name = spec["kind"], spec["character"]
    config.CHARACTER_ID = name
    if kind == "theme":
        from webapp.theme import build_theme
        result = build_theme(name, spec["weight"], event)
        event(100, "主题就绪", result=result)
        return
    out = Path(spec["directory"])
    for attr in ("RESULT_DIR", "TRAIN_MODELS_DIR", "CURVES_DIR", "LOGS_DIR"):
        Path(getattr(config, attr)).mkdir(parents=True, exist_ok=True)
    config.USE_CPU_OFFLOAD = True
    config.AUTO_PREPROCESS_DATASET = False
    event(2, "准备本地模型")
    if kind == "generate":
        import test as inference
        config.LORA_WEIGHT_PATH = spec["weight"]
        config.LORA_SELECTION_MODE = "explicit"  # Pin the model selected for this queued task.
        config.ALLOW_BASE_MODEL_WITHOUT_LORA = False
        config.INFERENCE_SEED = spec.get("seed", 42)
        config.ENABLE_RESULT_EVALUATION = False
        inference._resolve_input_images = lambda: spec["images"]
        original_loader = inference.model_utils.load_inference_pipeline
        count = len(spec["images"])
        class ProgressPipeline:
            def __init__(self, pipe):
                self.pipe, self.index = pipe, 0
            def __getattr__(self, key):
                return getattr(self.pipe, key)
            def __call__(self, **kwargs):
                def step(pipe, i, timestep, callback_kwargs):
                    total = max(1, pipe.num_timesteps)
                    event(10 + 88 * (self.index + (i + 1) / total) / count,
                          f"转化图片 {self.index + 1}/{count} · {i + 1}/{total}")
                    return callback_kwargs
                kwargs["callback_on_step_end"] = step
                result = self.pipe(**kwargs)
                self.index += 1
                return result
        def loader(*args, **kwargs):
            pipe, control = original_loader(*args, **kwargs)
            event(10, "模型就绪")
            return ProgressPipeline(pipe), control
        inference.model_utils.load_inference_pipeline = loader
        inference.main()
        files = [Path(config.RESULT_DIR) / f"{Path(p).stem}_result.png" for p in spec["images"]]
        if not files or not all(p.is_file() for p in files):
            raise RuntimeError("生成未产出图片")
        event(100, "转化完成", result={"images": [f"/outputs/result/{p.name}" for p in files], "weight": spec["weight"]})
    else:
        config.USE_CPU_OFFLOAD = False
        config.LOG_EVERY_N_STEPS = 1
        cfg = dict(config.get_character_config(name))
        config.CHARACTER_PRESETS[name] = {"category": cfg["category"], "resolution": spec["resolution"],
                                          "max_train_steps": spec["steps"]}
        captions = ROOT / "dataset" / name / "captions.txt"
        if not captions.exists():
            from vlm.vlm_utils import VLMClient
            from webapp.theme import image_files
            client = VLMClient(config.VLM_PATH, dtype=config.VLM_DTYPE,
                               load_in_4bit=config.VLM_LOAD_IN_4BIT, offload_cpu=config.VLM_OFFLOAD_CPU)
            if not client.available:
                raise RuntimeError("缺少 captions.txt 且自动标注模型不可用，请上传标签文件后重试")
            lines = []
            for path in image_files(captions.parent / "images"):
                result = client.analyze_image(str(path), 'Return JSON {"tags":["short visual tags"]}. Describe viewpoint, framing, hair, clothes, colors; identify isolated accessories without inventing a person.', temperature=0)
                tags = result.get("tags", []) if isinstance(result, dict) else []
                if not tags or not all(isinstance(t, str) for t in tags):
                    raise RuntimeError(f"自动标注失败：{path.name}")
                lines.append(path.name + ", " + ", ".join(t.replace("\n", " ") for t in tags))
            captions.write_text("\n".join(lines), encoding="utf-8")
            client.unload()
            del client
        import train
        import train_utils
        original_log = train_utils.log_training
        def log(epoch, num_epochs, step, max_steps, *args, **kwargs):
            original_log(epoch, num_epochs, step, max_steps, *args, **kwargs)
            event(10 + 88 * step / max_steps, f"训练 {step}/{max_steps}")
        train_utils.log_training = log
        train.main()
        weight = Path(config.TRAIN_MODELS_DIR) / f"{name}_final.safetensors"
        if not weight.exists():
            raise RuntimeError("训练结束但缺少最终权重")
        result = {"weight": str(weight), "images": [f"/outputs/curves/{p.name}" for p in
                  [Path(config.CURVES_DIR) / f"{name}_{s}.png" for s in ("loss_by_step", "loss_by_epoch", "learning_rate")] if p.exists()],
                  "precompute_theme": spec.get("precompute_theme", True)}
        event(100, "新角色添加完成", result=result)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
