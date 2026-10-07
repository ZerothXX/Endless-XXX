"""Isolated training lifecycle shared by the standard and 24GB CLI entries."""
import contextlib
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import config


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        return self.log.write(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_isolated(trainer, character, run_root=None):
    """Stage data before preprocessing; retain source hashes and restore config on exit.

    网页（web/）写入的角色描述文件（marks.txt / mark_refs.txt）与 captions.txt 一样
    属于训练输入，必须一起复制到隔离目录：漏掉它们会让训练在隔离副本里看不到
    用户在网页上写的描述，产出与网页流程不一致的权重。
    """
    root = Path(config.PROJECT_ROOT)
    source = Path(config.DATASET_DIR) / character
    if not (source / "images").is_dir():
        raise FileNotFoundError(source / "images")
    parent = Path(run_root or Path(config.LOGS_DIR) / "training")
    out = parent / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    attrs = ("DATASET_DIR", "OUTPUT_DIR", "TRAIN_MODELS_DIR", "CURVES_DIR",
             "LOGS_DIR", "SEMANTIC_DIR", "RESULT_DIR", "CHARACTER_ID")
    previous = {key: getattr(config, key) for key in attrs}
    staged_files = ("captions.txt", getattr(config, "MARKS_FILE", "marks.txt"),
                    getattr(config, "MARK_REFS_FILE", "mark_refs.txt"))
    originals = [p for p in (source / "images").iterdir() if p.is_file()]
    originals += [source / name for name in staged_files]
    before = {str(p.resolve()): digest(p) for p in originals if p.is_file()}
    manifest = {"status": "preparing", "character": character,
                "started_at": datetime.now().isoformat(), "source_hashes": before,
                "note": "Fresh UNet LoRA training; shared configured outputs; source data is staged."}
    try:
        staged = out / "dataset" / character
        shutil.copytree(source / "images", staged / "images")
        for name in staged_files:
            if (source / name).is_file():
                shutil.copy2(source / name, staged / name)
        shutil.copy2(Path(config.DATASET_DIR) / "data_preprocessing.py", out / "dataset/data_preprocessing.py")
        config.DATASET_DIR = str(out / "dataset")
        config.CHARACTER_ID = character
        snapshot = out / "source"
        snapshot.mkdir()
        for relative in ("config.py", "train.py", "training_run.py", "dataset.py", "train_utils.py",
                         "model_utils.py", "image_utils.py", "web_inputs.py",
                         "character/package.py", "character/semantics.py",
                         "G24/config_24.py", "G24/train_24.py"):
            path = root / relative
            if path.is_file():
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        manifest.update(status="running", config={k: v for k, v in vars(config).items() if k.isupper()})
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train] 独立运行目录: {out}", flush=True)
        with open(out / "run.log", "w", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                trainer()
        weight = Path(config.TRAIN_MODELS_DIR) / f"{character}_final.safetensors"
        if not weight.is_file():
            raise RuntimeError("训练返回但未生成 final 权重")
        manifest.update(status="complete", weight_directory=str(weight.parent),
                        weights={p.name: digest(p) for p in weight.parent.glob(f"{character}_*.safetensors")})
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for key, value in previous.items():
            setattr(config, key, value)
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["originals_unchanged"] = all(Path(p).is_file() and digest(p) == sha for p, sha in before.items())
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
