"""GPU 工作进程：只调用项目正式的训练、转化入口。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def event(percent, phase, **extra):
    print("@EVENT " + json.dumps(
        {"percent": round(percent, 1), "phase": phase, **extra},
        ensure_ascii=False), flush=True)


def run(spec):
    name = spec["character"]
    config.CHARACTER_ID = name
    config.TRIGGER_WORD = config.get_trigger(name)
    if spec["kind"] == "generate":
        import test as inference
        config.INPUT_IMAGE = spec["images"]
        event(1, f"图片转化中……（0/{len(spec['images'])}）")
        inference.main()
        folder = Path(config.RESULT_DIR) / f"{name}_result"
        files = [folder / f"{name}_{Path(path).stem}_result.png"
                 for path in spec["images"]]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            raise RuntimeError(f"生成未产出图片: {', '.join(missing)}")
        event(100, "已完成", result={
            "images": [f"/api/results/{name}/{path.name}" for path in files]})
        return

    if spec["kind"] != "train":
        raise ValueError("未知任务")
    import train
    import train_utils
    # 网页短程训练，避免使用默认的完整训练步数。
    config.MAX_TRAIN_STEPS = 80
    config.CONCEPT_TRAIN_STEPS = 80
    config.LOG_EVERY_N_STEPS = 1
    original = train_utils.log_training

    def log(epoch, num_epochs, step, max_steps, *args, **kwargs):
        original(epoch, num_epochs, step, max_steps, *args, **kwargs)
        event(5 + 90 * step / max(1, max_steps),
              f"正在训练角色……（{step}/{max_steps}）")

    train_utils.log_training = log
    try:
        event(1, "正在训练角色……")
        train.main()
    finally:
        train_utils.log_training = original
    from character import semantics
    audit = semantics.audit_switch(name)
    if not audit.get("ready"):
        raise RuntimeError("训练完成，但角色包尚未通过完整性检查")
    event(100, "已完成", result={"character": name})


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
