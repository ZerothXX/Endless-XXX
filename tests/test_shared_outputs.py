import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import config
from weight_selection import resolve_weight
from webapp.server import create_app, characters
from webapp import worker


class SharedOutputTests(unittest.TestCase):
    def test_model_source_is_independent_of_final_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            standard, high = Path(tmp) / "train_models", Path(tmp) / "24/train_models"
            for directory in (standard, high):
                directory.mkdir(parents=True)
                for suffix in ("final", "best"):
                    (directory / f"37_{suffix}.safetensors").write_bytes(b"fixture")
            with patch.multiple(config, TRAIN_MODELS_DIR=str(standard), TRAIN_MODELS_24_DIR=str(high),
                                ALLOW_BASE_MODEL_WITHOUT_LORA=False):
                for source, directory in (("standard", standard), ("24", high)):
                    with patch.object(config, "LORA_MODEL_SOURCE", source):
                        for mode in ("final", "best"):
                            self.assertEqual(resolve_weight("37", mode=mode), str((directory / f"37_{mode}.safetensors").resolve()))
                (high / "37_final.safetensors").unlink()
                with patch.object(config, "LORA_MODEL_SOURCE", "24"):
                    with self.assertRaises(FileNotFoundError):
                        resolve_weight("37", mode="final")
                    self.assertEqual(resolve_weight("37", mode="auto"), str((high / "37_best.safetensors").resolve()))

    def test_selection_modes_and_no_accidental_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final, best = [root / f"37_{suffix}.safetensors" for suffix in ("final", "best")]
            best.write_bytes(b"best")
            (root / "37_step_999.safetensors").write_bytes(b"checkpoint")
            with patch.multiple(config, TRAIN_MODELS_DIR=tmp, PROJECT_ROOT=tmp,
                                LORA_SELECTION_MODE="final", ALLOW_BASE_MODEL_WITHOUT_LORA=False):
                with self.assertRaises(FileNotFoundError):
                    resolve_weight("37")
                self.assertEqual(resolve_weight("37", mode="auto"), str(best.resolve()))
                final.write_bytes(b"final")
                self.assertEqual(resolve_weight("37"), str(final.resolve()))
                self.assertEqual(resolve_weight("37", mode="best"), str(best.resolve()))
                self.assertEqual(resolve_weight("37", mode="explicit", explicit_path="{character}_best.safetensors"), str(best.resolve()))
                with self.assertRaises(ValueError):
                    resolve_weight("other", mode="explicit", explicit_path=str(final))
                with self.assertRaises(FileNotFoundError):
                    resolve_weight("37", mode="explicit", explicit_path="37_missing.safetensors")

    def test_web_registry_uses_same_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset/other/images").mkdir(parents=True)
            (root / "other_final.safetensors").write_bytes(b"final")
            with patch("webapp.server.ROOT", root), patch.multiple(config, TRAIN_MODELS_DIR=tmp, LORA_SELECTION_MODE="final"):
                self.assertEqual(characters(), {"other": resolve_weight("other")})

    def test_result_serving_restricted_to_image_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a_result.png").write_bytes(b"fixture")
            (root / "secret.json").write_text("{}")
            with patch.multiple(config, RESULT_DIR=tmp, CURVES_DIR=tmp):
                client = create_app(start_jobs=False).test_client()
                with client.get("/outputs/result/a_result.png") as response:
                    self.assertEqual(response.status_code, 200)
                self.assertEqual(client.get("/outputs/result/secret.json").status_code, 404)
                self.assertEqual(client.get("/outputs/weights/a_result.png").status_code, 404)
                self.assertEqual(client.get("/outputs/result/../config.py").status_code, 404)

    def test_worker_generation_and_training_share_config_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = {key: str(root / sub) for key, sub in (("RESULT_DIR", "result"), ("TRAIN_MODELS_DIR", "train_models"),
                    ("CURVES_DIR", "curves"), ("LOGS_DIR", "logs"))}
            role = root / "dataset/role"
            role.mkdir(parents=True)
            (role / "captions.txt").write_text("image.png, tags")
            events = []
            def generate():
                self.assertEqual(config.RESULT_DIR, dirs["RESULT_DIR"])
                self.assertEqual(config.LORA_SELECTION_MODE, "explicit")
                (Path(config.RESULT_DIR) / "image_result.png").write_bytes(b"new")
            def train():
                self.assertEqual(config.TRAIN_MODELS_DIR, dirs["TRAIN_MODELS_DIR"])
                (Path(config.TRAIN_MODELS_DIR) / "role_final.safetensors").write_bytes(b"model")
                (Path(config.CURVES_DIR) / "role_loss_by_step.png").write_bytes(b"curve")
            fake_inference = SimpleNamespace(main=generate, model_utils=SimpleNamespace(load_inference_pipeline=lambda: None))
            # Restore all worker-overridden global settings after this mock integration test.
            settings = {k: v for k, v in vars(config).items() if k.isupper()}
            settings["CHARACTER_PRESETS"] = dict(config.CHARACTER_PRESETS)
            settings.update(dirs)
            with patch.multiple(config, **settings), patch.object(worker, "ROOT", root), patch.object(worker, "event", side_effect=lambda *args, **kw: events.append(kw)), patch.dict("sys.modules", {
                    "test": fake_inference, "train": SimpleNamespace(main=train),
                    "train_utils": SimpleNamespace(log_training=lambda *args, **kwargs: None)}):
                spec = {"character": "role", "directory": str(root / "logs/web_tasks/task"), "weight": str(root / "train_models/role_final.safetensors"), "images": ["image.png"]}
                worker.run({**spec, "kind": "generate"})
                self.assertEqual(events[-1]["result"]["images"], ["/outputs/result/image_result.png"])
                worker.run({**spec, "kind": "train", "steps": 50, "resolution": 640})
                self.assertEqual(events[-1]["result"]["weight"], str(root / "train_models/role_final.safetensors"))
                self.assertFalse((root / "logs/web_tasks/task/weights").exists())


if __name__ == "__main__":
    unittest.main()
