import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import config
from G24 import config_24
from G24.train_24 import check_device
from training_run import run_isolated


class TrainingEntryTests(unittest.TestCase):
    def test_24gb_profile_targets_selected_character(self):
        target = SimpleNamespace(CHARACTER_PRESETS={"37": {"resolution": 640}, "other": {"category": "cat"}},
                                 OUTPUT_24_DIR="output/24", TRAIN_MODELS_24_DIR="output/24/train_models")
        config_24.apply(target, "other")
        self.assertEqual(target.CHARACTER_PRESETS["37"]["resolution"], 640)
        self.assertEqual(target.CHARACTER_PRESETS["other"]["resolution"], 1024)
        self.assertEqual(target.CHARACTER_PRESETS["other"]["category"], "cat")
        self.assertEqual(target.LORA_RANK, 64)
        self.assertTrue(target.TRAIN_CACHE_TEXT_ENCODERS)
        self.assertEqual(Path(target.TRAIN_MODELS_DIR), Path("output/24/train_models"))
        self.assertEqual(Path(target.CURVES_DIR), Path("output/24/curves"))
        self.assertEqual(Path(target.LOGS_DIR), Path("output/24/logs"))

    def test_device_gate(self):
        for gib, accepted in [(8, False), (24, True)]:
            cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                                   get_device_properties=lambda index: SimpleNamespace(total_memory=gib * 1024**3))
            if accepted:
                check_device(cuda)
            else:
                with self.assertRaises(RuntimeError):
                    check_device(cuda)
        with self.assertRaises(RuntimeError):
            check_device(SimpleNamespace(is_available=lambda: False))

    def run_fixture(self, failing=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "dataset/example/images"
            images.mkdir(parents=True)
            (images / "001.png").write_bytes(b"source image")
            (root / "dataset/data_preprocessing.py").write_text("# fixture", encoding="utf-8")
            weights = root / "original_weights"
            weights.mkdir()
            (weights / "example_final.safetensors").write_bytes(b"original weight")
            settings = {"PROJECT_ROOT": str(root), "DATASET_DIR": str(root / "dataset"),
                        "TRAIN_MODELS_DIR": str(weights), "LOGS_DIR": str(root / "output/logs")}
            def fake_train():
                staged = Path(config.DATASET_DIR) / "example"
                (staged / "images/001.png").write_bytes(b"preprocessed copy")
                (staged / "captions.txt").write_text("001.png, generated tags", encoding="utf-8")
                if failing:
                    raise RuntimeError("fixture failure")
                Path(config.TRAIN_MODELS_DIR).mkdir(exist_ok=True)
                (Path(config.TRAIN_MODELS_DIR) / "example_final.safetensors").write_bytes(b"new weight")
                print("fixture training finished")
            with patch.multiple(config, **settings):
                if failing:
                    with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                        run_isolated(fake_train, "example")
                    out = next((root / "output/logs/training").iterdir())
                else:
                    out = run_isolated(fake_train, "example")
                self.assertEqual(config.DATASET_DIR, str(root / "dataset"))
                self.assertEqual(config.TRAIN_MODELS_DIR, str(weights))
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed" if failing else "complete")
            self.assertTrue(manifest["originals_unchanged"])
            self.assertEqual((images / "001.png").read_bytes(), b"source image")
            self.assertFalse((images.parent / "captions.txt").exists())
            self.assertEqual((weights / "example_final.safetensors").read_bytes(),
                             b"original weight" if failing else b"new weight")

    def test_isolation_keeps_sources_and_restores_config(self):
        self.run_fixture()

    def test_failure_records_manifest_and_restores_config(self):
        self.run_fixture(failing=True)


if __name__ == "__main__":
    unittest.main()
