from pathlib import Path
import sys
import unittest
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from character.transfer_constraints import composition_constraints, fit_prompt, region_visible
from character.semantic_memory import AccessoryAdaptationPlanner, load_character_memory, card_from_memory
from character.semantic_memory import style_policy_for_subject
from character.inference_policy import generation_settings
from character.character_parser import pick_representative_images
import dataset
import image_utils
from PIL import Image
from vision.accessory_refiner import prepare_local_condition, refine_accessory, validate_box
from character.object_policy import object_plan, validate_object_plan
from vision.background_guard import preserve_background


class TransferRegressionTests(unittest.TestCase):
    def test_generation_settings_separate_human_from_nonhuman(self):
        human = generation_settings("human")
        self.assertEqual(human["strength"], config.HUMAN_IMG2IMG_STRENGTH)
        self.assertEqual(human["control_end"], config.HUMAN_CONTROL_GUIDANCE_END)
        self.assertEqual(generation_settings("object")["strength"], config.OBJECT_IMG2IMG_STRENGTH)
        self.assertEqual(generation_settings("animal")["control_end"], 1.0)
        self.assertEqual(generation_settings("unknown")["lora_scale"], config.NON_HUMAN_LORA_SCALE)

    def test_human_character_style_does_not_change_nonhuman_policy(self):
        card = config.get_character_card("37")
        self.assertEqual(style_policy_for_subject(card, {"subject_type": "human"}), "character")
        for kind in ("animal", "object", "unknown"):
            self.assertEqual(style_policy_for_subject(card, {"subject_type": kind}), "preserve")
        sr = {"subject_type": "human", "visible_regions": ["face", "upper body"],
              "style": "oil painting"}
        planner = AccessoryAdaptationPlanner()
        prompt = planner.build_card_transfer_prompt(sr, planner.plan(sr, card["accessory"]),
                                                    card, "ch37", "1girl", max_tokens=200)
        self.assertIn("anime style", prompt)
        self.assertNotIn("oil painting", prompt)

    def test_background_guard_preserves_every_pixel_outside_mask(self):
        import numpy as np
        source = Image.new("RGB", (64, 64), (90, 100, 110))
        generated = Image.new("RGB", source.size, (220, 120, 30))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (16, 16, 48, 48))
        result, audit = preserve_background(source, generated, mask)
        outside = np.asarray(mask) == 0
        self.assertTrue(np.array_equal(np.asarray(source)[outside], np.asarray(result)[outside]))
        self.assertEqual(result.getpixel((32, 32)), generated.getpixel((32, 32)))
        self.assertFalse(audit["semantic_mask_verified"])
        for invalid in (None, Image.new("L", source.size, 0), Image.new("L", source.size, 255)):
            with self.assertRaises(ValueError):
                preserve_background(source, generated, invalid)
    def test_training_reference_captions_and_fixed_diagnostics(self):
        import torch
        ds = dataset.get_train_dataset("37", 640)
        self.assertEqual([Path(ds.image_paths[i]).name for i in ds.evaluation_indices()],
                         ["001.png", "006.png", "012.png", "013.png"])
        camera = ds.get_prompt(11)
        emblem = ds.get_prompt(12)
        self.assertIn("cyan compact camera", camera)
        self.assertNotIn("orange strap", camera)  # Only visible in whole-character refs, not 012.
        self.assertIn("cyan teardrops", emblem)
        self.assertNotIn("1girl", camera)
        self.assertNotIn("1girl", emblem)
        self.assertTrue(torch.equal(ds.get_item(0, deterministic=True)["pixel_values"],
                                    ds.get_item(0, deterministic=True)["pixel_values"]))
    def test_portrait_does_not_request_unseen_legs(self):
        sr = {"subject_type": "human", "visible_regions": ["face", "upper body", "hands"],
              "coverage": {"chest": "covered"}, "hands": "overlapping hands"}
        positive, negative = composition_constraints(sr)
        self.assertIn("upper body", positive)
        self.assertIn("overlapping hands", positive)
        self.assertIn("bare thighs", negative)
        self.assertFalse(region_visible(sr, "legs"))
        planner = AccessoryAdaptationPlanner()
        card = config.get_character_card("37")
        plan = planner.plan(sr, card["accessory"])
        self.assertEqual(plan["target_region"], "shoulder")
        prompt = planner.build_card_transfer_prompt(sr, plan, card, "ch37", "1girl", max_tokens=200)
        for forbidden in ("thighhighs", "boots", "waist bag", "pleated skirt", "clover"):
            self.assertNotIn(forbidden, prompt)
        for feature in ("bangs", "white blouse", "blue jacket", "detached sleeves",
                        "four-petal brooch on shoulder"):
            self.assertIn(feature, prompt)

    def test_full_body_does_not_mean_bare_legs(self):
        sr = {"subject_type": "human", "visible_regions": ["full body"],
              "coverage": {"legs": "covered"}}
        self.assertTrue(region_visible(sr, "feet"))
        positive, negative = composition_constraints(sr)
        self.assertIn("covered legs", positive)
        self.assertIn("bare legs", negative)

    def test_budget_preserves_required_and_checks_both_encoders(self):
        class Tokenizer:
            def __init__(self, extra): self.extra = extra
            def __call__(self, text, **kwargs):
                return {"input_ids": list(range(len(text.split()) + self.extra))}
        prompt, dropped = fit_prompt(["symbol", "upper body"], ["quality", "optional detail"],
                                      (Tokenizer(2), Tokenizer(4)), 8)
        self.assertIn("upper body", prompt)
        self.assertEqual(dropped, ["optional detail"])
        with self.assertRaises(ValueError):
            fit_prompt(["one two three"], [], (Tokenizer(4),), 6)

    def test_all_accessory_references_are_selected(self):
        paths = dataset.scan_images(str(ROOT / "dataset/37"))
        captions = dataset.load_captions(str(ROOT / "dataset/37/captions.txt"))
        chosen = [Path(p).name for p in pick_representative_images(paths, captions)]
        self.assertIn("012.png", chosen)
        self.assertIn("013.png", chosen)
        self.assertEqual(chosen[0], "001.png")

    def test_accessory_training_is_not_labelled_as_person(self):
        text = dataset.build_text("ch37", "1girl", ["shoulder accessory close-up"])
        self.assertNotIn("1girl", text)
        self.assertIn("accessory", text)
        self.assertIn("1girl", dataset.build_text("ch37", "1girl", ["full body"]))

    def test_full_image_bucket_does_not_stretch(self):
        image = Image.new("RGB", (100, 300), "black")
        transform = dataset.MixedContentCrop(size=640, full_prob=1)
        result = transform(image)
        import numpy as np
        ys, xs = np.where(np.asarray(result).min(axis=2) < 10)
        ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        self.assertAlmostEqual(ratio, 1 / 3, delta=0.005)

    def test_reviewed_memory_replaces_stale_vlm_at_read_time(self):
        memory = load_character_memory("37")
        self.assertEqual(memory["source"], "reviewed_card")
        self.assertEqual(len(memory["accessories"]), 2)
        self.assertIn("012.png", memory["accessories"][1]["source_images"])

    def test_other_character_has_no_37_defaults(self):
        card = card_from_memory({"appearance": {"hair": "orange", "eyes": "green"},
                                 "accessories": [{"name": "star", "shape": "five points",
                                                  "color": "red"}]})
        self.assertIn("orange hair", card["core_identity_tags"])
        self.assertNotIn("cyan", str(card))
        self.assertEqual(card["accessory"]["description"], "star, five points, red")

    def test_generic_objects_use_material_and_observed_surface(self):
        cases = [
            ("mug", "ceramic", "outer wall", "glaze pattern"),
            ("backpack", "canvas", "front panel", "embroidered patch"),
            ("notebook", "leather", "front cover", "embossed ornament"),
            ("headphones", "plastic", "outer earcup", "painted decoration"),
            ("chair", "wood", "backrest panel", "painted decoration"),
            ("storage box", "metal", "lid", "embossed ornament"),
            ("vase", "ceramic", "vase body", "glaze pattern"),
            ("unknown", "unknown", "exterior", "surface pattern"),
        ]
        planner = AccessoryAdaptationPlanner()
        card = config.get_character_card("37")
        for category, material, surface, form in cases:
            with self.subTest(category=category):
                subject = {"subject_type": "object", "object_category": category,
                           "material": material, "surface_regions": [surface],
                           "structure": ["original functional parts"], "style": "photo"}
                plan = planner.plan(subject, card["accessory"])
                self.assertEqual(plan["adaptation_type"], form)
                self.assertEqual(plan["target_region"], surface)
                self.assertFalse(plan["placement_verified"])
                prompt = planner.build_card_transfer_prompt(subject, plan, card, "ch37", "1girl",
                                                            max_tokens=200)
                self.assertIn("blue pink white color scheme", prompt)
                self.assertIn("original shape and parts", prompt)
                for unwanted in ("pink hair", "1girl", "skirt", "cup body"):
                    self.assertNotIn(unwanted, prompt)
                if material != "ceramic":
                    self.assertNotIn("ceramic", prompt)

    def test_object_validation_keeps_existing_regions_not_anatomy_substrings(self):
        subject = {"material": "plastic", "surface_regions": ["outer earcup"],
                   "attachment_regions": []}
        plan = validate_object_plan({"adaptation_type": "painted decoration",
                                     "target_region": "outer earcup"}, subject)
        self.assertIsNotNone(plan)  # 'ear' does not make headphones a person.
        for form, region in (("glaze pattern", "outer earcup"), ("bag charm", "neck"),
                             ("surface pattern", "cup body")):
            self.assertIsNone(validate_object_plan({"adaptation_type": form,
                                                    "target_region": region}, subject))
        class Client:
            available = True
            def analyze_image(self, *args, **kwargs):
                return {"adaptation_type": "glaze pattern", "target_region": "cup body"}
        planner = AccessoryAdaptationPlanner(client=Client(), subject_image=object())
        actual = planner.plan(dict(subject, subject_type="object"), {})
        self.assertEqual(actual["adaptation_type"], "painted decoration")
        self.assertEqual(actual["target_region"], "outer earcup")

    def test_hanging_requires_observed_fastener_and_unknown_does_not_invent_one(self):
        result = {"adaptation_type": "zipper ornament", "target_region": "zipper pull"}
        self.assertIsNone(validate_object_plan(result, {"material": "canvas"}))
        self.assertIsNotNone(validate_object_plan(result, {"material": "canvas",
                                                "attachment_regions": ["zipper pull"]}))
        self.assertEqual(object_plan({"material": "ceramic and fabric"})["adaptation_type"],
                         "surface pattern")
        unknown = AccessoryAdaptationPlanner().plan({"subject_type": "unknown"}, {})
        self.assertNotIn("neck", str(unknown))

    def test_unreviewed_character_retains_nonhuman_palette(self):
        card = card_from_memory({"color_palette": {"main": "orange", "secondary": "black",
                                                    "accent": "unknown"},
                                 "accessories": [{"name": "red star"}]})
        subject = {"subject_type": "object", "object_category": "box", "material": "wood"}
        planner = AccessoryAdaptationPlanner()
        prompt = planner.build_card_transfer_prompt(subject, planner.plan(subject, {}), card,
                                                    "other_character", max_tokens=200)
        self.assertIn("orange black color scheme", prompt)
        self.assertNotIn("pink", prompt)

    def test_local_refinement_is_grayscale_condition_not_rgb_paste(self):
        import numpy as np
        scene = Image.new("RGB", (640, 640), (100, 120, 130))
        reference = Image.new("RGB", (64, 64), "white")
        reference.paste((0, 180, 240), (12, 12, 52, 52))
        crop, mask, guide, box = prepare_local_condition(scene, reference, [0.4, 0.4, 0.6, 0.6])
        arr = np.asarray(guide)
        self.assertTrue(np.array_equal(arr[:, :, 0], arr[:, :, 1]))
        self.assertTrue(np.array_equal(arr[:, :, 1], arr[:, :, 2]))
        self.assertGreater(arr.max(), 0)
        class FakePipe:
            def __call__(self, **kwargs):
                return type("Result", (), {"images": [Image.new("RGB", crop.size, "red")]})()
        result, _ = refine_accessory(FakePipe(), scene, reference, [0.4, 0.4, 0.6, 0.6], "", "", None)
        self.assertEqual(result.getpixel((0, 0)), scene.getpixel((0, 0)))
        self.assertEqual(result.getpixel((639, 639)), scene.getpixel((639, 639)))
        self.assertEqual(result.getpixel((320, 320)), (255, 0, 0))

    def test_local_placement_rejects_invalid_coordinates(self):
        for box in ([0, 0, 2, 1], [0.5, 0.5, 0.1, 0.8], [0, 0, float("nan"), 1]):
            with self.assertRaises(ValueError):
                validate_box(box)

    def test_nonhuman_has_no_human_identity(self):
        planner = AccessoryAdaptationPlanner()
        sr = {"subject_type": "object", "object_category": "cup", "material": "ceramic",
              "style": "realistic photo"}
        plan = planner.plan(sr, config.get_character_card("37")["accessory"])
        prompt = planner.build_card_transfer_prompt(sr, plan, config.get_character_card("37"),
                                                    "ch37", "1girl", max_tokens=200)
        self.assertNotIn("1girl", prompt)
        self.assertNotIn("pink hair", prompt)
        self.assertIn("photorealistic", prompt)
        self.assertIn("white center", prompt)

    def test_real_sdxl_tokenizers_on_all_existing_subjects(self):
        from transformers import CLIPTokenizer
        from huggingface_hub import snapshot_download
        try:
            cache = Path(snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                                           local_files_only=True))
        except Exception as exc:
            self.skipTest(f"No local SDXL tokenizer cache: {exc}")
        tokenizers = [CLIPTokenizer.from_pretrained(str(cache / name), local_files_only=True)
                      for name in ("tokenizer", "tokenizer_2")]
        card = config.get_character_card("37")
        planner = AccessoryAdaptationPlanner()
        for category, material, surface in (("backpack", "canvas", "front panel"),
                                            ("headphones", "plastic", "outer earcup"),
                                            ("notebook", "leather", "front cover"),
                                            ("chair", "wood", "backrest panel"),
                                            ("vase", "ceramic", "vase body")):
            subject = {"subject_type": "object", "object_category": category,
                       "material": material, "surface_regions": [surface], "style": "photo"}
            plan = planner.plan(subject, card["accessory"])
            prompt = planner.build_card_transfer_prompt(subject, plan, card, "ch37", "1girl",
                        tokenizer=tokenizers[0], tokenizer_2=tokenizers[1])
            self.assertIn("blue pink white color scheme", prompt)
            self.assertIn("white center", prompt)
            for tokenizer in tokenizers:
                self.assertLessEqual(len(tokenizer(prompt)["input_ids"]), 76, category)
        for path in sorted((ROOT / "output/result").glob("*_subject.json")):
            subject = json.loads(path.read_text(encoding="utf-8"))
            plan = planner.plan(subject, card["accessory"])
            prompt = planner.build_card_transfer_prompt(subject, plan, card, "ch37", "1girl",
                        tokenizer=tokenizers[0], tokenizer_2=tokenizers[1])
            for tokenizer in tokenizers:
                self.assertLessEqual(len(tokenizer(prompt)["input_ids"]), 76, path.name)
            if subject.get("subject_type") == "human":
                if region_visible(subject, "head"):
                    self.assertIn("bangs", prompt)
                self.assertIn("white blouse", prompt)
                self.assertIn("blue jacket", prompt)
            else:
                self.assertIn("white center", prompt)


if __name__ == "__main__":
    unittest.main()
