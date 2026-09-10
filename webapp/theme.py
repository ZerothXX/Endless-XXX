"""Resource-derived themes. No character IDs, fixed palettes or learned weights here."""
import colorsys
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import config

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "output/web_themes"
VERSION = 5


def image_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def fingerprint(folder, weight):
    paths = image_files(folder / "images") + [folder / "captions.txt", Path(weight)]
    h = hashlib.sha256(str(VERSION).encode())
    for p in paths:
        if p.exists():
            h.update(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
    h.update(json.dumps(config.get_character_card(folder.name), sort_keys=True).encode())
    return h.hexdigest()[:20]


def transparent_subject(image):
    """Use genuine alpha, or remove only border-connected near-white studio background."""
    rgba = image.convert("RGBA")
    a = np.array(rgba)
    if (a[:, :, 3] < 128).mean() > .015:
        return rgba, "source_alpha"
    rgb = a[:, :, :3]
    near_white = ((rgb.min(2) > 230) & (rgb.max(2) - rgb.min(2) < 18)).astype(np.uint8)
    count, labels = cv2.connectedComponents(near_white)
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    bg = np.isin(labels, edge_ids[edge_ids != 0])
    white_edge = np.concatenate([near_white[0], near_white[-1], near_white[:, 0], near_white[:, -1]]).mean()
    if bg.mean() < .12 or white_edge < .85:
        return None, "needs_sam"
    a[bg, 3] = 0
    return Image.fromarray(a), "border_connected_white"


def palette_from_pixels(pixels):
    """Aggregate hue families, not background or outline frequency.

    Raw statistics and a highlight-derived design swatch are both retained.
    Near-black ink and skin-like orange pixels cannot displace a saturated outfit.
    """
    p = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    hsv = cv2.cvtColor(p[None], cv2.COLOR_RGB2HSV)[0].astype(float)
    h, s, v = hsv[:, 0] * 2, hsv[:, 1] / 255, hsv[:, 2] / 255
    chroma = (s > .20) & (v > .20)
    bins = ((h + 15) // 30).astype(int) % 12
    # Suppress flesh-colored low-chroma pixels, but not genuinely orange costumes.
    weight = np.where((h < 55) & (s < .48), .30, 1.0)
    scores = np.bincount(bins[chroma], weights=weight[chroma], minlength=12)
    if scores.max() == 0:
        raw = np.median(p, axis=0)
        dominant = raw
        selected = np.ones(len(p), bool)
    else:
        key = int(scores.argmax())
        selected = chroma & (bins == key)
        raw = np.median(p[selected], axis=0)
        # A theme uses the lit textile color rather than its dark folds/line art.
        lit = selected & (v >= np.quantile(v[selected], .60))
        dominant = np.median(p[lit], axis=0)
    dh, dl, ds = colorsys.rgb_to_hls(*(dominant / 255))
    neutral = (s < .20) & (v > .60)
    secondary = np.median(p[neutral], axis=0) if neutral.any() else np.array([242, 245, 248])
    distance = np.minimum(abs(h - dh * 360), 360 - abs(h - dh * 360))
    accent_ok = chroma & (distance > 40)
    accent_scores = np.bincount(bins[accent_ok], weights=weight[accent_ok], minlength=12)
    if accent_scores.max() > 0:
        accent_mask = accent_ok & (bins == accent_scores.argmax())
        accent = np.median(p[accent_mask & (v >= np.quantile(v[accent_mask], .45))], axis=0)
    else:
        accent_mask = np.zeros(len(p), bool)
        accent = np.array(colorsys.hls_to_rgb((dh + .12) % 1, .70, max(ds, .3))) * 255
    def entry(role, color, fraction):
        return {"role": role, "hex": "#" + "".join(f"{int(x):02x}" for x in np.clip(color, 0, 255)),
                "ratio": round(float(fraction), 4)}
    return [entry("dominant", dominant, selected.mean()), entry("secondary", secondary, neutral.mean()),
            entry("tertiary", accent, accent_mask.mean())], {
                "raw_dominant": entry("raw", raw, 0)["hex"], "mode": "light" if dl >= .55 else "dark",
                "sample_count": len(p), "method": "foreground hue families; highlight textile swatch",
                "ratios_are_selected_pixel_fractions_not_page_area": True}


def mascot_assets():
    target = CACHE / "shared"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("pm", "bb"):
        image = Image.open(ROOT / "f_example" / f"{name}.png").convert("RGBA")
        image = image.crop(image.getbbox())
        image.save(target / f"{name}.png")
        if name == "bb":
            arr = np.array(image)
            hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
            # Only the supplied mascot's colored fabric/ear/lamp pixels; preserve white lettering.
            accent = (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 100) & (arr[:, :, 3] > 0)
            layer = np.full_like(arr, 255)
            layer[:, :, 3] = np.where(accent, arr[:, :, 3], 0).astype(np.uint8)
            Image.fromarray(layer).save(target / "bb_accent.png")
    return target


def infer_reference_roles(files, target):
    """Use existing VLM only when references have no useful labels; unload before SAM."""
    from vlm.vlm_utils import VLMClient
    from PIL import ImageDraw
    board = Image.new("RGB", (1000, ((len(files) + 4) // 5) * 230), "white")
    draw = ImageDraw.Draw(board)
    for index, path in enumerate(files):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
        image.thumbnail((185, 195))
        x, y = (index % 5) * 200, (index // 5) * 230
        board.paste(image, (x, y + 25), image)
        draw.text((x + 4, y + 5), str(index), fill="black")
    board.save(target / "reference_inventory.png")
    client = VLMClient(config.VLM_PATH, dtype=config.VLM_DTYPE,
                       load_in_4bit=config.VLM_LOAD_IN_4BIT, offload_cpu=config.VLM_OFFLOAD_CPU)
    try:
        if not client.available:
            raise RuntimeError("参考图没有有效标注，且 Qwen2.5-VL 不可用")
        result = client.analyze_image(board,
            'Classify the numbered reference tiles. Return JSON {"items":[{"index":0,"kind":"full body|clothing|face close-up|accessory close-up|other","signature_emblem":false}]}. '
            'signature_emblem is true ONLY for an isolated distinctive wearable emblem or ornament, never a whole person, camera, generic item or background. Do not invent references.', temperature=0)
        items = result.get("items", []) if isinstance(result, dict) else []
        lines, candidates = [], []
        for item in items:
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(files):
                continue
            kind = item.get("kind")
            if kind not in {"full body", "clothing", "face close-up", "accessory close-up", "other"}:
                continue
            lines.append(f"{files[index].name}, {kind}")
            if item.get("signature_emblem") is True and kind == "accessory close-up":
                candidates.append(files[index])
        (target / "reference_roles.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return "\n".join(lines), candidates
    finally:
        client.unload()


def build_theme(name, weight, progress=lambda percent, phase: None):
    folder = ROOT / "dataset" / name
    stamp = fingerprint(folder, weight)
    target = CACHE / name / stamp
    if (target / "theme.json").exists():
        return json.loads((target / "theme.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    files = image_files(folder / "images")
    if not files:
        raise ValueError("角色没有图片")
    card = config.get_character_card(name) or {}
    refs = (card.get("accessory") or {}).get("source_images", [])
    caption_file = folder / "captions.txt"
    captions = caption_file.read_text(encoding="utf-8-sig") if caption_file.exists() else ""
    candidates = [p for p in files if p.name in refs]
    if not candidates:
        candidates = [p for p in files if any(p.name in line and any(t in line.lower() for t in
                      ("accessory", "emblem", "ornament", "symbol", "配饰")) for line in captions.splitlines())]
    if not captions.strip() and getattr(config, "WEB_THEME_VLM_FOR_UNLABELED", True):
        progress(1, "Qwen2.5-VL 识别参考图用途")
        captions, inferred = infer_reference_roles(files, target)
        candidates = candidates or inferred
    # Preserve all source evidence. Never use a whole-image fallback in color statistics.
    sam = None
    pixels, records, assets = [], [], []
    progress(3, "读取角色参考与标注")
    try:
        for index, path in enumerate(files):
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
            image.thumbnail((640, 640))
            cutout, method = transparent_subject(image)
            if cutout is None:
                if sam is None:
                    from vision.segmentation import load_sam2
                    sam = load_sam2()
                    if sam is None:
                        raise RuntimeError("SAM2 不可用，无法剔除复杂背景；未使用整图伪造主题配色")
                from webapp.segmentation import segment_theme_subject
                mask = segment_theme_subject(image.convert("RGB"), sam)
                if mask is None:
                    records.append({"file": path.name, "status": "rejected_no_mask"})
                    continue
                cutout = image.copy()
                cutout.putalpha(mask)
                method = "sam2"
            arr = np.array(cutout)
            mask = arr[:, :, 3] > 128
            coverage = float(mask.mean())
            edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()
            accepted = .015 < coverage < .95 and (edge < .60 or method != "sam2")
            records.append({"file": path.name, "method": method, "coverage": coverage,
                            "border_coverage": float(edge), "status": "accepted" if accepted else "rejected_geometry"})
            if not accepted:
                continue
            cutout.save(target / f"source_{path.stem}.png")
            # Equal per-image sampling prevents high-resolution sheets dominating the result.
            sample = arr[:, :, :3][mask]
            caption = next((line.lower() for line in captions.splitlines() if line.startswith(path.name + ",")), "")
            # Close-ups are deliberately overrepresented in LoRA datasets; correct
            # that acquisition bias so 6 face crops cannot outweigh the whole outfit.
            sampling_weight = (1.0 if any(t in caption for t in ("full body", "clothing", "three-quarter"))
                               else .25 if any(t in caption for t in ("close-up", "accessory")) else .5)
            budget = int(6000 * sampling_weight)
            stride = max(1, len(sample) // budget)
            pixels.append(sample[::stride][:budget])
            records[-1]["sampling_weight"] = sampling_weight
            if path in candidates:
                # Isolated compact references outrank complete-body context references.
                box = cutout.getbbox()
                aspect = (box[2] - box[0]) / max(1, box[3] - box[1])
                score = 2 * (method != "sam2") + min(aspect, 1 / aspect)
                assets.append((score, cutout.crop(box), path.name))
            progress(5 + 75 * (index + 1) / len(files), f"主体分割与配色采样 {index + 1}/{len(files)}")
    finally:
        if sam is not None:
            del sam
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
    if len(pixels) < min(3, len(files)):
        raise RuntimeError("有效主体分割不足；请检查参考图，主题未写入成功缓存")
    palette, stats = palette_from_pixels(np.concatenate(pixels))
    progress(88, "生成角色配色与透明装饰素材")
    base = f"/themes/{name}/{stamp}"
    accessory = None
    warnings = []
    if assets:
        _, cutout, source = max(assets, key=lambda item: item[0])
        cutout.save(target / "accessory.png")
        accessory = base + "/accessory.png"
    else:
        source = None
        warnings.append("未发现可靠的配饰特写；已生成配色主题，未伪造角色标志")
    data = {"name": name, "version": VERSION, "fingerprint": stamp, "palette": palette,
            "mode": stats["mode"], "assets": {"accessory": accessory, "signature": None},
            "provenance": {"weight": str(weight), "accessory_source": source,
                           "segmentation": records, "palette": stats},
            "warnings": warnings, "quality": "automatic_needs_visual_review"}
    (target / "theme.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(100, "主题已缓存")
    return data
