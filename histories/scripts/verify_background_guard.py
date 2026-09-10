"""Audit automatic SAM masks and protect backgrounds of completed raw comparisons.

Usage: python histories/scripts/verify_background_guard.py COMPARISON_DIRECTORY
Raw results are never overwritten. This does not correct anatomy or accessory identity.
"""
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


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Provide a completed weight comparison directory")
    comparison = Path(sys.argv[1]).resolve(strict=True)
    manifest = json.loads((comparison / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Comparison is incomplete; run this audit after generation")
    out = ROOT / "output/experiments" / ("background_guard_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"AUDIT_DIR={out}", flush=True)
    import numpy as np
    import image_utils
    from vision.segmentation import load_sam2, segment_subject
    from vision.background_guard import preserve_background, validate_subject_mask
    reports = []
    with open(out / "run.log", "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            sam = load_sam2()
            if sam is None:
                raise RuntimeError("SAM required; no full-image fallback allowed")
            for stem, filename in (("human", "human.png"), ("cup", "cup.jpg"), ("cat", "cat.png")):
                original = image_utils.load_image(str(ROOT / "input" / filename))
                raw_old = image_utils.load_image(str(comparison / "old" / f"{stem}.png"))
                source = image_utils.resize_fit(original, 512, max_long=768)
                if source.size != raw_old.size:
                    raise ValueError("Comparison resolution differs from audit recipe")
                mask = segment_subject(source, sam)
                mask, _ = validate_subject_mask(mask, source.size)
                mask.save(out / f"{stem}_mask.png")
                for arm in ("old", "new"):
                    raw = image_utils.load_image(str(comparison / arm / f"{stem}.png"))
                    result, audit = preserve_background(source, raw, mask)
                    target = out / arm
                    target.mkdir(exist_ok=True)
                    result.save(target / f"{stem}.png")
                    outside = np.asarray(mask) == 0
                    diff = np.abs(np.asarray(result).astype(int) - np.asarray(source).astype(int))
                    audit.update(arm=arm, sample=stem, max_outside_pixel_error=int(diff[outside].max()),
                                 source=str(ROOT / "input" / filename),
                                 raw_result=str(comparison / arm / f"{stem}.png"))
                    if audit["max_outside_pixel_error"] != 0:
                        raise AssertionError("Background pixel preservation failed")
                    reports.append(audit)
                    print(f"COMPLETE {arm}/{stem}: {audit}", flush=True)
    (out / "audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"COMPLETE={out}", flush=True)


if __name__ == "__main__":
    main()
