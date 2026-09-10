"""Reference contours condition a local diffusion repaint; reference RGB is never composited.

The caller must supply a checked placement box. This module does not guess a
screen-space location from a label such as 'shoulder' or 'cup body'.
"""
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import image_utils


def validate_box(box):
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError("Accessory placement requires normalized [left, top, right, bottom]")
    box = tuple(float(v) for v in box)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("Accessory box coordinates must be finite and within [0, 1]")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Accessory placement box has no area")
    return box


def prepare_local_condition(scene, reference, box, resolution=512):
    box = validate_box(box)
    x0, y0, x1, y1 = [round(v * d) for v, d in zip(box, (scene.width, scene.height) * 2)]
    if min(x1 - x0, y1 - y0) < 8:
        raise ValueError("Accessory placement is too small for local refinement")
    side = min(max(x1 - x0, y1 - y0) * 2, scene.width, scene.height)
    left = max(0, min(scene.width - side, (x0 + x1 - side) // 2))
    top = max(0, min(scene.height - side, (y0 + y1 - side) // 2))
    crop_box = (left, top, left + side, top + side)
    crop = scene.crop(crop_box).resize((resolution, resolution), Image.Resampling.LANCZOS)
    scale = resolution / side
    target = tuple(round(v * scale) for v in (x0 - left, y0 - top, x1 - left, y1 - top))
    tx0, ty0, tx1, ty1 = target
    reference_fit = ImageOps.contain(reference, (tx1 - tx0, ty1 - ty0),
                                     method=Image.Resampling.LANCZOS)
    # Only edges travel from reference to diffusion. All RGB output comes from the model.
    edge = image_utils.extract_canny(reference_fit, 50, 150)
    guide = image_utils.extract_canny(crop, 50, 150)
    mask = Image.new("L", crop.size, 0)
    margin = max(6, round(resolution * 0.035))
    paint_box = (max(0, tx0 - margin), max(0, ty0 - margin),
                 min(resolution, tx1 + margin), min(resolution, ty1 + margin))
    guide.paste((0, 0, 0), paint_box)
    px = tx0 + (tx1 - tx0 - edge.width) // 2
    py = ty0 + (ty1 - ty0 - edge.height) // 2
    guide.paste(edge, (px, py))
    # Follow the emblem silhouette, not a rectangular edit window. Otherwise
    # diffusion's color drift inside the rectangle becomes a visible patch on the object.
    contours, _ = cv2.findContours(np.asarray(edge.convert("L")), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros((edge.height, edge.width), dtype=np.uint8)
    if not contours:
        raise ValueError("Reference contains no usable contours for an accessory mask")
    cv2.drawContours(silhouette, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
    mask.paste(Image.fromarray(silhouette), (px, py))
    mask = mask.filter(ImageFilter.MaxFilter(9))
    return crop, mask, guide, crop_box


def refine_accessory(pipe, scene, reference, box, prompt, negative_prompt, generator,
                     resolution=512, steps=30, strength=0.95, control_scale=0.85,
                     guidance_scale=7.0):
    crop, mask, guide, crop_box = prepare_local_condition(scene, reference, box, resolution)
    output = pipe(prompt=prompt, negative_prompt=negative_prompt, image=crop,
                  mask_image=mask, control_image=guide, width=resolution, height=resolution,
                  num_inference_steps=steps, strength=strength,
                  controlnet_conditioning_scale=control_scale, guidance_scale=guidance_scale,
                  generator=generator).images[0]
    size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    generated_patch = output.resize(size, Image.Resampling.LANCZOS)
    blend = mask.filter(ImageFilter.GaussianBlur(3)).resize(size, Image.Resampling.LANCZOS)
    result = scene.copy()
    result.paste(generated_patch, crop_box[:2], blend)
    return result, {"crop": crop, "mask": mask, "control": guide, "generated_patch": output}
