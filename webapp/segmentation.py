"""Theme-specific full-subject prompt: box + foreground point + background corners.

The old unconstrained highest-IoU grid can select the sky or a single limb.
This adapter uses the project's installed SAM2 video model's single-frame API.
"""
import numpy as np
from PIL import Image
import torch


def segment_theme_subject(image, sam):
    model, processor = sam
    w, h = image.size
    points = [[[[.50*w, .35*h], [.02*w, .02*h], [.98*w, .02*h], [.02*w, .98*h], [.98*w, .98*h]]]]
    labels = [[[1, 0, 0, 0, 0]]]
    inputs = processor(images=image, input_points=points, input_labels=labels,
                       input_boxes=[[[.03*w, .015*h, .97*w, .985*h]]], return_tensors="pt")
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model._single_frame_forward(pixel_values=inputs["pixel_values"].to(device, dtype=torch.float16),
                     input_points=inputs["input_points"].to(device), input_labels=inputs["input_labels"].to(device),
                     input_boxes=inputs["input_boxes"].to(device), multimask_output=False)
    post = processor.post_process_masks([output.high_res_masks.float()], inputs["original_sizes"].tolist(),
                                        mask_threshold=0.0, binarize=False)
    mask = post[0].reshape(-1, h, w)[0].cpu().numpy() > 0
    if not .02 < mask.mean() < .95:
        return None
    return Image.fromarray(mask.astype(np.uint8)*255)
