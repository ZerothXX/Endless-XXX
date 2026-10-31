# Endless-XXX · FLUX.2 Research

[中文](README.md) · [English](README.en.md)

This branch briefly introduces FLUX.2 research into character visual transfer for people, animals and objects. The work remains experimental.

## Current method

Generation uses the **official distilled FLUX.2 Klein 4B model**. The current focus is **fixed multi-image references without LoRA**: 2–3 whole reference images for a character are passed to the model together with the target image.

| Input | Method |
|---|---|
| People | Place the target first, followed by whole character references; use prompt constraints for pose and handheld props. |
| Objects | Use whole character references and prompts to guide the transfer of colors, patterns and materials. |
| Animals | Condition generation on whole character references; Qwen3-VL also reads the same references to provide a character design description. |

Local inference uses NF4 4bit quantization, BF16 computation, 4 sampling steps and CFG 1.

## Research directions

Explored alternatives include character QLoRA fine-tuning, reference ordering and attention weights, local design boards, and character understanding followed by design descriptions. These remain candidate approaches.

[Main project and research entry: Endlexx](https://github.com/ZerothXX/Endlexx#readme)
