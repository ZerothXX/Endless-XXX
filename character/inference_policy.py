"""One parameter policy shared by hybrid and fallback inference paths."""
import config


def generation_settings(subject_type):
    prefix = {"human": "HUMAN", "object": "OBJECT", "animal": "ANIMAL"}.get(subject_type)
    strength = getattr(config, f"{prefix}_IMG2IMG_STRENGTH", config.IMG2IMG_STRENGTH)
    control = getattr(config, f"{prefix}_CN_SCALE", config.CONTROLNET_CONDITIONING_SCALE)
    end = getattr(config, f"{prefix}_CONTROL_GUIDANCE_END", 1.0)
    scale = (getattr(config, "HUMAN_LORA_SCALE", config.LORA_SCALE) if subject_type == "human"
             else config.NON_HUMAN_LORA_SCALE)
    if not 0 < strength <= 1 or not 0 < end <= 1 or control < 0 or scale < 0:
        raise ValueError("Invalid generation strength, control schedule or LoRA scale")
    return {"strength": strength, "control_scale": control, "control_end": end,
            "lora_scale": scale}
