"""Material/surface-based transfer policy, independent of character or object category.

These are semantic plans, not verified pixel locations. A named surface must never
be interpreted as an automatic inpainting box.
"""
import re


def known_text(value):
    return (value.strip().lower() if isinstance(value, str) and
            value.strip().lower() not in {"unknown", "none", "null", "n/a"} else "")


def observed_regions(subject, key):
    values = subject.get(key)
    return list(dict.fromkeys(known_text(v) for v in values if known_text(v))) \
        if isinstance(values, list) else []


def material_forms(material):
    words = set(re.findall(r"[a-z]+", known_text(material)))
    families = [
        ({"fabric", "canvas", "cotton", "textile", "cloth", "plush", "denim"},
         ["embroidered patch", "surface pattern"]),
        ({"ceramic", "porcelain", "stoneware"},
         ["glaze pattern", "painted decoration", "surface pattern"]),
        ({"metal", "steel", "aluminum", "aluminium", "leather", "rubber"},
         ["embossed ornament", "surface pattern", "painted decoration"]),
        ({"wood", "wooden", "plastic", "glass", "paper", "cardboard"},
         ["painted decoration", "surface pattern"]),
    ]
    matches = [forms for tokens, forms in families if words & tokens]
    # Mixed/unknown materials: do not assume which material belongs to a surface.
    return matches[0] if len(matches) == 1 else ["surface pattern"]


def object_plan(subject):
    surfaces = observed_regions(subject, "surface_regions")
    material = known_text(subject.get("material"))
    forms = material_forms(material)
    return {
        "adaptation_type": forms[0],
        "target_region": surfaces[0] if surfaces else "visible exterior surface",
        "material": material or "original material",
        "scale": "small",
        "candidates": forms,
        "preserve_regions": observed_regions(subject, "structure"),
        "palette_policy": "recolor existing surfaces; preserve texture and functional parts",
        "placement_verified": False,
        "reason": "Adapt the character design to observed material and surface; "
                  "do not add anatomy, garments, attachment points or functional parts.",
        "source": "material_surface_rule",
    }


def validate_object_plan(result, subject):
    """Reject incompatible material operations and invented attachment locations.

    Returning None asks the caller to use the conservative material/surface rule.
    VLM localization remains unverified even when its semantic labels are valid.
    """
    form = known_text(result.get("adaptation_type"))
    region = known_text(result.get("target_region"))
    surfaces = observed_regions(subject, "surface_regions")
    attachments = observed_regions(subject, "attachment_regions")
    hanging = form in {"bag charm", "zipper ornament", "small ornament"}
    allowed_regions = attachments if hanging else surfaces
    if not region or (allowed_regions and region not in allowed_regions):
        return None
    if hanging:
        if not attachments or region not in attachments:
            return None
        # Only a clearly identified fastener/loop is evidence of a hanging point.
        if not set(re.findall(r"[a-z]+", region)) & {"zipper", "loop", "ring", "hook"}:
            return None
    elif form not in material_forms(subject.get("material")):
        return None
    if not allowed_regions:
        # No observed location: no model-invented cup body, neck, handle, etc.
        return None
    plan = object_plan(subject)
    plan.update(adaptation_type=form, target_region=region, source="vlm",
                reason=known_text(result.get("reason")) or plan["reason"])
    if hanging:
        plan["material"] = known_text(result.get("material")) or "matching material"
    plan["scale"] = result.get("scale") if result.get("scale") in {"small", "medium"} else "small"
    return plan
