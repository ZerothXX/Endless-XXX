"""Source-visible composition and bounded prompt assembly (no model imports)."""
import re


def visible_regions(subject):
    values = subject.get("visible_regions") or []
    if not isinstance(values, list):
        values = [values]
    return {str(v).strip().lower().replace("_", " ") for v in values}


def region_visible(subject, region):
    regions = visible_regions(subject)
    if region in regions:
        return True
    aliases = {
        "head": {"face", "hair"},
        "torso": {"upper body", "chest", "shoulders", "body"},
        "arms": {"upper body", "hands"},
        "waist": {"hips", "hip"},
        "legs": {"leg", "thighs", "knees", "lower legs"},
        "feet": {"foot", "shoes", "boots"},
    }
    # 'full body' permits clothing, but says nothing about exposed skin.
    return "full body" in regions or bool(regions & aliases.get(region, set()))


def composition_constraints(subject):
    """Use observed framing/coverage; unknown is never interpreted as bare skin."""
    positive, negative = [], []
    if subject.get("subject_type") != "human":
        return positive, negative
    framing = str(subject.get("framing") or "").lower()
    regions = visible_regions(subject)
    upper = framing in {"upper body", "bust", "portrait", "close-up"}
    upper = upper or (bool(regions) and not region_visible(subject, "legs")
                      and not region_visible(subject, "feet"))
    if upper:
        positive.append("upper body")
        negative.extend(["full body", "bare legs", "bare thighs", "feet"])
    coverage = subject.get("coverage") or {}
    if not isinstance(coverage, dict):
        coverage = {}
    if coverage.get("legs") == "covered":
        positive.append("covered legs")
        negative.extend(["bare legs", "bare thighs", "skirt lift"])
    if coverage.get("chest") == "covered":
        positive.append("covered chest")
        negative.extend(["cleavage", "open shirt"])
    hands = str(subject.get("hands") or "").lower()
    if hands in {"overlapping hands", "folded hands", "crossed hands"}:
        positive.append(hands)
    return positive, negative


def fit_prompt(required, optional, tokenizers=(), max_tokens=77):
    """Never discard anatomy/symbol constraints to keep decorative quality words.

    Counts both SDXL tokenizers including BOS/EOS. An oversized required block
    fails explicitly instead of silently truncating essential conditions.
    """
    tokenizers = tuple(t for t in tokenizers if t is not None)
    def count(text):
        if tokenizers:
            return max(len(t(text, truncation=False, verbose=False)["input_ids"]) for t in tokenizers)
        return len(re.findall(r"\w+|[^\w\s]", text)) + 2
    parts = [p.strip() for p in required if p and p.strip()]
    if count(", ".join(parts)) > max_tokens:
        raise ValueError("Required transfer conditions exceed the CLIP token budget; "
                         "shorten the character accessory description or subject conditions.")
    dropped = []
    for phrase in optional:
        if not phrase or phrase in parts:
            continue
        if count(", ".join(parts + [phrase])) <= max_tokens:
            parts.append(phrase)
        else:
            dropped.append(phrase)
    return ", ".join(parts), dropped
