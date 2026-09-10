"""Flatten alpha onto neutral backgrounds for inspecting actual sampled foreground."""
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]

def audit(folder):
    sources = sorted(folder.glob("source_*.png"))
    sheet = Image.new("RGB", (1000, ((len(sources) + 4) // 5) * 250), "#d5d5d5")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(sources):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((190, 220))
        x, y = (i % 5) * 200, (i // 5) * 250
        sheet.paste(image, (x + (200 - image.width) // 2, y + 22), image)
        draw.text((x + 8, y + 4), path.stem, fill="black")
    sheet.save(folder / "segmentation_audit.png")

if __name__ == "__main__":
    import sys
    audit(Path(sys.argv[1]))
