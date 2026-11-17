"""Regenerate the README example images from the files in test/.

    .venv/bin/python docs/make_examples.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import repixel as rp  # noqa: E402

DOCS, TEST = ROOT / "docs", ROOT / "test"
TREES = TEST / "Gemini_Generated_Image.jpeg"
SOLDIERS = TEST / "Gemini_Generated_Image_kewh32kewh32kewh.jpg"
VIDEO = TEST / "gemini_generated_video_78be5980.mp4"
GREY, DARK = (120, 120, 120), (24, 24, 28)


def on(rgba: np.ndarray, colour, k: int) -> Image.Image:
    im = Image.fromarray(rgba, "RGBA" if rgba.shape[2] == 4 else "RGB").convert("RGBA")
    bg = Image.new("RGBA", im.size, colour + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB").resize((im.width * k, im.height * k), Image.NEAREST)


def run(*argv) -> None:
    if rp.main([str(a) for a in argv]) != 0:
        raise SystemExit(f"repixel failed: {argv}")


def trees():
    src = np.asarray(Image.open(TREES).convert("RGB"))
    plain = rp.repixel_image(src)
    cut = rp.repixel_image(src, rp.Options(transparent="magenta"))
    box, k = (slice(0, 134), slice(0, 120)), 4            # the first tree, in art pixels
    s = plain.scale_x
    crop = src[int(box[0].start * s):int(box[0].stop * s), int(box[1].start * s):int(box[1].stop * s)]
    size = (120 * k, 134 * k)
    panels = [Image.fromarray(crop).resize(size, Image.BICUBIC),
              on(plain.pixels[box], GREY, k), on(cut.pixels[box], DARK, k)]
    sheet = Image.new("RGB", (size[0] * 3 + 16, size[1]), (255, 255, 255))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * (size[0] + 8), 0))
    sheet.save(DOCS / "trees_before_after.png", optimize=True)


def soldiers():
    out = DOCS / "soldiers_112x33.png"
    run(SOLDIERS, "-t", "--trim", "-o", out, "--no-ask")
    on(np.asarray(Image.open(out).convert("RGBA")), GREY, 6).save(DOCS / "soldiers_6x.png", optimize=True)


def walk():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run(VIDEO, "-t", "--trim", "-o", tmp / "walk.png", "--no-ask")
        im = Image.open(tmp / "walk.png")
        frames, ms = [], []
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(on(np.asarray(im.convert("RGBA")), GREY, 4).convert("P", palette=Image.ADAPTIVE))
            ms.append(int(im.info.get("duration", 100)))
        frames[0].save(DOCS / "walk_4x.gif", save_all=True, append_images=frames[1:], duration=ms, loop=0)
        run(VIDEO, "-t", "--split", "--loop", "--sheet", "-o", tmp / "sheet.png", "--no-ask")
        on(np.asarray(Image.open(tmp / "sheet.png").convert("RGBA")), GREY, 4).save(
            DOCS / "walk_sheet_4x.png", optimize=True)


if __name__ == "__main__":
    trees()
    soldiers()
    walk()
