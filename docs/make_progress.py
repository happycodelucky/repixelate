"""Regenerate the version-to-version comparison images in docs/progress/.

    .venv/bin/python docs/make_progress.py

Each earlier version is taken from git history and run on the same inputs as the
current code, so the pictures always compare like with like. To add a version, commit
it, add its commit to VERSIONS, and add a section to docs/PROGRESS.md.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import bench  # noqa: E402
import synth  # noqa: E402

# (label, git commit). The working tree is always the last, newest version.
VERSIONS = [("v1", "d6fec30")]
CURRENT = "v2"

OUT = ROOT / "docs" / "progress"
TREES = ROOT / "test" / "Gemini_Generated_Image.jpeg"
VIDEO = ROOT / "test" / "gemini_generated_video_78be5980.mp4"
DARK, GREY, PAPER = (24, 24, 28), (120, 120, 120), (255, 255, 255)
FONT = ImageFont.load_default(size=15)


def scripts(tmp: Path) -> dict[str, Path]:
    """Every version's repixel.py, oldest first."""
    out = {}
    for label, rev in VERSIONS:
        path = tmp / f"repixel_{label}.py"
        path.write_bytes(subprocess.run(["git", "show", f"{rev}:repixel.py"], cwd=ROOT,
                                        capture_output=True, check=True).stdout)
        out[label] = path
    out[CURRENT] = ROOT / "repixel.py"
    return out


def run(script: Path, *argv) -> str:
    r = subprocess.run([sys.executable, str(script), *map(str, argv)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{script.name} failed: {r.stderr}")
    return r.stdout


def on(rgba: np.ndarray, colour, size) -> Image.Image:
    im = Image.fromarray(rgba).convert("RGBA")
    bg = Image.new("RGBA", im.size, colour + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB").resize(size, Image.NEAREST)


def labelled(panels: list[tuple[str, Image.Image]], cols: int) -> Image.Image:
    """Lay panels out in a grid with a caption above each."""
    pad, cap = 10, 24
    w, h = max(p.width for _, p in panels), max(p.height for _, p in panels)
    rows = -(-len(panels) // cols)
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + cap) + (rows + 1) * pad), PAPER)
    draw = ImageDraw.Draw(sheet)
    for i, (text, p) in enumerate(panels):
        x, y = pad + (i % cols) * (w + pad), pad + (i // cols) * (h + cap + pad)
        draw.text((x, y + 2), text, fill=(0, 0, 0), font=FONT)
        sheet.paste(p, (x, y + cap))
    return sheet


def edges(ver: dict[str, Path], tmp: Path):
    """Background removal on the real trees image, seen on a dark background."""
    box, k = (0, 55, 62, 125), 6
    panels = []
    for label, script in ver.items():
        out = tmp / f"trees_{label}.png"
        run(script, TREES, "-t", "magenta", "-o", out, "--no-ask")
        crop = np.asarray(Image.open(out).convert("RGBA").crop(box))
        panels.append((f"{label}: background removed", on(crop, DARK, (crop.shape[1] * k, crop.shape[0] * k))))
    labelled(panels, len(panels)).save(OUT / "edges.png", optimize=True)


def detection(ver: dict[str, Path], tmp: Path):
    """A small-pixel sprite sheet with blur and JPEG damage, where the answer is known."""
    art, mask = bench.arts()["sheet"]
    img = synth.degrade(art, 2.4, blur=0.5, jpeg=75, rng=np.random.default_rng(7))
    src = tmp / "sheet.png"
    Image.fromarray(img).save(src)
    size = (art.shape[1] * 6, art.shape[0] * 6)
    truth = np.dstack([art, np.where(mask, 255, 0).astype(np.uint8)])
    panels = [("Input: 2.4 screen pixels per art pixel, blurred, JPEG",
               Image.fromarray(img).resize(size, Image.BICUBIC)),
              ("What was actually drawn", on(truth, GREY, size))]
    for label, script in ver.items():
        out = tmp / f"sheet_{label}.png"
        found = re.search(r"pixel size ([\d.]+)", run(script, src, "-t", "-o", out, "--no-ask")).group(1)
        got = np.asarray(Image.open(out).convert("RGBA"))
        acc = synth.score(got, art, mask)["acc"]
        panels.append((f"{label}: found {found} px, {acc:.0%} of sprite pixels right", on(got, GREY, size)))
    labelled(panels, 2).save(OUT / "detection.png", optimize=True)


def video(ver: dict[str, Path], tmp: Path):
    """The real clip: what each version writes for the first walking soldier."""
    k, n, box = 5, 12, (4, 0, 26, 34)
    panels = []
    for label, script in ver.items():
        frames_dir = tmp / f"frames_{label}"
        report = run(script, VIDEO, "-t", "--trim", "-o", tmp / f"walk_{label}.png",
                     "--frames", frames_dir, "--no-ask")
        count = re.search(r"(\d+) unique frames", report).group(1)
        files = sorted(frames_dir.glob("*.png"))
        start = next(i for i in range(len(files) - 1)            # first frame where he moves
                     if (np.asarray(Image.open(files[i]).convert("RGBA").crop(box))
                         != np.asarray(Image.open(files[i + 1]).convert("RGBA").crop(box))).mean() > 0.05)
        strip = Image.new("RGB", (n * (22 * k + 4) - 4, 34 * k), GREY)
        for j, f in enumerate(files[start:start + n]):
            crop = np.asarray(Image.open(f).convert("RGBA").crop(box))
            strip.paste(on(crop, GREY, (22 * k, 34 * k)), (j * (22 * k + 4), 0))
        panels.append((f"{label}: {count} output frames for the 10-second clip; {n} in a row from where he starts walking", strip))
    labelled(panels, 1).save(OUT / "video.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        versions = scripts(Path(t))
        edges(versions, Path(t))
        detection(versions, Path(t))
        video(versions, Path(t))
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
