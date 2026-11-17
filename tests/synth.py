"""Synthetic ground truth for Repixel: procedural pixel art plus the damage that
upscaling, AI generation, JPEG and video codecs do to it.

Every generator returns true-resolution art (uint8 RGB, one array element per art
pixel). `degrade` renders it the way a fake-pixel-art image looks: each art pixel
covers a fractional number of screen pixels, with optional grid wobble, blur, noise
and JPEG compression. Because the truth is known, results can be scored exactly.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

MAGENTA = (255, 0, 255)


def _ramp(rng, n=4):
    """A dark-to-light colour ramp with a random hue (never close to magenta)."""
    while True:
        base = rng.integers(40, 230, 3).astype(float)
        if not (base[0] > 150 and base[2] > 150 and base[1] < 120):
            break
    return [tuple(np.clip(base * f, 0, 255).astype(int)) for f in np.linspace(0.45, 1.15, n)]


def _blob(rng, w, h):
    """Random left-right symmetric blob mask."""
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    m = ((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2 < 0.85
    for _ in range(6):
        bx, by = rng.integers(1, w // 2 + 1), rng.integers(1, h - 1)
        bw, bh = rng.integers(1, max(2, w // 4)), rng.integers(1, max(2, h // 4))
        val = rng.random() < 0.6
        m[by:by + bh, bx:bx + bw] = val
    m[:, : w // 2] = m[:, ::-1][:, : w // 2]
    m[0], m[-1], m[:, 0], m[:, -1] = False, False, False, False
    return m


def _outline(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    inner = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return mask & ~inner


def make_sprite(rng, w=16, h=24, bg=MAGENTA):
    """One outlined, shaded sprite on a flat background. Returns (rgb, mask)."""
    mask = _blob(rng, w, h)
    ramp = _ramp(rng)
    accent = _ramp(rng, 2)[1]
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:] = bg
    shade = np.clip((np.mgrid[0:h, 0:w][0] * 3 // h) + rng.integers(0, 2, (h, w)) * (rng.random((h, w)) < 0.15), 0, 2)
    for i in range(3):
        rgb[mask & (shade == i)] = ramp[3 - i]
    rgb[_outline(mask)] = tuple(int(c * 0.25) for c in ramp[0])
    for _ in range(4):
        y, x = rng.integers(2, h - 2), rng.integers(2, w - 2)
        if mask[y, x]:
            rgb[y, x] = accent
    return rgb, mask


def make_sheet(rng, n=5, sw=16, sh=24, gap=6, bg=MAGENTA):
    """A row of sprites on a flat background. Returns (rgb, mask)."""
    w, h = n * sw + (n + 1) * gap, sh + 2 * gap
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:] = bg
    mask = np.zeros((h, w), bool)
    for i in range(n):
        s, m = make_sprite(rng, sw, sh, bg)
        x = gap + i * (sw + gap)
        rgb[gap:gap + sh, x:x + sw] = s
        mask[gap:gap + sh, x:x + sw] = m
    return rgb, mask


def make_scene(rng, w=160, h=90):
    """A full-frame landscape: banded sky, hills, textured ground, trees."""
    sky = _ramp(rng, 5)
    ground = _ramp(rng, 3)
    leaf = _ramp(rng, 3)
    trunk = (70, 45, 30)
    rgb = np.empty((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    band = np.clip((yy * 5 // (h * 2 // 3)), 0, 4)
    dither = ((xx + yy) % 2 == 0) & ((yy * 5 % (h * 2 // 3)) > (h * 2 // 3) * 0.7)
    band = np.clip(band + dither, 0, 4)
    for i in range(5):
        rgb[band == i] = sky[4 - i]
    ph = rng.random(3) * 6.28
    hill = (h * 0.6 + 6 * np.sin(xx[0] / 13 + ph[0]) + 3 * np.sin(xx[0] / 5 + ph[1])).astype(int)
    g = yy >= hill[None, :]
    tex = rng.integers(0, 3, (h, w))
    tex[rng.random((h, w)) < 0.7] = 1
    for i in range(3):
        rgb[g & (tex == i)] = ground[i]
    rgb[g & (yy == hill[None, :])] = tuple(int(c * 0.5) for c in ground[0])
    for _ in range(max(3, w // 20)):
        x = int(rng.integers(6, w - 6))
        base = hill[x]
        th = int(rng.integers(5, 10))
        rgb[base - th:base, x:x + 2] = trunk
        r = int(rng.integers(3, 6))
        d2 = (xx - x - 0.5) ** 2 + (yy - (base - th - r + 1)) ** 2
        can = d2 <= r * r
        rgb[can] = leaf[1]
        rgb[can & (xx < x) & (yy < base - th - r + 1)] = leaf[2]
        rgb[can & (d2 > (r - 1) ** 2)] = leaf[0]
    return rgb


def _coverage(bounds: np.ndarray, n: int) -> np.ndarray:
    """A[X, k] = how much of screen pixel [X, X+1) cell k (bounds[k]..bounds[k+1]) covers."""
    X = np.arange(n)[:, None]
    lo, hi = bounds[None, :-1], bounds[None, 1:]
    return np.clip(np.minimum(X + 1, hi) - np.maximum(X, lo), 0, None)


def _wobble(rng, n, amount, period):
    """Smooth random drift of grid lines, in units of one cell."""
    if amount <= 0:
        return np.zeros(n + 1)
    walk = np.cumsum(rng.normal(0, 1, n + 1))
    k = np.ones(max(3, n // 8)) / max(3, n // 8)
    walk = np.convolve(walk, k, mode="same")
    walk -= np.linspace(walk[0], walk[-1], n + 1)
    return walk / (np.abs(walk).max() + 1e-9) * amount * period


def degrade(art: np.ndarray, scale: float, *, offset=(0.0, 0.0), wobble=0.0, blur=0.0,
            noise=0.0, jpeg: int | None = None, rng=None, scale_y: float | None = None) -> np.ndarray:
    """Render true-resolution art as a degraded 'fake pixel art' image (uint8 RGB).

    offset   fraction of a cell cropped from the left/top (shifts the grid phase)
    wobble   peak grid-line drift in cells (AI-style uneven grid)
    blur     Gaussian sigma in screen pixels
    noise    Gaussian noise sigma (0-255) added before compression
    jpeg     JPEG quality (4:2:0 chroma), or None for lossless
    """
    rng = rng or np.random.default_rng(0)
    h, w, _ = art.shape
    sx, sy = scale, scale_y or scale
    bx = np.arange(w + 1) * sx - offset[0] * sx + _wobble(rng, w, wobble, sx)
    by = np.arange(h + 1) * sy - offset[1] * sy + _wobble(rng, h, wobble, sy)
    W, H = int(np.floor(bx[-1])), int(np.floor(by[-1]))
    ax, ay = _coverage(bx, W), _coverage(by, H)
    ax /= ax.sum(1, keepdims=True)
    ay /= ay.sum(1, keepdims=True)
    img = np.einsum("Yy,yxc,Xx->YXc", ay, art.astype(float), ax, optimize=True)
    if noise:
        img = img + rng.normal(0, noise, img.shape)
    im = Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    if jpeg:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=jpeg, subsampling=2)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(im)


def score(out: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None, max_shift=2) -> dict:
    """Compare a result with the truth, allowing a small border crop/shift.

    acc     fraction of pixels whose nearest truth-palette colour is the true colour
    err     mean absolute RGB error on the overlap
    shape   True if the output has the truth's size to within one border cell per side
    If `mask` is given and `out` has alpha, alpha_acc is the fraction of correct
    transparent/opaque decisions and colour scores cover truly opaque pixels only.
    """
    pal, tidx = np.unique(truth.reshape(-1, 3), axis=0, return_inverse=True)
    tidx = tidx.reshape(truth.shape[:2])
    rgb = out[..., :3].astype(int)
    best = None
    H, W = truth.shape[:2]
    h, w = rgb.shape[:2]
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0, x0 = max(0, dy), max(0, dx)
            y1, x1 = min(H, h + dy), min(W, w + dx)
            if y1 - y0 < 0.8 * H or x1 - x0 < 0.8 * W:
                continue
            o = rgb[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            t = truth[y0:y1, x0:x1].astype(int)
            e = np.abs(o - t).mean()
            if best is None or e < best[0]:
                best = (e, dy, dx, y0, y1, x0, x1)
    if best is None:
        return {"acc": 0.0, "err": 255.0, "shape": False, "alpha_acc": 0.0}
    e, dy, dx, y0, y1, x0, x1 = best
    o = rgb[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    ti = tidx[y0:y1, x0:x1]
    near = np.argmin(((o[:, :, None, :] - pal[None, None].astype(int)) ** 2).sum(-1), axis=-1)
    ok = near == ti
    res = {"shape": abs(h - H) <= 2 and abs(w - W) <= 2, "size": (w, h)}
    if mask is not None and out.shape[2] == 4:
        m = mask[y0:y1, x0:x1]
        a = out[y0 - dy:y1 - dy, x0 - dx:x1 - dx, 3] > 0
        res["alpha_acc"] = float((a == m).mean())
        sel = m & a
        res["acc"] = float(ok[sel].mean()) if sel.any() else 0.0
        res["err"] = float(np.abs(o - truth[y0:y1, x0:x1].astype(int))[sel].mean()) if sel.any() else 255.0
    else:
        res["acc"] = float(ok.mean())
        res["err"] = float(e)
    return res


def make_cycle(rng, poses=6, sw=16, sh=24, n=3, gap=6, bg=MAGENTA):
    """A looping animation of `n` sprites: returns (frames, masks), each (poses, h, w, ..).

    Every pose shifts the sprite's lower third sideways by a different amount (legs)
    and bobs the whole sprite on alternate poses, so all poses are distinct and the
    last leads back into the first.
    """
    sprites = [make_sprite(rng, sw, sh, bg) for _ in range(n)]
    w, h = n * sw + (n + 1) * gap, sh + 2 * gap
    frames = np.empty((poses, h, w, 3), np.uint8)
    masks = np.zeros((poses, h, w), bool)
    frames[:] = bg
    swing = np.round(2 * np.sin(2 * np.pi * np.arange(poses) / poses)).astype(int)
    for p in range(poses):
        for i, (rgb, mask) in enumerate(sprites):
            x0, y0 = gap + i * (sw + gap), gap - (p % 2)
            legs = slice(2 * sh // 3, sh)
            rgb2, mask2 = rgb.copy(), mask.copy()
            rgb2[legs] = np.roll(rgb[legs], swing[p], axis=1)
            mask2[legs] = np.roll(mask[legs], swing[p], axis=1)
            rgb2[~mask2] = bg
            frames[p, y0:y0 + sh, x0:x0 + sw] = rgb2
            masks[p, y0:y0 + sh, x0:x0 + sw] = mask2
    return frames, masks


def write_video(path, poses: np.ndarray, order, scale=8.0, hold=3, fps=24, crf=18, rng=None):
    """Encode poses[order[0]], poses[order[1]], ... as an H.264 video, each held for
    `hold` frames, rendered at `scale` screen pixels per art pixel with a little blur
    and noise. Needs ffmpeg."""
    import subprocess
    rng = rng or np.random.default_rng(0)
    first = degrade(poses[0], scale, blur=0.4)
    h, w = first.shape[0] // 2 * 2, first.shape[1] // 2 * 2
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    for p in order:
        base = degrade(poses[p], scale, blur=0.4)[:h, :w].astype(float)
        for _ in range(hold):
            f = np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
            proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
