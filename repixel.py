#!/usr/bin/env python3
"""
Repixel - recover true-resolution pixel art from upscaled / JPEG-damaged images.

Pipeline
  1. Edge profiles   - per axis, sum colour jumps between neighbouring columns/rows,
                       ignoring the small jumps JPEG noise produces.
  2. Grid period     - score every candidate cell size (fractional sizes allowed) by
                       how coherently edges line up on a grid of that size (a DFT of
                       the edge profile). Harmonics are resolved by preferring the
                       largest period that still explains the edges.
  3. Grid alignment  - phase of the winning frequency gives the grid offset; for
                       larger cells each grid line is then snapped to the nearest real
                       edge, so slightly uneven (AI-generated) grids still line up.
  4. Sampling        - each cell's colour is the median of its centre region, which
                       avoids the blurred/ringing borders.
  5. Palette cleanup - near-identical colours (JPEG noise) are merged, or the image is
                       quantised to N colours with --colors.
  6. Output          - lossless PNG at 1 art pixel = 1 image pixel, optionally with
                       the background keyed out to transparency.

Videos (needs ffmpeg) are handled frame by frame: the grid is re-detected on every
frame (so camera zooms/pans are followed), each frame is placed on one shared art-pixel
canvas, a single palette is used for all frames, repeated frames are merged, and the
result is written as a lossless animated PNG (or GIF / a folder of PNG frames).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Grid detection
# --------------------------------------------------------------------------- #

EDGE_NOISE = 24.0  # sum-of-|dRGB| below this is treated as JPEG noise


def edge_profile(img: np.ndarray, axis: int) -> np.ndarray:
    """Edge strength at each boundary along `axis`.

    Returned array has length n+1 where index b is the boundary between pixel b-1
    and pixel b (so 0 and n are the image borders and are always zero).
    axis=1 -> vertical lines (x positions), axis=0 -> horizontal lines (y positions).
    """
    d = np.abs(np.diff(img, axis=axis)).sum(axis=2)
    d = np.maximum(d - EDGE_NOISE, 0.0)
    prof = d.sum(axis=0) if axis == 1 else d.sum(axis=1)
    out = np.zeros(prof.size + 2)
    out[1:-1] = prof
    return out


def _coherence(prof: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Complex, weight-normalised DFT of an edge profile at the given frequencies."""
    idx = np.nonzero(prof)[0]
    if idx.size == 0:
        return np.zeros(freqs.size, complex)
    w = prof[idx]
    total = w.sum()
    out = np.empty(freqs.size, complex)
    chunk = max(1, 4_000_000 // idx.size)
    for s in range(0, freqs.size, chunk):
        f = freqs[s:s + chunk, None]
        out[s:s + chunk] = (w * np.exp(-2j * np.pi * f * idx)).sum(axis=1) / total
    return out


@dataclass
class Candidate:
    period: float
    score: float


def _score(profs, periods: np.ndarray) -> np.ndarray:
    """Coherence magnitude averaged over the given axis profiles (weighted by edge mass)."""
    freqs = 1.0 / periods
    num = np.zeros(periods.size)
    den = 0.0
    for p in profs:
        mass = p.sum()
        if mass <= 0:
            continue
        num += mass * np.abs(_coherence(p, freqs))
        den += mass
    return num / den if den else num


def _refine(profs, period: float, n: int) -> Candidate:
    """Fine search around a period."""
    f0 = 1.0 / period
    half = 1.0 / n
    freqs = np.linspace(f0 - half, f0 + half, 81)
    freqs = freqs[freqs > 0]
    periods = 1.0 / freqs
    s = _score(profs, periods)
    i = int(np.argmax(s))
    return Candidate(float(periods[i]), float(s[i]))


def find_periods(profs, n: int, pmin=2.0, pmax=None, top=5) -> list[Candidate]:
    """Return ranked candidate cell sizes (largest-plausible first)."""
    # Edges are sampled at integer positions, so periods below 2 px alias onto real
    # ones (f and 1-f look identical). 2 px is therefore the smallest detectable cell.
    pmin = max(pmin, 2.0)
    if pmax is None:
        pmax = min(128.0, n / 4)
    step = 1.0 / (4 * n)
    freqs = np.arange(1.0 / pmax, 1.0 / pmin, step)
    periods = 1.0 / freqs
    s = _score(profs, periods)

    # local maxima of the coarse score
    peaks = np.nonzero((s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1
    peaks = peaks[np.argsort(-s[peaks])][:12]
    cands = [_refine(profs, periods[i], n) for i in peaks]
    if not cands:
        return []
    best = max(cands, key=lambda c: c.score)

    # Harmonic resolution: a grid of size P also lines up perfectly at P/2, P/3 ...
    # so walk up integer multiples of the best period and keep the largest one that
    # still explains most of the edges.
    fundamental = best
    for k in range(2, 9):
        if best.period * k > pmax:
            break
        c = _refine(profs, best.period * k, n)
        if c.score >= 0.6 * best.score:
            fundamental = c

    # Build a ranked list: the chosen fundamental first, then other distinct peaks.
    ranked = [fundamental]
    for c in sorted(cands, key=lambda c: -c.score):
        if all(abs(c.period - r.period) / r.period > 0.03 for r in ranked):
            ranked.append(c)
    return ranked[:top]


def _is_harmonic(p: float, fundamental: float) -> bool:
    """True if p is (roughly) fundamental / k for an integer k."""
    k = fundamental / p
    return k > 1.5 and abs(k - round(k)) < 0.04 * k


def find_period_near(profs, n: int, p0: float, rel: float = 0.08) -> Candidate:
    """Best period within +-rel of p0 (for tracking a slowly zooming video)."""
    f0 = 1.0 / p0
    freqs = np.arange(f0 / (1 + rel), f0 / (1 - rel), 1.0 / (4 * n))
    s = _score(profs, 1.0 / freqs)
    return _refine(profs, 1.0 / freqs[int(np.argmax(s))], n)


def grid_offset(prof: np.ndarray, period: float) -> float:
    """Where the grid lines sit, in [0, period): phase of the edge profile at 1/period."""
    c = _coherence(prof, np.array([1.0 / period]))[0]
    if abs(c) < 1e-9:
        return 0.0
    # edges sit where 2*pi*x/p == -arg(c)  (mod 2*pi)
    return float((-np.angle(c) / (2 * np.pi) * period) % period)


def grid_lines(prof: np.ndarray, period: float, n: int, snap: bool = True,
               offset: float | None = None) -> tuple[np.ndarray, int]:
    """Cell boundaries (in boundary coordinates 0..n) covering the image.

    Returns (lines, k_first): cell j of the result is local grid cell k_first + j,
    where local cell k spans [offset + k*period, offset + (k+1)*period).
    """
    if offset is None:
        offset = grid_offset(prof, period)
    k0 = int(np.floor((0 - offset) / period))
    k1 = int(np.ceil((n - offset) / period))
    ks = np.arange(k0, k1 + 1)
    lines = offset + period * ks

    if snap and period >= 5:
        lines = _snap_lines(prof, lines, period)

    # Clip to the image, dropping slivers narrower than half a cell at the borders.
    keep = (lines > 0) & (lines < n)
    inner, kin = lines[keep], ks[keep]
    if inner.size == 0:
        return np.array([0.0, float(n)]), 0
    out = list(inner)
    k_first = int(kin[0])
    if out[0] >= period * 0.5:   # partial first cell is wide enough to keep
        out.insert(0, 0.0)
        k_first -= 1
    if n - out[-1] >= period * 0.5:
        out.append(float(n))
    return np.array(out), k_first


def _snap_lines(prof: np.ndarray, lines: np.ndarray, period: float) -> np.ndarray:
    """Pull each predicted grid line onto the strongest nearby real edge."""
    n = prof.size - 1
    sm = np.convolve(prof, np.ones(3) / 3, mode="same")
    strong = sm[sm > 0]
    if strong.size == 0:
        return lines
    thr = 0.15 * np.percentile(strong, 90)
    win = max(1, int(round(period * 0.2)))
    resid = np.full(lines.size, np.nan)
    for k, x in enumerate(lines):
        c = int(round(x))
        a, b = max(1, c - win), min(n - 1, c + win)
        if a > b:
            continue
        seg = prof[a:b + 1]
        j = int(np.argmax(seg))
        if seg[j] > thr:
            resid[k] = (a + j) - x
    ok = ~np.isnan(resid)
    if ok.sum() < 2:
        return lines
    ks = np.arange(lines.size)
    # snapped lines keep their measured position, the rest follow interpolated drift
    drift = np.interp(ks, ks[ok], resid[ok])
    out = lines + np.where(ok, resid, drift)
    # keep monotonic with a sane minimum spacing
    for k in range(1, out.size):
        out[k] = max(out[k], out[k - 1] + 0.5 * period)
    return out


# --------------------------------------------------------------------------- #
# Sampling & palette
# --------------------------------------------------------------------------- #

def sample_cells(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Median colour of the centre region of every cell."""
    h, w, _ = img.shape
    cw = np.diff(xs)
    ch = np.diff(ys)
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    bw = max(1, int(np.median(cw) * 0.5))
    bh = max(1, int(np.median(ch) * 0.5))
    ox = np.arange(bw) - (bw - 1) / 2
    oy = np.arange(bh) - (bh - 1) / 2
    px = np.clip(np.floor(cx[:, None] + ox[None, :]).astype(int), 0, w - 1)  # (nx, bw)
    py = np.clip(np.floor(cy[:, None] + oy[None, :]).astype(int), 0, h - 1)  # (ny, bh)
    block = img[py[:, None, :, None], px[None, :, None, :]]  # (ny, nx, bh, bw, 3)
    ny, nx = block.shape[:2]
    return np.median(block.reshape(ny, nx, bh * bw, 3), axis=2)


def merge_similar(cells: np.ndarray, tol: float) -> np.ndarray:
    """Greedy, frequency-ordered merge of colours closer than `tol` (RGB distance)."""
    flat = cells.reshape(-1, 3)
    q = np.round(flat).astype(np.int32)
    uniq, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(-counts)
    pal_sum = []   # weighted colour sums
    pal_n = []     # weights
    pal = np.empty((0, 3))
    assign = np.empty(uniq.shape[0], int)
    for i in order:
        col = uniq[i].astype(float)
        if pal.shape[0]:
            d = np.sqrt(((pal - col) ** 2).sum(1))
            j = int(np.argmin(d))
            if d[j] <= tol:
                assign[i] = j
                pal_sum[j] += col * counts[i]
                pal_n[j] += counts[i]
                continue
        assign[i] = len(pal_n)
        pal_sum.append(col * counts[i])
        pal_n.append(counts[i])
        pal = np.vstack([pal, col])
    palette = np.array(pal_sum) / np.array(pal_n)[:, None]
    return palette[assign[inv]].reshape(cells.shape)


def kmeans_quantize(cells: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    flat = cells.reshape(-1, 3).astype(float)
    uniq, inv, counts = np.unique(np.round(flat), axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    k = min(k, uniq.shape[0])
    rng = np.random.default_rng(seed)
    # k-means++ init, weighted by pixel count
    cent = [uniq[rng.choice(len(uniq), p=counts / counts.sum())]]
    for _ in range(1, k):
        d = np.min(((uniq[:, None] - np.array(cent)[None]) ** 2).sum(2), axis=1) * counts
        cent.append(uniq[rng.choice(len(uniq), p=d / d.sum())] if d.sum() else uniq[0])
    cent = np.array(cent, float)
    for _ in range(iters):
        lab = np.argmin(((uniq[:, None] - cent[None]) ** 2).sum(2), axis=1)
        new = np.array([
            np.average(uniq[lab == j], axis=0, weights=counts[lab == j]) if (lab == j).any() else cent[j]
            for j in range(k)
        ])
        if np.allclose(new, cent):
            break
        cent = new
    lab = np.argmin(((uniq[:, None] - cent[None]) ** 2).sum(2), axis=1)
    return cent[lab][inv].reshape(cells.shape)


def detect_background(img: np.ndarray) -> np.ndarray:
    """Most common (coarsely quantised) colour along the image border."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, 3)
    q = (border // 16).astype(int)
    keys, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    top = np.argmax(counts)
    return np.median(border[inv.reshape(-1) == top], axis=0)


def to_image(out: np.ndarray) -> tuple[Image.Image, int]:
    """Exact (lossless) PIL image; uses an indexed PNG when <= 256 colours."""
    ch = out.shape[2]
    flat = out.reshape(-1, ch)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    mode = "RGBA" if ch == 4 else "RGB"
    if len(uniq) > 256:
        return Image.fromarray(out, mode), len(uniq)
    im = Image.fromarray(inv.reshape(out.shape[:2]).astype(np.uint8), "P")
    im.putpalette(uniq[:, :3].astype(np.uint8).tobytes(), rawmode="RGB")
    if ch == 4:
        im.info["transparency"] = bytes(uniq[:, 3].astype(np.uint8))
    return im, len(uniq)


def parse_color(s: str) -> np.ndarray:
    """'ff00ff', '#ff00ff', '0xff00ff', '#f0f', 'magenta', 'rgb(255,0,255)' -> RGB array."""
    from PIL import ImageColor
    t = s.strip()
    if t[:2].lower() == "0x":
        t = t[2:]
    if t and all(ch in "0123456789abcdefABCDEF" for ch in t) and len(t) in (3, 6):
        t = "#" + t
    try:
        return np.array(ImageColor.getrgb(t)[:3], float)
    except ValueError:
        raise SystemExit(f"error: can't read colour {s!r} (try ff00ff, #ff00ff or magenta)")




# --------------------------------------------------------------------------- #
# Shared steps
# --------------------------------------------------------------------------- #

def choose_period(profs, w: int, h: int, args) -> tuple[float, float, str] | None:
    """Pixel size from --scale, detection, or by asking the user."""
    if args.scale:
        return args.scale, args.scale_y or args.scale, "given"
    cands = find_periods(profs, max(w, h))
    if not cands:
        print("No pixel grid found (image has no edges?). Use --scale.", file=sys.stderr)
        return None
    best = cands[0]
    rivals = [c for c in cands[1:] if not _is_harmonic(c.period, best.period)]
    uncertain = best.score < 0.25 or any(c.score > 0.9 * best.score for c in rivals)
    interactive = sys.stdin.isatty() and not args.no_ask
    if args.ask or (uncertain and interactive):
        if uncertain:
            print("Grid detection is uncertain.", file=sys.stderr)
        p = ask_period(cands, w, h)
        return p, p, "chosen"
    if uncertain:
        print("warning: low-confidence detection; alternatives: "
              + ", ".join(f"{c.period:.3f}" for c in rivals) + " (use --scale or --ask)",
              file=sys.stderr)
    return best.period, best.period, f"detected, confidence {best.score:.2f}"


def clean_colors(cells: np.ndarray, args) -> np.ndarray:
    if args.colors:
        cells = kmeans_quantize(cells, args.colors)
    elif args.merge > 0:
        cells = merge_similar(cells, args.merge)
    return np.clip(np.round(cells), 0, 255).astype(np.uint8)


def key_background(rgb: np.ndarray, bg: np.ndarray, tol: float) -> np.ndarray:
    """RGB -> RGBA with cells near `bg` made fully transparent."""
    dist = np.sqrt(((rgb.astype(float) - bg) ** 2).sum(-1))
    alpha = np.where(dist <= tol, 0, 255).astype(np.uint8)
    rgb = rgb.copy()
    rgb[alpha == 0] = 0
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


_YCC = np.array([[0.299, 0.587, 0.114],
                 [-0.168736, -0.331264, 0.5],
                 [0.5, -0.418688, -0.081312]])


def _neighbourhood(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """Dilate a boolean mask by r pixels (8-connected)."""
    out = mask.copy()
    H, W = mask.shape
    for _ in range(r):
        p = np.pad(out, 1, constant_values=False)
        out = np.any([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return out


def despill(img: np.ndarray, bg: np.ndarray, near: int = 2, snap_tol: float = 28.0) -> np.ndarray:
    """Remove the background's tint from pixels within `near` px of transparency.

    JPEG smears colour, so edge pixels pick up the background hue (a dark outline on
    magenta turns purple). A colour threshold can't cut these without eating the
    outline, so instead the bg-hue part of their colour is subtracted (brightness
    kept), capped by how much of that hue the clean interior pixels nearby have.
    Corrected colours snap to the nearest untinted colour used inside the sprite.
    Alpha is never changed.
    """
    out = img.copy()
    op = out[..., 3] > 0
    if op.all():
        return out
    H, W = op.shape
    zone = op & _neighbourhood(~op, near)             # pixels close to transparency
    if not zone.any():
        return out
    bc = (_YCC @ bg)[1:]
    if np.linalg.norm(bc) < 10:                       # grey/white/black bg has no hue to spill
        return out
    bdir = bc / np.linalg.norm(bc)
    ycc = out[..., :3].astype(float) @ _YCC.T
    spill = ycc[..., 1:] @ bdir
    # reference: how much of that hue the clean interior pixels nearby legitimately have
    interior = op & ~_neighbourhood(~op, near + 1)
    vals = np.where(interior, spill, np.nan)
    r = 3
    pv = np.pad(vals, r, constant_values=np.nan)
    stack = np.stack([pv[r + dy:r + dy + H, r + dx:r + dx + W]
                      for dy in range(-r, r + 1) for dx in range(-r, r + 1)], 0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(stack, axis=0)
    cap = np.clip(np.nan_to_num(ref, nan=0.0), 0, None)
    excess = np.where(zone, np.clip(spill - cap, 0, None), 0)
    ycc[..., 1:] -= excess[..., None] * bdir
    rgb = np.clip(np.round(ycc @ np.linalg.inv(_YCC).T), 0, 255)
    # Snap corrected colours to the nearest colour already used inside the sprite, so
    # despilling doesn't bloat the palette with one-off in-between shades.
    changed = zone & (excess > 0)
    pal = np.unique(out[interior][:, :3], axis=0).astype(float) if interior.any() else None
    if pal is not None:  # don't snap onto colours that carry the bg tint themselves
        pal = pal[((pal @ _YCC.T)[:, 1:] @ bdir) <= 20]
        pal = pal if len(pal) else None
    if pal is not None and changed.any():
        cols, inv = np.unique(rgb[changed], axis=0, return_inverse=True)
        snapped = cols.copy()
        for i, c in enumerate(cols):
            d = np.sqrt(((pal - c) ** 2).sum(1))
            j = int(np.argmin(d))
            if d[j] <= snap_tol:
                snapped[i] = pal[j]
        rgb[changed] = snapped[inv.reshape(-1)]
    out[..., :3] = np.where(changed[..., None], rgb, out[..., :3])
    return out


def trim_box(alpha: np.ndarray) -> tuple[slice, slice] | None:
    """Bounding box of non-transparent pixels; alpha may be (h, w) or (t, h, w)."""
    a = alpha.reshape(-1, *alpha.shape[-2:]).any(0)
    if not a.any():
        return None
    r = np.nonzero(a.any(1))[0]
    c = np.nonzero(a.any(0))[0]
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


def resolve_bg(args, img: np.ndarray) -> np.ndarray:
    """Background colour: auto-detected, or the given colour snapped to what the image
    really contains (JPEG shifts flat colours, e.g. ff00ff often becomes fd33f8)."""
    if args.transparent == "auto":
        bg = detect_background(img)
    else:
        want = parse_color(args.transparent)
        px = img[::3, ::3].reshape(-1, 3)
        near = px[np.sqrt(((px - want) ** 2).sum(1)) <= 100]
        bg = np.median(near, axis=0) if len(near) >= 0.01 * len(px) else want
    print(f"  background #{''.join(f'{int(round(v)):02x}' for v in bg)}", file=sys.stderr)
    return bg


# --------------------------------------------------------------------------- #
# Still images
# --------------------------------------------------------------------------- #

def process_image(args) -> int:
    img = np.asarray(Image.open(args.input).convert("RGB")).astype(np.float32)
    h, w, _ = img.shape
    px_prof = edge_profile(img, axis=1)
    py_prof = edge_profile(img, axis=0)
    chosen = choose_period([px_prof, py_prof], w, h, args)
    if chosen is None:
        return 1
    sx, sy, how = chosen

    snap = not args.no_snap
    xs, _ = grid_lines(px_prof, sx, w, snap)
    ys, _ = grid_lines(py_prof, sy, h, snap)
    raw = sample_cells(img, xs, ys)
    out = clean_colors(raw, args)

    if args.transparent:
        bg = resolve_bg(args, img)
        out = key_background(out, bg, args.key_tol)
        if args.despill:
            out = despill(out, bg)
        box = trim_box(out[..., 3]) if args.trim else None
        if box:
            out = out[box]
    mode = "RGBA" if out.shape[2] == 4 else "RGB"

    result, ncolors = to_image(out)
    outp = args.output or args.input.with_name(args.input.stem + "_repixel.png")
    result.save(outp, optimize=True)

    # Reconstruction error: how well the recovered grid explains the source
    xi = np.searchsorted(xs, np.arange(w) + 0.5, side="right") - 1
    yi = np.searchsorted(ys, np.arange(h) + 0.5, side="right") - 1
    valid_x = (xi >= 0) & (xi < len(xs) - 1)
    valid_y = (yi >= 0) & (yi < len(ys) - 1)
    recon = raw[np.clip(yi, 0, raw.shape[0] - 1)][:, np.clip(xi, 0, raw.shape[1] - 1)]
    m = valid_y[:, None] & valid_x[None, :]
    err = np.abs(recon - img).mean(axis=2)[m].mean()

    print(f"{args.input.name}: {w}x{h}  pixel size {sx:.3f}"
          + (f" x {sy:.3f}" if sy != sx else "") + f" ({how})")
    print(f"  -> {outp}  {out.shape[1]}x{out.shape[0]}, {ncolors} colours, "
          f"mean reconstruction error {err:.1f}/255")

    if args.preview is not None:
        k = args.preview
        if k:
            size = (out.shape[1] * k, out.shape[0] * k)
        else:
            size = (int(round(out.shape[1] * sx)), int(round(out.shape[0] * sy)))
        prev = outp.with_name(outp.stem + "_preview.png")
        Image.fromarray(out, mode).resize(size, Image.NEAREST).save(prev)
        print(f"  preview -> {prev}")
    return 0


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif"}


def video_info(path: Path) -> tuple[int, int, float, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    num, den = (st.get("avg_frame_rate") or st["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 24.0
    return int(st["width"]), int(st["height"]), fps, int(st.get("nb_read_packets", 0))


def iter_frames(path: Path, w: int, h: int, fps: float | None = None):
    vf = ["-vf", f"fps={fps:g}"] if fps else []
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), *vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32)
    finally:
        proc.stdout.close()
        proc.wait()


def gif_frames(frames: np.ndarray) -> list[Image.Image] | None:
    """Exact indexed frames sharing one palette (transparent = last index), or None
    if the clip has too many colours for GIF."""
    rgba = frames.shape[-1] == 4
    flat = frames.reshape(-1, frames.shape[-1])
    opaque = flat[:, 3] > 0 if rgba else np.ones(len(flat), bool)
    uniq, inv = np.unique(flat[opaque][:, :3], axis=0, return_inverse=True)
    limit = 255 if rgba else 256
    if len(uniq) > limit:
        return None
    idx = np.full(len(flat), len(uniq), np.uint8)
    idx[opaque] = inv.reshape(-1)
    idx = idx.reshape(frames.shape[:3])
    pal = uniq.astype(np.uint8)
    if rgba:
        pal = np.vstack([pal, [[0, 0, 0]]]).astype(np.uint8)
    imgs = []
    for f in idx:
        im = Image.fromarray(f, "P")
        im.putpalette(pal.tobytes(), rawmode="RGB")
        if rgba:
            im.info["transparency"] = len(uniq)
        imgs.append(im)
    return imgs


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Running median (edge-padded) to steady per-frame measurements."""
    if x.size < 3 or win < 3:
        return x
    r = win // 2
    pad = np.pad(x, r, mode="edge")
    return np.array([np.median(pad[i:i + win]) for i in range(x.size)])


def _progress(i: int, total: int, label: str):
    if sys.stderr.isatty():
        print(f"\r  {label} {i}/{total or '?'}", end="", file=sys.stderr, flush=True)


def process_video(args) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("Video input needs ffmpeg/ffprobe on PATH (brew install ffmpeg).", file=sys.stderr)
        return 1
    w, h, src_fps, total = video_info(args.input)
    fps = args.fps or src_fps
    if args.fps:
        total = int(round(total * args.fps / src_fps))
    n = max(w, h)
    snap = not args.no_snap

    # Pass 1: pixel size per frame (follows zooms), smoothed over time.
    periods, confs = [], []
    first = None
    for t, img in enumerate(iter_frames(args.input, w, h, args.fps)):
        _progress(t + 1, total, "analysing frame")
        profs = [edge_profile(img, 1), edge_profile(img, 0)]
        if t == 0:
            first = img
            chosen = choose_period(profs, w, h, args)
            if chosen is None:
                return 1
            sx0, sy0, how = chosen
            periods.append(sx0)
            confs.append(1.0)
        elif args.scale:          # fixed scale given: don't track
            periods.append(sx0)
            confs.append(1.0)
        else:
            c = find_period_near(profs, n, periods[-1])
            periods.append(c.period)
            confs.append(c.score)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    nframes = len(periods)
    if nframes == 0:
        print("No frames decoded.", file=sys.stderr)
        return 1
    aspect = sy0 / sx0
    px_t = _smooth(np.array(periods), 9)

    # Pass 2: grid per frame, tracked onto one global art-pixel canvas.
    frames, origins = [], []
    ax_prev = ay_prev = None
    sx_shift = sy_shift = 0
    cx, cy = w / 2, h / 2
    for t, img in enumerate(iter_frames(args.input, w, h, args.fps)):
        if t >= nframes:
            break
        _progress(t + 1, nframes, "sampling frame")
        p_x, p_y = px_t[t], px_t[t] * aspect
        prof_x, prof_y = edge_profile(img, 1), edge_profile(img, 0)
        ox, oy = grid_offset(prof_x, p_x), grid_offset(prof_y, p_y)
        # Art-space coordinate of the frame centre in this frame's local grid. The grid
        # phase only fixes it modulo one cell, so unwrap against the previous frame.
        ax, ay = (cx - ox) / p_x, (cy - oy) / p_y
        if ax_prev is not None:
            sx_shift = int(round(ax_prev - ax))
            sy_shift = int(round(ay_prev - ay))
        ax_prev, ay_prev = ax + sx_shift, ay + sy_shift
        xs, kx = grid_lines(prof_x, p_x, w, snap, ox)
        ys, ky = grid_lines(prof_y, p_y, h, snap, oy)
        frames.append(sample_cells(img, xs, ys))
        origins.append((ky + sy_shift, kx + sx_shift))
    if sys.stderr.isatty():
        print(file=sys.stderr)

    # Shared canvas: by default only the area visible in every frame.
    y0s = np.array([o[0] for o in origins]); x0s = np.array([o[1] for o in origins])
    y1s = y0s + np.array([f.shape[0] for f in frames])
    x1s = x0s + np.array([f.shape[1] for f in frames])
    if args.extent == "intersect" and y0s.max() < y1s.min() and x0s.max() < x1s.min():
        Y0, Y1, X0, X1 = y0s.max(), y1s.min(), x0s.max(), x1s.min()
    else:
        Y0, Y1, X0, X1 = y0s.min(), y1s.max(), x0s.min(), x1s.max()
    H, W = int(Y1 - Y0), int(X1 - X0)
    stack = np.zeros((nframes, H, W, 3), np.float32)
    covered = np.zeros((nframes, H, W), bool)
    for i, (f, (oy_, ox_)) in enumerate(zip(frames, origins)):
        ya, xa = oy_ - Y0, ox_ - X0
        sy_ = slice(max(0, -ya), min(f.shape[0], H - ya))
        sx_ = slice(max(0, -xa), min(f.shape[1], W - xa))
        dy = slice(sy_.start + ya, sy_.stop + ya)
        dx = slice(sx_.start + xa, sx_.stop + xa)
        stack[i, dy, dx] = f[sy_, sx_]
        covered[i, dy, dx] = True

    # One palette for the whole clip, so colours don't flicker between frames.
    rgb = clean_colors(stack.reshape(nframes * H, W, 3), args).reshape(nframes, H, W, 3)

    if args.transparent or not covered.all():
        if args.transparent:
            bg = resolve_bg(args, first)
            out = key_background(rgb, bg, args.key_tol)
        else:
            out = np.concatenate([rgb, np.full((*rgb.shape[:3], 1), 255, np.uint8)], -1)
        out[~covered] = 0
        if args.transparent and args.despill:
            for i in range(nframes):
                out[i] = despill(out[i], bg)
        box = trim_box(out[..., 3]) if args.trim else None
        if box:
            out = out[:, box[0], box[1]]
    else:
        out = rgb

    # Remove one-frame blips: a pixel that changes for a single frame and then
    # returns to its previous value is compression noise, not animation.
    if not args.no_deblip and nframes >= 3:
        for i in range(1, nframes - 1):
            same_around = (out[i - 1] == out[i + 1]).all(-1)
            blip = same_around & (out[i] != out[i - 1]).any(-1)
            out[i][blip] = out[i - 1][blip]

    # Merge runs of (near-)identical frames into one longer frame.
    keep, durations = [0], [1]
    for i in range(1, nframes):
        diff = (out[i] != out[keep[-1]]).any(-1).mean()
        if diff <= args.dedupe:
            durations[-1] += 1
        else:
            keep.append(i)
            durations.append(1)
    ms_per = 1000.0 / fps
    cum = np.round(np.cumsum(durations) * ms_per).astype(int)
    ms = np.diff(np.concatenate([[0], cum])).tolist()

    stem = args.input.stem + "_repixel"
    outp = args.output or args.input.with_name(stem + ".png")
    mode = "RGBA" if out.shape[-1] == 4 else "RGB"
    imgs = [Image.fromarray(out[i], mode) for i in keep]
    ncolors = len(np.unique(out[keep].reshape(-1, out.shape[-1]), axis=0))
    if outp.suffix.lower() == ".gif":
        imgs = gif_frames(out[keep])
        if imgs is None:
            print(f"warning: {ncolors} colours > 256, GIF will be lossy; use --colors 255 "
                  "or a .png output", file=sys.stderr)
            imgs = [Image.fromarray(out[i], mode) for i in keep]
        imgs[0].save(outp, save_all=True, append_images=imgs[1:], duration=ms, loop=0,
                     disposal=2, optimize=False)
    else:
        if len(imgs) == 1:
            to_image(out[keep[0]])[0].save(outp, optimize=True)
        else:
            imgs[0].save(outp, save_all=True, append_images=imgs[1:], duration=ms, loop=0,
                         disposal=0, blend=0)
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(keep):
            to_image(out[i])[0].save(args.frames / f"{stem}_{j:04d}.png", optimize=True)

    print(f"{args.input.name}: {w}x{h} {src_fps:g}fps"
          + (f" -> {fps:g}fps" if args.fps else "") + f", {nframes} frames  pixel size "
          f"{px_t.min():.3f}-{px_t.max():.3f} ({how} on frame 1, tracked after)")
    print(f"  -> {outp}  {out.shape[2]}x{out.shape[1]}, {len(keep)} unique frames, "
          f"{ncolors} colours")
    if min(confs) < 0.3:
        print(f"warning: grid was weak on some frames (min confidence {min(confs):.2f})",
              file=sys.stderr)
    if args.frames:
        print(f"  frames -> {args.frames}/")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def ask_period(cands: list[Candidate], w: int, h: int) -> float:
    print("\nCandidate pixel sizes (source px per art pixel):", file=sys.stderr)
    for i, c in enumerate(cands, 1):
        print(f"  {i}) {c.period:7.3f}  ->  ~{w / c.period:.0f} x {h / c.period:.0f}   "
              f"(confidence {c.score:.2f})", file=sys.stderr)
    while True:
        ans = input(f"Pick 1-{len(cands)}, or type a size [1]: ").strip()
        if not ans:
            return cands[0].period
        try:
            if ans.isdigit() and 1 <= int(ans) <= len(cands):
                return cands[int(ans) - 1].period
            v = float(ans)
            if v >= 1:
                return v
        except ValueError:
            pass
        print("  ?", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Recover true-resolution pixel art from an upscaled/JPEG image or video.")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path,
                    help="output file (default: <input>_repixel.png; for video, a .gif "
                         "name writes a GIF instead of an animated PNG)")
    ap.add_argument("-s", "--scale", type=float,
                    help="source pixels per art pixel; skips detection (fractions OK, e.g. 2.87)")
    ap.add_argument("--scale-y", type=float,
                    help="separate vertical scale for non-square pixels (requires --scale)")
    ap.add_argument("--ask", action="store_true",
                    help="always show candidates and ask which pixel size to use")
    ap.add_argument("--no-ask", action="store_true",
                    help="never prompt, even if detection is uncertain")
    ap.add_argument("--no-snap", action="store_true",
                    help="use a perfectly uniform grid (don't snap lines to edges)")
    ap.add_argument("--merge", type=float, default=18.0,
                    help="merge colours closer than this RGB distance (0 = off, default 18)")
    ap.add_argument("--colors", type=int,
                    help="quantise to at most N colours (k-means) instead of --merge")
    ap.add_argument("-t", "--transparent", nargs="?", const="auto", metavar="RRGGBB|auto",
                    help="make this background colour transparent, e.g. ff00ff, '#f0f' or magenta "
                         "(auto-detects from the border if no colour given)")
    ap.add_argument("--key-tol", type=float, default=100.0,
                    help="with --transparent: pixels within this RGB distance of the background "
                         "colour are cut (default 100; raise to cut more pinkish edge pixels)")
    ap.add_argument("--no-despill", dest="despill", action="store_false",
                    help="with --transparent: don't remove the background's tint from pixels "
                         "near the sprite edge (removed by default, e.g. purple-tinted "
                         "outlines on magenta)")
    ap.add_argument("--trim", action="store_true",
                    help="crop transparent margins (with --transparent)")
    ap.add_argument("--preview", type=int, nargs="?", const=0, metavar="N",
                    help="images: also save a nearest-neighbour upscale (N x, default: source size)")
    vg = ap.add_argument_group("video")
    vg.add_argument("--frames", type=Path, metavar="DIR",
                    help="also write each unique frame as a PNG into DIR")
    vg.add_argument("--dedupe", type=float, default=0.002, metavar="FRAC",
                    help="merge consecutive frames differing in at most this fraction of "
                         "pixels (default 0.002; 0 = exact only, -1 = keep every frame)")
    vg.add_argument("--fps", type=float,
                    help="resample to this frame rate first (e.g. 8 or 12 for sprite-style animation)")
    vg.add_argument("--no-deblip", action="store_true",
                    help="keep single-frame pixel flickers (removed by default)")
    vg.add_argument("--extent", choices=["intersect", "union"], default="intersect",
                    help="canvas = area seen in every frame (default) or in any frame")
    args = ap.parse_args(argv)

    if args.input.suffix.lower() in VIDEO_EXTS:
        return process_video(args)
    return process_image(args)


if __name__ == "__main__":
    sys.exit(main())
