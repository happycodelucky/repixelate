#!/usr/bin/env python3
"""
Repixel - recover true-resolution pixel art from upscaled / JPEG-damaged images and video.

Still images
  1. Edge profiles   - per axis, sum colour jumps between neighbouring columns/rows,
                       ignoring the small jumps JPEG noise produces.
  2. Grid period     - candidate cell sizes (fractional sizes allowed) are the peaks of
                       the edge profiles' spectrum. Each candidate is then checked by
                       actually redrawing the image on that grid: candidates that
                       reproduce the image much worse than the others (sprite spacing,
                       JPEG blocks) are thrown out, and harmonics are resolved by
                       preferring the largest period that still explains the edges.
  3. Grid alignment  - the phase of the winning frequency gives the grid offset. Slow
                       drift of the grid across the image is followed, and for larger
                       cells each line is snapped to the nearest real edge, so uneven
                       (AI-generated) grids still line up.
  4. Sampling        - each cell's colour is the median of its centre region, which
                       avoids the blurred/ringing borders.
  5. Palette cleanup - colours that differ only by noise are merged. How far apart
                       they may be is measured from the image's own noise, so clean
                       input is returned exactly; or quantise to N colours (--colors).
  6. Transparency    - optionally the background colour is cut (everywhere, or only
                       where it is connected to the border) and its tint is cleaned
                       off the sprite edges. Alpha is always fully on or fully off.
  7. Output          - lossless PNG at 1 art pixel = 1 image pixel.

Video (needs ffmpeg)
  The grid is measured on every frame (so camera zooms are followed), every frame is
  placed on one shared art-pixel canvas and all frames share one palette. Poses that
  the video holds for several frames (animation "on threes") are detected and become
  one clean frame each. Optionally each sprite is separated (--split), the repeating
  cycle is found (--loop) and the animation is reduced to a set number of frames
  (--poses). Output is a lossless animated PNG, a GIF, a sprite sheet (--sheet) or a
  folder of PNG frames.

Use as a library:  repixel_image(array, Options(...)) -> Result
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

__version__ = "0.3.0"

# --------------------------------------------------------------------------- #
# Options and results
# --------------------------------------------------------------------------- #


@dataclass
class Options:
    """Settings for `repixel_image` (the CLI flags map onto these one to one)."""
    scale: float | None = None        # source px per art px; None = detect
    scale_y: float | None = None      # separate vertical scale (needs scale)
    snap: bool = True                 # follow uneven grids
    merge: float | None = None        # colour merge distance; None = from image noise
    colors: int | None = None         # quantise to at most N colours instead
    transparent: str | None = None    # background colour, or "auto"
    key: str = "full"                 # "full": cut the colour everywhere; "contiguous": only where it touches the border
    key_tol: float = 100.0            # RGB distance from the background that is cut
    despill: bool = True              # remove background tint from edge pixels
    trim: bool = False                # crop transparent margins


@dataclass
class Candidate:
    period: float
    score: float                      # edge coherence, 0..1
    error: float = 0.0                # mean abs error when the image is redrawn on it


@dataclass
class Detection:
    period_x: float
    period_y: float
    confidence: float
    uncertain: bool
    candidates: list[Candidate] = field(default_factory=list)   # best first
    rivals: list[Candidate] = field(default_factory=list)       # plausible alternatives


@dataclass
class Result:
    pixels: np.ndarray                # (h, w, 3) or (h, w, 4) uint8, 1 px per art pixel
    scale_x: float
    scale_y: float
    error: float                      # mean abs reconstruction error, 0..255
    detection: Detection | None       # None when the scale was given
    background: np.ndarray | None     # colour that was keyed out, if any
    merge_tol: float                  # colour merge distance that was used


# --------------------------------------------------------------------------- #
# Grid detection
# --------------------------------------------------------------------------- #

EDGE_NOISE = 24.0   # sum-of-|dRGB| below this is treated as JPEG noise
VALID_ERR = (2.5, 1.0)   # a candidate is plausible if error <= a * best error + b
HARMONIC_KEEP = 0.6      # a k-times-larger period must keep this much coherence ...
HARMONIC_ERR = (1.65, 0.3)   # ... and must not redraw the image much worse


def edge_profile(img: np.ndarray, axis: int) -> np.ndarray:
    """Edge strength at each boundary along `axis`.

    Returned array has length n+1 where index b is the boundary between pixel b-1
    and pixel b (so 0 and n are the image borders and are always zero).
    axis=1 -> vertical lines (x positions), axis=0 -> horizontal lines (y positions).
    """
    d = np.abs(np.diff(img, axis=axis)).sum(axis=2)
    np.maximum(d - EDGE_NOISE, 0.0, out=d)
    prof = d.sum(axis=0, dtype=np.float64) if axis == 1 else d.sum(axis=1, dtype=np.float64)
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


def _refine(profs, period: float, n: int, span: float = 1.0) -> Candidate:
    """Fine search around a period (span = half-width in units of 1/n cycles/px)."""
    f0 = 1.0 / period
    half = span / n
    freqs = np.linspace(f0 - half, f0 + half, 81)
    freqs = freqs[freqs > 0]
    periods = 1.0 / freqs
    s = _score(profs, periods)
    i = int(np.argmax(s))
    return Candidate(float(periods[i]), float(s[i]))


def _spectrum_peaks(profs, n: int, pmin: float, pmax: float, top: int) -> list[Candidate]:
    """Strongest distinct peaks of the combined edge-profile spectrum."""
    nfft = 8 * (1 << int(np.ceil(np.log2(n + 2))))
    num, den = np.zeros(nfft // 2 + 1), 0.0
    for p in profs:
        mass = p.sum()
        if mass > 0:
            num += np.abs(np.fft.rfft(p, nfft))
            den += mass
    if den == 0:
        return []
    s = num / den
    k0, k1 = max(1, int(np.ceil(nfft / pmax))), int(np.floor(nfft / pmin))
    seg = s[k0:k1 + 1]
    if seg.size < 3:
        return []
    pk = np.nonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
    out: list[Candidate] = []
    for i in pk[np.argsort(-seg[pk])][:4 * top]:
        c = _refine(profs, nfft / (k0 + i), n)
        if all(abs(c.period - o.period) / o.period > 0.02 for o in out):
            out.append(c)
        if len(out) >= top:
            break
    return out


def grid_offset(prof: np.ndarray, period: float) -> float:
    """Where the grid lines sit, in [0, period): phase of the edge profile at 1/period."""
    c = _coherence(prof, np.array([1.0 / period]))[0]
    if abs(c) < 1e-9:
        return 0.0
    # edges sit where 2*pi*x/p == -arg(c)  (mod 2*pi)
    return float((-np.angle(c) / (2 * np.pi) * period) % period)


class _Redraw:
    """Quick estimate of how well a uniform grid of a given size reproduces the image:
    every pixel is compared with the pixel at the centre of its cell. Works on a fixed
    random subset of rows and columns so the cost doesn't depend on the image size."""

    def __init__(self, img: np.ndarray, profs, limit: int = 512):
        self.img, self.profs = img, profs
        h, w, _ = img.shape
        rng = np.random.default_rng(0)
        self.rows = np.sort(rng.choice(h, min(h, limit), replace=False))
        self.cols = np.sort(rng.choice(w, min(w, limit), replace=False))
        self.sub = img[self.rows][:, self.cols]

    def error(self, px: float, py: float | None = None) -> float:
        py = py or px
        h, w, _ = self.img.shape
        ox, oy = grid_offset(self.profs[0], px), grid_offset(self.profs[1], py)
        cx = np.floor(ox + (np.floor((self.cols + 0.5 - ox) / px) + 0.5) * px)
        cy = np.floor(oy + (np.floor((self.rows + 0.5 - oy) / py) + 0.5) * py)
        cx = np.clip(cx, 0, w - 1).astype(int)
        cy = np.clip(cy, 0, h - 1).astype(int)
        return float(np.abs(self.sub - self.img[cy][:, cx]).mean())


def _is_harmonic(p: float, fundamental: float) -> bool:
    """True if p is (roughly) fundamental / k for an integer k."""
    k = fundamental / p
    return k > 1.5 and abs(k - round(k)) < 0.04 * k


def detect_grid(img: np.ndarray, profs=None, pmin: float = 2.0, pmax: float | None = None) -> Detection | None:
    """Find the art-pixel size of an image. Returns None if it has no usable edges."""
    h, w, _ = img.shape
    if profs is None:
        profs = [edge_profile(img, 1), edge_profile(img, 0)]
    n = max(w, h)
    # Edges are sampled at integer positions, so periods below 2 px alias onto real
    # ones (f and 1-f look identical). 2 px is therefore the smallest detectable cell.
    pmin = max(pmin, 2.0)
    if pmax is None:
        pmax = min(128.0, n / 4)
    peaks = _spectrum_peaks(profs, n, pmin, pmax, top=16)
    if not peaks:
        return None

    # Redraw the image on every candidate grid. Peaks that come from the layout
    # (evenly spaced sprites, JPEG's 8 px blocks) reproduce it far worse than a real
    # pixel grid or its harmonics do, so they are dropped here.
    redraw = _Redraw(img, profs)
    for c in peaks:
        c.error = redraw.error(c.period)
    e_min = min(c.error for c in peaks)
    limit = VALID_ERR[0] * e_min + VALID_ERR[1]
    valid = [c for c in peaks if c.error <= limit] or peaks
    best = max(valid, key=lambda c: c.score)

    # Harmonic resolution: a grid of size P also lines up perfectly at P/2, P/3 ...
    # so walk up integer multiples of the best period and keep the largest one that
    # still explains most of the edges without redrawing the image much worse.
    fundamental = best
    for k in range(2, 9):
        if best.period * k > pmax:
            break
        c = _refine(profs, best.period * k, n)
        if c.score < HARMONIC_KEEP * best.score:
            continue
        c.error = redraw.error(c.period)
        if c.error <= HARMONIC_ERR[0] * best.error + HARMONIC_ERR[1]:
            fundamental = c

    ranked = [fundamental]
    for c in sorted(valid, key=lambda c: -c.score):
        if all(abs(c.period - r.period) / r.period > 0.03 for r in ranked):
            ranked.append(c)
    rivals = [c for c in ranked[1:] if not _is_harmonic(c.period, fundamental.period)]
    close = [c for c in rivals if c.score > 0.9 * fundamental.score]
    uncertain = fundamental.score < 0.25 or bool(close)

    # The axes normally share one period. Only when each axis, measured on its own,
    # clearly disagrees (non-square pixels) are they allowed to differ.
    px = py = fundamental.period
    ax = _refine([profs[0]], px, w, span=0.5).period if profs[0].sum() > 0 else px
    ay = _refine([profs[1]], py, h, span=0.5).period if profs[1].sum() > 0 else py
    if 0.003 < abs(ax - ay) / px < 0.02:
        px, py = ax, ay
    return Detection(px, py, fundamental.score, uncertain, ranked[:6], rivals[:5])


def find_period_near(profs, n: int, p0: float, rel: float = 0.08) -> Candidate:
    """Best period within +-rel of p0 (for tracking a slowly zooming video)."""
    f0 = 1.0 / p0
    freqs = np.arange(f0 / (1 + rel), f0 / (1 - rel), 1.0 / (4 * n))
    s = _score(profs, 1.0 / freqs)
    return _refine(profs, 1.0 / freqs[int(np.argmax(s))], n)


# --------------------------------------------------------------------------- #
# Grid lines
# --------------------------------------------------------------------------- #

def _capture(prof: np.ndarray, lines: np.ndarray) -> float:
    """Edge strength lying on the given lines (linear interpolation)."""
    return float(np.interp(lines, np.arange(prof.size), prof).sum())


def _drift(prof: np.ndarray, period: float, offset: float, cells: float = 4.0) -> np.ndarray:
    """How far the real grid has drifted from the uniform one at each position.

    The edge profile is demodulated at the grid frequency; the phase of the result,
    smoothed over a few cells, is the local grid offset.
    """
    x = np.arange(prof.size)
    z = prof * np.exp(-2j * np.pi * (x - offset) / period)
    sigma = max(cells * period, 6.0)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    c = np.convolve(z, k)[r:r + z.size]
    c = c + 0.05 * np.abs(c).mean()        # where there are no edges, assume no drift
    return -np.unwrap(np.angle(c)) * period / (2 * np.pi)


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

    if snap:
        # Follow slow drift, but only when that clearly puts more edges on the lines:
        # on an even grid the uniform lines are already the best estimate.
        drift = _drift(prof, period, offset)
        xs = np.arange(prof.size)
        moved = lines + np.interp(lines, xs, drift)
        moved = lines + np.interp(moved, xs, drift)
        inside = (lines > 0) & (lines < n)
        if _capture(prof, moved[inside]) > 1.15 * _capture(prof, lines[inside]):
            lines = moved
        if period >= 5:
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

def _sample_box(xs: np.ndarray, ys: np.ndarray) -> tuple[int, int]:
    """Size in pixels of the centre region sampled in each cell (half the cell)."""
    return max(1, int(np.median(np.diff(xs)) * 0.5)), max(1, int(np.median(np.diff(ys)) * 0.5))


def sample_count(xs: np.ndarray, ys: np.ndarray) -> int:
    bw, bh = _sample_box(xs, ys)
    return bw * bh


def sample_cells(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Median colour of the centre region of every cell."""
    h, w, _ = img.shape
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    bw, bh = _sample_box(xs, ys)
    ox = np.arange(bw) - (bw - 1) / 2
    oy = np.arange(bh) - (bh - 1) / 2
    px = np.clip(np.floor(cx[:, None] + ox[None, :]).astype(int), 0, w - 1)  # (nx, bw)
    py = np.clip(np.floor(cy[:, None] + oy[None, :]).astype(int), 0, h - 1)  # (ny, bh)
    block = img[py[:, None, :, None], px[None, :, None, :]]  # (ny, nx, bh, bw, 3)
    ny, nx = block.shape[:2]
    if bw * bh == 1:
        return block.reshape(ny, nx, -1).astype(np.float64)
    return np.median(block.reshape(ny, nx, bh * bw, -1), axis=2)


def reconstruction_error(img: np.ndarray, cells: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """Mean abs difference between the source and the cells drawn back on the grid."""
    h, w, _ = img.shape
    xi = np.searchsorted(xs, np.arange(w) + 0.5, side="right") - 1
    yi = np.searchsorted(ys, np.arange(h) + 0.5, side="right") - 1
    vx = (xi >= 0) & (xi < len(xs) - 1)
    vy = (yi >= 0) & (yi < len(ys) - 1)
    recon = cells[np.clip(yi, 0, cells.shape[0] - 1)][:, np.clip(xi, 0, cells.shape[1] - 1)]
    m = vy[:, None] & vx[None, :]
    return float(np.abs(recon[..., :3] - img[..., :3]).mean(axis=2)[m].mean())


def noise_level(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """Per-channel noise (sigma, 0..255) inside cells.

    Looks at neighbouring pixel pairs that both lie wholly inside one cell: in clean
    pixel art they are identical, so any difference is compression noise or blur.
    Pairs that are exactly equal are ignored, so flat JPEG blocks don't hide the noise
    elsewhere, and a clean image measures exactly zero.
    """
    h, w, _ = img.shape

    def inside_pairs(lines, n):
        # Large cells have their lines snapped to whole pixels, so the true boundary can
        # be up to a pixel away: keep clear of it. Small cells have no room to spare.
        m = 1.0 if np.median(np.diff(lines)) >= 5 else 0.0
        cell = np.searchsorted(lines, np.arange(n) - m, side="right") - 1      # cell of left edge
        cell_r = np.searchsorted(lines, np.arange(n) + 1 + m, side="left") - 1  # cell of right edge
        whole = (cell == cell_r) & (cell >= 0) & (cell < len(lines) - 1)
        return whole, np.nonzero(whole[:-1] & whole[1:] & (cell[:-1] == cell[1:]))[0]

    wx, pairs_x = inside_pairs(xs, w)
    wy, pairs_y = inside_pairs(ys, h)
    rows, cols = np.nonzero(wy)[0], np.nonzero(wx)[0]
    diffs = []
    if pairs_x.size and rows.size:
        r = rows[:: max(1, rows.size // 400)]
        diffs.append(np.abs(img[r][:, pairs_x] - img[r][:, pairs_x + 1]).ravel())
    if pairs_y.size and cols.size:
        c = cols[:: max(1, cols.size // 400)]
        diffs.append(np.abs(img[pairs_y][:, c] - img[pairs_y + 1][:, c]).ravel())
    if not diffs:
        return 0.0
    d = np.concatenate(diffs)
    d = d[d > 0]
    if d.size < 0.01 * sum(x.size for x in diffs):
        return 0.0
    d = np.minimum(d, 3 * np.median(d))     # pairs straddling a real edge don't count
    return float(d.mean()) / 1.128          # mean |difference of two samples| -> sigma


MERGE_TIGHT = 2.5       # colours within this many noise sigmas are always one colour
MERGE_WIDE = 6.0        # stragglers are absorbed by a much commoner colour up to here
MERGE_MIN = 18.0        # lower limit for the wide distance once any noise is present:
                        # generated art also varies from cell to cell by about this much
MERGE_MAX = 30.0        # upper limit for the wide distance
STRAGGLER = 0.2         # "much commoner" = the straggler has under this share of it


def merge_tolerance(sigma: float) -> float:
    """Widest colour merge distance for a measured noise level. A clean image (no
    noise at all) gets 0, so its colours are kept exactly as they are."""
    if sigma <= 0:
        return 0.0
    return float(np.clip(MERGE_WIDE * sigma, MERGE_MIN, MERGE_MAX))


def _greedy_clusters(cols: np.ndarray, weight: np.ndarray, order: np.ndarray, tol: float,
                     share: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign colours, in the given order, to the nearest existing cluster within `tol`,
    else start a new cluster. With `share`, a colour may only join a cluster whose
    founding colour outweighs it (its weight is under `share` of the founder's).
    Returns (assignment, cluster means, cluster weights).

    Colours are tested against the current clusters a batch at a time; everything up
    to the first colour that needs a new cluster is settled in one step, which gives
    exactly the one-at-a-time result.
    """
    seed = np.empty((len(cols), 3))
    seed_w = np.empty(len(cols))
    n = 0
    assign = np.empty(len(cols), int)
    tol2 = tol * tol
    i, batch = 0, 64
    while i < len(order):
        idx = order[i:i + batch]
        first = 0
        if n:
            d = ((cols[idx, None, :] - seed[None, :n, :]) ** 2).sum(-1)
            if share is not None:
                d = np.where(weight[idx, None] < share * seed_w[None, :n], d, np.inf)
            j = d.argmin(1)
            miss = d[np.arange(len(idx)), j] > tol2
            first = int(np.argmax(miss)) if miss.any() else len(idx)
            assign[idx[:first]] = j[:first]
        if first < len(idx):
            k = idx[first]
            seed[n], seed_w[n] = cols[k], weight[k]
            assign[k] = n
            n += 1
            i += first + 1
            batch = max(64, batch // 2)
        else:
            i += first
            batch = min(8192, batch * 2)
    total = np.zeros((n, 3))
    mass = np.zeros(n)
    np.add.at(total, assign, cols * weight[:, None])
    np.add.at(mass, assign, weight)
    return assign, total / np.maximum(mass, 1e-9)[:, None], mass


def merge_similar(cells: np.ndarray, tol: float, tight: float | None = None) -> np.ndarray:
    """Merge colours that differ only by noise.

    Two passes, both greedy with the commonest colours first:
      1. colours within `tight` of each other become one colour (noise makes almost
         every sample unique, so "common" is counted over a coarse colour bin);
      2. a rare colour within `tol` of a much commoner one is absorbed by it. This
         clears noise tails and one-off blends, while two shades that are both in
         real use stay separate even when they are close.
    `tight` defaults to tol * MERGE_TIGHT / MERGE_WIDE.
    """
    flat = cells.reshape(-1, 3)
    q = np.round(flat).astype(np.int64)
    if tol <= 0:
        return q.reshape(cells.shape).astype(float)
    if tight is None:
        tight = tol * MERGE_TIGHT / MERGE_WIDE
    key = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    ukey, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    uniq = np.stack([ukey >> 16, (ukey >> 8) & 255, ukey & 255], axis=1).astype(float)
    coarse = ((ukey >> 19) << 10) | (((ukey >> 11) & 31) << 5) | ((ukey >> 3) & 31)
    _, cinv = np.unique(coarse, return_inverse=True)
    density = np.bincount(cinv, weights=counts)[cinv]
    a1, mean1, mass1 = _greedy_clusters(uniq, counts.astype(float), np.lexsort((-counts, -density)), tight)
    a2, mean2, _ = _greedy_clusters(mean1, mass1, np.argsort(-mass1, kind="stable"), tol, STRAGGLER)
    return mean2[a2[a1[inv]]].reshape(cells.shape)


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


def absorb_blends(out: np.ndarray, tol: float) -> np.ndarray:
    """Replace rare in-between colours with the neighbour they are a blend of.

    Blur and chroma subsampling leave a cell on the border between two colours with a
    mixture of both. Such a cell has a rare colour that lies on the line between the
    colours of two opposite neighbours; it takes the nearer of the two. Colours in
    common use (real shading and anti-aliasing) are never touched.
    """
    if tol <= 0:
        return out
    H, W, _ = out.shape
    q = out.astype(np.int64)
    key = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    cnt = counts[inv].reshape(H, W).astype(float)
    c = out.astype(float)
    pc = np.pad(c, ((1, 1), (1, 1), (0, 0)), mode="edge")
    pn = np.pad(cnt, 1, mode="edge")
    best_d = np.full((H, W), np.inf)
    best = c.copy()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = pc[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        b = pc[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        na = pn[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        nb = pn[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        ab = b - a
        len2 = (ab ** 2).sum(-1)
        t = ((c - a) * ab).sum(-1) / np.maximum(len2, 1e-9)
        dist = np.sqrt(((c - (a + t[..., None] * ab)) ** 2).sum(-1))
        ok = ((len2 > (2 * tol) ** 2) & (t > 0.1) & (t < 0.9) & (dist <= tol)
              & (cnt < STRAGGLER * np.minimum(na, nb)) & (dist < best_d))
        best_d = np.where(ok, dist, best_d)
        best = np.where(ok[..., None], np.where((t < 0.5)[..., None], a, b), best)
    return best.astype(np.uint8)


def clean_colors(cells: np.ndarray, tol: float, colors: int | None = None) -> np.ndarray:
    """Palette cleanup: merge within `tol`, or quantise to `colors` colours."""
    if colors:
        cells = kmeans_quantize(cells, colors)
    else:
        cells = merge_similar(cells, tol)
    out = np.clip(np.round(cells), 0, 255).astype(np.uint8)
    return out if colors else absorb_blends(out, tol)


# --------------------------------------------------------------------------- #
# Transparency
# --------------------------------------------------------------------------- #

def detect_background(img: np.ndarray) -> np.ndarray:
    """Most common (coarsely quantised) colour along the image border."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, 3)
    q = (border // 16).astype(int)
    keys, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    top = np.argmax(counts)
    return np.median(border[inv.reshape(-1) == top], axis=0)


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
        raise ValueError(f"can't read colour {s!r} (try ff00ff, #ff00ff or magenta)") from None


def resolve_background(spec: str, img: np.ndarray) -> np.ndarray:
    """Background colour: auto-detected, or the given colour snapped to what the image
    really contains (JPEG shifts flat colours, e.g. ff00ff often becomes fd33f8)."""
    if spec == "auto":
        return detect_background(img)
    want = parse_color(spec)
    px = img[::3, ::3].reshape(-1, 3)
    near = px[np.sqrt(((px - want) ** 2).sum(1)) <= 100]
    return np.median(near, axis=0) if len(near) >= 0.01 * len(px) else want


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """The part of `mask` that is connected (4-way) to the image border."""
    reach = np.zeros_like(mask)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = mask[0], mask[-1], mask[:, 0], mask[:, -1]
    while True:
        grow = reach.copy()
        grow[1:] |= reach[:-1]
        grow[:-1] |= reach[1:]
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow &= mask
        if (grow == reach).all():
            return reach
        reach = grow


def key_background(rgb: np.ndarray, bg: np.ndarray, tol: float, contiguous: bool = False) -> np.ndarray:
    """RGB -> RGBA with cells near `bg` made fully transparent.

    rgb may be (h, w, 3) or (t, h, w, 3). With `contiguous`, only background that is
    connected to the image border is cut, so the same colour used inside the art
    (and any enclosed gap) is kept.
    """
    dist = np.sqrt(((rgb.astype(float) - bg) ** 2).sum(-1))
    cut = dist <= tol
    if contiguous:
        if cut.ndim == 2:
            cut = _border_connected(cut)
        else:
            cut = np.stack([_border_connected(c) for c in cut])
    alpha = np.where(cut, 0, 255).astype(np.uint8)
    rgb = rgb.copy()
    rgb[cut] = 0
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


_YCC = np.array([[0.299, 0.587, 0.114],
                 [-0.168736, -0.331264, 0.5],
                 [0.5, -0.418688, -0.081312]])
_YCC_INV = np.linalg.inv(_YCC)
SPILL_MIN = 8.0     # background tint below this (chroma units) is left alone
SPILL_CUT = 0.35    # an unexplained edge pixel at least this much background is cut


def _neighbourhood(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """Dilate a boolean mask by r pixels (8-connected)."""
    out = mask.copy()
    H, W = mask.shape
    for _ in range(r):
        p = np.pad(out, 1, constant_values=False)
        out = np.any([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return out


def _window(a: np.ndarray, r: int, fill) -> list[np.ndarray]:
    """The (2r+1)^2 shifted copies of a 2-D (or 2-D + channels) array."""
    H, W = a.shape[:2]
    pad = [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2)
    p = np.pad(a, pad, constant_values=fill)
    return [p[r + dy:r + dy + H, r + dx:r + dx + W]
            for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def despill(img: np.ndarray, bg: np.ndarray, near: int = 2, snap_tol: float = 28.0) -> np.ndarray:
    """Clean the background's tint off pixels within `near` px of transparency.

    JPEG and video smear colour, so edge pixels pick up the background hue (a dark
    outline on magenta turns purple), and blur leaves pixels that are part sprite,
    part background. For each edge pixel:
      * how much background hue it has beyond what the art around it has is measured
        (so a sprite that really is purple next to a magenta background is left alone);
      * that excess hue is removed, keeping brightness;
      * if the result matches a clean colour used within 3 px, it takes that colour;
      * if not, and the pixel could be mostly background, it is cut (made transparent)
        rather than left as an off-colour speck; otherwise it keeps the corrected colour.
    Colour is only ever changed towards what the sprite already contains.
    """
    out = img.copy()
    op = out[..., 3] > 0
    if op.all() or not op.any():
        return out
    ybg = _YCC @ bg
    mag = float(np.linalg.norm(ybg[1:]))
    if mag < 10:                                      # grey/white/black bg has no hue to spill
        return out
    bdir = ybg[1:] / mag
    zone = op & _neighbourhood(~op, near)             # pixels close to transparency
    ycc = out[..., :3].astype(float) @ _YCC.T
    spill = ycc[..., 1:] @ bdir
    # How much of that hue the surrounding art legitimately has: a low percentile over
    # the opaque pixels within 3 px, so a few tinted neighbours don't raise it.
    near_spill = np.sort(np.stack(_window(np.where(op, spill, np.inf), 3, np.inf)), axis=0)
    count = np.isfinite(near_spill).sum(0)
    quarter = np.clip(((count - 1) * 0.25).astype(int), 0, None)
    ref = np.take_along_axis(near_spill, quarter[None], 0)[0]
    ref = np.where(count > 0, ref, 0.0)
    excess = np.where(zone, np.clip(spill - np.clip(ref, 0, None), 0, None), 0)
    excess[excess < SPILL_MIN] = 0
    tinted = excess > 0
    if not tinted.any():
        return out
    ycc[..., 1:] -= excess[..., None] * bdir
    fixed = np.clip(np.round(ycc @ _YCC_INV.T), 0, 255)

    # Nearest clean colour in the neighbourhood.
    clean = op & ~tinted
    best = np.full(op.shape, np.inf)
    snap = fixed.copy()
    for col, ok in zip(_window(out[..., :3].astype(float), 3, 0.0), _window(clean, 3, False)):
        d = np.where(ok, ((fixed - col) ** 2).sum(-1), np.inf)
        better = d < best
        best = np.where(better, d, best)
        snap = np.where(better[..., None], col, snap)
    explained = tinted & (best <= snap_tol ** 2)

    # Unexplained pixels: cut those that may be mostly background. The hue gives one
    # estimate of the background share; brightness gives the most it could be.
    y = ycc[..., 0]
    share_max = np.where(y < ybg[0], y / max(ybg[0], 1e-9), (255 - y) / max(255 - ybg[0], 1e-9))
    share = np.minimum(excess / mag, share_max)
    cut = tinted & ~explained & (share >= SPILL_CUT)

    # What is left keeps its corrected colour, moved onto the nearest clean colour used
    # anywhere in the image if one is close, so the palette doesn't fill with one-offs.
    left = tinted & ~explained & ~cut
    if left.any() and clean.any():
        pal = np.unique(out[clean][:, :3], axis=0).astype(float)
        cols, inv = np.unique(fixed[left], axis=0, return_inverse=True)
        d = ((cols[:, None, :] - pal[None]) ** 2).sum(-1)
        j = d.argmin(1)
        near_pal = d[np.arange(len(cols)), j] <= snap_tol ** 2
        cols = np.where(near_pal[:, None], pal[j], cols)
        fixed[left] = cols[inv.reshape(-1)]
    rgb = np.where(explained[..., None], snap, fixed)
    out[..., :3] = np.where(tinted[..., None], rgb, out[..., :3])
    out[cut] = 0
    return out


def trim_box(alpha: np.ndarray) -> tuple[slice, slice] | None:
    """Bounding box of non-transparent pixels; alpha may be (h, w) or (t, h, w)."""
    a = alpha.reshape(-1, *alpha.shape[-2:]).any(0)
    if not a.any():
        return None
    r = np.nonzero(a.any(1))[0]
    c = np.nonzero(a.any(0))[0]
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


# --------------------------------------------------------------------------- #
# Still images
# --------------------------------------------------------------------------- #

def _split_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """float32 RGB plus, if the input has real transparency, its alpha channel.
    Transparent areas are filled with a flat colour the art doesn't use, so that the
    sprite outline still shows up as an edge."""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    rgb = img[..., :3].astype(np.float32)
    if img.shape[2] < 4 or (img[..., 3] >= 255).all():
        return rgb, None
    alpha = img[..., 3].astype(np.float32)
    opaque = rgb[alpha >= 128]
    fills = np.array([[255, 0, 255], [0, 255, 0], [0, 255, 255], [255, 255, 0]], np.float32)
    if len(opaque):
        sample = opaque[:: max(1, len(opaque) // 20000)]
        far = [np.sqrt(((sample - f) ** 2).sum(1)).min() for f in fills]
        fill = fills[int(np.argmax(far))]
    else:
        fill = fills[0]
    a = (alpha / 255.0)[..., None]
    return rgb * a + fill * (1 - a), alpha


def repixel_image(img: np.ndarray, opts: Options | None = None,
                  detection: Detection | None = None) -> Result | None:
    """Recover the true-resolution art from an image array (h, w, 3 or 4).

    Returns None when no pixel grid can be found. Pass `detection` to reuse a grid
    found earlier with `detect_grid`.
    """
    opts = opts or Options()
    rgb, alpha = _split_alpha(np.asarray(img))
    h, w, _ = rgb.shape
    profs = [edge_profile(rgb, 1), edge_profile(rgb, 0)]
    if opts.scale:
        sx, sy = opts.scale, opts.scale_y or opts.scale
        detection = None
    else:
        detection = detection or detect_grid(rgb, profs)
        if detection is None:
            return None
        sx, sy = detection.period_x, detection.period_y

    xs, _ = grid_lines(profs[0], sx, w, opts.snap)
    ys, _ = grid_lines(profs[1], sy, h, opts.snap)
    raw = sample_cells(rgb, xs, ys)
    err = reconstruction_error(rgb, raw, xs, ys)
    tol = opts.merge if opts.merge is not None else merge_tolerance(noise_level(rgb, xs, ys))
    out = clean_colors(raw, tol, opts.colors)

    bg = None
    if opts.transparent:
        bg = resolve_background(opts.transparent, rgb if alpha is None else rgb[alpha >= 128])
        out = key_background(out, bg, opts.key_tol, opts.key == "contiguous")
        # Smearing needs noise or blur: a clean image has none, so any background-like
        # hue at an edge is part of the art. It only reaches a second ring of cells
        # when cells are small.
        if opts.despill and (tol > 0 or opts.merge is not None):
            out = despill(out, bg, near=2 if min(sx, sy) < 4 else 1)
    if alpha is not None:
        a = sample_cells(alpha[..., None], xs, ys)[..., 0] >= 128
        if out.shape[2] == 3:
            out = np.concatenate([out, np.full((*out.shape[:2], 1), 255, np.uint8)], -1)
        out[~a] = 0
    if out.shape[2] == 4 and opts.trim:
        box = trim_box(out[..., 3])
        if box:
            out = out[box]
    return Result(out, sx, sy, err, detection, bg, tol)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

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


def save_animation(frames: np.ndarray, ms: list[int], path: Path) -> bool:
    """Write frames (t, h, w, 3|4) as a GIF (by extension) or animated PNG.
    Returns False if the result had to be lossy (GIF with too many colours)."""
    mode = "RGBA" if frames.shape[-1] == 4 else "RGB"
    if len(frames) == 1:
        if path.suffix.lower() == ".gif":
            (gif_frames(frames) or [Image.fromarray(frames[0], mode)])[0].save(path)
        else:
            to_image(frames[0])[0].save(path, optimize=True)
        return True
    if path.suffix.lower() == ".gif":
        imgs = gif_frames(frames)
        exact = imgs is not None
        if imgs is None:
            imgs = [Image.fromarray(f, mode) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0,
                     disposal=2, optimize=False)
        return exact
    imgs = [Image.fromarray(f, mode) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0,
                 disposal=0, blend=0)
    return True


def sheet_image(rows: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Lay animations out as a sprite sheet: one row per animation, one column per
    frame, every cell the same size, sprites centred and standing on the cell's
    bottom edge. Returns (rgba, cell_width, cell_height)."""
    cw = max(r.shape[2] for r in rows)
    ch = max(r.shape[1] for r in rows)
    cols = max(r.shape[0] for r in rows)
    sheet = np.zeros((ch * len(rows), cw * cols, 4), np.uint8)
    for j, r in enumerate(rows):
        h, w = r.shape[1:3]
        x0, y0 = (cw - w) // 2, ch - h
        for i, f in enumerate(r):
            sheet[j * ch + y0:j * ch + y0 + h, i * cw + x0:i * cw + x0 + w, :f.shape[-1]] = f
            if f.shape[-1] == 3:
                sheet[j * ch + y0:j * ch + y0 + h, i * cw + x0:i * cw + x0 + w, 3] = 255
    return sheet, cw, ch


# --------------------------------------------------------------------------- #
# Animation analysis
# --------------------------------------------------------------------------- #

def _frame_ids(frames: np.ndarray) -> np.ndarray:
    """(t, pixels) array with one integer per pixel colour."""
    flat = frames.reshape(frames.shape[0], -1, frames.shape[-1]).astype(np.int64)
    ids = np.zeros(flat.shape[:2], np.int64)
    for c in range(flat.shape[-1]):
        ids = (ids << 8) | flat[..., c]
    return ids


def frame_distances(frames: np.ndarray) -> np.ndarray:
    """D[i, j] = share of pixels that differ between frames i and j, counted over the
    pixels that change at all (so a small sprite on a large canvas still scores 0..1)."""
    T = frames.shape[0]
    ids = _frame_ids(frames)
    ids = ids[:, (ids != ids[0]).any(0)]
    if ids.shape[1] == 0:
        return np.zeros((T, T), np.float32)
    D = np.empty((T, T), np.float32)
    for i in range(T):
        D[i] = (ids != ids[i]).mean(1)
    return D


def rest_pose(ids: np.ndarray) -> np.ndarray:
    """The most common value of every pixel over time (per-pixel mode)."""
    s = np.sort(ids, axis=0)
    t = np.arange(s.shape[0])[:, None]
    start = np.maximum.accumulate(np.where(np.vstack([np.ones((1, s.shape[1]), bool),
                                                      s[1:] != s[:-1]]), t, 0), axis=0)
    return np.take_along_axis(s, np.argmax(t - start, axis=0)[None], 0)[0]


def _unpack_ids(ids: np.ndarray, channels: int) -> np.ndarray:
    return np.stack([(ids >> (8 * (channels - 1 - c))) & 255 for c in range(channels)], -1).astype(np.uint8)


def find_holds(frames: np.ndarray) -> list[int] | None:
    """Detect animation "on twos/threes": poses held for several video frames.

    Generated and hand-made pixel animation usually runs at 6-12 poses a second inside
    a 24 or 30 fps video. The picture then only really changes every few frames, and
    the frames between differ by compression noise alone, so the frame-to-frame change
    falls into two groups: small (same pose) and large (new pose). The two are split
    with Otsu's threshold. Returns the frame index at which each pose starts, or None
    when the clip has no such rhythm (smooth motion, or no motion).
    """
    T = frames.shape[0]
    if T < 6:
        return None
    ids = _frame_ids(frames)
    ids = ids[:, (ids != ids[0]).any(0)]
    if ids.shape[1] == 0:
        return None
    d = (ids[1:] != ids[:-1]).mean(1)          # d[i]: change from frame i to i + 1
    order = np.sort(d)
    csum = np.cumsum(order)
    n = np.arange(1, T - 1)                    # size of the low group
    lo = csum[:-1] / n
    hi = (csum[-1] - csum[:-1]) / (T - 1 - n)
    between = n * (T - 1 - n) * (hi - lo) ** 2
    i = int(np.argmax(between))
    thr = (order[i] + order[i + 1]) / 2
    jump = d > thr
    share = jump.mean()
    if not 0.05 <= share <= 0.6 or d[jump].mean() < 3 * d[~jump].mean() + 1e-4:
        return None
    starts = [0] + (np.nonzero(jump)[0] + 1).tolist()
    if np.median(np.diff(starts + [T])) < 2:
        return None
    return starts


def collapse_holds(frames: np.ndarray, starts: list[int]) -> tuple[np.ndarray, list[int]]:
    """Turn every held pose into one frame: each pixel takes the value it has in most
    of the pose's frames, which also removes the compression flicker between them.
    Returns (poses, how many source frames each was held for)."""
    T, C = frames.shape[0], frames.shape[-1]
    poses, runs = [], []
    for a, b in zip(starts, starts[1:] + [T]):
        block = frames[a:b]
        if b - a >= 3:
            ids = _frame_ids(block)
            mode = rest_pose(ids)
            agree = (ids == mode).sum(0) > 1
            mode = np.where(agree, mode, ids[(b - a) // 2])
            poses.append(_unpack_ids(mode, C).reshape(block.shape[1:]))
        else:
            poses.append(block[0])
        runs.append(b - a)
    return np.stack(poses), runs


def active_span(frames: np.ndarray, max_gap: int = 6) -> tuple[int, int]:
    """The stretch of frames (start, end exclusive) in which an animation is really
    moving. A sprite that stands still, apart from a flickering pixel or two, before
    and after it acts is cut down to the action; one that never moves gives one frame.
    """
    T = frames.shape[0]
    if T < 3:
        return 0, T
    ids = _frame_ids(frames)
    rest = rest_pose(ids)
    moving = (ids != rest).any(0)
    if not moving.any():
        return 0, 1
    act = (ids[:, moving] != rest[moving]).mean(1)
    top = float(np.percentile(act, 90))
    if act.max() < 0.05:
        return 0, 1
    above = act > 0.35 * max(top, 0.5 * act.max())
    idx = np.nonzero(above)[0]
    if idx.size == 0:
        return 0, 1
    # runs of activity, bridging short returns to the rest pose (a walk passes through it)
    breaks = np.nonzero(np.diff(idx) > max_gap)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k]) + 1


def absorb_temporal_blends(out: np.ndarray, tol: float = 24.0) -> None:
    """In place: remove single-frame pixel values that are not real animation.

    A pixel that changes for one frame and then returns to its previous value is
    compression noise. A pixel that, for one frame, shows a mixture of what it was
    and what it becomes is a motion-blurred in-between; it takes whichever of the
    two it is closer to, which turns a smeared transition into a clean step.
    """
    rgba = out.shape[-1] == 4
    for i in range(1, out.shape[0] - 1):
        a, x, b = out[i - 1], out[i], out[i + 1]
        single = (x != a).any(-1) & (x != b).any(-1)
        if not single.any():
            continue
        same = (a == b).all(-1)
        fa, fx, fb = a[..., :3].astype(float), x[..., :3].astype(float), b[..., :3].astype(float)
        ab = fb - fa
        len2 = (ab ** 2).sum(-1)
        t = ((fx - fa) * ab).sum(-1) / np.maximum(len2, 1e-9)
        dist = np.sqrt(((fx - (fa + t[..., None] * ab)) ** 2).sum(-1))
        blend = (len2 > (2 * tol) ** 2) & (t > 0.15) & (t < 0.85) & (dist <= tol)
        if rgba:
            blend &= (a[..., 3] > 0) & (x[..., 3] > 0) & (b[..., 3] > 0)
        take_a = single & (same | (blend & (t < 0.5)))
        take_b = single & ~same & blend & (t >= 0.5)
        x[take_a] = a[take_a]
        x[take_b] = b[take_b]


def merge_repeats(frames: np.ndarray, threshold: float, runs: list[int] | None = None) -> tuple[list[int], list[int]]:
    """Runs of (near-)identical consecutive frames -> (kept indices, total run lengths).
    `runs` gives how long each input frame is already held (default 1 each)."""
    runs = runs or [1] * frames.shape[0]
    keep, total = [0], [runs[0]]
    for i in range(1, frames.shape[0]):
        if threshold >= 0 and (frames[i] != frames[keep[-1]]).any(-1).mean() <= threshold:
            total[-1] += runs[i]
        else:
            keep.append(i)
            total.append(runs[i])
    return keep, total


LOOP_DIP = 0.85     # a cycle must make frames this much more alike than a typical lag does


def find_loop(D: np.ndarray, min_len: int, runs: list[int] | None = None) -> tuple[int, int] | None:
    """Find a repeating cycle in an animation from its frame-distance matrix.

    `runs` says how long each frame is shown; the search is done in that real time, so
    poses held for different lengths don't distort the rhythm. For every lag, measure
    how much the picture differs from the picture that lag later. Nearby moments are
    always alike, so the curve starts low and climbs; after that, a cycle shows as a
    dip at its length. Generated video never repeats exactly, so a dip only has to be
    clear, not zero.

    The shortest dip that is nearly as deep as the deepest is taken as the cycle (longer
    ones are multiples of it). A walk, though, dips at one step as
    well as at the full two-step cycle, because one step looks much like the other. So
    if there is also a dip at about twice that length, the two halves really are
    different pictures, and the longer cycle joins up about as well as the shorter one,
    the longer one is the cycle.

    Returns (first frame, number of frames) of one cycle. Of all the places it could
    start, the one chosen is where its end leads back into its start and the cycle best
    matches the cycles just before and after it, compared with how much happens inside
    it. So a stretch where nothing
    moves, which repeats perfectly, is not mistaken for the cycle, and neither is one
    that runs off the end of the action. `min_len` is the shortest cycle considered,
    in time steps.
    """
    P = D.shape[0]
    runs = np.asarray(runs if runs is not None else np.ones(P, int))
    shown = np.repeat(np.arange(P), runs)              # which frame is on screen at each step
    F = shown.size
    begins = np.concatenate([[0], np.cumsum(runs)[:-1]])
    Dt = D[np.ix_(shown, shown)]
    lags = np.arange(1, F - max(3, int(np.ceil(0.08 * F))) + 1)
    if lags.size < 3:
        return None
    hold = max(1, int(np.median(runs)))
    # Only moments when something is moving are compared: a still start and a still end
    # match each other and everything in between badly, which would bury the real cycle.
    t = np.arange(F)
    moving = (Dt[t, np.minimum(t + hold, F - 1)] > 0) | (Dt[t, np.maximum(t - hold, 0)] > 0)
    m = np.full(lags.size, np.inf)
    for i, L in enumerate(lags):
        both = moving[:F - L] & moving[L:]
        if both.sum() >= 3:
            m[i] = np.diagonal(Dt, int(L))[both].mean()
    if not np.isfinite(m).any():
        return None
    m[~np.isfinite(m)] = m[np.isfinite(m)].max()
    if hold > 1:                                        # held frames make the curve stepped
        k = hold | 1
        m = np.convolve(np.pad(m, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    dips = [i for i in range(1, len(m) - 1)
            if lags[i] >= min_len and m[i] == m[max(0, i - hold):i + hold + 1].min()
            and m[:i].max() > m[i]]
    clear = [i for i in dips if m[i] <= LOOP_DIP * float(np.median(m))]
    if not clear:
        return None
    # The shortest dip that is nearly as deep as the deepest. Depth is judged within half
    # the clip: beyond that few moments are compared, and a still start that matches a
    # still end looks like a perfect repeat.
    near = [i for i in clear if lags[i] <= F / 2] or clear
    deepest = min(m[i] for i in near)
    base = next(i for i in near if m[i] <= 1.3 * deepest + 0.02)

    def best_start(L: int) -> tuple[int, int, float] | None:
        """(first frame, frame count, how badly it joins) of the best cycle of L time steps."""
        starts = begins[begins + L < F]
        if starts.size == 0:
            return None
        steps = np.arange(L)
        repeat = np.empty(starts.size)                  # how well the cycle matches the one after
        for k, t in enumerate(starts):                  # and the one before it
            nxt = t + steps[t + steps + L < F]
            prv = t + steps[t + steps - L >= 0]
            repeat[k] = np.concatenate([Dt[nxt, nxt + L], Dt[prv, prv - L]]).mean()
        seam = (repeat + Dt[starts, starts + L]) / 2    # the join itself counts as much again
        inside = np.array([Dt[t, t + 1:t + L].mean() if L > 1 else 0.0 for t in starts])
        lively = inside >= 0.5 * inside.max()
        score = np.where(lively, seam / np.maximum(inside, 1e-6), np.inf)
        k = int(np.argmin(score))
        t = int(starts[k])
        first, last = int(shown[t]), int(shown[t + L])
        if t + L - begins[last] > runs[last] / 2:       # the cycle ends past the middle of a hold
            last += 1
        return first, max(1, last - first), float(score[k])

    L = int(lags[base])
    found = best_start(L)
    twice = [j for j in dips if 1.7 * L <= lags[j] <= 2.3 * L and m[j] <= 1.25 * m[base]]
    if twice:
        L2 = int(lags[min(twice, key=lambda j: m[j])])
        longer = best_start(L2)
        if longer and found and longer[2] <= 1.15 * found[2]:        # and it joins about as well
            t = int(begins[longer[0]])
            half = np.arange(t, min(t + L, F - L))
            if half.size and Dt[half, half + L].mean() > VOTE_SAME:   # two different steps
                found = longer
    return found[:2] if found else None


GHOST_SPREAD = 0.25   # a colour seen in under this share of an animation's frames is transient
GHOST_EXTRA = 0.02    # ghosted = this much more transient colour than a typical frame has
GHOST_DROP = 2.0      # this many times further out, a frame is too ghosted to keep


def transient_pixels(frames: np.ndarray) -> np.ndarray:
    """(t, h, w) mask of pixels whose colour the animation hardly ever uses.

    A sprite keeps its palette as it moves, so nearly every colour turns up in most
    frames. A faint copy of another pose left in a frame (a ghost) is a mixture, and
    mixtures are colours of their own that appear in that frame and nowhere else.
    """
    T = frames.shape[0]
    ids = _frame_ids(frames)
    opaque = (frames[..., 3] > 0).reshape(T, -1) if frames.shape[-1] == 4 else np.ones(ids.shape, bool)
    uniq, inv = np.unique(ids, return_inverse=True)
    inv = inv.reshape(ids.shape)
    seen = np.zeros((uniq.size, T), bool)
    for t in range(T):
        seen[inv[t][opaque[t]], t] = True
    rare = seen.mean(1) < GHOST_SPREAD
    return (rare[inv] & opaque).reshape(frames.shape[:3])


def ghost_scores(frames: np.ndarray) -> np.ndarray:
    """How ghosted each frame of an animation looks: the share of its (opaque) pixels
    that have a transient colour. A clean frame scores near 0."""
    T = frames.shape[0]
    count = (frames[..., 3] > 0).reshape(T, -1).sum(1) if frames.shape[-1] == 4 else np.prod(frames.shape[1:3])
    return transient_pixels(frames).reshape(T, -1).sum(1) / np.maximum(count, 1)


def _ghost_limit(score: np.ndarray) -> float:
    """The ghost score above which a frame stands out against the animation's typical one."""
    typical = float(np.median(score))
    return max(2 * typical, typical + GHOST_EXTRA)


def _ghosted(score: np.ndarray) -> np.ndarray:
    return score > _ghost_limit(score)


def drop_ghosts(frames: np.ndarray, runs: list[int], ghost: np.ndarray,
                limit: float) -> tuple[np.ndarray, list[int], np.ndarray, int]:
    """Leave out the frames whose ghost score is over `limit`, showing the frame before
    for longer.

    Repair removes a ghost's odd colours, but a frame that was badly ghosted is still a
    muddle of two poses. Dropping it costs one step of motion and keeps the animation
    clean. Two neighbouring frames are never both dropped, and at least three frames
    are always kept. Returns (frames, runs, ghost scores, frames dropped).
    """
    T = frames.shape[0]
    heavy = ghost > limit
    keep = np.ones(T, bool)
    for t in np.argsort(-ghost):
        if heavy[t] and keep.sum() > 3 and keep[t - 1] and keep[(t + 1) % T]:
            keep[t] = False
    if keep.all():
        return frames, runs, ghost, 0
    runs = list(runs)
    for t in np.nonzero(~keep)[0]:                 # its time goes to the kept frame before it
        prev = next(i for i in ((t - k) % T for k in range(1, T)) if keep[i])
        runs[prev] += runs[t]
    return frames[keep], [r for r, k in zip(runs, keep) if k], ghost[keep], int((~keep).sum())


def repair_ghosts(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Paint over the ghost pixels of ghosted frames.

    In a frame that stands out as ghosted, every pixel with a transient colour takes
    the commonest value among its neighbours that have a normal colour (which may be
    transparent). This is done from the outside of a patch inwards. Pixels with normal
    colours are never touched, and neither are frames that are not ghosted.
    Returns (frames, ghost score of each frame before the repair, frames repaired).
    """
    score = ghost_scores(frames)
    bad = np.nonzero(_ghosted(score))[0]
    if bad.size == 0:
        return frames, score, 0
    out = frames.copy()
    transient = transient_pixels(frames)
    H, W, C = frames.shape[1:]
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for t in bad:
        ids = _frame_ids(frames[t][None])[0].reshape(H, W)
        todo = transient[t].copy()
        for _ in range(3):
            if not todo.any():
                break
            pi = np.pad(ids, 1, constant_values=-1)
            pt = np.pad(todo, 1, constant_values=True)
            new, left = ids.copy(), todo.copy()
            for y, x in np.argwhere(todo):
                good = [pi[1 + y + dy, 1 + x + dx] for dy, dx in shifts if not pt[1 + y + dy, 1 + x + dx]]
                if len(good) >= 3:
                    vals, counts = np.unique(good, return_counts=True)
                    new[y, x] = vals[np.argmax(counts)]
                    left[y, x] = False
            ids, todo = new, left
        out[t] = _unpack_ids(ids.reshape(-1), C).reshape(H, W, C)
    return out, score, int(bad.size)


VOTE_SAME = 0.10    # passes of a cycle this alike show the same drawing: vote pixel by pixel
SWAP_CLOSE = 0.35   # a pass this alike may stand in for a ghosted pose of the cycle


def vote_cycle(frames: np.ndarray, start: int, length: int, D: np.ndarray,
               ghost: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Use the other passes of a looping animation to clean up one cycle.

    The cycle frames[start:start+length] is the reference. Every other frame is matched
    to the reference pose it looks most like. Then, per pose:
      * if at least two other passes show the same drawing (they differ only by noise),
        every pixel takes the value most of them agree on;
      * otherwise, if the pose is ghosted and another pass shows much the same pose
        without the ghost, that frame is used instead.
    Passes are never mixed unless they are the same drawing: generated video redraws the
    character on every pass, and mixing different drawings gives limbs from both.
    Returns (cycle, poses voted, poses swapped). `ghost` is updated in place for a pose
    that was swapped, so it then describes the frame now in the cycle.
    """
    T = frames.shape[0]
    ref = np.arange(start, start + length)
    cycle = frames[ref].copy()
    others = np.array([t for t in range(T) if not start <= t < start + length], int)
    if others.size == 0:
        return cycle, 0, 0
    nearest = D[np.ix_(others, ref)].argmin(1)
    ghosted = _ghosted(ghost)
    voted = swapped = 0
    for phase in range(length):
        r = start + phase
        passes = others[nearest == phase]
        same = passes[D[passes, r] <= VOTE_SAME]
        if same.size >= 2:
            ids = _frame_ids(frames[np.concatenate([[r], same])])
            mode = rest_pose(ids)
            majority = (ids == mode).sum(0) * 2 > ids.shape[0]
            mode = np.where(majority, mode, ids[0])
            new = _unpack_ids(mode, frames.shape[-1]).reshape(frames.shape[1:])
            voted += int((new != cycle[phase]).any())
            cycle[phase] = new
            continue
        if not ghosted[r]:
            continue
        clean = passes[(D[passes, r] <= SWAP_CLOSE) & ~ghosted[passes] & (ghost[passes] < 0.5 * ghost[r])]
        if clean.size:
            t = min(clean, key=lambda t: (ghost[t], D[t, r]))
            cycle[phase] = frames[t]
            ghost[r] = ghost[t]
            swapped += 1
    return cycle, voted, swapped


def pick_poses(D: np.ndarray, n: int, ghost: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Reduce an animation to n frames.

    The frames are split into n consecutive stretches so that every frame is as close
    as possible to the one chosen to stand for its stretch (dynamic programming, so the
    split is the best possible, not a greedy guess). The stand-in is the stretch's most
    typical frame, which is never a blurred in-between: those resemble nothing else.
    With `ghost` scores, the least ghosted of the frames that are nearly as typical is
    chosen instead. Returns (frame, start, end) per stretch, end exclusive.
    """
    T = D.shape[0]
    n = max(1, min(n, T))
    cost = np.full((T, T), np.inf)
    rep = np.zeros((T, T), int)
    for i in range(T):
        c = np.cumsum(D[i:, i:], axis=0)                 # c[j-i, m-i] = sum_{t=i..j} D[t, m]
        c = np.where(np.tri(T - i, dtype=bool), c, np.inf)   # the stand-in lies inside i..j
        cost[i, i:] = c.min(1)
        rep[i, i:] = c.argmin(1) + i
    best = np.full((n, T), np.inf)
    back = np.zeros((n, T), int)
    best[0] = cost[0]
    for k in range(1, n):
        for j in range(k, T):
            cand = best[k - 1, k - 1:j] + cost[k:j + 1, j]
            b = int(np.argmin(cand))
            best[k, j] = cand[b]
            back[k, j] = b + k
    out, j = [], T - 1
    for k in range(n - 1, -1, -1):
        i = back[k, j] if k else 0
        r = int(rep[i, j])
        if ghost is not None:
            c = D[i:j + 1, i:j + 1].sum(0)
            ok = np.nonzero(c <= 1.15 * c.min() + 1e-9)[0]
            r = i + int(min(ok, key=lambda m: (ghost[i + m], c[m])))
        out.append((r, int(i), int(j) + 1))
        j = i - 1
    return out[::-1]


def split_sprites(alpha: np.ndarray) -> list[tuple[slice, slice, np.ndarray]]:
    """Separate the sprites of a clip from its (t, h, w) opacity.

    Sprites are told apart where they usually are (pixels opaque in at least half the
    frames), so two that only touch mid-stride stay separate. Each then claims the
    pixels it reaches in any frame, nearest sprite first. Returns, in reading order,
    (rows, columns, mask) per sprite: its bounding box and which pixels in it are its.
    """
    usual = alpha.mean(0) >= 0.5
    ever = alpha.any(0)
    H, W = usual.shape
    lab = np.zeros((H, W), int)
    sizes = []
    left = usual.copy()
    while left.any():
        y, x = np.unravel_index(int(np.argmax(left)), left.shape)
        blob = np.zeros_like(left)
        blob[y, x] = True
        while True:
            grow = _neighbourhood(blob, 1) & left
            if (grow == blob).all():
                break
            blob = grow
        left &= ~blob
        sizes.append(int(blob.sum()))
        lab[blob] = len(sizes)
    if not sizes:
        return []
    keep = [k + 1 for k, n in enumerate(sizes) if n >= max(6, 0.1 * max(sizes))]
    lab[~np.isin(lab, keep)] = 0
    while True:                                  # grow every sprite into what it ever covers
        p = np.pad(lab, 1)
        new = lab.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                take = (new == 0) & ever & (cand > 0)
                new[take] = cand[take]
        if (new == lab).all():
            break
        lab = new
    boxes = []
    for k in keep:
        m = lab == k
        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        box = (slice(int(ys[0]), int(ys[-1]) + 1), slice(int(xs[0]), int(xs[-1]) + 1))
        boxes.append((*box, m[box]))
    boxes.sort(key=lambda b: b[0].start)
    rows: list[list] = []
    for b in boxes:                        # group into rows of vertically overlapping boxes
        if rows and b[0].start < max(r[0].stop for r in rows[-1]):
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b[1].start)]


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif"}
MAX_ANALYSIS_FRAMES = 480      # loop / pose analysis is cubic in the frame count


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
    """Decoded frames as (h, w, 3) uint8 arrays."""
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
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _ordered_map(fn, items, workers: int):
    """Run fn over items on a thread pool, yielding results in order while keeping
    only a few frames in memory at a time (numpy releases the GIL in the heavy parts)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        pending: deque = deque()
        for it in items:
            pending.append(ex.submit(fn, it))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _video_profiles(frame: np.ndarray) -> list[np.ndarray]:
    """Edge profiles from every second row / column: plenty for tracking the grid of a
    video frame, at a quarter of the cost."""
    return [edge_profile(frame[::2].astype(np.float32), 1),
            edge_profile(frame[:, ::2].astype(np.float32), 0)]


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


def _progress_end():
    if sys.stderr.isatty():
        print(file=sys.stderr)


def _ms(runs: list[int], fps: float) -> list[int]:
    """Frame counts -> display times in ms that add up without rounding drift."""
    cum = np.round(np.cumsum(runs) * 1000.0 / fps).astype(int)
    return np.diff(np.concatenate([[0], cum])).tolist()


def reduce_animation(frames: np.ndarray, args, fps: float) -> tuple[np.ndarray, list[int], str]:
    """Turn one animation (t, h, w, c) into its output frames: held poses become one
    frame each, ghosted poses are repaired, then --loop (with cross-cycle clean-up),
    --poses and repeat merging are applied.
    Returns (frames, how many source frames each lasts, note for the report)."""
    notes = []
    runs = [1] * frames.shape[0]
    starts = None if args.no_hold else find_holds(frames)
    if starts:
        frames, runs = collapse_holds(frames, starts)
        notes.append(f"{len(runs)} poses held about {int(np.median(runs))} frames each")
    if (args.loop or args.poses) and frames.shape[0] > MAX_ANALYSIS_FRAMES:
        step = int(np.ceil(frames.shape[0] / MAX_ANALYSIS_FRAMES))
        runs = [sum(runs[i:i + step]) for i in range(0, len(runs), step)]
        frames = frames[::step]
    ghost = np.zeros(frames.shape[0])
    heavy = np.inf                      # ghost score beyond which a pose is left out
    if not args.no_repair and frames.shape[0] >= 4:
        frames, ghost, repaired = repair_ghosts(frames)
        heavy = GHOST_DROP * _ghost_limit(ghost)
        if repaired:
            notes.append(f"{repaired} ghosted repaired")
    if args.loop and frames.shape[0] >= 4:
        D = frame_distances(frames)
        lp = find_loop(D, max(3, int(round(0.25 * fps))), runs)
        if lp:
            cycle = slice(lp[0], lp[0] + lp[1])
            notes.append(f"loop of {lp[1]}")
            if args.no_vote:
                frames = frames[cycle]
            else:
                frames, voted, swapped = vote_cycle(frames, lp[0], lp[1], D, ghost)
                if voted:
                    notes.append(f"{voted} voted across cycles")
                if swapped:
                    notes.append(f"{swapped} ghosted swapped")
            runs, ghost = runs[cycle], ghost[cycle]
        else:
            notes.append("no loop found")
    if not args.no_repair:
        frames, runs, ghost, dropped = drop_ghosts(frames, runs, ghost, heavy)
        if dropped:
            notes.append(f"{dropped} ghosted dropped")
    if args.poses and frames.shape[0] > args.poses:
        segs = pick_poses(frame_distances(frames), args.poses, ghost)
        runs = [sum(runs[s:e]) for _, s, e in segs]
        frames = frames[[r for r, _, _ in segs]]
        notes.append(f"reduced to {len(segs)}")
    else:
        keep, runs = merge_repeats(frames, args.dedupe, runs)
        frames = frames[keep]
    return frames, runs, ", ".join(notes)


def process_video(args, opts: Options) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("Video input needs ffmpeg/ffprobe on PATH (brew install ffmpeg).", file=sys.stderr)
        return 1
    import os
    w, h, src_fps, total = video_info(args.input)
    fps = args.fps or src_fps
    args.out_fps = fps
    if args.fps:
        total = int(round(total * args.fps / src_fps))
    n = max(w, h)
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    # Pass 1: edge profiles of every frame, and a few whole frames to detect the grid on.
    probe_at = sorted({0, total // 4, total // 2, (3 * total) // 4}) if total else [0]
    probes: dict[int, np.ndarray] = {}
    profs: list[list[np.ndarray]] = []

    def first_pass(item):
        t, frame = item
        if t in probe_at:
            probes[t] = frame
        return _video_profiles(frame)

    for t, p in enumerate(_ordered_map(first_pass, enumerate(iter_frames(args.input, w, h, args.fps)), workers)):
        _progress(t + 1, total, "analysing frame")
        profs.append(p)
    _progress_end()
    nframes = len(profs)
    if nframes == 0:
        print("No frames decoded.", file=sys.stderr)
        return 1

    # Pixel size: detected on the clearest of the probe frames, then tracked outwards
    # frame to frame (so zooms are followed) and smoothed over time.
    ref = 0
    if opts.scale:
        sx0, sy0, how = opts.scale, opts.scale_y or opts.scale, "given"
        periods = np.full(nframes, sx0)
        confs = np.ones(nframes)
    else:
        found = {}
        for t, frame in probes.items():
            if t < nframes:
                d = detect_grid(frame.astype(np.float32))
                if d is not None:
                    found[t] = d
        if not found:
            print("No pixel grid found (video has no edges?). Use --scale.", file=sys.stderr)
            return 1
        ref = max(found, key=lambda t: found[t].confidence)
        det = found[ref]
        chosen = choose_period(det, w, h, args)
        sx0, sy0, how = chosen
        periods = np.empty(nframes)
        confs = np.ones(nframes)
        periods[ref] = sx0
        for span in (range(ref + 1, nframes), range(ref - 1, -1, -1)):
            prev = [sx0]                      # follow the median of the last few frames,
            for t in span:                    # so one bad frame can't lead the track astray
                c = find_period_near(profs[t], n, float(np.median(prev[-5:])))
                periods[t], confs[t] = c.period, c.score
                prev.append(c.period)
    aspect = sy0 / sx0
    px_t = _smooth(periods, 9)

    # Pass 2: grid and colours of every frame.
    cx, cy = w / 2, h / 2

    def second_pass(item):
        t, frame = item
        img = frame.astype(np.float32)
        p_x, p_y = px_t[t], px_t[t] * aspect
        ox, oy = grid_offset(profs[t][0], p_x), grid_offset(profs[t][1], p_y)
        xs, kx = grid_lines(profs[t][0], p_x, w, opts.snap, ox)
        ys, ky = grid_lines(profs[t][1], p_y, h, opts.snap, oy)
        sigma = noise_level(img, xs, ys) if t == ref else 0.0
        return sample_cells(img, xs, ys), kx, ky, (cx - ox) / p_x, (cy - oy) / p_y, sigma

    frames, origins = [], []
    ax_prev = ay_prev = None
    sx_shift = sy_shift = 0
    sigma = 0.0
    for t, (cells, kx, ky, ax, ay, sg) in enumerate(
            _ordered_map(second_pass, enumerate(iter_frames(args.input, w, h, args.fps)), workers)):
        if t >= nframes:
            break
        _progress(t + 1, nframes, "sampling frame")
        # Art-space coordinate of the frame centre in this frame's local grid. The grid
        # phase only fixes it modulo one cell, so unwrap against the previous frame.
        if ax_prev is not None:
            sx_shift = int(round(ax_prev - ax))
            sy_shift = int(round(ay_prev - ay))
        ax_prev, ay_prev = ax + sx_shift, ay + sy_shift
        frames.append(cells)
        origins.append((ky + sy_shift, kx + sx_shift))
        sigma = max(sigma, sg)
    _progress_end()
    nframes = len(frames)

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
    tol = opts.merge if opts.merge is not None else merge_tolerance(sigma)
    if opts.colors:
        rgb = np.clip(np.round(kmeans_quantize(stack.reshape(nframes * H, W, 3), opts.colors)), 0, 255)
    else:
        rgb = np.clip(np.round(merge_similar(stack.reshape(nframes * H, W, 3), tol)), 0, 255)
    rgb = rgb.astype(np.uint8).reshape(nframes, H, W, 3)
    if not opts.colors:
        for i in range(nframes):
            rgb[i] = absorb_blends(rgb[i], tol)

    bg = None
    if opts.transparent or not covered.all():
        if opts.transparent:
            bg = resolve_background(opts.transparent, probes[ref].astype(np.float32))
            print(f"  background #{''.join(f'{int(round(v)):02x}' for v in bg)}", file=sys.stderr)
            out = key_background(rgb, bg, opts.key_tol, opts.key == "contiguous")
        else:
            out = np.concatenate([rgb, np.full((*rgb.shape[:3], 1), 255, np.uint8)], -1)
        out[~covered] = 0
        if opts.transparent and opts.despill:
            near = 2 if px_t.min() * min(1.0, aspect) < 4 else 1
            for i in range(nframes):
                out[i] = despill(out[i], bg, near=near)
        box = trim_box(out[..., 3]) if opts.trim else None
        if box:
            out = out[:, box[0], box[1]]
    else:
        out = rgb

    if not args.no_deblip and nframes >= 3:
        absorb_temporal_blends(out)

    print(f"{args.input.name}: {w}x{h} {src_fps:g}fps"
          + (f" -> {fps:g}fps" if args.fps else "") + f", {nframes} frames  pixel size "
          f"{px_t.min():.3f}-{px_t.max():.3f} ({how}"
          + (f" on frame {ref + 1}, tracked across the clip)" if not opts.scale else ")"))

    stem = args.input.stem + "_repixel"
    outp = args.output or args.input.with_name(stem + ".png")
    ncolors = lambda a: len(np.unique(a.reshape(-1, a.shape[-1]), axis=0))
    lossy = False

    def write_frames(fr: np.ndarray, tag: str):
        if args.frames:
            args.frames.mkdir(parents=True, exist_ok=True)
            for j, f in enumerate(fr):
                to_image(f)[0].save(args.frames / f"{stem}{tag}_{j:04d}.png", optimize=True)

    if args.split:
        if out.shape[-1] != 4:
            print("error: --split needs a transparent background (add -t)", file=sys.stderr)
            return 1
        boxes = split_sprites(out[..., 3] > 0)
        if not boxes:
            print("error: nothing left after removing the background", file=sys.stderr)
            return 1
        rows = []
        for k, (ys_, xs_, own) in enumerate(boxes, 1):
            crop = out[:, ys_, xs_].copy()
            crop[:, ~own] = 0
            a, b = active_span(crop, max(2, int(round(0.25 * fps))))
            fr, fr_runs, note = reduce_animation(crop[a:b], args, fps)
            rows.append(fr)
            line = (f"  sprite {k}: {fr.shape[2]}x{fr.shape[1]} at {xs_.start},{ys_.start}  "
                    f"moves in frames {a + 1}-{b}"
                    + (f", {note}" if note else "") + f", {len(fr)} frames")
            if not args.sheet:
                p = outp.with_name(f"{outp.stem}_{k}{outp.suffix}")
                lossy |= not save_animation(fr, _ms(fr_runs, fps), p)
                line += f"  -> {p}"
            print(line)
            write_frames(fr, f"_{k}")
        if args.sheet:
            sheet, cw, ch = sheet_image(rows)
            to_image(sheet)[0].save(outp, optimize=True)
            print(f"  -> {outp}  sheet of {sheet.shape[1] // cw}x{len(rows)} cells, each {cw}x{ch}, "
                  f"{ncolors(sheet)} colours")
    else:
        fr, fr_runs, note = reduce_animation(out, args, fps)
        write_frames(fr, "")
        if args.sheet:
            sheet, cw, ch = sheet_image([fr])
            to_image(sheet)[0].save(outp, optimize=True)
            print(f"  -> {outp}  sheet of {len(fr)}x1 cells, each {cw}x{ch}, {ncolors(sheet)} colours"
                  + (f" ({note})" if note else ""))
        else:
            lossy = not save_animation(fr, _ms(fr_runs, fps), outp)
            print(f"  -> {outp}  {fr.shape[2]}x{fr.shape[1]}, {len(fr)} unique frames, "
                  f"{ncolors(fr)} colours" + (f" ({note})" if note else ""))
    if lossy:
        print("warning: more than 256 colours, so the GIF is lossy; use --colors 255 "
              "or a .png output", file=sys.stderr)
    if confs.min() < 0.3:
        print(f"warning: grid was weak on some frames (min confidence {confs.min():.2f})",
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
        print(f"Pick 1-{len(cands)}, or type a size [1]: ", end="", file=sys.stderr, flush=True)
        ans = input().strip()
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


def choose_period(det: Detection, w: int, h: int, args) -> tuple[float, float, str]:
    """Pixel size from a detection, asking the user when asked to or when unsure."""
    interactive = sys.stdin.isatty() and not args.no_ask
    if args.ask or (det.uncertain and interactive):
        if det.uncertain:
            print("Grid detection is uncertain.", file=sys.stderr)
        p = ask_period(det.candidates, w, h)
        if abs(p - det.candidates[0].period) < 1e-9:
            return det.period_x, det.period_y, "chosen"
        return p, p, "chosen"
    if det.uncertain:
        alts = det.rivals or det.candidates[1:]
        print("warning: low-confidence detection; alternatives: "
              + ", ".join(f"{c.period:.3f}" for c in alts) + " (use --scale or --ask)",
              file=sys.stderr)
    return det.period_x, det.period_y, f"detected, confidence {det.confidence:.2f}"


def process_image(args, opts: Options) -> int:
    with Image.open(args.input) as im:
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        img = np.asarray(im.convert("RGBA" if has_alpha else "RGB"))
    h, w = img.shape[:2]
    how = "given"
    detection = None
    if not opts.scale:
        rgb, _ = _split_alpha(img)
        detection = detect_grid(rgb)
        if detection is None:
            print("No pixel grid found (image has no edges?). Use --scale.", file=sys.stderr)
            return 1
        sx, sy, how = choose_period(detection, w, h, args)
        if how == "chosen":                     # the user's pick overrides the detection
            opts.scale, opts.scale_y = sx, (sy if sy != sx else None)
    res = repixel_image(img, opts, detection)
    if res.background is not None:
        print(f"  background #{''.join(f'{int(round(v)):02x}' for v in res.background)}", file=sys.stderr)
    out = res.pixels
    result, ncolors = to_image(out)
    outp = args.output or args.input.with_name(args.input.stem + "_repixel.png")
    result.save(outp, optimize=True)

    size = f"{res.scale_x:.3f}" + (f" x {res.scale_y:.3f}" if abs(res.scale_y - res.scale_x) > 5e-4 else "")
    print(f"{args.input.name}: {w}x{h}  pixel size {size} ({how})")
    print(f"  -> {outp}  {out.shape[1]}x{out.shape[0]}, {ncolors} colours, "
          f"mean reconstruction error {res.error:.1f}/255")

    if args.preview is not None:
        k = args.preview
        if k:
            psize = (out.shape[1] * k, out.shape[0] * k)
        else:
            psize = (int(round(out.shape[1] * res.scale_x)), int(round(out.shape[0] * res.scale_y)))
        prev = outp.with_name(outp.stem + "_preview.png")
        mode = "RGBA" if out.shape[2] == 4 else "RGB"
        Image.fromarray(out, mode).resize(psize, Image.NEAREST).save(prev)
        print(f"  preview -> {prev}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Recover true-resolution pixel art from an upscaled/JPEG image or video.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--version", action="version", version=f"repixel {__version__}")
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
                    help="use a perfectly uniform grid (don't follow drift or snap lines to edges)")
    ap.add_argument("--merge", type=float, default=None,
                    help="merge colours closer than this RGB distance (default: chosen from the "
                         "image's noise, 0 for a clean image; 0 = keep every colour)")
    ap.add_argument("--colors", type=int,
                    help="quantise to at most N colours (k-means) instead of --merge")
    ap.add_argument("-t", "--transparent", nargs="?", const="auto", metavar="RRGGBB|auto",
                    help="make this background colour transparent, e.g. ff00ff, '#f0f' or magenta "
                         "(auto-detects from the border if no colour given)")
    ap.add_argument("--key", choices=["full", "contiguous"], default="full",
                    help="with --transparent: 'full' cuts the background colour wherever it "
                         "appears (default); 'contiguous' cuts only background connected to the "
                         "image border, keeping that colour inside the art")
    ap.add_argument("--key-tol", type=float, default=100.0,
                    help="with --transparent: pixels within this RGB distance of the background "
                         "colour are cut (default 100; raise to cut more pinkish edge pixels)")
    ap.add_argument("--no-despill", dest="despill", action="store_false",
                    help="with --transparent: don't clean the background's tint off pixels "
                         "at the sprite edge (cleaned by default, e.g. purple-tinted "
                         "outlines on magenta)")
    ap.add_argument("--trim", action="store_true",
                    help="crop transparent margins")
    ap.add_argument("--preview", type=int, nargs="?", const=0, metavar="N",
                    help="images: also save a nearest-neighbour upscale (N x, default: source size)")
    vg = ap.add_argument_group("video")
    vg.add_argument("--fps", type=float,
                    help="resample to this frame rate first (e.g. 8 or 12 for sprite-style animation)")
    vg.add_argument("--no-hold", action="store_true",
                    help="keep every video frame even when poses are held for several frames "
                         "(by default held poses are detected and become one clean frame each)")
    vg.add_argument("--poses", type=int, metavar="N",
                    help="reduce the animation to its N most representative frames "
                         "(blurred in-between frames are never picked)")
    vg.add_argument("--loop", action="store_true",
                    help="find the repeating cycle (e.g. a walk cycle) and keep exactly one")
    vg.add_argument("--no-repair", action="store_true",
                    help="leave ghosted poses as they are. By default, in a pose that has many "
                         "colours the animation uses nowhere else (a faint copy of another "
                         "pose), those pixels are painted over from their neighbours, and a "
                         "badly ghosted pose is left out")
    vg.add_argument("--no-vote", action="store_true",
                    help="with --loop: take the cycle exactly as one pass of it shows it. By "
                         "default the other passes are used to clean it up: identical passes "
                         "are voted pixel by pixel, and a ghosted pose is swapped for a cleaner "
                         "pass of the same pose")
    vg.add_argument("--split", action="store_true",
                    help="treat each separate sprite as its own animation, covering only the "
                         "frames in which it moves (needs --transparent)")
    vg.add_argument("--sheet", action="store_true",
                    help="write a sprite sheet PNG instead of an animation: one column per "
                         "frame, and with --split one row per sprite")
    vg.add_argument("--frames", type=Path, metavar="DIR",
                    help="also write each output frame as a PNG into DIR")
    vg.add_argument("--dedupe", type=float, default=0.002, metavar="FRAC",
                    help="merge consecutive frames differing in at most this fraction of "
                         "pixels (default 0.002; 0 = exact only, -1 = keep every frame)")
    vg.add_argument("--no-deblip", action="store_true",
                    help="keep single-frame pixel flickers and blurred in-between pixels "
                         "(removed by default)")
    vg.add_argument("--extent", choices=["intersect", "union"], default="intersect",
                    help="canvas = area seen in every frame (default) or in any frame")
    args = ap.parse_args(argv)
    if args.scale_y and not args.scale:
        ap.error("--scale-y requires --scale")
    if args.scale is not None and args.scale < 1:
        ap.error("--scale must be at least 1")
    if args.poses is not None and args.poses < 1:
        ap.error("--poses must be at least 1")

    if not args.input.exists():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1

    opts = Options(scale=args.scale, scale_y=args.scale_y, snap=not args.no_snap,
                   merge=args.merge, colors=args.colors, transparent=args.transparent,
                   key=args.key, key_tol=args.key_tol, despill=args.despill, trim=args.trim)
    try:
        if opts.transparent and opts.transparent != "auto":
            parse_color(opts.transparent)
        if args.input.suffix.lower() in VIDEO_EXTS:
            return process_video(args, opts)
        return process_image(args, opts)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
