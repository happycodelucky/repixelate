"""Ground-truth benchmark: degrade known pixel art, repixel it, score against truth.

    .venv/bin/python tests/bench.py            # summary table
    .venv/bin/python tests/bench.py -v         # every case
    .venv/bin/python tests/bench.py --impl other.py     # another implementation

To compare against the first version of the tool:
    git show d6fec30:repixel.py > out/repixel_v1.py
    .venv/bin/python tests/bench.py --impl out/repixel_v1.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402

SCALES = [2.4, 2.867, 3.0, 4.0, 5.37, 8.0, 10.35, 22.93]
CONDITIONS = {
    "clean":        dict(),
    "jpeg90":       dict(jpeg=90),
    "blur+jpeg75":  dict(blur=0.5, jpeg=75),
    "offset+noise": dict(offset=(0.37, 0.61), blur=0.7, noise=3, jpeg=70),
    "wobble":       dict(wobble=0.35, blur=0.5, jpeg=80),
    "soft":         dict(blur=-0.12, jpeg=85),      # negative = fraction of the cell size
}


def arts():
    out = {}
    out["sheet"] = synth.make_sheet(np.random.default_rng(1))
    out["bigsheet"] = synth.make_sheet(np.random.default_rng(2), n=4, sw=24, sh=32)
    out["scene"] = (synth.make_scene(np.random.default_rng(3)), None)
    # Real art (not procedural): a fixed copy, so the benchmark never depends on a
    # file that the tool itself regenerates.
    real = Path(__file__).resolve().parent / "data" / "soldiers_truth.png"
    if real.exists():
        im = np.asarray(Image.open(real).convert("RGBA"))
        rgb = im[..., :3].copy()
        mask = im[..., 3] > 0
        rgb[~mask] = synth.MAGENTA
        pad = 4
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)
        m2 = np.pad(mask, pad)
        rgb[~m2] = synth.MAGENTA
        out["soldiers"] = (rgb, m2)
    return out


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def run_v1(mod, img, transparent):
    """Adapter for the first version of the tool, which had no `repixel_image`."""
    f = img.astype(np.float32)
    h, w, _ = f.shape
    px, py = mod.edge_profile(f, 1), mod.edge_profile(f, 0)
    cands = mod.find_periods([px, py], max(w, h))
    if not cands:
        return None, 0.0
    p = cands[0].period
    xs, _ = mod.grid_lines(px, p, w, True)
    ys, _ = mod.grid_lines(py, p, h, True)
    raw = mod.sample_cells(f, xs, ys)
    out = np.clip(np.round(mod.merge_similar(raw, 18.0)), 0, 255).astype(np.uint8)
    if transparent:
        bg = mod.detect_background(f)
        out = mod.despill(mod.key_background(out, bg, 100.0), bg)
    return out, p


def run_new(mod, img, transparent):
    opts = mod.Options(transparent="auto" if transparent else None)
    r = mod.repixel_image(img, opts)
    return (r.pixels, r.scale_x) if r is not None else (None, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", type=Path, default=ROOT / "repixel.py")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--only", help="condition name")
    args = ap.parse_args()
    mod = load(args.impl)
    run = run_new if hasattr(mod, "repixel_image") else run_v1

    rows = []
    t0 = time.time()
    for cname, cond in CONDITIONS.items():
        if args.only and cname != args.only:
            continue
        for aname, (art, mask) in arts().items():
            for s in SCALES:
                if s > 12 and art.shape[1] > 130:
                    continue
                kw = dict(cond)
                if kw.get("blur", 0) < 0:
                    kw["blur"] = -kw["blur"] * s
                img = synth.degrade(art, s, rng=np.random.default_rng(7), **kw)
                out, p = run(mod, img, mask is not None)
                if out is None:
                    sc = {"acc": 0.0, "err": 255.0, "shape": False}
                else:
                    sc = synth.score(out, art, mask)
                    op = out[..., :3][out[..., 3] > 0] if out.shape[2] == 4 else out.reshape(-1, 3)
                    tp = art[mask] if mask is not None else art.reshape(-1, 3)
                    sc["pal"] = len(np.unique(op, axis=0)) / len(np.unique(tp, axis=0))
                sc.update(cond=cname, art=aname, scale=s, period=p,
                          period_ok=abs(p - s) / s < 0.01)
                rows.append(sc)
                if args.verbose:
                    print(f"{cname:13s} {aname:9s} s={s:6.3f} p={p:7.3f} "
                          f"{'ok ' if sc['period_ok'] else 'BAD'} shape={'ok ' if sc['shape'] else 'BAD'} "
                          f"acc={sc['acc']:.4f} pal={sc.get('pal', 0):.2f} alpha={sc.get('alpha_acc', float('nan')):.4f} err={sc['err']:.2f}")
    print(f"\n{'condition':13s} {'n':>3s} {'period':>7s} {'shape':>6s} {'acc':>7s} {'min':>7s} {'alpha':>7s} {'colour err':>10s} {'palette':>8s}")
    for cname in CONDITIONS:
        r = [x for x in rows if x["cond"] == cname]
        if not r:
            continue
        al = [x["alpha_acc"] for x in r if "alpha_acc" in x]
        print(f"{cname:13s} {len(r):3d} {np.mean([x['period_ok'] for x in r]):7.2f} "
              f"{np.mean([x['shape'] for x in r]):6.2f} {np.mean([x['acc'] for x in r]):7.4f} "
              f"{np.min([x['acc'] for x in r]):7.4f} {np.mean(al) if al else float('nan'):7.4f} "
              f"{np.mean([x['err'] for x in r]):10.2f} {np.mean([x.get('pal', 0) for x in r]):8.2f}")
    print(f"overall acc {np.mean([x['acc'] for x in rows]):.4f}  "
          f"period ok {np.mean([x['period_ok'] for x in rows]):.3f}  ({time.time() - t0:.1f}s)")
    return rows


if __name__ == "__main__":
    main()
