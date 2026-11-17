"""Tests for Repixel. Run with:  .venv/bin/python -m unittest discover tests -v

They use synthetic pixel art with known ground truth (tests/synth.py), so results are
checked against what the art really is, not against a previous output.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench   # noqa: E402
import repixel as rp  # noqa: E402
import synth   # noqa: E402


def run_cli(*argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = rp.main([str(a) for a in argv])
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class StillImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            sys.argv = ["bench"]
            cls.rows = bench.main()

    def rows_for(self, cond):
        return [r for r in self.rows if r["cond"] == cond]

    def test_pixel_size_is_found_in_every_case(self):
        bad = [(r["cond"], r["art"], r["scale"], round(r["period"], 3)) for r in self.rows if not r["period_ok"]]
        self.assertEqual(bad, [])

    def test_output_has_the_art_size(self):
        bad = [(r["cond"], r["art"], r["scale"], r.get("size")) for r in self.rows if not r["shape"]]
        self.assertEqual(bad, [])

    def test_clean_input_is_recovered_exactly(self):
        for r in self.rows_for("clean"):
            self.assertEqual(r["acc"], 1.0, (r["art"], r["scale"]))
            self.assertEqual(r["pal"], 1.0, (r["art"], r["scale"]))
            if "alpha_acc" in r:
                self.assertEqual(r["alpha_acc"], 1.0, (r["art"], r["scale"]))

    def test_damaged_input_accuracy(self):
        floors = {"jpeg90": 0.91, "blur+jpeg75": 0.85, "offset+noise": 0.82, "wobble": 0.85, "soft": 0.89}
        for cond, floor in floors.items():
            acc = np.mean([r["acc"] for r in self.rows_for(cond)])
            self.assertGreaterEqual(acc, floor, cond)
            alpha = np.mean([r["alpha_acc"] for r in self.rows_for(cond) if "alpha_acc" in r])
            self.assertGreaterEqual(alpha, 0.995, cond)

    def test_large_pixels_survive_heavy_damage(self):
        for r in self.rows:
            if r["scale"] >= 8 and r["art"] != "soldiers":
                self.assertGreaterEqual(r["acc"], 0.9, (r["cond"], r["art"], r["scale"]))


class Transparency(unittest.TestCase):
    def setUp(self):
        art, mask = synth.make_sheet(np.random.default_rng(5), n=2)
        # a background-coloured window inside the first sprite, fully enclosed by it
        ys, xs = np.nonzero(mask)
        cy, cx = int(np.median(ys)), int(xs.min() + 8)
        mask[cy - 2:cy + 3, cx - 2:cx + 3] = True
        art[cy - 2:cy + 3, cx - 2:cx + 3] = (40, 90, 60)
        art[cy - 1:cy + 2, cx - 1:cx + 2] = synth.MAGENTA
        self.art, self.mask, self.hole = art, mask, (slice(cy - 1, cy + 2), slice(cx - 1, cx + 2))

    def test_full_key_cuts_the_colour_everywhere(self):
        r = rp.repixel_image(synth.degrade(self.art, 6.0), rp.Options(transparent="magenta"))
        self.assertEqual(r.pixels.shape[:2], self.art.shape[:2])
        self.assertTrue((r.pixels[self.hole][..., 3] == 0).all())

    def test_contiguous_key_keeps_enclosed_colour(self):
        r = rp.repixel_image(synth.degrade(self.art, 6.0), rp.Options(transparent="magenta", key="contiguous"))
        self.assertTrue((r.pixels[self.hole][..., 3] == 255).all())
        self.assertTrue((r.pixels[self.hole][..., :3] == synth.MAGENTA).all())
        self.assertTrue((r.pixels[0, 0] == 0).all())
        np.testing.assert_array_equal(r.pixels[..., 3] > 0, self.mask)

    def test_background_given_as_hex_snaps_to_the_real_colour(self):
        img = synth.degrade(self.art, 4.0, jpeg=80)
        for spec in ("ff00ff", "#FF00FF", "0xff00ff", "f0f", "magenta", "auto"):
            r = rp.repixel_image(img, rp.Options(transparent=spec))
            self.assertLess(np.abs(r.background - synth.MAGENTA).max(), 40, spec)
            self.assertGreater(((r.pixels[..., 3] > 0) == (self.mask & ~self._hole_mask())).mean(), 0.99, spec)

    def _hole_mask(self):
        m = np.zeros_like(self.mask)
        m[self.hole] = True
        return m

    def test_despill_never_adds_pixels_or_touches_clean_art(self):
        img = synth.degrade(self.art, 3.0, blur=0.5, jpeg=75)
        a = rp.repixel_image(img, rp.Options(transparent="auto", despill=False)).pixels
        b = rp.repixel_image(img, rp.Options(transparent="auto")).pixels
        self.assertFalse(((b[..., 3] > 0) & (a[..., 3] == 0)).any())
        clean = rp.repixel_image(synth.degrade(self.art, 3.0), rp.Options(transparent="auto"))
        np.testing.assert_array_equal(clean.pixels[..., :3][self.mask & ~self._hole_mask()],
                                      self.art[self.mask & ~self._hole_mask()])

    def test_input_alpha_is_kept(self):
        art, mask = synth.make_sheet(np.random.default_rng(6), n=2)
        big = synth.degrade(art, 5.0)
        a = (synth.degrade(np.repeat(mask[..., None] * 255, 3, 2).astype(np.uint8), 5.0)[..., 0])
        rgba = np.dstack([big, a])
        rgba[a < 128, :3] = 0                      # whatever lies under transparency
        r = rp.repixel_image(rgba)
        self.assertEqual(r.pixels.shape[2], 4)
        np.testing.assert_array_equal(r.pixels[..., 3] > 0, mask)
        np.testing.assert_array_equal(r.pixels[..., :3][mask], art[mask])

    def test_trim(self):
        r = rp.repixel_image(synth.degrade(self.art, 4.0), rp.Options(transparent="auto", trim=True))
        ys, xs = np.nonzero(self.mask)
        self.assertEqual(r.pixels.shape[:2], (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))


class Palette(unittest.TestCase):
    def test_merge_leaves_clean_colours_alone(self):
        cells = np.random.default_rng(0).integers(0, 255, (20, 30, 3)).astype(float)
        np.testing.assert_array_equal(rp.merge_similar(cells, 0.0), cells)

    def test_merge_collapses_noise_but_keeps_real_shades(self):
        rng = np.random.default_rng(1)
        true = np.array([[200, 40, 40], [185, 40, 40], [20, 20, 90], [240, 240, 240]], float)
        idx = rng.integers(0, 4, 4000)
        cells = (true[idx] + rng.normal(0, 2.0, (4000, 3))).reshape(40, 100, 3)
        out = rp.merge_similar(cells, rp.merge_tolerance(2.0))
        got = np.unique(np.round(out.reshape(-1, 3)), axis=0)
        self.assertEqual(len(got), 4)
        self.assertLess(np.abs(np.sort(got, 0) - np.sort(true, 0)).max(), 1.5)

    def test_colors_limit(self):
        art = synth.make_scene(np.random.default_rng(3))
        r = rp.repixel_image(synth.degrade(art, 4.0, jpeg=85), rp.Options(colors=6))
        self.assertLessEqual(len(np.unique(r.pixels.reshape(-1, 3), axis=0)), 6)

    def test_indexed_png_is_exact(self):
        art, mask = synth.make_sheet(np.random.default_rng(4))
        rgba = np.dstack([art, np.where(mask, 255, 0).astype(np.uint8)])
        rgba[~mask] = 0
        im, n = rp.to_image(rgba)
        self.assertEqual(im.mode, "P")
        buf = io.BytesIO()
        im.save(buf, "PNG")
        np.testing.assert_array_equal(np.asarray(Image.open(buf).convert("RGBA")), rgba)


class Animation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.poses, self.masks = synth.make_cycle(rng, poses=6)
        self.rgba = np.concatenate([self.poses, np.where(self.masks, 255, 0).astype(np.uint8)[..., None]], -1)
        self.rgba[~self.masks] = 0

    def noisy_holds(self, order, hold=3, flips=3):
        rng = np.random.default_rng(9)
        frames = np.repeat(self.rgba[order], hold, axis=0).copy()
        opaque = np.argwhere(frames[..., 3] > 0)
        for t, y, x in opaque[rng.choice(len(opaque), flips * len(frames), replace=False)]:
            frames[t, y, x, :3] = rng.integers(0, 255, 3)       # single-frame flicker
        return frames

    def test_held_poses_are_found_and_cleaned(self):
        order = list(range(6)) * 3
        frames = self.noisy_holds(order)
        starts = rp.find_holds(frames)
        self.assertEqual(starts, list(range(0, len(frames), 3)))
        poses, runs = rp.collapse_holds(frames, starts)
        self.assertEqual(runs, [3] * len(order))
        np.testing.assert_array_equal(poses, self.rgba[order])

    def test_smooth_motion_has_no_holds(self):
        self.assertIsNone(rp.find_holds(self.rgba[list(range(6)) * 4]))

    def test_loop_is_found(self):
        seq = self.rgba[[0, 0, 0] + list(range(6)) * 3 + [0, 0]]
        start, length = rp.find_loop(rp.frame_distances(seq), 3)
        self.assertEqual(length, 6)
        self.assertGreaterEqual(start, 2)
        np.testing.assert_array_equal(seq[start], seq[start + length])

    def test_no_loop_in_a_one_way_animation(self):
        rng = np.random.default_rng(3)
        seq = rng.integers(0, 255, (12, 8, 8, 3)).astype(np.uint8)
        self.assertIsNone(rp.find_loop(rp.frame_distances(seq), 3))

    def test_pick_poses(self):
        seq = np.repeat(self.rgba, [2, 3, 1, 4, 2, 3], axis=0)
        segs = rp.pick_poses(rp.frame_distances(seq), 6)
        self.assertEqual([(s, e) for _, s, e in segs], [(0, 2), (2, 5), (5, 6), (6, 10), (10, 12), (12, 15)])
        for rep, s, e in segs:
            self.assertTrue(s <= rep < e)
        three = rp.pick_poses(rp.frame_distances(seq), 3)
        self.assertEqual(len(three), 3)
        self.assertEqual(three[0][1], 0)
        self.assertEqual(three[-1][2], len(seq))

    def test_sprites_that_touch_briefly_are_still_split(self):
        alpha = np.repeat((self.rgba[:1, ..., 3] > 0), 10, axis=0)
        boxes = rp.split_sprites(alpha)
        self.assertEqual(len(boxes), 3)
        gap = slice(boxes[0][1].stop, boxes[1][1].start)
        alpha[4, 12:14, gap] = True                 # an arm reaches across for one frame
        again = rp.split_sprites(alpha)
        self.assertEqual(len(again), 3)
        self.assertEqual([b[1].start for b in again], sorted(b[1].start for b in again))

    def test_active_span(self):
        seq = self.rgba[[0] * 10 + list(range(6)) * 2 + [0] * 10]
        a, b = rp.active_span(seq)
        self.assertTrue(10 <= a <= 11 and 21 <= b <= 22, (a, b))
        self.assertEqual(rp.active_span(self.rgba[[0] * 8]), (0, 1))

    def test_blended_in_between_pixels_are_removed(self):
        a, b = self.rgba[0], self.rgba[3]
        mix = a.copy()
        both = (a[..., 3] > 0) & (b[..., 3] > 0)
        mix[both, :3] = (a[both, :3].astype(int) * 2 // 5 + b[both, :3].astype(int) * 3 // 5).astype(np.uint8)
        seq = np.stack([a, a, mix, b, b])
        rp.absorb_temporal_blends(seq)
        changed = both & (np.abs(a[..., :3].astype(int) - b[..., :3]).sum(-1) > 150)
        np.testing.assert_array_equal(seq[2][changed], b[changed])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class Video(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.poses, cls.masks = synth.make_cycle(np.random.default_rng(2), poses=6)
        cls.order = [0] * 4 + list(range(6)) * 3 + [0] * 4
        cls.clip = cls.tmp / "walk.mp4"
        synth.write_video(cls.clip, cls.poses, cls.order, scale=8.0, hold=3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def frames_of(self, path):
        with Image.open(path) as im:
            return [np.asarray(im.convert("RGBA")) for i in range(getattr(im, "n_frames", 1))
                    if im.seek(i) is None]

    def test_animation_is_recovered(self):
        out = self.tmp / "walk.png"
        code, stdout, _ = run_cli(self.clip, "-t", "-o", out, "--no-ask")
        self.assertEqual(code, 0)
        self.assertRegex(stdout, r"pixel size (7\.99\d|8\.00\d)-(7\.99\d|8\.00\d) \(detected")
        self.assertIn("19 poses held about 3 frames each", stdout)
        frames = self.frames_of(out)
        # the standing lead-in runs straight into the first walk pose (the same picture),
        # then 17 more walk poses and the standing pose again
        want = [0, 1, 2, 3, 4, 5] * 3 + [0]
        self.assertEqual(len(frames), len(want))
        self.assertEqual(frames[0].shape[:2], self.poses.shape[1:3])
        for f, p in zip(frames, want):
            self.assertGreater(((f[..., 3] > 0) == self.masks[p]).mean(), 0.995)
            sc = synth.score(f, self.poses[p], self.masks[p], max_shift=0)
            self.assertGreater(sc["acc"], 0.93)

    def test_gif_and_apng_are_lossless(self):
        gif, png, frames_dir = self.tmp / "a.gif", self.tmp / "a.png", self.tmp / "frames"
        self.assertEqual(run_cli(self.clip, "-t", "-o", gif, "--frames", frames_dir, "--no-ask")[0], 0)
        self.assertEqual(run_cli(self.clip, "-t", "-o", png, "--no-ask")[0], 0)
        ref = [np.asarray(Image.open(f).convert("RGBA")) for f in sorted(frames_dir.glob("*.png"))]
        for path in (gif, png):
            got = self.frames_of(path)
            self.assertEqual(len(got), len(ref))
            for a, b in zip(got, ref):
                np.testing.assert_array_equal(a[..., 3] > 0, b[..., 3] > 0)
                np.testing.assert_array_equal(a[..., :3][b[..., 3] > 0], b[..., :3][b[..., 3] > 0])

    def test_split_loop_sheet(self):
        out = self.tmp / "sheet.png"
        code, stdout, _ = run_cli(self.clip, "-t", "--split", "--loop", "--sheet", "-o", out, "--no-ask")
        self.assertEqual(code, 0, stdout)
        self.assertEqual(stdout.count("  sprite "), 3)
        self.assertEqual(stdout.count("loop of 6"), 3)
        sheet = np.asarray(Image.open(out).convert("RGBA"))
        cw = sheet.shape[1] // 6
        self.assertEqual(sheet.shape[0] % 3, 0)
        self.assertEqual(sheet.shape[1], cw * 6)

    def test_poses_reduces_the_frame_count(self):
        out = self.tmp / "four.png"
        code, stdout, _ = run_cli(self.clip, "-t", "--loop", "--poses", "4", "-o", out, "--no-ask")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.frames_of(out)), 4)


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        art, _ = synth.make_sheet(np.random.default_rng(1))
        self.src = self.tmp / "art.png"
        Image.fromarray(synth.degrade(art, 5.0, jpeg=85)).save(self.src)
        self.art = art

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_output_and_report(self):
        code, out, _ = run_cli(self.src, "--no-ask")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"art\.png: \d+x\d+  pixel size 5\.00\d \(detected, confidence 0\.\d\d\)")
        self.assertRegex(out, r"-> .*art_repixel\.png  116x36, \d+ colours, mean reconstruction error \d+\.\d/255")
        with Image.open(self.tmp / "art_repixel.png") as im:
            self.assertEqual(im.size, (116, 36))

    def test_scale_and_preview(self):
        code, out, _ = run_cli(self.src, "--scale", "5", "--preview", "3", "-o", self.tmp / "o.png")
        self.assertEqual(code, 0)
        self.assertIn("(given)", out)
        with Image.open(self.tmp / "o_preview.png") as im:
            self.assertEqual(im.size, (116 * 3, 36 * 3))

    def test_errors(self):
        self.assertEqual(run_cli(self.tmp / "missing.png")[0], 1)
        code, _, err = run_cli(self.src, "-t", "notacolour", "--no-ask")
        self.assertEqual(code, 1)
        self.assertIn("can't read colour", err)
        self.assertEqual(run_cli(self.src, "--scale-y", "3")[0], 2)
        flat = self.tmp / "flat.png"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(flat)
        self.assertEqual(run_cli(flat, "--no-ask")[0], 1)


if __name__ == "__main__":
    unittest.main()
