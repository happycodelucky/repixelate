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

    def two_step_walk(self):
        """An 8-pose walk: the second step repeats the first with part of each sprite
        drawn differently (as the far leg differs from the near one)."""
        poses, masks = synth.make_cycle(np.random.default_rng(5), poses=4)
        step1 = np.concatenate([poses, np.where(masks, 255, 0).astype(np.uint8)[..., None]], -1)
        step1[~masks] = 0
        step2 = step1.copy()
        for t in range(4):
            ys, xs = np.nonzero(step2[t, ..., 3] > 0)
            pick = slice(len(ys) // 2, len(ys) // 2 + 150)
            step2[t, ys[pick], xs[pick], :3] = (30, 60, 200)
        return np.concatenate([step1, step2])

    def test_two_step_walk_gives_the_full_cycle(self):
        walk = self.two_step_walk()
        seq = walk[list(range(8)) * 3]
        D = rp.frame_distances(seq)
        self.assertGreater(D[0, 4], rp.VOTE_SAME)          # the two steps do differ
        self.assertLess(D[0, 4], 0.5 * D[0, 1])            # but are much alike
        start, length = rp.find_loop(D, 3)
        self.assertEqual(length, 8)

    def test_loop_is_measured_in_real_time(self):
        # a still lead-in and ending, and poses held for different lengths
        order = [0] + list(range(1, 6)) + list(range(6)) * 2 + [0]
        runs = [15] + [3, 2, 4, 3, 3] + [3] * 12 + [12]
        D = rp.frame_distances(self.rgba[order])
        start, length = rp.find_loop(D, 6, runs)
        self.assertEqual(length, 6)
        self.assertTrue(1 <= start and start + length <= len(order) - 1)
        self.assertEqual(order[start], order[start + length])

    def test_ghosted_pose_is_repaired(self):
        frames = self.rgba[self.GHOST_ORDER * 2].copy()
        clean = frames.copy()
        frames[3], n = self.ghosted(clean[2], clean[3], share=1)
        fixed, score, repaired = rp.repair_ghosts(frames)
        self.assertEqual(repaired, 1)
        self.assertEqual(int(np.argmax(score)), 3)
        # Every ghost pixel now has a colour the sprite really uses, taken from next to
        # it. That is a plausible colour, not always the one that was drawn.
        wrong_before = (frames[3] != clean[3]).any(-1).sum()
        wrong_after = (fixed[3] != clean[3]).any(-1).sum()
        self.assertEqual(wrong_before, n)
        self.assertLess(wrong_after, wrong_before)
        self.assertLess(rp.ghost_scores(fixed)[3], 0.25 * score[3])
        palette = {tuple(c) for c in clean.reshape(-1, 4)}
        self.assertTrue(all(tuple(c) in palette for c in fixed[3].reshape(-1, 4)))
        others = [t for t in range(len(frames)) if t != 3]
        np.testing.assert_array_equal(fixed[others], frames[others])
        # an animation without ghosts is returned untouched
        np.testing.assert_array_equal(rp.repair_ghosts(clean)[0], clean)

    def test_badly_ghosted_poses_are_dropped(self):
        frames = self.rgba.copy()
        ghost = np.array([0.0, 0.3, 0.0, 0.2, 0.25, 0.0])
        out, runs, g, dropped = rp.drop_ghosts(frames, [3] * 6, ghost, 0.1)
        self.assertEqual(dropped, 2)                        # 3 and 4 are neighbours: only one goes
        np.testing.assert_array_equal(out, frames[[0, 2, 3, 5]])
        self.assertEqual(runs, [6, 3, 6, 3])
        self.assertEqual(sum(runs), 18)
        few = rp.drop_ghosts(frames[:3], [3] * 3, np.array([0.5, 0.5, 0.5]), 0.1)
        self.assertEqual(few[3], 0)                         # never fewer than three frames

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

    def corrupted(self, order, flips=3, seed=11):
        """The given poses with a few random wrong pixels in every frame."""
        rng = np.random.default_rng(seed)
        frames = self.rgba[order].copy()
        opaque = np.argwhere(frames[..., 3] > 0)
        for t, y, x in opaque[rng.choice(len(opaque), flips * len(frames), replace=False)]:
            frames[t, y, x, :3] = rng.integers(0, 255, 3)
        return frames

    def test_voting_across_cycles_removes_random_errors(self):
        order = [0, 0] + list(range(6)) * 4
        frames = self.corrupted(order)
        D = rp.frame_distances(frames)
        start, length = rp.find_loop(D, 3)
        self.assertEqual(length, 6)
        cycle, voted, swapped = rp.vote_cycle(frames, start, length, D, np.zeros(len(frames)))
        want = self.rgba[[order[start + i] for i in range(6)]]
        self.assertFalse((frames[start:start + 6] == want).all())     # one pass alone is wrong
        np.testing.assert_array_equal(cycle, want)
        self.assertGreater(voted, 0)
        self.assertEqual(swapped, 0)

    def test_voting_needs_three_passes(self):
        frames = self.corrupted(list(range(6)) * 2)
        D = rp.frame_distances(frames)
        cycle, voted, _ = rp.vote_cycle(frames, 0, 6, D, np.zeros(len(frames)))
        np.testing.assert_array_equal(cycle, frames[:6])
        self.assertEqual(voted, 0)

    def test_voting_never_mixes_different_drawings(self):
        other, masks = synth.make_cycle(np.random.default_rng(77), poses=6)
        other = np.concatenate([other, np.where(masks, 255, 0).astype(np.uint8)[..., None]], -1)
        other[~masks] = 0
        frames = np.concatenate([self.rgba, other, other])        # same rhythm, redrawn sprites
        D = rp.frame_distances(frames)
        cycle, voted, swapped = rp.vote_cycle(frames, 0, 6, D, np.zeros(len(frames)))
        np.testing.assert_array_equal(cycle, self.rgba)
        self.assertEqual((voted, swapped), (0, 0))

    def ghosted(self, prev, pose, share=3):
        """`pose` with a faint copy of `prev` left in every `share`-th changed pixel."""
        out = pose.copy()
        both = (prev[..., 3] > 0) & (pose[..., 3] > 0)
        changed = np.argwhere(both & (np.abs(prev[..., :3].astype(int) - pose[..., :3]).sum(-1) > 150))
        for y, x in changed[::share]:
            out[y, x, :3] = (prev[y, x, :3].astype(int) * 2 // 5 + pose[y, x, :3].astype(int) * 3 // 5)
        return out, len(changed[::share])

    # A ghost shows where a pixel changes and then stays: pose 3 is held for two frames
    # here, and the first of them keeps a faint copy of pose 2.
    GHOST_ORDER = [0, 1, 2, 3, 3, 4, 5] * 2

    def test_ghost_score(self):
        clean = self.rgba[self.GHOST_ORDER]
        self.assertLess(rp.ghost_scores(clean).max(), 0.01)
        frames = clean.copy()
        frames[3], n = self.ghosted(clean[2], clean[3], share=1)
        self.assertGreater(n, 20)
        score = rp.ghost_scores(frames)
        self.assertEqual(int(np.argmax(score)), 3)
        self.assertGreater(score[3], 0.05)
        self.assertGreater(score[3], 5 * np.delete(score, 3).max())

    def test_ghosted_pose_is_swapped_for_a_clean_pass(self):
        frames = self.rgba[self.GHOST_ORDER].copy()
        for t in (4, 11):                       # the second held frame differs a little
            ys, xs = np.nonzero(frames[t, ..., 3] > 0)
            frames[t, ys[:60], xs[:60], :3] = (10, 200, 200)
        want = frames[:7].copy()
        frames[3], _ = self.ghosted(frames[2], frames[3], share=2)
        D = rp.frame_distances(frames)
        cycle, voted, swapped = rp.vote_cycle(frames, 0, 7, D, rp.ghost_scores(frames))
        self.assertEqual((voted, swapped), (0, 1))
        np.testing.assert_array_equal(cycle, want)

    def test_pick_poses_avoids_ghosted_frames(self):
        seq = np.repeat(self.rgba[[0, 3]], 3, axis=0)
        D = rp.frame_distances(seq)
        self.assertEqual([r for r, _, _ in rp.pick_poses(D, 2)], [0, 3])
        ghost = np.array([0.5, 0.0, 0.5, 0.5, 0.5, 0.0])
        self.assertEqual([r for r, _, _ in rp.pick_poses(D, 2, ghost)], [1, 5])

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

    def test_loop_with_and_without_cross_cycle_voting(self):
        out = self.tmp / "loop.png"
        code, stdout, _ = run_cli(self.clip, "-t", "--loop", "-o", out, "--no-ask")
        self.assertEqual(code, 0)
        self.assertIn("loop of 6", stdout)
        voted = self.frames_of(out)
        code, stdout, _ = run_cli(self.clip, "-t", "--loop", "--no-vote", "-o", out, "--no-ask")
        self.assertEqual(code, 0)
        self.assertNotIn("voted", stdout)
        plain = self.frames_of(out)
        self.assertEqual(len(voted), 6)

        def accuracy(frames):
            return np.mean([max(synth.score(f, self.poses[p], self.masks[p], max_shift=0)["acc"]
                                for p in range(6)) for f in frames])
        self.assertGreaterEqual(accuracy(voted), accuracy(plain))

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
