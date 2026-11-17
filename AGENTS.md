# AGENTS.md

Repixel recovers true-resolution pixel art from upscaled, JPEG-damaged or AI-generated images and videos.

**To use the tool** (commands, flags, reading the output, choosing settings), read [llms.txt](llms.txt). That file is the user-facing documentation for LLMs and agents. Keep it in sync whenever the CLI changes.

This file is for agents **changing the code**.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

- The existing `.venv` was built with Homebrew Python 3.14. The system `/usr/bin/python3` (3.9) has no numpy and is too old: the code uses `X | None` types and needs Python 3.10+.
- Video needs `ffmpeg` and `ffprobe` on PATH. The video tests are skipped without them.

## Layout

| Path | What |
|---|---|
| `repixel.py` | The whole tool: one file, CLI entry point `main()`, library entry point `repixel_image()`, version in `__version__` |
| `pyproject.toml` | Packaging for `pip install` / `uv tool install`; installs the single module and a `repixel` command. The version is read from `repixel.py` |
| `tests/synth.py` | Ground truth: procedural pixel art and animations, the damage model (`degrade`), scoring (`score`), a synthetic video writer |
| `tests/bench.py` | Accuracy benchmark over art × pixel size × damage; prints a table |
| `tests/data/` | Fixed real-art ground truth for the benchmark (never regenerate it: the scores are only comparable while it stays the same) |
| `tests/test_repixel.py` | `unittest` suite (41 tests, about 25 s) built on the two files above |
| `llms.txt` | Usage docs for LLMs/agents |
| `README.md` | Usage docs for humans, with example images |
| `docs/` | README example images, and `make_examples.py` which regenerates them from `test/` |
| `docs/PROGRESS.md` | Version-by-version results: benchmark table and comparison images. `docs/make_progress.py` regenerates the images in `docs/progress/` by running each earlier version from git history on the same inputs |
| `requirements.txt` | numpy, Pillow |
| `test/` | Real sample inputs: two Gemini JPEGs (trees, soldier sheet) and one Gemini mp4 (walking soldiers) |
| `out/` | Scratch outputs from manual runs; git-ignored, safe to delete |

## How `repixel.py` is organised

1. **Grid detection**
   - `edge_profile`: per-axis edge strength, ignoring small JPEG-noise jumps (`EDGE_NOISE`).
   - `detect_grid`: candidates are the peaks of the edge profiles' spectrum (`_spectrum_peaks`, a zero-padded FFT, refined by `_refine`). Each candidate is then redrawn on a pixel subset (`_Redraw`): candidates whose error is far above the best are dropped (`VALID_ERR`). The most coherent survivor is walked up its integer multiples; a multiple is taken if it keeps `HARMONIC_KEEP` of the coherence and redraws the image no more than `HARMONIC_ERR` worse. Returns a `Detection`.
   - `find_period_near`: same scoring in a narrow window, used to track the pixel size from frame to frame in video.
   - `grid_offset` / `grid_lines`: grid phase; then `_drift` follows slow drift of the grid (local phase of the demodulated edge profile), used only if it puts clearly more edge on the lines; then `_snap_lines` pulls each line onto a real edge when the period is 5 px or more. `grid_lines` returns `(lines, k_first)`; video needs `k_first` to place frames on a shared canvas.
2. **Sampling and palette**
   - `sample_cells`: median of each cell's centre region. `reconstruction_error` is the reported fit.
   - `noise_level`: noise sigma from neighbouring pixel pairs that lie wholly inside one cell. Exactly 0 for a clean image.
   - `merge_tolerance` → `merge_similar`: two greedy passes (`_greedy_clusters`, batched but equal to one-at-a-time): colours within the tight distance become one; then a rare colour within the wide distance of a much commoner one is absorbed. `absorb_blends` replaces rare in-between colours with the opposite neighbour they are a mix of. `kmeans_quantize` for `--colors`.
3. **Transparency**
   - `resolve_background` snaps a given colour to the image's real background colour.
   - `key_background` cuts pixels by RGB distance (binary alpha); `contiguous=True` keeps only the part connected to the border (`_border_connected`).
   - `despill`: for edge pixels, measures background hue in excess of what the opaque pixels within 3 px have, removes it, snaps to the nearest clean local colour, and cuts unexplained pixels that could be mostly background.
   - `_split_alpha` handles inputs that already have alpha; `trim_box` finds the crop.
4. **Output**: `to_image` (exact indexed PNG), `gif_frames` (exact shared-palette GIF), `save_animation`, `sheet_image`.
5. **Animation analysis** (all on art-resolution frames)
   - `absorb_temporal_blends`: single-frame flickers and single-frame in-between blends.
   - `find_holds` / `collapse_holds`: Otsu split of frame-to-frame change into "same pose" and "new pose"; each held pose becomes its per-pixel majority frame.
   - `split_sprites`: sprites separated where they usually are, then grown into what they ever cover. `active_span`: the stretch where a sprite differs from its rest pose (`rest_pose` = per-pixel mode).
   - `find_loop`: dips of mean frame distance against lag, measured in real time (each pose weighted by how long it is held) over moving moments only. Takes the shortest dip that is nearly as deep as the deepest, then the dip at twice that if the halves differ and it joins as well. The start is where the cycle joins and matches its neighbouring cycles best. `pick_poses`: optimal split into N consecutive stretches (dynamic programming), each represented by its medoid, or the least ghosted near-medoid.
   - `transient_pixels` / `ghost_scores`: pixels whose colour appears in under `GHOST_SPREAD` of the animation's poses, and their share per pose. `repair_ghosts` paints them over from stable neighbours in poses that stand out; `drop_ghosts` leaves out poses over `GHOST_DROP` times the limit. `vote_cycle`: cleans the kept cycle using its other passes: a per-pixel majority where at least two other passes are the same drawing (`VOTE_SAME`), else a ghosted pose is swapped for a cleaner close pass (`SWAP_CLOSE`).
6. **Flows**
   - `repixel_image`: the still-image path as a library call. `process_image`: its CLI wrapper.
   - `process_video`: two ffmpeg passes on a thread pool (`_ordered_map`). Pass 1 collects edge profiles and four probe frames; the grid is detected on the clearest probe and tracked outwards. Pass 2 samples every frame; frames are then placed on one canvas by unwrapping each frame's grid phase, share one palette, and go through `reduce_animation` (holds, `--loop`, `--poses`, repeat merging), per sprite with `--split`.

## Design rules

- **Outputs must be lossless.** Never let Pillow quantise behind your back: `Image.quantize` and RGB→P conversion can shift colours. Build palettes explicitly, as `to_image` and `gif_frames` do.
- **Clean input must come back exact.** Every cleanup step measures whether there is anything to clean (noise sigma, blends, spill) and does nothing when there isn't. The `clean` benchmark row must stay at 1.0000 / palette 1.00.
- **Transparency is binary.** The user explicitly rejected blended or partial alpha at edges. Edge cleanup is a colour-threshold cut plus despill (recolour, or cut).
- **Measure, don't eyeball alone.** Change detection, sampling, palette or keying only with `tests/bench.py` before and after. Then also look at the real images: the benchmark's damage model is not everything AI art does.
- **Stay vectorised.** Use numpy; keep Python loops to per-frame, per-cluster or per-lag work.
- **Single file, minimal dependencies** (numpy, Pillow, optional ffmpeg). No scipy or OpenCV.
- **CLI output stays parseable.** Results go to stdout in the formats shown in `llms.txt`. Warnings, progress, prompts and the `background #rrggbb` line go to stderr. Never prompt when stdin is not a TTY.

## Why it is still Python

Rewriting in a compiled language or moving to the GPU was considered and rejected on measurements (M4 Pro): the 1376×768 still takes 0.9 s and the 2752×1536 one 0.5 s, mostly numpy sorting and gathers that already run on SIMD. The 240-frame 720p video takes about 6.5 s, of which roughly half is ffmpeg decoding and pipe I/O. The work per frame is a few million array operations, far below where a GPU pays for its transfer cost. What did matter was algorithmic: computing edge profiles once per frame instead of four times, profiling on half the rows, threading the per-frame work, and replacing `np.nanpercentile` and a per-colour Python loop with vectorised code (the video went from 15.7 s to 6.5 s). If speed becomes a problem again, profile first (`python -m cProfile`).

## Gotchas learned the hard way

- **Aliasing below 2 px.** Edges sit at integer positions, so frequency `f` and `1-f` look identical. Searching periods under 2 px produced a ghost 1.536 px result for a real 2.867 px grid. `detect_grid` clamps `pmin` to 2.
- **Harmonics.** A grid of size P also lines up perfectly at P/2 and P/3. Pick the largest well-scoring multiple, and don't count harmonics as rivals when judging uncertainty (`_is_harmonic`).
- **Layout looks like a grid.** Evenly spaced sprites give stronger spectral peaks than a blurred 2.4 px grid. Coherence alone picked the sprite spacing; only redrawing the image tells them apart. A phase-shifted "null grid" does not: layout is phase-sensitive too.
- **Redraw error grows with cell size** even for the right grid (a correct 8 px grid scores about 1.5× its own 4 px harmonic), so candidates are compared with generous ratios, and harmonics with their own tighter one. False multiples measured 1.7× and up; true ones 1.6× at most.
- **Don't lock the period to "a whole number of cells".** It looked free for Gemini's exact upscales but mis-locked small images whose last cell is cut off.
- **JPEG noise is not independent per pixel.** It is correlated over 8×8 blocks, so a cell median is not much steadier than one pixel. Scaling the merge distance down by the sample count doubled the palette.
- **Generated art varies from cell to cell beyond its noise.** Measured noise on the soldier sheet is about 1.2, yet it needs a merge distance of about 18 to reach its real palette. Hence `MERGE_MIN`, applied only when some noise is present.
- **A least-squares "deconvolution" sampler is not worth it.** Solving for the cells through a blur model, with the true grid known, gained 0 to 3 points over the centre median and lost in some cases.
- **JPEG chroma subsampling.** Colour is stored at half resolution, so edge pixels have correct brightness but colour smeared from neighbours. Magenta bleeds into dark outlines and makes them purple. A distance threshold can't remove that without deleting the outline; that's what `despill` is for.
- **Despill must judge hue against the local art.** Removing all background hue turned a purple trunk grey and left light grey specks (visible on a dark background) where half-background pixels were recoloured instead of cut.
- **The background is not the colour the user names.** `ff00ff` arrives as about `fd33f8` or `ff2bfc` after JPEG, so `resolve_background` snaps it.
- **Video grids drift.** The test video zooms from 10.35 to 10.63 px per pixel, so one grid can't serve every frame. Each frame's grid phase is only known modulo one cell; the frame-centre art coordinate is unwrapped against the previous frame.
- **AI video is animated "on threes", but not strictly.** Holds in the test video vary between 2 and 4 frames and the rhythm shifts, so a fixed cadence fails; `find_holds` thresholds the actual frame-to-frame change. With several sprites moving out of step the whole-canvas signal is muddier than each sprite's own, which is why `--split` runs it per sprite.
- **Repeated cycles in AI video are not the same drawing.** In the test clip the same pose in different passes shares only 30–80% of its pixels, with two or three passes each. A plain per-pixel vote across passes mixes limbs from different drawings, so `vote_cycle` only votes among passes that differ by noise and otherwise swaps whole frames. On that clip it swaps one pose and votes none; on exact repeats (the synthetic clip) it votes.
- **Ghosts show in the palette, not in the motion.** "Colour lies between the previous and the next frame's" looked plausible but, checked against the poses that are visibly ghosted, found none of them (it flags real motion instead). What works: colours that appear in under a quarter of the sprite's poses. A ghost is a mixture, and mixtures are one-off colours. Check any new detector against poses you have looked at, not against its own scores.
- **Loop rules each fail alone.** The deepest dip is fooled by a still start matching a still end, and prefers three steps to two. The shortest clear dip is fooled by partial repeats (a bob every other pose). Measuring in poses instead of time breaks when holds vary (2 to 6 frames in the test clip). And a full walk cycle must join about as well as one step, or a soldier with one clean pass gets a loop with a bad seam.
- **A standing sprite "loops" perfectly.** `find_loop` picks its start by closure relative to how much happens inside the cycle, or it lands on a motionless stretch.
- **`np.nanpercentile` is extremely slow** on stacked windows (a Python-level loop per column). Sort and index instead.
- **zsh doesn't word-split** `$VAR` holding a command. Use a shell function for repeated commands.

## Verifying changes

Run the tests and the benchmark:

```bash
.venv/bin/python -m unittest discover tests
.venv/bin/python tests/bench.py
```

Benchmark as of this version (186 cases; the first implementation scored 0.779 overall and found the right pixel size in 86% of cases):

```
condition       n  period  shape     acc     min   alpha colour err  palette
clean          31    1.00   1.00  1.0000  1.0000  1.0000       0.00     1.00
jpeg90         31    1.00   1.00  0.9325  0.6597  0.9995       1.98     1.77
blur+jpeg75    31    1.00   1.00  0.8793  0.3483  0.9988       3.18     2.36
offset+noise   31    1.00   1.00  0.8513  0.3142  0.9983       4.07     2.86
wobble         31    1.00   1.00  0.8785  0.4791  0.9989       3.30     2.43
soft           31    1.00   1.00  0.9116  0.5587  0.9994       2.54     2.07
overall acc 0.9088  period ok 1.000
```

`acc` is the share of pixels whose nearest true-palette colour is the true colour; `palette` is output colours over true colours. The low minimums are all one case family: the soldier sprite, whose palette has many near-identical dark greens, at under 4 px with blur.

Then run the real files and compare against the expected numbers:

```bash
.venv/bin/python repixel.py test/Gemini_Generated_Image.jpeg -o out/trees.png --no-ask
# expect: pixel size 2.867 -> 480x268, 365 colours, reconstruction error 3.8/255

.venv/bin/python repixel.py test/Gemini_Generated_Image.jpeg -t magenta -o out/trees_t.png --no-ask
# expect: background #ff2bfc, 225 colours

.venv/bin/python repixel.py test/Gemini_Generated_Image_kewh32kewh32kewh.jpg -o out/soldiers.png --no-ask
# expect: pixel size 22.931, confidence 0.98 -> 120x67, 59 colours, error 1.6/255

.venv/bin/python repixel.py test/Gemini_Generated_Image_kewh32kewh32kewh.jpg -t --trim -o out/soldiers_t.png --no-ask
# expect: 112x33

.venv/bin/python repixel.py test/gemini_generated_video_78be5980.mp4 -t --trim -o out/video.png --no-ask
# expect: pixel size 10.347-10.629 -> 112x34, 85 unique frames (85 poses held about 3 frames each)

.venv/bin/python repixel.py test/gemini_generated_video_78be5980.mp4 -t --split --loop --sheet -o out/sheet.png --no-ask
# expect: 5 sprites, loops of 5, 11, 9, 8 and 8 poses; 4, 1 and 2 ghosted repaired on sprites 2, 3 and 5,
#         1 dropped on each of sprites 2 and 3; 5, 10, 8, 8 and 8 frames; sheet of 10x5 cells, each 22x34
```

Then look at the results. The numbers alone don't catch visual regressions.

- Upscale the outputs ×4–8 with nearest-neighbour and compare against a same-scale crop of the source.
- Composite transparent ones on a dark background `(24,24,28)` as well as grey `(120,120,120)`: light specks and leftover tint at edges only show on dark.
- For GIF/APNG changes, decode every frame and diff against `--frames` PNGs: zero mismatched pixels is required (the test suite does this on a synthetic clip).

The installable package is checked by hand: `pip install .` into a fresh venv, then `repixel --version`. Bump `__version__` in `repixel.py` with each recorded version.

When a change moves the benchmark or the real-file results, record it as a new version: commit it, add the previous version's commit to `VERSIONS` in `docs/make_progress.py` and bump `CURRENT`, run the script, and add a row and a section to `docs/PROGRESS.md` (and the short table in `README.md`).

When you add or change a flag, update `llms.txt` (commands, settings, output format) and the options table in `README.md` in the same change. If a change visibly alters the output, regenerate the images with `.venv/bin/python docs/make_examples.py`.
