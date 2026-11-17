# AGENTS.md

Repixel recovers true-resolution pixel art from upscaled, JPEG-damaged or AI-generated images and videos.

**To use the tool** (commands, flags, reading the output, choosing settings), read [llms.txt](llms.txt). That file is the user-facing documentation for LLMs and agents. Keep it in sync whenever the CLI changes.

This file is for agents **changing the code**.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

- The existing `.venv` was built with Homebrew Python 3.14. The system `/usr/bin/python3` (3.9) has no numpy and is too old: the code uses `X | None` types and needs Python 3.10+.
- Video needs `ffmpeg` and `ffprobe` on PATH.

## Layout

| Path | What |
|---|---|
| `repixel.py` | The whole tool: one file, CLI entry point `main()` |
| `llms.txt` | Usage docs for LLMs/agents |
| `README.md` | Usage docs for humans, with example images |
| `docs/` | README example images, generated from the `test/` inputs: before/after tree crop, soldier sheet, walking animation |
| `requirements.txt` | numpy, Pillow |
| `test/` | Sample inputs: two Gemini JPEGs (trees, soldier sheet) and one Gemini mp4 (walking soldiers) |
| `out/` | Scratch outputs from manual runs; safe to delete |

## How `repixel.py` is organised

1. **Grid detection**
   - `edge_profile`: per-axis edge strength, ignoring small JPEG-noise jumps (`EDGE_NOISE`).
   - `find_periods`: DFT-coherence score over candidate frequencies, then harmonic resolution: it walks up integer multiples of the best period and keeps the largest one scoring at least 0.6 of the best.
   - `find_period_near`: same scoring in a narrow window, used to track the pixel size from frame to frame in video.
   - `grid_offset` / `grid_lines`: grid phase, then `_snap_lines` pulls each line onto a real edge when the period is 5 px or more. `grid_lines` returns `(lines, k_first)`; video needs `k_first` to place frames on a shared canvas.
2. **Sampling and palette**
   - `sample_cells`: median of each cell's centre region.
   - `merge_similar` (greedy, frequency-ordered) or `kmeans_quantize`.
3. **Transparency**
   - `resolve_bg` snaps a given colour to the image's real background colour.
   - `key_background` cuts pixels by RGB distance (binary alpha).
   - `despill` recolours tinted edge pixels in YCbCr and snaps them to untinted interior colours.
   - `trim_box` finds the crop.
4. **Output**
   - `to_image` builds exact indexed PNGs.
   - `gif_frames` builds exact shared-palette GIF frames.
5. **Flows**
   - `process_image`: the still-image path.
   - `process_video`: two ffmpeg passes. Pass 1 measures the pixel size per frame, then median-smooths it. Pass 2 finds the grid per frame and unwraps each frame's grid phase so all frames land on one art-pixel canvas. Then the steps above with one palette across all frames, the one-frame blip filter, and merging of repeated frames.
   - `choose_period`, `clean_colors`: steps shared by both paths.

## Design rules

- **Outputs must be lossless.** Never let Pillow quantise behind your back: `Image.quantize` and RGB→P conversion can shift colours. Build palettes explicitly, as `to_image` and `gif_frames` do.
- **Transparency is binary.** The user explicitly rejected blended or partial alpha at edges. Edge cleanup is a colour-threshold cut plus despill (recolouring only).
- **Stay vectorised.** Use numpy; keep Python loops to per-frame or per-unique-colour work. The 2752×1536 still takes about 0.4 s; the 240-frame 720p video takes about 15 s.
- **Single file, minimal dependencies** (numpy, Pillow, optional ffmpeg). No scipy or OpenCV.
- **CLI output stays parseable.** Results go to stdout in the formats shown in `llms.txt`. Warnings, progress and the `background #rrggbb` line go to stderr. Never prompt when stdin is not a TTY.

## Gotchas learned the hard way

- **Aliasing below 2 px.** Edges sit at integer positions, so frequency `f` and `1-f` look identical. Searching periods under 2 px produced a ghost 1.536 px result for a real 2.867 px grid. `find_periods` clamps `pmin` to 2.
- **Harmonics.** A grid of size P also lines up perfectly at P/2 and P/3. Pick the largest well-scoring multiple, and don't count harmonics as rivals when judging uncertainty (`_is_harmonic`).
- **JPEG chroma subsampling.** Colour is stored at half resolution, so edge pixels have correct brightness but colour smeared from neighbours. Magenta bleeds into dark outlines and makes them purple. A distance threshold can't remove that without deleting the outline; that's what `despill` is for.
- **The background is not the colour the user names.** `ff00ff` arrives as about `fd33f8` or `ff2bfc` after JPEG, so `resolve_bg` snaps it.
- **Video grids drift.** The test video zooms from 10.35 to 10.63 px per pixel, so one grid can't serve every frame. Each frame's grid phase is only known modulo one cell; the frame-centre art coordinate is unwrapped against the previous frame.
- **zsh doesn't word-split** `$VAR` holding a command. Use a shell function for repeated commands.

## Verifying changes

There is no automated test suite. After changing detection, sampling, keying or output, run these and compare against the expected numbers:

```bash
.venv/bin/python repixel.py test/Gemini_Generated_Image.jpeg -o out/trees.png --no-ask
# expect: pixel size 2.867 -> 480x268, reconstruction error ~3.8/255

.venv/bin/python repixel.py test/Gemini_Generated_Image_kewh32kewh32kewh.jpg -o out/soldiers.png --no-ask
# expect: pixel size 22.933, confidence ~0.98 -> 120x67, ~45 colours, error ~1.6/255

.venv/bin/python repixel.py test/Gemini_Generated_Image_kewh32kewh32kewh.jpg -t --trim -o out/soldiers_t.png --no-ask
# expect: 112x33

.venv/bin/python repixel.py test/gemini_generated_video_78be5980.mp4 --fps 12 -t --trim -o out/video.png --no-ask
# expect: pixel size 10.347-10.631 -> 112x34, 120 unique frames
```

Then look at the results. The numbers alone don't catch visual regressions.

- Upscale the outputs ×6–8 with nearest-neighbour, composite transparent ones on grey `(120,120,120)`, and compare against a same-scale crop of the source.
- For transparency changes, count leftover background-tinted pixels: opaque pixels whose YCbCr chroma, projected onto the background's chroma direction, exceeds 20. On the trees with `-t magenta` that count is about 33 with despill and about 1,968 without.
- For GIF/APNG changes, decode every frame and diff against `--frames` PNGs: zero mismatched pixels is required.

When you add or change a flag, update `llms.txt` (commands, settings, output format) and the options table in `README.md` in the same change. If a change visibly alters the output, regenerate the images in `docs/`.
