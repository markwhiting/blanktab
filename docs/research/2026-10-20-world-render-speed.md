# World rendering speed in a real browser (spike 3)

**Date:** 2026-10-20.

## Summary
The world draws and reads back one frame in well under a millisecond per step in a real browser worker. The bar was 60 steps per second per room; every room runs at roughly 900 to 2,300 steps per second, so at least 15 times faster than needed even in the worst single run (hideseek, 907/s). Nothing needed fixing for speed. The look of each room was checked at 3x: all four are legible and visibly different from each other. Looking at them did find one real problem, now fixed: the font had no descenders, so p, y and g looked like capitals ("maP"). The glyph cell is now 9 rows tall with real tails.

## Method
- `npm run bench:render -w @blanktab/world` serves `packages/world/bench/` with Vite and opens it in Playwright Chromium.
- A dedicated Worker makes an `OffscreenCanvas(320,200)` and `createWorld({ seed: 1, canvas })`.
- For each room: `bringToFront`, 30 warm-up steps, then 600 timed iterations of `act(random valid action, seeded rng)` + `observe()` (render, full-frame `getImageData`, 80x50 downsample, 32x32 fovea).
- The page was served by the Vite dev server (not a production build). `--runs 3` makes `run.mjs` load the page three times and print the median per room.
- Machine: Apple M4 Mac, Chromium 156.0.8078.4 (headless, Playwright). Another benchmark (WebGPU) was running on the same machine, so numbers are noisy; load average was about 4 to 6. Each figure below is the median of three runs.

## Results (final code, median of 3 runs via `--runs 3`)
```
{"room":"crib","runs":3,"msPerStep":0.473,"stepsPerSec":2114.2}
{"room":"toys","runs":3,"msPerStep":0.523,"stepsPerSec":1912}
{"room":"hideseek","runs":3,"msPerStep":1.006,"stepsPerSec":993.5}
{"room":"places","runs":3,"msPerStep":0.866,"stepsPerSec":1154.3}
```
Steps per second in each run (crib / toys / hideseek / places):

| Run | crib | toys | hideseek | places |
|---|---|---|---|---|
| 1 | 1686 | 1664 | 907 | 1031 |
| 2 | 2114 | 1912 | 994 | 1154 |
| 3 | 2314 | 2037 | 1063 | 1191 |

Earlier, before the font change, the medians were aggregated by hand from three runs (crib 2281, toys 1947, hideseek 1099, places 1407) plus a fourth run with hideseek at 921/s; those are no longer the headline numbers because the taller glyphs changed the drawing slightly.

**Verdict:** target (>= 60 steps/s per room) met in every room. Worst case seen in any run: hideseek at 907/s, a margin of about 15x.

## What was fixed
- Font: descenders added (see above); font tests check the tails and that g/9, y/Y, p/P, q/9, j/J masks differ.
- Speed: nothing. The suspected costs (one `fillRect` per lit text pixel, area rescans, redrawing static layers every step) are real but cheap at this size; areas are cached and only computed by `state()`, not on the step path. Optimising would only risk the byte-identical pixel output, so it was left alone.
- Carried from the Places review: the link-word pool is now deduplicated and the room throws a `RangeError` if there are fewer unique words than pages. New tests (seeds 1-50) check that link words are distinct per page and that link boxes lie inside the viewport and do not overlap. Unused imports and variables in `places.test.ts` were removed.

## Looking at the rooms (3x, seed 1)
Pictures are in `docs/research/2026-10-20-world-render-speed/` (`crib.png`, `toys.png`, `hideseek.png`, `places.png`).
- **Crib:** coloured circles and squares on a cream background; clearly distinct from the others.
- **Toys:** a labelled button with a counter field, two sliders with knobs, a toggle, plain coloured blocks and loose words; widgets are recognisable and words are readable.
- **Hide-and-seek:** a title, tabs, a content panel, a scrollbar, and a pop-up window with a title bar and close box in front; the pop-up is clearly visible.
- **Places:** a back button, a large title, lines of text and four underlined blue links.
- **Descenders (fixed):** the font cell grew from 7 to 9 rows (baseline after row 7). p, y, g, q and j now have real tails below the baseline, so "map", "king" and "young" read as lowercase. Tab, header, button and pop-up title text moved up one pixel to keep the tails inside their bars. The pictures above were regenerated after this change and checked by eye.
