# World rendering speed in a real browser (spike 3)

**Date:** 2026-10-20.

## Summary
The world draws and reads back one frame in well under a millisecond per step in a real browser worker. The bar was 60 steps per second per room; every room runs at roughly 1,000 to 2,300 steps per second, so 18 to 40 times faster than needed. Nothing needed fixing for speed. The look of each room was checked at 3x: all four are legible and visibly different from each other. One cosmetic limit is noted below (the font's descenders).

## Method
- `npm run bench:render -w @blanktab/world` serves `packages/world/bench/` with Vite and opens it in Playwright Chromium.
- A dedicated Worker makes an `OffscreenCanvas(320,200)` and `createWorld({ seed: 1, canvas })`.
- For each room: `bringToFront`, 30 warm-up steps, then 600 timed iterations of `act(random valid action, seeded rng)` + `observe()` (render, full-frame `getImageData`, 80x50 downsample, 32x32 fovea).
- Machine: Apple M4 Mac, Chromium 156.0.8078.4 (headless, Playwright). Another benchmark (WebGPU) was running on the same machine, so numbers are noisy; load average was about 4 to 6. Each figure below is the median of three runs.

## Results (median of 3)
```
{"room":"crib","msPerStep":0.438,"stepsPerSec":2281.4}
{"room":"toys","msPerStep":0.514,"stepsPerSec":1946.8}
{"room":"hideseek","msPerStep":0.91,"stepsPerSec":1098.9}
{"room":"places","msPerStep":0.711,"stepsPerSec":1406.8}
```
All three runs, steps per second (crib / toys / hideseek / places):

| Run | crib | toys | hideseek | places |
|---|---|---|---|---|
| 1 | 2281 | 1947 | 1099 | 1155 |
| 2 | 1866 | 1709 | 1071 | 1407 |
| 3 | 2655 | 2479 | 1303 | 1434 |

**Verdict:** target (>= 60 steps/s per room) met in every room, with a margin of at least 17x even in the worst single run (hideseek, 1,071/s).

## What was fixed
- Speed: nothing. The suspected costs (one `fillRect` per lit text pixel, area rescans, redrawing static layers every step) are real but cheap at this size; areas are cached and only computed by `state()`, not on the step path. Optimising would only risk the byte-identical pixel output, so it was left alone.
- Carried from the Places review: the link-word pool is now deduplicated and the room throws a `RangeError` if there are fewer unique words than pages. New tests (seeds 1-50) check that link words are distinct per page and that link boxes lie inside the viewport and do not overlap. Unused imports and variables in `places.test.ts` were removed.

## Looking at the rooms (3x, seed 1)
Pictures are in `docs/research/2026-10-20-world-render-speed/` (`crib.png`, `toys.png`, `hideseek.png`, `places.png`).
- **Crib:** coloured circles and squares on a cream background; clearly distinct from the others.
- **Toys:** a labelled button with a counter field, two sliders with knobs, a toggle, plain coloured blocks and loose words; widgets are recognisable and words are readable.
- **Hide-and-seek:** a title, tabs, a content panel, a scrollbar, and a pop-up window with a title bar and close box in front; the pop-up is clearly visible.
- **Places:** a back button, a large title, lines of text and four underlined blue links.
- **Known cosmetic limit:** the 5x7 font has no room for descenders, so lowercase `p`, `y` and `g` sit one row lower than other letters and can look like capitals at a glance ("map" reads like "maP"). Words stay readable in context. Redesigning the glyphs was left out of this task; it would change pixel output.
