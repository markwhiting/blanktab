# Spike 2: can a tab train a learner-sized network fast enough?

**Date:** 2026-10-20. **Method:** `npm run bench:gpu` (`packages/tensor/bench/step.gpu.bench.ts`), a stand-in model with the spec §5.5 layer shapes, trained in headless Chromium with our own `@blanktab/tensor`.

## What the number means for the project

**Yes, with room to spare.** On a MacBook Air (Apple M4) the learner-sized training step runs at **about 108 steps per second on WebGPU**. The spec's target is **20**, so the default width in spec §5.5 stands and nothing needs to shrink. The roughly 5× headroom leaves room for the parts the stand-in leaves out: low-rank adapters, the second forward pass for the learning-progress reward, and replay during sleep.

**Without WebGPU, the target is out of reach at default width.** The same step on our CPU backend in the same tab runs at **11–17 steps per second** (it varies between runs), below 20. A tab without a usable GPU needs about half width (`enc` 64, `hidden` 128), which measured 20.2 steps per second (see "If a device falls short").

Two things got fixed along the way:
- **A GPU memory leak in our library.** Each step leaked two buffers. They came from the convolution on the input frame. The fix means repeated training steps no longer grow GPU memory (checked over 50 steps).
- **A speed-up from 80 to 108 steps per second.** Before the fix, two thirds of each step was spent creating one tiny settings buffer per GPU operation.

**Caveat:** this is one machine, a recent Apple laptop. An older integrated GPU (Intel, low-end AMD) has not been measured and may be several times slower. The 5× headroom should cover that, but this is unverified.

## Setup

| | |
|---|---|
| Machine | MacBook Air, Mac16,12, Apple M4 (10-core GPU, Metal 4), 32 GB, macOS 27.2 (26B5112a) |
| Browser | Playwright 1.64.1 Chromium, `HeadlessChrome/156.0.8078.4`, flags `--enable-unsafe-webgpu --use-angle=metal` |
| Adapter (`adapterInfo`) | `{"vendor":"apple","architecture":"metal-3","device":"","description":"","isFallbackAdapter":false}`, a hardware adapter. The benchmark fails on a software adapter unless `GPU_ALLOW_SOFTWARE=1`. |
| Model | spec §5.5 stand-in, 2,622,661 parameters (inside the spec's 1–3 M range). See `packages/tensor/bench/learner-model.ts`. |
| Inputs | periphery 3×50×80, fovea 3×32×32, proprioception 3. Fresh inputs and targets are uploaded every step. |
| Step | forward, MSE on both frame predictions, dummy policy cross-entropy and value MSE, `backward`, `Adam.step`, `zeroGrad`, `disposeGraph`. The GRU state is carried across steps. |
| Timing | 20 warm-up steps, then 200 timed steps. No readback inside the timed window except the loss after the last step. That read waits for all queued GPU work, so the GPU's own time is included. |
| Work per step | 55,997,436 FLOPs = **0.056 CU** (1 CU = 10⁹ FLOPs, counted exactly per primitive). Both backends give the same count. |

Model layers: periphery conv 3→16 (k5 s2 p2) → relu → conv 16→32 (k3 s2 p1) → relu → linear 8320→128. Fovea conv 3→16 (k3 s2 p1) → relu → conv 16→32 (k3 s2 p1) → relu → linear 2048→128. Core input = sum of the two plus linear(3→128). GRU 128→256. Heads: 256→3000, 256→768, 256→76 (25+3+3+45), 256→1.

## Results (final code, three consecutive runs)

| Backend | steps/s (runs) | ms/step | final loss |
|---|---|---|---|
| WebGPU | 108.8, 107.5, 107.5 | 9.2–9.3 | 0.0576539 |
| CPU (same tab) | 14.8, 16.6, 16.3 | 60–67 | 0.0576539 |

A later pre-commit run gave WebGPU 110.6 and CPU 10.8 steps/s. CPU timing varies between runs more than GPU timing does. Other work was running on the machine at the same time.

The two backends' final losses agree to 7 digits after 220 steps, a whole-model parity check. `liveBufferGrowth` over the 200 timed steps is 0 on both.

JSON lines from the first run:

```
{"backend":"webgpu","stepsPerSec":108.8,"paramCount":2622661,"flopsPerStep":55997436,"cuPerStep":0.055997436,"msPerStep":9.19,"steps":200,"warmup":20,"finalLoss":0.057653941214084625,"liveBufferGrowth":0,"widths":{"conv1":16,"conv2":32,"enc":128,"hidden":256},"adapter":{"vendor":"apple","architecture":"metal-3","device":"","description":"","isFallbackAdapter":false},"userAgent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/156.0.8078.4 Safari/537.36"}
{"backend":"cpu","stepsPerSec":14.8,"paramCount":2622661,"flopsPerStep":55997436,"cuPerStep":0.055997436,"msPerStep":67.37,"steps":200,"warmup":20,"finalLoss":0.05765393376350403,"liveBufferGrowth":0,"widths":{"conv1":16,"conv2":32,"enc":128,"hidden":256},"adapter":null,"userAgent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/156.0.8078.4 Safari/537.36"}
```

## Where the time goes (WebGPU)

A step issues 198 GPU dispatches: 46 matmul, 31 Adam (one per parameter tensor), and the rest are elementwise, reduction and im2col/col2im passes.

| Suspect | Measured | Action |
|---|---|---|
| One uniform buffer created per dispatch | 7.5 ms of the 10.7 ms of CPU-side issue time per step (~38 µs per dispatch) | **Fixed:** one uniform ring per backend. Each pass gets a 256-byte slot, uploaded with one `writeBuffer` per batch. **78.6–80.3 → 107.5–111 steps/s.** |
| Compensated (Neumaier) summation in matmul | Swapping in plain summation (experiment only, not committed) gave 113–114 steps/s, against 95–111 with compensation: ≤ ~15% | Kept. The accuracy was a deliberate choice, and the step meets the target with compensation on. |
| Pipeline creation | Pipelines are cached per kernel key and created during warm-up. Pipeline lookup plus encoder: 0.09 ms/step | None needed |
| Per-step readbacks | None in the step. Only the final loss is read. | None needed |
| Bind-group creation | 1.2 ms/step | Not changed. Next candidate if CPU-bound again. |

After the fix, CPU-side issue time is about 1.3–1.8 ms per step, against about 9 ms per step overall, so **the step is now bound by GPU-side execution**. 56 MFLOP in ~9 ms is only ~6 GFLOP/s, far below what the GPU can do. GPU throughput barely changes when the work shrinks: a run with half the `enc` and `hidden` widths, at 61% of the FLOPs, gave 111.8 steps/s. So the cost is mostly per pass (198 serialized passes, roughly 45 µs each), not arithmetic. Matmuls with little parallelism add to it. For example, the periphery linear layer, [1×8320]·[8320×128], runs as 8 workgroups, each looping over 520 tiles. Fewer, fused passes would raise the ceiling: one Adam pass for all parameters, fused elementwise chains, split-k matmul. None of this is needed for the target. (This is the agent's reading of the numbers, not profiled per kernel; GPU timestamp queries would confirm it.)

The measurements in this table came from temporary instrumentation (timers inside `dispatch`, per-kernel dispatch counters). None of it is committed.

## GPU memory hygiene (Review Focus 2 and 4)

`packages/tensor/test/gpu/resources.gpu.test.ts`:
- **50 training steps of this model:** `stats.liveBuffers` after step 50 equals its value after step 1, and returns to the pre-model value after `model.dispose()`. This failed at first: 97 after step 1, 195 after step 50.
- `zeros(maxStorageBufferBindingSize / 4 + 1)` throws `ResourceError` naming the byte size and the limit (4,294,967,292 on this adapter), and allocates nothing.
- `createWebGpuBackend()` resolves `null` when `navigator.gpu` is `undefined`, and when `requestAdapter()` returns `null`.

**The leak and its fix.** `conv2d` on a non-grad input (the frame) computes `im2col(frame)`. That tensor has no graph node, so `disposeGraph` treated it as a caller-owned leaf and never freed it. But the caller never sees it, so nothing ever freed it: one buffer per input conv per step. `layerNorm` on a constant input had the same problem, leaking 9 buffers per call. The fix is in the library, not the benchmark. Every composite (`linear`, `conv2d`, `gruCell`, `layerNorm`, `mse`) now runs inside `composite(...)`, which settles the node-less tensors it created:
- If a graph node references one, it is marked graph-owned, and `disposeGraph` frees it.
- Otherwise it is dead and freed immediately.

The composite's result, and views of the caller's own tensors, are never touched. Constants the caller creates are still the caller's (unchanged semantics). Node tests in `packages/tensor/test/composite.test.ts` cover conv2d, layerNorm, a fully-constant composite, constant reuse across steps, and views of caller inputs. `composite` is exported, so the learner's own layers can use it.

## If a device falls short

Not needed on this machine. For slower devices, per spec §5.5/§6: shrink widths first, resolution last. The benchmark takes `BENCH_WIDTHS='{"hidden":128,"enc":64}'` (fields `conv1`, `conv2`, `enc`, `hidden`), so a slower machine can find its width directly.

One measured point (single run, same machine): `enc` 64, `hidden` 128 (1,244,997 parameters, 0.034 CU/step) gives **CPU 20.2 steps/s**, just at the target, and WebGPU 111.8 steps/s. So a tab without WebGPU reaches the target at roughly half width. WebGPU speed is nearly flat in width on this machine (see above), so width cuts help a GPU tab only once it is FLOP-bound.

## Not yet verified
- Any machine other than an Apple M4, in particular an Intel or AMD integrated GPU, and any non-Chromium browser.
- Headed (visible) Chrome, and a background tab, where throttling may change the numbers (spike 4).
- Sustained runs. These are 220-step runs, so thermal throttling over hours is not measured.
