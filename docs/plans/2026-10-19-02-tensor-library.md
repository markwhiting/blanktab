# Plan 02 — Tensor library (our own math engine)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**In plain terms:** the learner needs a way to do neural-network math inside the browser tab, fast, using the computer's graphics card. We're writing that ourselves, with no ML libraries, so nothing pretrained can sneak in and so we can count exactly how much computation each learner uses. This plan delivers that engine and proves it is correct and fast enough. It plugs into the end-to-end loop that Plan 01 already runs on GitHub Pages.

**Prerequisite:** Plan 01 is complete. The repo, walls, CI and `@blanktab/interface` already exist.

**Goal:** our own tensor/autodiff library with CPU and WebGPU backends, proven by parity tests, gradient checks and a learner-sized throughput benchmark.

**Architecture:** npm workspaces under `packages/`. `tensor` is a dynamic-graph autodiff library (micrograd-style: each result tensor records its parents and a backward closure). Ops are written once against a storage-level `Backend` interface, implemented twice: plain TypeScript (reference, fallback, CI) and WebGPU/WGSL (fast path). Every backend primitive adds to an exact FLOP counter.

**Tech Stack:** TypeScript (strict), Node 22, npm workspaces, Vite, Vitest (node + browser mode via Playwright/Chromium), ESLint flat config, GitHub Actions.

**Spec:** `docs/specs/2026-10-19-blanktab-v1-design.md` (§4, §5.2–5.3, §6, §12). Read it and `AGENTS.md` first.

## Global Constraints

- No pretrained weights or assets of any kind, and no ML libraries (no TF.js, ONNX, transformers.js). All math is ours. (AGENTS.md invariant 1; spec §6.)
- Walls: `learner` may import only `@blanktab/interface` and `@blanktab/tensor`. `world` may import only `@blanktab/interface`. `tensor` and `interface` import no workspace package. `harness` and `app` may import anything. A violation fails `npm run lint`. (Spec §4.)
- The compute unit is 1 CU = 10⁹ FLOPs. FLOPs are counted exactly per primitive. (Spec §6.)
- `SCHEMA_VERSION = 1`.
- Dependencies: after any `npm install`, run `npm run lint && npm run typecheck && npm test`. If npm reports "removed N" or "changed N" packages, stop and investigate. (User's global rule.)
- Commit after every task. Never push unless asked.

## Review Focus

1. **Broadcasting mismatches:** `add` of shapes `[2,3]` and `[4]` must throw `ShapeError` naming both shapes, not silently produce garbage. Test in Task 2.
2. **GPU memory growth:** a full training step repeated 50 times must not grow the live GPU buffer count. Test in Task 6.
3. **Numeric stability:** `logSoftmax` and `softmax` on logits around ±1000 must return finite values. Test in Task 2.
4. **WebGPU unavailable or over limits:** `createWebGpuBackend()` resolves to `null` when there is no adapter. Allocating more than `maxStorageBufferBindingSize` throws a `ResourceError` naming the size and the limit. Test in Task 6.
5. **Gradient accumulation:** a tensor used twice in one graph (e.g. `mul(x, x)`) must receive the summed gradient `2x`, and `zeroGrad` must reset it. Test in Task 3.

---

## File structure

```
package.json                     workspaces, scripts: lint, typecheck, test, test:gpu, bench:gpu
tsconfig.base.json               strict, ES2022, moduleResolution bundler
eslint.config.js                 wall rules (per-package no-restricted-imports)
vitest.config.ts                 node tests for all packages (CPU)
vitest.gpu.config.ts             browser-mode tests: *.gpu.test.ts, *.gpu.bench.ts
.github/workflows/ci.yml
tools/walls.test.ts
packages/interface/src/{index,constants,action,observation}.ts  + test/
packages/tensor/src/
  backend.ts       Backend + Storage interfaces, op-name unions, errors
  cpu.ts           createCpuBackend()
  webgpu.ts        createWebGpuBackend()
  kernels/*.wgsl.ts  WGSL source strings, one file per primitive family
  tensor.ts        Tensor class, factories, default backend
  ops.ts           differentiable primitive ops
  composite.ts     conv2d, linear, gruCell, layerNorm, mse
  autograd.ts      backward(), noGrad()
  optim.ts         Adam
  random.ts        createRng(), randn init helpers
  gradcheck.ts     finite-difference checker (exported for tests)
  index.ts
packages/tensor/test/*.test.ts            CPU
packages/tensor/test/gpu/*.gpu.test.ts    WebGPU parity
packages/tensor/bench/step.gpu.bench.ts   spike 2 benchmark
packages/{learner,world,harness}/src/index.ts   empty placeholders (`export {}`) so walls can be tested
```

---

### Task 1: Tensor backend interface and CPU backend

**Files:**
- Create: `packages/tensor/src/backend.ts`, `cpu.ts`, `random.ts`; `packages/tensor/test/cpu-backend.test.ts`

**Interfaces:**
- Produces (exact, in `backend.ts`):

```ts
export class ShapeError extends Error {}
export class ResourceError extends Error {}
export type UnaryOp = "neg"|"relu"|"step"|"tanh"|"sigmoid"|"exp"|"log"|"recip"|"sqrt"|"square"
export type BinaryOp = "add"|"sub"|"mul"|"div"
export interface Storage { readonly length: number; readonly backend: Backend }
export interface Backend {
  readonly name: "cpu" | "webgpu"
  readonly stats: { flops: number; liveBuffers: number }
  upload(data: Float32Array): Storage
  zeros(length: number): Storage
  read(s: Storage): Promise<Float32Array>
  dispose(s: Storage): void
  unary(op: UnaryOp, a: Storage): Storage
  scalar(op: BinaryOp, a: Storage, c: number): Storage                 // a op c elementwise
  binary(op: BinaryOp, a: Storage, aShape: number[], b: Storage, bShape: number[], outShape: number[]): Storage  // NumPy broadcasting
  sumToShape(a: Storage, fromShape: number[], toShape: number[]): Storage  // inverse of broadcast
  matmul(a: Storage, b: Storage, m: number, k: number, n: number, transA: boolean, transB: boolean): Storage
  reduceLast(op: "sum" | "max", a: Storage, rows: number, cols: number): Storage   // -> length rows
  im2col(x: Storage, c: number, h: number, w: number, kh: number, kw: number, stride: number, pad: number): Storage
    // x is CHW; output [c*kh*kw, outH*outW] row-major, zero padding
  col2im(cols: Storage, c: number, h: number, w: number, kh: number, kw: number, stride: number, pad: number): Storage  // adjoint of im2col (sums overlaps)
  gatherRows(a: Storage, rows: number, cols: number, idx: Int32Array): Storage
  scatterAddRows(grad: Storage, rows: number, cols: number, idx: Int32Array): Storage  // adjoint of gatherRows
  adam(p: Storage, g: Storage, m: Storage, v: Storage, lr: number, b1: number, b2: number, eps: number, t: number): void  // in-place, bias-corrected
}
export function broadcastShape(a: number[], b: number[]): number[]   // throws ShapeError naming both shapes
```

- FLOP accounting (exact): `unary`, `scalar` and `binary` add the output length. `matmul` adds `2*m*k*n`. `reduceLast` adds `rows*cols`. `sumToShape` adds the input length. `adam` adds `10 * p.length`. `im2col`, `col2im`, `gatherRows`, `scatterAddRows`, `upload` and `read` add 0.
- `liveBuffers` counts storages allocated and not yet disposed.
- `random.ts`: `createRng(seed: number): () => number` (mulberry32, uniform on [0,1)); `randn(rng, n): Float32Array` (Box–Muller).

- [ ] **Step 1: Write the failing tests.** These are exact-value tests on small inputs:
  - `matmul` with all four `transA`/`transB` combinations against a hand-computed 2×3·3×2 product. Assert `stats.flops` increases by exactly `2*2*3*2 = 24`.
  - `binary("add")` with shapes `[2,3]` + `[3]` → `[[1+10,2+20,3+30],[4+10,5+20,6+30]]`.
  - `broadcastShape([2,3],[4])` throws `ShapeError` whose message contains `[2,3]` and `[4]`.
  - `sumToShape` from `[2,3]` to `[3]` gives column sums, and from `[2,3]` to `[2,1]` gives row sums.
  - `reduceLast("max")` over rows.
  - `im2col` on a 1×3×3 input, 2×2 kernel, stride 1, pad 0 → a 4×4 matrix with known entries. `col2im` is its adjoint: `dot(im2col(x), y) == dot(x, col2im(y))` for random x, y (within 1e-4).
  - `gatherRows`/`scatterAddRows` adjointness, checked the same way, including repeated indices.
  - `adam` one step matches a hand-computed value for `p=1, g=0.5, lr=0.1, t=1` (→ `p = 0.9`).
  - `liveBuffers` goes up by one on `upload` and back down on `dispose`.
  - `createRng(42)` is deterministic, and `randn` over 10⁵ samples has mean ≈ 0 (±0.02) and std ≈ 1 (±0.02).
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/tensor/test/cpu-backend.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** `createCpuBackend(): Backend` in `cpu.ts` with plain loops over `Float32Array`, plus `random.ts`.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(tensor): backend interface and CPU reference backend"`

---

### Task 2: Tensor class and differentiable ops (forward)

**Files:**
- Create: `packages/tensor/src/tensor.ts`, `ops.ts`, `index.ts`; `packages/tensor/test/ops.test.ts`

**Interfaces:**
- Consumes: `Backend`, `Storage`, `broadcastShape` (Task 1).
- Produces:

```ts
export class Tensor {
  readonly shape: readonly number[]; readonly size: number
  readonly storage: Storage; readonly backend: Backend
  requiresGrad: boolean; grad: Tensor | null
  read(): Promise<Float32Array>
  dispose(): void
}
export function setDefaultBackend(b: Backend): void
export function getDefaultBackend(): Backend          // CPU backend until set
export function tensor(data: Float32Array | number[], shape: number[], opts?: { requiresGrad?: boolean; backend?: Backend }): Tensor
export function zeros(shape: number[], opts?): Tensor
export function param(data: Float32Array, shape: number[]): Tensor   // requiresGrad true
// ops.ts — all return new Tensors and record their parents when any input requiresGrad (and grad mode is on)
add, sub, mul, div(a: Tensor, b: Tensor): Tensor          // broadcasting
scale(a: Tensor, c: number): Tensor; addScalar(a: Tensor, c: number): Tensor
neg, relu, tanh, sigmoid, exp, log, sqrt, square(a: Tensor): Tensor
matmul(a: Tensor, b: Tensor): Tensor                     // 2-D only: [m,k]·[k,n]
sum(a: Tensor): Tensor                                   // scalar, shape []
mean(a: Tensor): Tensor
sumLast(a: Tensor): Tensor                               // [..., n] -> [...]
softmax(a: Tensor): Tensor; logSoftmax(a: Tensor): Tensor   // over the last axis, max-subtracted
reshape(a: Tensor, shape: number[]): Tensor              // shares storage; -1 allowed once
gatherRows(a: Tensor, idx: Int32Array): Tensor           // a: [rows, cols]
```

- [ ] **Step 1: Write the failing tests:**
  - forward values for each op on small inputs;
  - `matmul` on shapes `[2,3]`·`[2,3]` throws `ShapeError`;
  - `add` of `[2,3]` and `[4]` throws `ShapeError` (Review Focus 1);
  - `softmax(tensor([1000, 0, -1000], [3]))` and `logSoftmax` of the same are all finite, and softmax ≈ `[1, 0, 0]` (Review Focus 3);
  - `reshape([2,3] → [3,-1])` gives shape `[3,2]` and shares storage (`.storage` is the same object);
  - ops on a tensor with `requiresGrad: false` record no parents.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/tensor/test/ops.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Each op calls one or a few backend primitives. `softmax` = `exp(a − max)` / `sumLast`, using `reduceLast("max")` and broadcasting over the last axis. A tensor keeps its graph node as an internal `_node?: { parents: Tensor[]; backward: (g: Tensor) => Tensor[] }`. The backward closures are written in Task 3. In this task they may throw "not implemented".
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(tensor): Tensor class and forward ops"`

---

### Task 3: Autodiff, gradient checker, Adam

**Files:**
- Create: `packages/tensor/src/autograd.ts`, `gradcheck.ts`, `optim.ts`; modify `ops.ts` (backward closures); `packages/tensor/test/autograd.test.ts`, `optim.test.ts`

**Interfaces:**
- Produces:

```ts
export function backward(loss: Tensor): void      // loss must be shape []; seeds grad 1; topological order; accumulates into .grad of leaves with requiresGrad
export function noGrad<T>(fn: () => T): T
export function disposeGraph(loss: Tensor, keep: Tensor[]): void   // disposes every non-leaf tensor reachable from loss, plus its grads, except those in keep
export function gradCheck(f: (...xs: Tensor[]) => Tensor, inputs: { data: Float32Array; shape: number[] }[], eps?: number): number  // returns max relative error (CPU only)
export class Adam {
  constructor(params: Tensor[], opts: { lr: number; beta1?: number; beta2?: number; eps?: number })  // defaults 0.9, 0.999, 1e-8
  step(): void        // uses backend.adam; skips params whose grad is null
  zeroGrad(): void    // disposes grads and sets them to null
  setLr(lr: number): void
}
```

- Backward rules: broadcasting ops reduce gradients with `sumToShape`. The `relu` gradient uses `step`. The `tanh` gradient is `1 − y²` and the `sigmoid` gradient is `y(1−y)`. The `log` gradient uses `recip`. `matmul` gradients use the `transA`/`transB` flags (no explicit transposes). `gatherRows` uses `scatterAddRows`. `reshape` reshapes the gradient. `softmax`/`logSoftmax` use their standard Jacobian-vector forms.

- [ ] **Step 1: Write the failing tests:**
  - `gradCheck` < 1e-3 for every op in Task 2, each on a random 3×4 input. Use positive inputs for `log`, `sqrt`, `recip` and `div`. Include the broadcasting `add [3,4]+[4]`, `matmul [3,4]·[4,2]`, and `logSoftmax` followed by `gatherRows`.
  - Review Focus 5: for `x = param([3],[1])`, `backward(sum(mul(x,x)))` gives `x.grad = 6`. A second backward without `zeroGrad` gives 12. After `zeroGrad`, `grad` is null.
  - `noGrad(() => mul(x,x))` has no graph node.
  - `disposeGraph`: record `liveBuffers` before a forward pass over params. Then run forward, `backward`, `disposeGraph(loss, [])` and `opt.zeroGrad()`. `liveBuffers` is back to the recorded value.
  - Adam fits `y = 3x − 2` on 64 points from a seeded rng: within 500 steps at `lr = 0.05`, MSE < 1e-3.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/tensor/test/autograd.test.ts packages/tensor/test/optim.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** `backward`, `noGrad`, `disposeGraph`, the backward closures, `gradCheck` (central differences in float64 over the CPU backend) and `Adam`.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(tensor): reverse-mode autodiff, gradient checker, Adam"`

---

### Task 4: Composite layers — linear, conv2d, GRU cell, layerNorm, MSE

**Files:**
- Create: `packages/tensor/src/composite.ts`, add an `im2col` op to `ops.ts`; `packages/tensor/test/composite.test.ts`

**Interfaces:**
- Produces:

```ts
export function im2col(x: Tensor, kh: number, kw: number, stride: number, pad: number): Tensor   // x [C,H,W] -> [C*kh*kw, outH*outW]; backward via col2im
export function linear(x: Tensor, w: Tensor, b?: Tensor): Tensor          // x [B,in], w [in,out], b [out]
export function conv2d(x: Tensor, w: Tensor, b: Tensor | undefined, opts: { kh: number; kw: number; stride: number; pad: number }): Tensor
  // x [C,H,W]; w [O, C*kh*kw]; b [O, 1]; out [O,outH,outW]. Implemented as matmul(w, im2col(x)) + b, then reshape.
export interface GruParams { wz: Tensor; uz: Tensor; bz: Tensor; wr: Tensor; ur: Tensor; br: Tensor; wh: Tensor; uh: Tensor; bh: Tensor }
export function initGru(rng: () => number, inDim: number, hidDim: number): GruParams   // param() tensors, uniform ±1/sqrt(hidDim)
export function gruCell(x: Tensor, h: Tensor, p: GruParams): Tensor
  // z=σ(xWz+hUz+bz), r=σ(xWr+hUr+br), ĥ=tanh(xWh+(r⊙h)Uh+bh), h' = (1−z)⊙h + z⊙ĥ ; x [B,in], h [B,hid]
export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps?: number): Tensor   // over the last axis, eps 1e-5
export function mse(pred: Tensor, target: Tensor): Tensor   // mean of squared differences, shape []
```


- [ ] **Step 1: Write the failing tests:**
  - `conv2d` against a naive nested-loop convolution for `C=3, H=10, W=8, O=4, k=3, stride=2, pad=1` (max abs diff < 1e-5), with output shape `[4,5,4]`;
  - `gradCheck` < 1e-3 for `conv2d`, `gruCell` (B=2, in=5, hid=4), `layerNorm` and `mse`;
  - `gruCell` with `z` forced to 0 (bz = −50) returns `h` unchanged within 1e-4.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/tensor/test/composite.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** The composites are built only from Task 2–5 ops plus the new `im2col` op. They have no backward code of their own.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(tensor): linear, conv2d, GRU cell, layerNorm, MSE"`

---

### Task 5: WebGPU backend with browser-mode parity tests

**Files:**
- Create: `packages/tensor/src/webgpu.ts`, `packages/tensor/src/kernels/{elementwise,binary,matmul,reduce,im2col,gather,adam}.wgsl.ts`, `vitest.gpu.config.ts`, `packages/tensor/test/gpu/parity.gpu.test.ts`; modify `package.json` (scripts `test:gpu`, `bench:gpu`)

**Interfaces:**
- Consumes: the `Backend` interface (Task 1).
- Produces: `createWebGpuBackend(opts?: { powerPreference?: "high-performance" | "low-power" }): Promise<Backend | null>`. It resolves `null` when `navigator.gpu` or an adapter is missing. It reads `device.limits.maxStorageBufferBindingSize`, and any allocation over it throws `ResourceError` with a message containing both numbers. It counts FLOPs exactly as the CPU backend does. `read()` copies to a `MAP_READ` staging buffer.

- [ ] **Step 1: Write the failing parity tests.** For each primitive in `Backend`, run the same seeded random inputs through the CPU and GPU backends and compare with `max |cpu − gpu| ≤ 1e-4 · max(1, |cpu|)`. Use shapes that aren't multiples of the workgroup size: matmul with `m=37, k=129, n=65`, all four transpose flags; and im2col for the periphery encoder's first layer (`C=3, H=50, W=80, k=5, stride=2, pad=2`). Also:
  - a full training step of the Task 3 regression on GPU reaches MSE < 1e-3;
  - `stats.flops` is identical between backends for the same op sequence.
- [ ] **Step 2: Configure browser mode and verify the tests fail.** Configure `vitest.gpu.config.ts` with browser mode, the Playwright provider and Chromium launched with `--enable-unsafe-webgpu`. Include `**/*.gpu.test.ts` and `**/*.gpu.bench.ts`. Follow the installed Vitest version's browser-mode docs, since the provider setup differs between major versions. If headless Chromium on macOS has no WebGPU adapter, run headed. Run: `npm run test:gpu`. Expected: FAIL (the backend is missing). Confirm the failure is not "no adapter"; if it is, fix the browser launch first.
- [ ] **Step 3: Implement** the WGSL kernels and `webgpu.ts`. Use one compute pipeline per primitive variant, cached by key; 256-thread workgroups for elementwise kernels; and a 16×16 tiled matmul with the transpose flags as uniforms. Broadcasting passes shapes and strides as a uniform buffer (max rank 4; throw `ShapeError` above that).
- [ ] **Step 4: Run the tests to verify they pass.** Run: `npm run test:gpu`. Expected: PASS. Also run `npm test` (CPU and CI suite) and expect PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(tensor): WebGPU backend with CPU parity tests"`

---

### Task 6: Learner-sized throughput benchmark (spike 2) and GPU memory hygiene

**Files:**
- Create: `packages/tensor/bench/step.gpu.bench.ts`, `packages/tensor/test/gpu/resources.gpu.test.ts`, `docs/research/2026-10-XX-webgpu-step-throughput.md` (use the date the benchmark is run)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: a recorded steps-per-second figure. This decides whether spec §5.5's default width stands.

**The benchmark model** follows spec §5.5. It is a stand-in, not the learner:
- **Periphery encoder:** conv 3→16 (k5, s2, p2) → relu → conv 16→32 (k3, s2, p1) → relu → linear to 128.
- **Fovea encoder:** conv 3→16 (k3, s2, p1) → relu → conv 16→32 (k3, s2, p1) → relu → linear to 128.
- **Core input:** sum of the two encodings plus a linear projection of the 3-d proprioception.
- **Core:** GRU, 128 → 256.
- **Heads:** predictor linear 256 → 3000 (40·25·3) and 256 → 768 (16·16·3); policy logits 256 → 25+3+3+45; value 256 → 1.

**Per step:** forward, MSE on both predictions plus a dummy policy/value loss, `backward`, `Adam.step`, `zeroGrad`, then `disposeGraph`.

- [ ] **Step 1: Write the failing resource test** (Review Focus 2 and 4):
  - Run the benchmark step 50 times on GPU. `stats.liveBuffers` after step 50 must equal its value after step 1.
  - Try to `zeros()` a length of `maxStorageBufferBindingSize / 4 + 1` and expect a `ResourceError`.
  - With `navigator.gpu` stubbed to `undefined`, `createWebGpuBackend()` resolves `null`.
- [ ] **Step 2: Run it to verify it fails.** Run: `npm run test:gpu -- resources`. Expected: FAIL until the bench model file exists and leaks are fixed.
- [ ] **Step 3: Implement the benchmark model and timing loop.** Warm up for 20 steps, then time 200 steps. Print `{backend, stepsPerSec, paramCount, flopsPerStep, cuPerStep}` as one JSON line for both CPU and GPU. Fix any leak the resource test finds; the fix is in `disposeGraph` or the backend, not the test.
- [ ] **Step 4: Run and record.** Run: `npm run bench:gpu`. Expected: the JSON lines print, and the resource test passes. Write the results note, including machine, browser version, both JSON lines, and a verdict against the spec's target of **≥20 steps/sec on GPU**. If it falls short, note the width reduction that reaches the target, following the spec's order: widths first, resolution last.
- [ ] **Step 5: Commit.** `git commit -m "bench(tensor): learner-sized step throughput (spike 2) + GPU resource hygiene"`

---

## Self-review notes

- **Spec coverage:** §6 (own library, CPU/WebGPU, FLOP counting, CU, ≥20 steps/s target) is Tasks 1–6. §12 (GPU tests run locally) is Task 5. Everything else belongs to Plans 02–05 (see `README.md`).
- **The 1 CU = 10⁹ FLOPs conversion** is used in Task 6's output. The harness-level CU meter is Plan 03.
- **Types used across tasks** (`Backend`, `Storage`, `Tensor`, `ShapeError`, `ResourceError`, `disposeGraph`, `GruParams`) are each defined in exactly one task.
