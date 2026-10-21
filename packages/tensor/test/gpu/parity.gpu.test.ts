import { afterAll, describe, expect, it } from "vitest"
import {
  Adam, ResourceError, ShapeError, add, backward, conv2d, createCpuBackend, createRng, disposeGraph, getDefaultBackend,
  gruCell, initGru, layerNorm, matmul, mean, mse, randn, reshape, setDefaultBackend, square, sub, tensor,
} from "../../src/index"
import type { Backend, BinaryOp, Storage, Tensor, UnaryOp } from "../../src/index"
import { createWebGpuBackend, isSoftwareAdapter } from "../../src/webgpu"
import type { WebGpuBackend } from "../../src/webgpu"
import { describeBackendContract } from "../backend-contract"
import { createBenchModel } from "../../bench/learner-model"

// Runs only in real Chromium via `npm run test:gpu` (vitest.gpu.config.ts).

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" })
const gpuOrNull = await createWebGpuBackend({ powerPreference: "high-performance" })

const requireGpu = (): WebGpuBackend => {
  if (!gpuOrNull) throw new Error("WebGPU backend unavailable: no navigator.gpu or no adapter")
  return gpuOrNull
}

const TOL = 1e-4
/** Per-primitive worst error, max over cases of max_i |cpu - gpu| / max(1, |cpu|). Logged for the report. */
const worst: Record<string, number> = {}

/** max_i |c - g| / max(1, |c|); NaN or infinity must match exactly. */
function relErr(c: Float32Array, g: Float32Array): number {
  expect(g.length).toBe(c.length)
  let e = 0
  for (let i = 0; i < c.length; i++) {
    const a = c[i]!
    const b = g[i]!
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      if (!Object.is(a, b) && !(Number.isNaN(a) && Number.isNaN(b))) return Infinity
      continue
    }
    e = Math.max(e, Math.abs(a - b) / Math.max(1, Math.abs(a)))
  }
  return e
}

/**
 * Gradient error, max_i |c - g| / max(|c_i|, 1e-2 * max_j |c_j|): relative per element, with a floor
 * at 1% of the tensor's largest entry so near-zero entries are judged against the tensor's scale.
 * Unlike a check of weights after one Adam step (≈ lr·sign(g)), this catches a gradient that is
 * wrong by a constant factor. Non-finite values must match exactly.
 */
function gradErr(c: Float32Array, g: Float32Array): number {
  expect(g.length).toBe(c.length)
  let maxAbs = 0
  for (const v of c) maxAbs = Math.max(maxAbs, Math.abs(v))
  const floor = 1e-2 * maxAbs
  let e = 0
  for (let i = 0; i < c.length; i++) {
    const a = c[i]!
    const b = g[i]!
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      if (!Object.is(a, b) && !(Number.isNaN(a) && Number.isNaN(b))) return Infinity
      continue
    }
    const d = Math.abs(a - b)
    const den = Math.max(Math.abs(a), floor)
    if (den === 0) {
      if (d !== 0) return Infinity
      continue
    }
    e = Math.max(e, d / den)
  }
  return e
}
const GRAD_TOL = 1e-3

/** Reads every parameter's gradient (all must be populated). */
const readGrads = (params: readonly Tensor[]) =>
  Promise.all(params.map((p, i) => {
    if (!p.grad) throw new Error(`param ${i} [${p.shape.join(",")}] has no grad`)
    return p.grad.read()
  }))

const cpu = createCpuBackend()

/** Runs `op` on both backends with the same host inputs and checks parity; returns the error. */
async function parity(name: string, inputs: Float32Array[], op: (be: Backend, s: Storage[]) => Storage | Storage[]): Promise<number> {
  const gpu = requireGpu()
  const run = async (be: Backend) => {
    const s = inputs.map((x) => be.upload(x))
    const out = op(be, s)
    const outs = Array.isArray(out) ? out : [out]
    const res = await Promise.all(outs.map((o) => be.read(o)))
    for (const o of new Set([...s, ...outs])) be.dispose(o)
    return res
  }
  const [c, g] = [await run(cpu), await run(gpu)]
  let e = 0
  c.forEach((ci, i) => (e = Math.max(e, relErr(ci, g[i]!))))
  worst[name] = Math.max(worst[name] ?? 0, e)
  expect(e, name).toBeLessThanOrEqual(TOL)
  return e
}

const rng = createRng(20261020)
const rand = (n: number, scale = 1) => randn(rng, n).map((v) => v * scale)
const positive = (n: number) => rand(n).map((v) => Math.abs(v) + 0.1)

afterAll(() => {
  console.log("ADAPTER", JSON.stringify(adapter && { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize }))
  console.log("MAX_ERRORS", JSON.stringify(worst))
})

it("creates a WebGPU backend on a hardware adapter (opt out: GPU_ALLOW_SOFTWARE=1)", () => {
  expect(adapter, "navigator.gpu.requestAdapter() returned no adapter").toBeTruthy()
  const gpu = requireGpu()
  expect(gpu.name).toBe("webgpu")
  const info = gpu.adapterInfo
  console.log("BACKEND_ADAPTER", JSON.stringify(info))
  expect(typeof info.vendor).toBe("string")
  expect(typeof info.architecture).toBe("string")
  if ((import.meta as ImportMeta & { env: Record<string, string | undefined> }).env.GPU_ALLOW_SOFTWARE === "1") return
  // A software adapter (SwiftShader) hides GPU-compiler behaviour such as Metal fast math.
  expect(isSoftwareAdapter(info), `software WebGPU adapter ${JSON.stringify(info)}; set GPU_ALLOW_SOFTWARE=1 to allow`).toBe(false)
})

it("a GPU validation error rejects the read that observes it, and later reads recover", async () => {
  const gpu = requireGpu()
  const ok = gpu.upload(new Float32Array([1, 2]))
  const bad = gpu.upload(new Float32Array([3, 4]))
  // Simulate a corrupted storage: its buffer is destroyed behind the backend's back, so the
  // next submit using it is invalid.
  ;(bad as unknown as { buffer: GPUBuffer }).buffer.destroy()
  const out = gpu.unary("neg", bad)
  await expect(gpu.read(out)).rejects.toThrow(ResourceError)
  expect(Array.from(await gpu.read(gpu.unary("neg", ok)))).toEqual([-1, -2])
})

it("documented rule: a GPU error skips its whole batch, including in-place Adam updates recorded in it", async () => {
  const gpu = requireGpu()
  const p = gpu.upload(new Float32Array([1, 2]))
  await gpu.read(p) // starts a fresh batch
  const bad = gpu.upload(new Float32Array([3, 4]))
  ;(bad as unknown as { buffer: GPUBuffer }).buffer.destroy()
  gpu.adam(p, gpu.upload(new Float32Array([1, 1])), gpu.zeros(2), gpu.zeros(2), 0.5, 0.9, 0.999, 1e-8, 1) // same batch
  gpu.unary("neg", bad) // invalidates the batch's submit
  await expect(gpu.read(p)).rejects.toThrow(ResourceError)
  expect(Array.from(await gpu.read(p))).toEqual([1, 2]) // the Adam update never ran
})

describeBackendContract("webgpu", () => Promise.resolve(requireGpu()))

describe("webgpu parity with cpu", () => {
  const n = 37 * 65 + 3 // not a multiple of 256

  it("unary", async () => {
    const extremes = new Float32Array([-100, -20, -1e-6, 0, 1e-6, 20, 100])
    for (const op of ["neg", "relu", "step", "tanh", "sigmoid", "exp", "log", "recip", "sqrt", "square"] as UnaryOp[]) {
      const pos = op === "log" || op === "recip" || op === "sqrt"
      const x = pos ? positive(n) : op === "exp" ? rand(n, 3) : new Float32Array([...rand(n, 3), ...extremes])
      await parity(`unary:${op}`, [x], (be, [a]) => be.unary(op, a!))
    }
  })

  it("scalar (add/sub/mul bit-exact: both backends use the float32 constant)", async () => {
    for (const op of ["add", "sub", "mul", "div"] as BinaryOp[]) {
      for (const c of [0.37, -2.5, 0.1, -1 / 3]) {
        const e = await parity(`scalar:${op}`, [rand(n, 3)], (be, [a]) => be.scalar(op, a!, c))
        // One correctly rounded float32 operation; div is excluded (fast math may use a reciprocal).
        if (op !== "div") expect(e, `scalar:${op} ${c}`).toBe(0)
      }
    }
  })

  it("binary with broadcasting", async () => {
    const cases: [number[], number[], number[]][] = [
      [[37, 65], [37, 65], [37, 65]],
      [[37, 65], [65], [37, 65]],
      [[37, 1], [1, 65], [37, 65]],
      [[2, 3, 5, 7], [3, 1, 7], [2, 3, 5, 7]],
      [[5], [2, 1, 5], [2, 1, 5]],
      [[], [37, 3], [37, 3]],
    ]
    for (const op of ["add", "sub", "mul", "div"] as BinaryOp[]) {
      for (const [as, bs, os] of cases) {
        const size = (s: number[]) => s.reduce((p, d) => p * d, 1)
        const b = op === "div" ? positive(size(bs)) : rand(size(bs), 2)
        await parity(`binary:${op}`, [rand(size(as), 2), b], (be, [x, y]) => be.binary(op, x!, as, y!, bs, os))
      }
    }
  })

  it("binary above rank 4 throws ShapeError", () => {
    const gpu = requireGpu()
    const a = gpu.zeros(2)
    expect(() => gpu.binary("add", a, [1, 1, 1, 1, 2], a, [2], [1, 1, 1, 1, 2])).toThrow(ShapeError)
  })

  it("sumToShape", async () => {
    const cases: [number[], number[]][] = [
      [[37, 65], [65]], [[37, 65], [37, 1]], [[37, 65], [1, 1]], [[37, 65], []],
      [[2, 3, 5, 7], [3, 1, 7]], [[2, 3, 5, 7], [2, 1, 5, 1]], [[4000, 3], [3]],
    ]
    for (const [from, to] of cases) {
      await parity("sumToShape", [rand(from.reduce((p, d) => p * d, 1), 3)], (be, [a]) => be.sumToShape(a!, from, to))
    }
  })

  it("matmul m=37 k=129 n=65, all transpose flags", async () => {
    const [m, k, nn] = [37, 129, 65]
    for (const ta of [false, true]) {
      for (const tb of [false, true]) {
        await parity("matmul", [rand(m * k), rand(k * nn)], (be, [a, b]) => be.matmul(a!, b!, m, k, nn, ta, tb))
      }
    }
    // degenerate sizes
    await parity("matmul", [rand(3 * 1), rand(1 * 5)], (be, [a, b]) => be.matmul(a!, b!, 3, 1, 5, false, false))
    await parity("matmul", [rand(0), rand(0)], (be, [a, b]) => be.matmul(a!, b!, 3, 0, 5, false, false))
  })

  it("reduceLast sum and max", async () => {
    for (const op of ["sum", "max"] as const) {
      await parity(`reduceLast:${op}`, [rand(37 * 129, 3)], (be, [a]) => be.reduceLast(op, a!, 37, 129))
      await parity(`reduceLast:${op}`, [rand(20000, 3)], (be, [a]) => be.reduceLast(op, a!, 1, 20000))
      await parity(`reduceLast:${op}`, [rand(1000 * 3, 3)], (be, [a]) => be.reduceLast(op, a!, 1000, 3))
    }
  })

  it("im2col / col2im for the periphery encoder's first layer (C=3, H=50, W=80, k=5, stride=2, pad=2)", async () => {
    const [c, h, w, k, s, p] = [3, 50, 80, 5, 2, 2]
    const oh = Math.floor((h + 2 * p - k) / s) + 1
    const ow = Math.floor((w + 2 * p - k) / s) + 1
    await parity("im2col", [rand(c * h * w)], (be, [x]) => be.im2col(x!, c, h, w, k, k, s, p))
    await parity("col2im", [rand(c * k * k * oh * ow)], (be, [x]) => be.col2im(x!, c, h, w, k, k, s, p))
    await parity("im2col", [rand(2 * 7 * 9)], (be, [x]) => be.im2col(x!, 2, 7, 9, 3, 2, 1, 1))
    await parity("col2im", [rand(2 * 3 * 2 * 7 * 10)], (be, [x]) => be.col2im(x!, 2, 7, 9, 3, 2, 1, 1))
  })

  it("gatherRows / scatterAddRows with repeated indices", async () => {
    const [rows, cols] = [37, 65]
    const idx = Int32Array.from({ length: 100 }, () => Math.floor(rng() * rows))
    await parity("gatherRows", [rand(rows * cols)], (be, [a]) => be.gatherRows(a!, rows, cols, idx))
    await parity("scatterAddRows", [rand(idx.length * cols)], (be, [g]) => be.scatterAddRows(g!, rows, cols, idx))
    await parity("gatherRows", [rand(rows * cols)], (be, [a]) => be.gatherRows(a!, rows, cols, new Int32Array(0)))
  })

  it("adam over three steps", async () => {
    await parity("adam", [rand(n), rand(n), rand(n), rand(n), new Float32Array(n), new Float32Array(n)], (be, [p, g1, g2, g3, m, v]) => {
      be.adam(p!, g1!, m!, v!, 0.01, 0.9, 0.999, 1e-8, 1)
      be.adam(p!, g2!, m!, v!, 0.01, 0.9, 0.999, 1e-8, 2)
      be.adam(p!, g3!, m!, v!, 0.01, 0.9, 0.999, 1e-8, 3)
      return [p!, m!, v!]
    })
  })

  it("adam with decoupled decay over three steps", async () => {
    await parity("adam", [rand(n), rand(n), rand(n), rand(n), new Float32Array(n), new Float32Array(n)], (be, [p, g1, g2, g3, m, v]) => {
      be.adam(p!, g1!, m!, v!, 0.01, 0.9, 0.999, 1e-8, 1, 1e-3)
      be.adam(p!, g2!, m!, v!, 0.01, 0.9, 0.999, 1e-8, 2, 1e-3)
      be.adam(p!, g3!, m!, v!, 0.01, 0.9, 0.999, 1e-8, 3, 1e-3)
      return [p!, m!, v!]
    })
  })
})

/** Runs `fn` with `be` as the default backend (so param()/initGru() allocate there). */
async function onBackend<T>(be: Backend, fn: () => Promise<T>): Promise<T> {
  const prev = getDefaultBackend()
  setDefaultBackend(be)
  try {
    return await fn()
  } finally {
    setDefaultBackend(prev)
  }
}

describe("webgpu training", () => {
  it("the Task 3 regression (y = 3x - 2, Adam lr 0.05) reaches MSE < 1e-3 on GPU without leaking buffers", async () => {
    const gpu = requireGpu()
    await onBackend(gpu, async () => {
      const r = createRng(42)
      const xs = new Float32Array(64)
      const ys = new Float32Array(64)
      for (let i = 0; i < 64; i++) {
        xs[i] = r() * 2 - 1
        ys[i] = 3 * xs[i]! - 2
      }
      const X = tensor(xs, [64, 1])
      const Y = tensor(ys, [64, 1])
      const w = tensor(new Float32Array([0]), [1, 1], { requiresGrad: true })
      const b = tensor(new Float32Array([0]), [1], { requiresGrad: true })
      const opt = new Adam([w, b], { lr: 0.05 })
      const live = gpu.stats.liveBuffers
      let loss = Infinity
      for (let step = 0; step < 500 && loss >= 1e-3; step++) {
        opt.zeroGrad()
        const l = mean(square(sub(add(matmul(X, w), b), Y)))
        backward(l)
        loss = (await l.read())[0]!
        opt.step()
        disposeGraph(l, [])
      }
      opt.zeroGrad()
      expect(loss).toBeLessThan(1e-3)
      expect(gpu.stats.liveBuffers).toBe(live)
      expect((await w.read())[0]).toBeCloseTo(3, 1)
      expect((await b.read())[0]).toBeCloseTo(-2, 1)
    })
  })

  it("conv + layerNorm + GRU training step: identical flops and liveBuffers deltas, matching loss and every parameter gradient", async () => {
    const gpu = requireGpu()
    const step = (be: Backend) =>
      onBackend(be, async () => {
        const r = createRng(7)
        const u = (shape: number[], s = 0.3) => tensor(randn(r, shape.reduce((p, d) => p * d, 1)).map((v) => v * s), shape, { requiresGrad: true })
        const x = tensor(randn(r, 3 * 10 * 16), [3, 10, 16])
        const cw = u([4, 3 * 5 * 5])
        const cb = u([4, 1])
        const gamma = u([32])
        const beta = u([32])
        const gru = initGru(r, 32, 8)
        const params = [cw, cb, gamma, beta, ...Object.values(gru)]
        const opt = new Adam(params, { lr: 0.01 })
        const f0 = be.stats.flops
        const l0 = be.stats.liveBuffers
        const y = conv2d(x, cw, cb, { kh: 5, kw: 5, stride: 2, pad: 2 }) // [4,5,8]
        // Rows of 32 straddle the 40-element channels: with rows inside one channel, layerNorm would
        // cancel the conv bias exactly (true gradient 0, so only roundoff would be compared).
        const seq = layerNorm(reshape(y, [5, 32]), gamma, beta)
        // Non-zero h0: with h0 = 0 the recurrent weights and the reset gate get exactly-zero gradients.
        const h = gruCell(seq, tensor(randn(r, 5 * 8).map((v) => 0.5 * v), [5, 8]), gru)
        const loss = mse(h, tensor(new Float32Array(5 * 8).fill(0.5), [5, 8]))
        backward(loss)
        const grads = await readGrads(params)
        opt.step()
        const value = (await loss.read())[0]!
        disposeGraph(loss, [])
        opt.zeroGrad()
        const out = { flops: be.stats.flops - f0, live: be.stats.liveBuffers - l0, value, grads }
        opt.dispose()
        return out
      })
    const c = await step(cpu)
    const g = await step(gpu)
    expect(c.flops).toBeGreaterThan(0)
    expect(g.flops).toBe(c.flops)
    expect(g.live).toBe(c.live)
    expect(relErr(new Float32Array([c.value]), new Float32Array([g.value]))).toBeLessThanOrEqual(TOL)
    c.grads.forEach((cg, i) => {
      expect(cg.some((v) => v !== 0), `param ${i} has an all-zero gradient`).toBe(true)
      expect(gradErr(cg, g.grads[i]!), `grad of param ${i}`).toBeLessThanOrEqual(GRAD_TOL)
    })
  })

  it("learner-sized step (bench model): every parameter gradient matches CPU, and losses agree over 10 steps", async () => {
    // One learner-sized backward records hundreds of passes, so it crosses the 64-pass uniform-ring
    // auto-flush many times before the gradients are read back.
    const gpu = requireGpu()
    const STEPS = 10
    const run = async (be: Backend) => {
      const model = createBenchModel(be)
      const live = be.stats.liveBuffers
      const grads: Float32Array[][] = []
      const losses: number[] = []
      for (let i = 0; i < STEPS; i++) {
        const loss = await model.step(true, async (params) => void grads.push(await readGrads(params)))
        losses.push(loss!)
      }
      expect(be.stats.liveBuffers, `${be.name} liveBuffers`).toBe(live)
      const shapes = model.params.map((p) => p.shape)
      model.dispose()
      return { grads, losses, shapes }
    }
    const c = await run(cpu)
    const g = await run(gpu)
    const errs = c.grads.map((step, k) => step.map((cg, i) => gradErr(cg, g.grads[k]![i]!)))
    console.log("LEARNER_GRAD_ERR (worst per step)", JSON.stringify(errs.map((e) => Number(Math.max(...e).toExponential(2)))))
    console.log("LEARNER_LOSSES", JSON.stringify({ cpu: c.losses, gpu: g.losses }))
    // Steps 1-2 are compared: step 1 starts from identical weights; step 2 is the first with h != 0,
    // so it covers the recurrent weights and the reset gate, whose gradients are exactly 0 when h = 0.
    // Later steps are not: Adam (≈ lr·sign(g) early on) amplifies float32 roundoff in near-zero
    // gradients into weight differences of order lr, so CPU and GPU legitimately drift apart.
    for (const k of [0, 1]) {
      expect(c.grads[k]!.length).toBe(c.shapes.length)
      c.grads[k]!.forEach((_, i) => expect(errs[k]![i]!, `step ${k + 1} grad of param ${i} [${c.shapes[i]!.join(",")}]`).toBeLessThanOrEqual(GRAD_TOL))
    }
    c.shapes.forEach((shape, i) => {
      const nonZero = [0, 1].some((k) => c.grads[k]![i]!.some((v) => v !== 0))
      expect(nonZero, `param ${i} [${shape.join(",")}] has an all-zero gradient in steps 1-2 (comparison would be vacuous)`).toBe(true)
    })
    // Losses over 10 steps (Adam updates in between): agree to ~4 significant digits.
    c.losses.forEach((cl, i) => expect(g.losses[i]!, `loss at step ${i + 1}`).toBeCloseTo(cl, 4))
  })

  it("allocating beyond maxStorageBufferBindingSize throws ResourceError naming the size and the limit", () => {
    const gpu = requireGpu()
    const limit = adapter!.limits.maxStorageBufferBindingSize
    const len = Math.floor(limit / 4) + 1
    expect(() => gpu.zeros(len)).toThrow(ResourceError)
    expect(() => gpu.zeros(len)).toThrow(new RegExp(`${len * 4}.*${limit}`))
  })
})
