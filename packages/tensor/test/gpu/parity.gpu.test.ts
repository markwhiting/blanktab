import { afterAll, describe, expect, it } from "vitest"
import {
  Adam, ResourceError, ShapeError, add, backward, conv2d, createCpuBackend, createRng, disposeGraph, getDefaultBackend,
  gruCell, initGru, layerNorm, matmul, mean, mse, randn, reshape, setDefaultBackend, square, sub, tensor,
} from "../../src/index"
import type { Backend, BinaryOp, Storage, UnaryOp } from "../../src/index"
import { createWebGpuBackend } from "../../src/webgpu"
import { describeBackendContract } from "../backend-contract"

// Runs only in real Chromium via `npm run test:gpu` (vitest.gpu.config.ts).

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" })
const gpuOrNull = await createWebGpuBackend({ powerPreference: "high-performance" })

const requireGpu = (): Backend => {
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

it("creates a WebGPU backend on this machine", () => {
  expect(adapter, "navigator.gpu.requestAdapter() returned no adapter").toBeTruthy()
  expect(requireGpu().name).toBe("webgpu")
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

  it("scalar", async () => {
    for (const op of ["add", "sub", "mul", "div"] as BinaryOp[]) {
      for (const c of [0.37, -2.5]) await parity(`scalar:${op}`, [rand(n, 3)], (be, [a]) => be.scalar(op, a!, c))
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

  it("stats.flops and liveBuffers deltas are identical between backends for a conv + layerNorm + GRU training step", async () => {
    const gpu = requireGpu()
    const step = (be: Backend) =>
      onBackend(be, async () => {
        const r = createRng(7)
        const u = (shape: number[], s = 0.3) => tensor(randn(r, shape.reduce((p, d) => p * d, 1)).map((v) => v * s), shape, { requiresGrad: true })
        const x = tensor(randn(r, 3 * 10 * 16), [3, 10, 16])
        const cw = u([4, 3 * 5 * 5])
        const cb = u([4, 1])
        const gamma = u([20])
        const beta = u([20])
        const gru = initGru(r, 20, 8)
        const params = [cw, cb, gamma, beta, ...Object.values(gru)]
        const opt = new Adam(params, { lr: 0.01 })
        const f0 = be.stats.flops
        const l0 = be.stats.liveBuffers
        const y = conv2d(x, cw, cb, { kh: 5, kw: 5, stride: 2, pad: 2 }) // [4,5,8]
        const seq = layerNorm(reshape(y, [8, 20]), gamma, beta)
        const h = gruCell(seq, tensor(new Float32Array(8 * 8), [8, 8]), gru)
        const loss = mse(h, tensor(new Float32Array(8 * 8).fill(0.5), [8, 8]))
        backward(loss)
        opt.step()
        const value = (await loss.read())[0]!
        disposeGraph(loss, [])
        opt.zeroGrad()
        const out = { flops: be.stats.flops - f0, live: be.stats.liveBuffers - l0, value, cw: await cw.read() }
        opt.dispose()
        return out
      })
    const c = await step(cpu)
    const g = await step(gpu)
    expect(c.flops).toBeGreaterThan(0)
    expect(g.flops).toBe(c.flops)
    expect(g.live).toBe(c.live)
    expect(relErr(new Float32Array([c.value]), new Float32Array([g.value]))).toBeLessThanOrEqual(TOL)
    expect(relErr(c.cw, g.cw)).toBeLessThanOrEqual(TOL)
  })

  it("allocating beyond maxStorageBufferBindingSize throws ResourceError naming the size and the limit", () => {
    const gpu = requireGpu()
    const limit = adapter!.limits.maxStorageBufferBindingSize
    const len = Math.floor(limit / 4) + 1
    expect(() => gpu.zeros(len)).toThrow(ResourceError)
    expect(() => gpu.zeros(len)).toThrow(new RegExp(`${len * 4}.*${limit}`))
  })
})
