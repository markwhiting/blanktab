import { describe, expect, it } from "vitest"
import {
  backward, conv2d, createCpuBackend, createRng, disposeGraph, gradCheck, gruCell, initGru, layerNorm, linear, mse, noGrad, param,
  composite, matmul, relu, reshape, sum, tensor,
} from "../src/index"
import type { Tensor } from "../src/index"


const rng = createRng(77)
/** Values in ±[0.2, 1.2]. */
const rand = (shape: number[]) => {
  const n = shape.reduce((p, d) => p * d, 1)
  return { data: Float32Array.from({ length: n }, () => (rng() < 0.5 ? -1 : 1) * (0.2 + rng())), shape }
}

describe("conv2d", () => {
  it("matches a naive convolution, shape [4,5,4]", async () => {
    const [C, H, W, O, k, stride, pad] = [3, 10, 8, 4, 3, 2, 1] as const
    const x = rand([C, H, W]), w = rand([O, C * k * k]), b = rand([O, 1])
    const out = conv2d(tensor(x.data, x.shape), tensor(w.data, w.shape), tensor(b.data, b.shape), { kh: k, kw: k, stride, pad })
    expect(out.shape).toEqual([4, 5, 4])
    const got = await out.read()
    const oh = Math.floor((H + 2 * pad - k) / stride) + 1, ow = Math.floor((W + 2 * pad - k) / stride) + 1
    let maxDiff = 0
    for (let o = 0; o < O; o++)
      for (let i = 0; i < oh; i++)
        for (let j = 0; j < ow; j++) {
          let acc = b.data[o]!
          for (let c = 0; c < C; c++)
            for (let p = 0; p < k; p++)
              for (let q = 0; q < k; q++) {
                const yy = i * stride - pad + p, xx = j * stride - pad + q
                if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue
                acc += w.data[o * C * k * k + (c * k + p) * k + q]! * x.data[(c * H + yy) * W + xx]!
              }
          maxDiff = Math.max(maxDiff, Math.abs(acc - got[(o * oh + i) * ow + j]!))
        }
    expect(maxDiff).toBeLessThan(1e-5)
  })

  it("gradCheck", () => {
    const opts = { kh: 3, kw: 3, stride: 2, pad: 1 }
    const err = gradCheck((x, w, b) => conv2d(x, w, b, opts), [rand([2, 5, 4]), rand([3, 18]), rand([3, 1])])
    expect(err).toBeLessThan(1e-3)
  })
})

describe("linear", () => {
  it("computes xW + b and passes gradCheck", async () => {
    const out = linear(tensor([1, 2], [1, 2]), tensor([1, 0, 0, 1, 1, 1], [2, 3]), tensor([10, 20, 30], [3]))
    expect(Array.from(await out.read())).toEqual([13, 22, 32])
    expect(gradCheck((x, w, b) => linear(x, w, b), [rand([2, 3]), rand([3, 4]), rand([4])])).toBeLessThan(1e-3)
  })
})

describe("gruCell", () => {
  it("gradCheck over x, h and all parameters", () => {
    const p = initGru(createRng(5), 5, 4)
    const names = Object.keys(p) as (keyof typeof p)[]
    const shapes = names.map((n) => [...p[n].shape])
    const err = gradCheck(
      (x, h, ...ps) => gruCell(x, h, Object.fromEntries(names.map((n, i) => [n, ps[i]!])) as unknown as typeof p),
      [rand([2, 5]), rand([2, 4]), ...shapes.map(rand)],
      5e-2, // smooth deep graph: float32 forward noise dominates at the default eps (~1e-3); Richardson absorbs the larger step
    )
    expect(err).toBeLessThan(1e-3)
  })

  it("with z forced to 0 returns h unchanged", async () => {
    const p = initGru(createRng(6), 5, 4)
    p.bz.dispose()
    p.bz = param(new Float32Array(4).fill(-50), [4])
    const h = rand([2, 4])
    const out = noGrad(() => gruCell(tensor(rand([2, 5]).data, [2, 5]), tensor(h.data, h.shape), p))
    const got = await out.read()
    for (let i = 0; i < got.length; i++) expect(Math.abs(got[i]! - h.data[i]!)).toBeLessThan(1e-4)
  })

  it("initGru uses uniform +-1/sqrt(hid)", async () => {
    const p = initGru(createRng(7), 5, 4)
    expect(p.wz.shape).toEqual([5, 4])
    expect(p.uz.shape).toEqual([4, 4])
    expect(p.bz.shape).toEqual([4])
    expect(p.wz.requiresGrad).toBe(true)
    for (const v of await p.wz.read()) expect(Math.abs(v)).toBeLessThanOrEqual(0.5)
  })
})

describe("layerNorm", () => {
  it("normalizes the last axis", async () => {
    const out = layerNorm(tensor([1, 2, 3, 4], [1, 4]), tensor([1, 1, 1, 1], [4]), tensor([0, 0, 0, 0], [4]))
    const v = await out.read()
    expect(v.reduce((a, c) => a + c, 0)).toBeCloseTo(0, 5)
    expect(v.reduce((a, c) => a + c * c, 0) / 4).toBeCloseTo(1, 3)
  })
  it("gradCheck", () => {
    expect(gradCheck((x, g, b) => layerNorm(x, g, b), [rand([3, 4]), rand([4]), rand([4])])).toBeLessThan(1e-3)
  })
})

describe("mse", () => {
  it("is the mean squared difference", async () => {
    const out = mse(tensor([1, 2, 3, 4], [2, 2]), tensor([1, 1, 1, 1], [2, 2]))
    expect(out.shape).toEqual([])
    expect((await out.read())[0]).toBeCloseTo((0 + 1 + 4 + 9) / 4, 6)
  })
  it("gradCheck", () => {
    expect(gradCheck((p, t) => mse(p, t), [rand([2, 3]), rand([2, 3])])).toBeLessThan(1e-3)
  })
})

describe("composites leave no unreachable intermediates (non-grad inputs)", () => {
  // A composite applied to a constant (e.g. conv2d on an input frame) builds internal tensors that
  // have no graph node, such as im2col of the frame. They are invisible to the caller, so
  // disposeGraph must free them; they must not be treated as caller-owned leaves.
  const be = createCpuBackend()
  const r = createRng(3)
  const p = (shape: number[]) => tensor(Float32Array.from({ length: shape.reduce((a, d) => a * d, 1) }, () => r() - 0.5), shape, { requiresGrad: true, backend: be })
  const c = (shape: number[]) => tensor(Float32Array.from({ length: shape.reduce((a, d) => a * d, 1) }, () => r() - 0.5), shape, { backend: be })

  /** Live-buffer delta of build → backward → disposeGraph → clear grads; params and inputs pre-exist. */
  function leaked(params: Tensor[], build: () => Tensor): number {
    const before = be.stats.liveBuffers
    const loss = build()
    backward(loss)
    disposeGraph(loss, [])
    for (const t of params) {
      t.grad?.dispose()
      t.grad = null
    }
    return be.stats.liveBuffers - before
  }

  it("conv2d on a constant input", () => {
    const w = p([4, 27]), b = p([4, 1]), x = c([3, 8, 8])
    expect(leaked([w, b], () => sum(relu(conv2d(x, w, b, { kh: 3, kw: 3, stride: 2, pad: 1 }))))).toBe(0)
  })

  it("layerNorm on a constant input", () => {
    const g = p([5]), b = p([5]), x = c([3, 5])
    expect(leaked([g, b], () => sum(layerNorm(x, g, b)))).toBe(0)
  })

  it("a fully constant composite keeps only its result", () => {
    const g = c([5]), b = c([5]), x = c([3, 5])
    const before = be.stats.liveBuffers
    const y = layerNorm(x, g, b)
    expect(be.stats.liveBuffers).toBe(before + 1)
    y.dispose()
  })

  it("a constant computed by a composite outside the step survives disposeGraph and can be reused", async () => {
    const w = c([4, 27]), x = c([3, 8, 8])
    const feat = conv2d(x, w, undefined, { kh: 3, kw: 3, stride: 2, pad: 1 }) // caller-owned constant
    const v = p([4, 4, 4])
    for (let i = 0; i < 2; i++) {
      const loss = sum(mse(feat, v))
      backward(loss)
      disposeGraph(loss, [])
      v.grad?.dispose()
      v.grad = null
    }
    expect((await feat.read()).length).toBe(64)
    feat.dispose()
  })

  it("a composite never frees a view of the caller's input", async () => {
    const x = c([2, 6]), w = p([3, 2])
    const n = leaked([w], () => sum(composite(() => matmul(reshape(x, [4, 3]), w))))
    expect(n).toBe(0)
    expect((await x.read()).length).toBe(12)
    x.dispose()
  })
})
