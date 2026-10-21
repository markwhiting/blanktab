import { describe, expect, it } from "vitest"
import {
  backward, conv2d, createCpuBackend, createRng, disposeGraph, gradCheck, gruCell, initGru, layerNorm, leakyRelu, linear, mse, noGrad, param,
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

describe("leakyRelu", () => {
  it("is x for x > 0 and slope·x otherwise (default slope 0.01), propagating NaN and ±Infinity", async () => {
    const out = await leakyRelu(tensor([-2, -0.5, 0, 0.75, 3, Number.NaN, Infinity, -Infinity], [8])).read()
    expect(Array.from(out.slice(0, 5))).toEqual([Math.fround(-0.02), Math.fround(-0.005), 0, 0.75, 3])
    expect(Number.isNaN(out[5])).toBe(true)
    expect([out[6], out[7]]).toEqual([Infinity, -Infinity])
    expect(Array.from(await leakyRelu(tensor([-2, 2], [2]), 0.25).read())).toEqual([-0.5, 2])
  })
  it("has gradient 1 for x > 0 and slope for x < 0: never exactly zero (a unit cannot die)", async () => {
    const x = tensor([-3, -0.4, 0.4, 3], [4], { requiresGrad: true })
    const loss = sum(leakyRelu(x))
    backward(loss)
    expect(Array.from(await x.grad!.read())).toEqual([Math.fround(0.01), Math.fround(0.01), 1, 1])
    disposeGraph(loss, [])
  })
  it("gradCheck", () => {
    expect(gradCheck((x) => leakyRelu(x), [rand([3, 5])])).toBeLessThan(1e-3)
    expect(gradCheck((x) => leakyRelu(x, 0.3), [rand([3, 5])])).toBeLessThan(1e-3)
  })
  it("rejects a slope outside [0, 1]", () => {
    for (const s of [-0.1, 1.5, Number.NaN]) expect(() => leakyRelu(tensor([1], [1]), s)).toThrow(RangeError)
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

  it("leakyRelu on a constant input (inference: only its result survives)", () => {
    const w = p([5, 4]), x = c([3, 5])
    expect(leaked([w], () => sum(leakyRelu(matmul(x, w))))).toBe(0)
    const before = be.stats.liveBuffers
    const y = noGrad(() => leakyRelu(x))
    expect(be.stats.liveBuffers).toBe(before + 1)
    y.dispose()
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

  it("documented rule: a node-less tensor stashed outside composite() is freed; only the result survives", async () => {
    const x = c([2, 3]), w = c([3, 2])
    let stashed: Tensor | null = null
    const before = be.stats.liveBuffers
    const y = noGrad(() => composite(() => {
      stashed = relu(x)
      return matmul(stashed, w)
    }))
    expect(be.stats.liveBuffers).toBe(before + 1)
    await expect(stashed!.read()).rejects.toThrow()
    expect((await y.read()).length).toBe(4)
    y.dispose()
  })

  it("a composite never frees a view of the caller's input", async () => {
    const x = c([2, 6]), w = p([3, 2])
    const n = leaked([w], () => sum(composite(() => matmul(reshape(x, [4, 3]), w))))
    expect(n).toBe(0)
    expect((await x.read()).length).toBe(12)
    x.dispose()
  })
})

describe("a composite that throws part-way leaves nothing behind", () => {
  it("frees what it made (graph nodes and constants alike, never an outside tensor), and so does a nested one", async () => {
    const w = param(Float32Array.of(1, 2, 3, 4), [2, 2])
    const x = tensor(Float32Array.of(1, 1, 1, 1), [2, 2])
    const be = x.backend
    const live = be.stats.liveBuffers
    const boom = () => { throw new Error("device lost") }
    expect(() => composite(() => { const y = relu(matmul(x, w)); reshape(y, [4]); return boom() })).toThrow("device lost")
    expect(be.stats.liveBuffers).toBe(live)
    expect(() => noGrad(() => composite(() => { matmul(x, x); return composite(() => { relu(x); return boom() }) }))).toThrow("device lost")
    expect(be.stats.liveBuffers).toBe(live)
    expect(() => composite(() => { reshape(x, [4]); return boom() })).toThrow("device lost") // a view of x: x survives
    expect(be.stats.liveBuffers).toBe(live)
    expect([...await x.read()]).toEqual([1, 1, 1, 1])
  })
})
