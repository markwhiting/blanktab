import { describe, expect, it } from "vitest"
import {
  ShapeError, Tensor, add, addScalar, div, exp, gatherRows, getDefaultBackend, log, logSoftmax, matmul, mean, mul,
  neg, param, relu, reshape, scale, sigmoid, softmax, sqrt, square, sub, sum, sumLast, tanh, tensor, zeros, noGrad,
} from "../src/index"

const close = (got: Float32Array, want: number[], tol = 1e-5) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => expect(Math.abs(got[i]! - w)).toBeLessThan(tol))
}
const t = (d: number[], s: number[]) => tensor(d, s)

describe("tensor construction", () => {
  it("has shape, size, storage, backend and defaults", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3])
    expect(x.shape).toEqual([2, 3])
    expect(x.size).toBe(6)
    expect(x.backend).toBe(getDefaultBackend())
    expect(x.requiresGrad).toBe(false)
    expect(x.grad).toBeNull()
    close(await x.read(), [1, 2, 3, 4, 5, 6])
    expect(zeros([2]).size).toBe(2)
    expect(tensor([7], []).size).toBe(1)
    expect(param(new Float32Array([1, 2]), [2]).requiresGrad).toBe(true)
  })
  it("rejects data/shape mismatch", () => {
    expect(() => t([1, 2, 3], [2, 2])).toThrow(ShapeError)
  })
  it("dispose releases storage", () => {
    const be = getDefaultBackend()
    const before = be.stats.liveBuffers
    const x = t([1], [1])
    expect(be.stats.liveBuffers).toBe(before + 1)
    x.dispose()
    expect(be.stats.liveBuffers).toBe(before)
  })
})

describe("elementwise forward", () => {
  it("binary ops broadcast", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    const b = t([10, 20, 30], [3])
    close(await add(a, b).read(), [11, 22, 33, 14, 25, 36])
    close(await sub(a, b).read(), [-9, -18, -27, -6, -15, -24])
    close(await mul(a, b).read(), [10, 40, 90, 40, 100, 180])
    close(await div(b, t([2, 4, 5], [3])).read(), [5, 5, 6])
    expect(add(a, b).shape).toEqual([2, 3])
  })
  it("scalar and unary ops", async () => {
    const a = t([-1, 0, 2, 4], [4])
    close(await scale(a, 3).read(), [-3, 0, 6, 12])
    close(await addScalar(a, 1).read(), [0, 1, 3, 5])
    close(await neg(a).read(), [1, 0, -2, -4])
    close(await relu(a).read(), [0, 0, 2, 4])
    close(await tanh(a).read(), [-0.7615942, 0, 0.9640276, 0.9993293])
    close(await sigmoid(a).read(), [0.2689414, 0.5, 0.8807971, 0.9820138])
    close(await exp(a).read(), [0.3678794, 1, 7.3890561, 54.59815], 1e-4)
    close(await log(t([1, Math.E, 10], [3])).read(), [0, 1, 2.3025851])
    close(await sqrt(t([0, 4, 9], [3])).read(), [0, 2, 3])
    close(await square(a).read(), [1, 0, 4, 16])
  })
  it("add of [2,3] and [4] throws ShapeError naming both shapes", () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    const b = t([1, 2, 3, 4], [4])
    expect(() => add(a, b)).toThrow(ShapeError)
    expect(() => add(a, b)).toThrow(/\[2,3\].*\[4\]/)
  })
})

describe("matmul and reductions", () => {
  it("matmul forward", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    const b = t([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4])
    const c = matmul(a, b)
    expect(c.shape).toEqual([2, 4])
    close(await c.read(), [38, 44, 50, 56, 83, 98, 113, 128])
  })
  it("matmul [2,3]·[2,3] throws ShapeError", () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    expect(() => matmul(a, a)).toThrow(ShapeError)
    expect(() => matmul(t([1, 2], [2]), a)).toThrow(ShapeError)
  })
  it("sum, mean, sumLast", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    expect(sum(a).shape).toEqual([])
    close(await sum(a).read(), [21])
    close(await mean(a).read(), [3.5])
    const s = sumLast(a)
    expect(s.shape).toEqual([2])
    close(await s.read(), [6, 15])
    expect(() => sumLast(t([1], []))).toThrow(ShapeError)
  })
  it("sumLast over an empty last axis gives zeros over the leading dims", async () => {
    const s = sumLast(zeros([3, 0]))
    expect(s.shape).toEqual([3])
    close(await s.read(), [0, 0, 0])
  })
})

describe("softmax", () => {
  it("is finite and correct at +-1000", async () => {
    const x = t([1000, 0, -1000], [3])
    const p = await softmax(x).read()
    const lp = await logSoftmax(x).read()
    for (const v of [...p, ...lp]) expect(Number.isFinite(v)).toBe(true)
    close(p, [1, 0, 0])
    close(lp, [0, -1000, -2000])
  })
  it("rows sum to 1 and logSoftmax = log(softmax)", async () => {
    const x = t([1, 2, 3, 0, 0, 0], [2, 3])
    const p = await softmax(x).read()
    close(p, [0.0900306, 0.2447285, 0.6652410, 1 / 3, 1 / 3, 1 / 3])
    close(await logSoftmax(x).read(), Array.from(p, Math.log), 1e-5)
  })
  it("does not leak intermediate buffers", () => {
    const be = getDefaultBackend()
    const x = t([1, 2, 3], [3])
    const before = be.stats.liveBuffers
    const out = [softmax(x), logSoftmax(x)]
    expect(be.stats.liveBuffers).toBe(before + 2)
    out.forEach((o) => o.dispose())
  })
})

describe("reshape and gatherRows", () => {
  it("reshape [2,3] -> [3,-1] gives [3,2] and shares storage", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    const r = reshape(a, [3, -1])
    expect(r.shape).toEqual([3, 2])
    expect(r.storage).toBe(a.storage)
    close(await r.read(), [1, 2, 3, 4, 5, 6])
  })
  it("disposing a reshaped view releases the shared storage exactly once", async () => {
    const be = getDefaultBackend()
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    const r = reshape(a, [6])
    const before = be.stats.liveBuffers
    r.dispose()
    expect(be.stats.liveBuffers).toBe(before - 1)
    await expect(a.read()).rejects.toThrow()
    a.dispose()
    expect(be.stats.liveBuffers).toBe(before - 1)
  })
  it("reshape rejects bad shapes", () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3])
    expect(() => reshape(a, [4, -1])).toThrow(ShapeError)
    expect(() => reshape(a, [-1, -1])).toThrow(ShapeError)
    expect(() => reshape(a, [5])).toThrow(ShapeError)
  })
  it("gatherRows", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [3, 2])
    const g = gatherRows(a, new Int32Array([2, 0, 2]))
    expect(g.shape).toEqual([3, 2])
    close(await g.read(), [5, 6, 1, 2, 5, 6])
    expect(() => gatherRows(a, new Int32Array([3]))).toThrow(ShapeError)
    expect(() => gatherRows(t([1], [1]), new Int32Array([0]))).toThrow(ShapeError)
  })
})

describe("graph recording", () => {
  it("records no parents when nothing requires grad", () => {
    const a = t([1, 2], [2])
    const b = t([3, 4], [2])
    for (const o of [add(a, b), relu(a), matmul(reshape(a, [1, 2]), reshape(b, [2, 1])), softmax(a), sum(a)]) {
      expect(o._node).toBeUndefined()
      expect(o.requiresGrad).toBe(false)
    }
  })
  it("records parents when an input requires grad", () => {
    const a = param(new Float32Array([1, 2]), [2])
    const b = t([3, 4], [2])
    const o = mul(a, b)
    expect(o.requiresGrad).toBe(true)
    expect(o._node!.parents).toEqual([a, b])
    expect(typeof o._node!.backward).toBe("function")
    expect(o).toBeInstanceOf(Tensor)
  })
  it("records nothing under noGrad", () => {
    const a = param(new Float32Array([1, 2]), [2])
    const o = noGrad(() => relu(a))
    expect(o._node).toBeUndefined()
    expect(o.requiresGrad).toBe(false)
    expect(relu(a)._node).toBeDefined()
  })
})
