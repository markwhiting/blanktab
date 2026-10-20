import { describe, expect, it } from "vitest"
import { ShapeError, broadcastShape } from "../src/backend"
import { createCpuBackend } from "../src/cpu"
import { createRng, randn } from "../src/random"

const f = (...x: number[]) => new Float32Array(x)
const close = (got: Float32Array, want: number[], tol = 1e-5) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => expect(Math.abs(got[i]! - w)).toBeLessThan(tol))
}
const dot = (a: Float32Array, b: Float32Array) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!
  return s
}

describe("cpu backend", () => {
  it("matmul handles all transpose combinations and counts flops", async () => {
    const be = createCpuBackend()
    // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2)
    const want = [58, 64, 139, 154]
    const A = [1, 2, 3, 4, 5, 6]
    const At = [1, 4, 2, 5, 3, 6]
    const B = [7, 8, 9, 10, 11, 12]
    const Bt = [7, 9, 11, 8, 10, 12]
    for (const [ta, tb] of [[false, false], [true, false], [false, true], [true, true]] as const) {
      const a = be.upload(new Float32Array(ta ? At : A))
      const b = be.upload(new Float32Array(tb ? Bt : B))
      const before = be.stats.flops
      const c = be.matmul(a, b, 2, 3, 2, ta, tb)
      expect(be.stats.flops - before).toBe(24)
      close(await be.read(c), want)
    }
  })

  it("binary add broadcasts [2,3]+[3]", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(1, 2, 3, 4, 5, 6))
    const b = be.upload(f(10, 20, 30))
    const before = be.stats.flops
    const o = be.binary("add", a, [2, 3], b, [3], [2, 3])
    expect(be.stats.flops - before).toBe(6)
    close(await be.read(o), [11, 22, 33, 14, 25, 36])
  })

  it("binary ops and unary/scalar are correct", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(1, 2, 3, 4))
    const col = be.upload(f(10, 20))
    close(await be.read(be.binary("mul", a, [2, 2], col, [2, 1], [2, 2])), [10, 20, 60, 80])
    close(await be.read(be.binary("sub", a, [2, 2], a, [2, 2], [2, 2])), [0, 0, 0, 0])
    close(await be.read(be.scalar("div", a, 2)), [0.5, 1, 1.5, 2])
    close(await be.read(be.scalar("sub", a, 1)), [0, 1, 2, 3])
    const x = be.upload(f(-1, 0, 2))
    close(await be.read(be.unary("relu", x)), [0, 0, 2])
    close(await be.read(be.unary("step", x)), [0, 0, 1])
    close(await be.read(be.unary("neg", x)), [1, -0, -2])
    close(await be.read(be.unary("square", x)), [1, 0, 4])
    close(await be.read(be.unary("sigmoid", x)), [1 / (1 + Math.E), 0.5, 1 / (1 + Math.exp(-2))])
    close(await be.read(be.unary("tanh", x)), [Math.tanh(-1), 0, Math.tanh(2)])
    close(await be.read(be.unary("exp", x)), [Math.exp(-1), 1, Math.exp(2)])
    close(await be.read(be.unary("recip", be.upload(f(2, 4)))), [0.5, 0.25])
    close(await be.read(be.unary("sqrt", be.upload(f(4, 9)))), [2, 3])
    close(await be.read(be.unary("log", be.upload(f(1, Math.E)))), [0, 1])
  })

  it("broadcastShape throws ShapeError naming both shapes", () => {
    expect(() => broadcastShape([2, 3], [4])).toThrow(ShapeError)
    try {
      broadcastShape([2, 3], [4])
    } catch (e) {
      expect((e as Error).message).toContain("[2,3]")
      expect((e as Error).message).toContain("[4]")
    }
    expect(broadcastShape([2, 3], [3])).toEqual([2, 3])
    expect(broadcastShape([2, 1], [1, 3])).toEqual([2, 3])
  })

  it("sumToShape inverts broadcast", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(1, 2, 3, 4, 5, 6))
    const before = be.stats.flops
    close(await be.read(be.sumToShape(a, [2, 3], [3])), [5, 7, 9])
    expect(be.stats.flops - before).toBe(6)
    close(await be.read(be.sumToShape(a, [2, 3], [2, 1])), [6, 15])
  })

  it("matmul with m != n for all transpose combinations", async () => {
    const be = createCpuBackend()
    // A 2x3, B 3x4 (row-major), C 2x4 hand-computed
    const A = [1, 2, 3, 4, 5, 6]
    const At = [1, 4, 2, 5, 3, 6]
    const B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    const Bt = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
    const want = [38, 44, 50, 56, 83, 98, 113, 128]
    for (const [ta, tb] of [[false, false], [true, false], [false, true], [true, true]] as const) {
      const a = be.upload(new Float32Array(ta ? At : A))
      const b = be.upload(new Float32Array(tb ? Bt : B))
      close(await be.read(be.matmul(a, b, 2, 3, 4, ta, tb)), want)
    }
  })

  it("binary add broadcasts 3-D [2,1,3] with [4,1] to [2,4,3]", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(1, 2, 3, 4, 5, 6))
    const b = be.upload(f(10, 20, 30, 40))
    const out = be.binary("add", a, [2, 1, 3], b, [4, 1], [2, 4, 3])
    close(await be.read(out), [
      11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43,
      14, 15, 16, 24, 25, 26, 34, 35, 36, 44, 45, 46,
    ])
  })

  it("sumToShape reduces 3-D [2,4,3] to [4,1] and [2,1,3]", async () => {
    const be = createCpuBackend()
    const x = be.upload(f(
      11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43,
      14, 15, 16, 24, 25, 26, 34, 35, 36, 44, 45, 46,
    ))
    close(await be.read(be.sumToShape(x, [2, 4, 3], [4, 1])), [81, 141, 201, 261])
    close(await be.read(be.sumToShape(x, [2, 4, 3], [2, 1, 3])), [104, 108, 112, 116, 120, 124])
  })

  it("sumToShape accumulates in float64", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(16777216, 1, 1, 1, 1))
    const out = await be.read(be.sumToShape(a, [5], [1]))
    expect(out[0]).toBe(16777220)
  })

  it("reduceLast sum and max over rows", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(1, 5, 3, -4, -2, -9))
    close(await be.read(be.reduceLast("max", a, 2, 3)), [5, -2])
    close(await be.read(be.reduceLast("sum", a, 2, 3)), [9, -15])
  })

  it("im2col known entries and col2im adjoint", async () => {
    const be = createCpuBackend()
    const x = be.upload(f(1, 2, 3, 4, 5, 6, 7, 8, 9))
    const cols = be.im2col(x, 1, 3, 3, 2, 2, 1, 0)
    expect(cols.length).toBe(16)
    close(await be.read(cols), [
      1, 2, 4, 5,
      2, 3, 5, 6,
      4, 5, 7, 8,
      5, 6, 8, 9,
    ])
    // padded / strided adjoint check
    for (const [c, h, w, kh, kw, s, p] of [[1, 3, 3, 2, 2, 1, 0], [2, 5, 4, 3, 3, 2, 1], [3, 4, 4, 3, 2, 1, 1]] as const) {
      const rng = createRng(7)
      const oh = Math.floor((h + 2 * p - kh) / s) + 1
      const ow = Math.floor((w + 2 * p - kw) / s) + 1
      const xd = randn(rng, c * h * w)
      const yd = randn(rng, c * kh * kw * oh * ow)
      const xs = be.upload(xd)
      const ys = be.upload(yd)
      const lhs = dot(await be.read(be.im2col(xs, c, h, w, kh, kw, s, p)), yd)
      const rhs = dot(xd, await be.read(be.col2im(ys, c, h, w, kh, kw, s, p)))
      expect(Math.abs(lhs - rhs)).toBeLessThan(1e-4)
    }
  })

  it("gatherRows and scatterAddRows are adjoint, with repeated indices", async () => {
    const be = createCpuBackend()
    const a = be.upload(f(1, 2, 3, 4, 5, 6))
    const idx = new Int32Array([2, 0, 2, 2])
    close(await be.read(be.gatherRows(a, 3, 2, idx)), [5, 6, 1, 2, 5, 6, 5, 6])
    const rng = createRng(3)
    const xd = randn(rng, 6)
    const yd = randn(rng, 8)
    const lhs = dot(await be.read(be.gatherRows(be.upload(xd), 3, 2, idx)), yd)
    const rhs = dot(xd, await be.read(be.scatterAddRows(be.upload(yd), 3, 2, idx)))
    expect(Math.abs(lhs - rhs)).toBeLessThan(1e-4)
  })

  it("adam one step", async () => {
    const be = createCpuBackend()
    const p = be.upload(f(1))
    const g = be.upload(f(0.5))
    const m = be.zeros(1)
    const v = be.zeros(1)
    const before = be.stats.flops
    be.adam(p, g, m, v, 0.1, 0.9, 0.999, 1e-8, 1)
    expect(be.stats.flops - before).toBe(10)
    close(await be.read(p), [0.9], 1e-5)
    close(await be.read(m), [0.05])
    close(await be.read(v), [0.00025])
  })

  it("liveBuffers tracks upload and dispose", () => {
    const be = createCpuBackend()
    expect(be.stats.liveBuffers).toBe(0)
    const s = be.upload(f(1))
    expect(be.stats.liveBuffers).toBe(1)
    be.dispose(s)
    expect(be.stats.liveBuffers).toBe(0)
  })
})

describe("random", () => {
  it("createRng is deterministic and in [0,1)", () => {
    const a = createRng(42)
    const b = createRng(42)
    for (let i = 0; i < 100; i++) {
      const x = a()
      expect(x).toBe(b())
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
    expect(createRng(1)()).not.toBe(createRng(2)())
  })

  it("randn has mean 0 and std 1", () => {
    const x = randn(createRng(42), 100000)
    let mean = 0
    for (const v of x) mean += v
    mean /= x.length
    let varr = 0
    for (const v of x) varr += (v - mean) ** 2
    const std = Math.sqrt(varr / x.length)
    expect(Math.abs(mean)).toBeLessThan(0.02)
    expect(Math.abs(std - 1)).toBeLessThan(0.02)
  })
})
