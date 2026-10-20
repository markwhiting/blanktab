import { describe, expect, it } from "vitest"
import { ResourceError, ShapeError } from "../src/backend"
import type { Backend } from "../src/backend"

const f = (...x: number[]) => new Float32Array(x)

/**
 * Semantics every Backend must share, independent of numerics. Run against the CPU backend in node
 * (cpu-contract.test.ts) and the WebGPU backend in the browser (gpu/parity.gpu.test.ts).
 */
export function describeBackendContract(label: string, make: () => Promise<Backend>): void {
  describe(`${label} backend contract`, () => {
    it("liveBuffers counts storages; dispose decrements once; double dispose is a no-op", async () => {
      const be = await make()
      const base = be.stats.liveBuffers
      const a = be.upload(f(1, 2))
      const z = be.zeros(3)
      const u = be.unary("neg", a)
      expect(be.stats.liveBuffers).toBe(base + 3)
      be.dispose(u)
      be.dispose(u)
      expect(be.stats.liveBuffers).toBe(base + 2)
      be.dispose(a)
      be.dispose(z)
      expect(be.stats.liveBuffers).toBe(base)
    })

    it("use after dispose throws ResourceError; read of a disposed storage rejects", async () => {
      const be = await make()
      const a = be.upload(f(1, 2))
      be.dispose(a)
      expect(() => be.unary("neg", a)).toThrow(ResourceError)
      const p = be.read(a)
      expect(p).toBeInstanceOf(Promise)
      await expect(p).rejects.toThrow(ResourceError)
    })

    it("copy returns an independent storage with the same values and costs no FLOPs", async () => {
      const be = await make()
      const a = be.upload(f(1, NaN, -0, 4))
      const live = be.stats.liveBuffers
      const flops = be.stats.flops
      const c = be.copy(a)
      expect(c).not.toBe(a)
      expect(be.stats.liveBuffers).toBe(live + 1)
      expect(be.stats.flops).toBe(flops)
      be.adam(a, be.upload(f(1, 1, 1, 1)), be.zeros(4), be.zeros(4), 0.5, 0.9, 0.999, 1e-8, 1) // mutate a in place
      be.dispose(a)
      const got = Array.from(await be.read(c))
      expect(got[0]).toBe(1)
      expect(Number.isNaN(got[1])).toBe(true)
      expect(Object.is(got[2], -0)).toBe(true)
      expect(got[3]).toBe(4)
      expect(Array.from(await be.read(be.copy(be.zeros(0))))).toEqual([])
      be.dispose(c)
      expect(() => be.copy(a)).toThrow(ResourceError)
    })

    it("zero-length storages round-trip", async () => {
      const be = await make()
      expect(Array.from(await be.read(be.upload(f())))).toEqual([])
      expect(Array.from(await be.read(be.zeros(0)))).toEqual([])
    })

    it("reduceLast max: NaN in a row propagates; -Infinity is a value; cols=0 throws ShapeError", async () => {
      const be = await make()
      const a = be.upload(f(1, NaN, 3, NaN, 2, 0, -Infinity, -Infinity, -Infinity, 4, 5, NaN))
      const out = await be.read(be.reduceLast("max", a, 4, 3))
      expect(Number.isNaN(out[0])).toBe(true)
      expect(Number.isNaN(out[1])).toBe(true)
      expect(out[2]).toBe(-Infinity)
      expect(Number.isNaN(out[3])).toBe(true)
      expect(() => be.reduceLast("max", be.zeros(0), 2, 0)).toThrow(ShapeError)
      expect(Array.from(await be.read(be.reduceLast("sum", be.zeros(0), 2, 0)))).toEqual([0, 0])
    })

    it("relu and step propagate NaN; ±0 and ±Infinity follow x > 0", async () => {
      const be = await make()
      const a = be.upload(f(NaN, -2, -0, 0, 3, Infinity, -Infinity))
      const relu = Array.from(await be.read(be.unary("relu", a)))
      const step = Array.from(await be.read(be.unary("step", a)))
      expect(Number.isNaN(relu[0])).toBe(true)
      expect(Number.isNaN(step[0])).toBe(true)
      expect(relu.slice(1).map(Math.abs)).toEqual([0, 0, 0, 3, Infinity, 0])
      expect(step.slice(1)).toEqual([0, 0, 0, 1, 1, 0])
    })

    it("scalar uses the float32 value of c (equals binary with c uploaded as a 1-element storage)", async () => {
      const be = await make()
      // Inputs whose float64 result x op 0.1 rounds differently from x op fround(0.1).
      const xs = Float32Array.from({ length: 4096 }, (_, i) => 1 + i / 1024 - 2)
      const a = be.upload(xs)
      for (const c of [0.1, -1 / 3, 1e-7]) {
        const cs = be.upload(f(c))
        for (const op of ["add", "sub", "mul", "div"] as const) {
          const got = await be.read(be.scalar(op, a, c))
          const want = await be.read(be.binary(op, a, [xs.length], cs, [1], [xs.length]))
          let mismatches = 0
          for (let i = 0; i < xs.length; i++) if (!Object.is(got[i], want[i])) mismatches++
          expect(mismatches, `scalar ${op} ${c}`).toBe(0)
        }
      }
    })

    it("im2col/col2im geometry: ShapeError when the output is empty (oh or ow < 1) or stride < 1", async () => {
      const be = await make()
      const x = be.upload(new Float32Array(1 * 4 * 4))
      // h + 2*pad - kh = -1 -> floor(-1/2) + 1 = 0 output rows (previously accepted as an empty result).
      expect(() => be.im2col(x, 1, 4, 4, 5, 3, 2, 0)).toThrow(ShapeError)
      expect(() => be.im2col(x, 1, 4, 4, 3, 5, 2, 0)).toThrow(ShapeError)
      expect(() => be.col2im(be.zeros(0), 1, 4, 4, 5, 3, 2, 0)).toThrow(ShapeError)
      for (const stride of [0, -1]) {
        expect(() => be.im2col(x, 1, 4, 4, 3, 3, stride, 1), `im2col stride ${stride}`).toThrow(ShapeError)
        expect(() => be.col2im(be.zeros(9 * 16), 1, 4, 4, 3, 3, stride, 1), `col2im stride ${stride}`).toThrow(ShapeError)
      }
      // A valid geometry still works: 1x1 kernel, stride 3 over 4x4 -> 2x2.
      expect(Array.from(await be.read(be.im2col(be.upload(Float32Array.from({ length: 16 }, (_, i) => i)), 1, 4, 4, 1, 1, 3, 0)))).toEqual([0, 3, 12, 15])
    })

    it("sums accumulate beyond float32 precision", async () => {
      const be = await make()
      const a = be.upload(f(16777216, 1, 1, 1, 1))
      expect((await be.read(be.sumToShape(a, [5], [1])))[0]).toBe(16777220)
      expect((await be.read(be.reduceLast("sum", a, 1, 5)))[0]).toBe(16777220)
      expect((await be.read(be.matmul(a, be.upload(f(1, 1, 1, 1, 1)), 1, 5, 1, false, false)))[0]).toBe(16777220)
    })

    it("counts FLOPs exactly per primitive", async () => {
      const be = await make()
      const delta = (fn: () => void) => {
        const before = be.stats.flops
        fn()
        return be.stats.flops - before
      }
      const a = be.upload(new Float32Array(6))
      const b = be.upload(new Float32Array(3))
      expect(delta(() => be.unary("exp", a))).toBe(6)
      expect(delta(() => be.scalar("mul", a, 2))).toBe(6)
      expect(delta(() => be.binary("add", a, [2, 3], b, [3], [2, 3]))).toBe(6)
      expect(delta(() => be.sumToShape(a, [2, 3], [3]))).toBe(6)
      expect(delta(() => be.matmul(a, a, 2, 3, 2, false, true))).toBe(24)
      expect(delta(() => be.reduceLast("max", a, 2, 3))).toBe(6)
      expect(delta(() => be.im2col(a, 1, 2, 3, 1, 1, 1, 0))).toBe(0)
      expect(delta(() => be.col2im(a, 1, 2, 3, 1, 1, 1, 0))).toBe(0)
      expect(delta(() => be.gatherRows(a, 2, 3, new Int32Array([1, 0])))).toBe(0)
      expect(delta(() => be.scatterAddRows(a, 2, 3, new Int32Array([1, 1])))).toBe(0)
      const m = be.zeros(3)
      const v = be.zeros(3)
      const p = be.upload(f(1, 2, 3))
      expect(delta(() => be.adam(p, b, m, v, 0.1, 0.9, 0.999, 1e-8, 1))).toBe(30)
    })

    it("validates shapes and indices with ShapeError", async () => {
      const be = await make()
      const a = be.upload(new Float32Array(6))
      expect(() => be.binary("add", a, [2, 3], be.upload(new Float32Array(4)), [4], [2, 4])).toThrow(ShapeError)
      expect(() => be.matmul(a, a, 2, 2, 2, false, false)).toThrow(ShapeError)
      expect(() => be.gatherRows(a, 2, 3, new Int32Array([2]))).toThrow(ShapeError)
      expect(() => be.scatterAddRows(a, 2, 3, new Int32Array([0, -1]))).toThrow(ShapeError)
      expect(() => be.sumToShape(a, [2, 3], [2])).toThrow(ShapeError)
      expect(() => be.im2col(be.upload(new Float32Array(4)), 1, 2, 2, 5, 5, 1, 0)).toThrow(ShapeError)
      expect(() => be.col2im(be.upload(new Float32Array(100)), 1, 2, 2, 5, 5, 1, 0)).toThrow(ShapeError)
      const p = be.upload(f(1))
      expect(() => be.adam(p, p, be.zeros(1), be.zeros(1), 0.1, 0.9, 0.999, 1e-8, 1)).toThrow(ShapeError)
    })

    it("adam updates p, m and v in place", async () => {
      const be = await make()
      const p = be.upload(f(1))
      const m = be.zeros(1)
      const v = be.zeros(1)
      be.adam(p, be.upload(f(0.5)), m, v, 0.1, 0.9, 0.999, 1e-8, 1)
      expect((await be.read(p))[0]).toBeCloseTo(0.9, 5)
      expect((await be.read(m))[0]).toBeCloseTo(0.05, 6)
      expect((await be.read(v))[0]).toBeCloseTo(0.00025, 7)
    })
  })
}
