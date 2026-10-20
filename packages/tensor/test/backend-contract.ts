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
