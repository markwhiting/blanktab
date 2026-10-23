import { readFileSync, writeFileSync } from "node:fs"
import { describe, it, expect } from "vitest"
import { chiSquareSf, createRng, dexp, dlog, hashSeed, lnGamma, normCdf, normInv, sobolPoints } from "../src/index"

const GOLDEN = new URL("./fixtures/golden.json", import.meta.url)

/** The IEEE-754 bits of a double as 16 hex digits: exact for every value, including -0, NaN and ±Infinity. */
function bits(x: number): string {
  const v = new DataView(new ArrayBuffer(8))
  v.setFloat64(0, x)
  return v.getBigUint64(0).toString(16).padStart(16, "0")
}

/** The gap between |x| and the next double away from zero. */
function ulp(x: number): number {
  const v = new DataView(new ArrayBuffer(8))
  v.setFloat64(0, Math.abs(x))
  v.setBigUint64(0, v.getBigUint64(0) + 1n)
  return v.getFloat64(0) - Math.abs(x)
}

describe("dexp and dlog", () => {
  it("dexp/dlog agree with Math.exp/Math.log to 2e-16 relative on 10,000 seeded points", () => {
    // Math.exp and Math.log are themselves not correctly rounded (V8 is off by one ulp on roughly 1 point in 10
    // here), and one ulp is up to 2.2e-16 relative, so even a correctly rounded dexp could not meet a bare 2e-16
    // against them. The bound is 2e-16 relative or one ulp of Math's value, whichever is larger.
    const tol = (want: number) => Math.max(2e-16 * Math.abs(want), ulp(want))
    const rng = createRng(hashSeed(1, "detmath-test"))
    for (let i = 0; i < 10_000; i++) {
      const x = -700 + 1400 * rng()
      const want = Math.exp(x)
      expect(Math.abs(dexp(x) - want), `dexp(${x})`).toBeLessThanOrEqual(tol(want))
      // dlog on positive inputs spread over the whole range, plus inputs near 1 (where log is small).
      for (const y of [want, 0.5 + 1.5 * rng()]) {
        const ly = Math.log(y)
        expect(Math.abs(dlog(y) - ly), `dlog(${y})`).toBeLessThanOrEqual(tol(ly))
      }
    }
  })
  it("dexp(0) = 1, dlog(1) = 0, dlog(0) = −Infinity, dlog(−1) NaN", () => {
    expect(dexp(0)).toBe(1)
    expect(dlog(1)).toBe(0)
    expect(dlog(0)).toBe(-Infinity)
    expect(dlog(-1)).toBeNaN()
    expect(dlog(Number.NaN)).toBeNaN()
    expect(dlog(Infinity)).toBe(Infinity)
    expect(dexp(Number.NaN)).toBeNaN()
    expect(dexp(-Infinity)).toBe(0)
    expect(dexp(Infinity)).toBe(Infinity)
    expect(dexp(710)).toBe(Infinity)
    expect(dexp(-746)).toBe(0)
  })
})

describe("normal distribution", () => {
  it("normInv(normCdf(x)) = x to 1e-12 on [−8, 8]", () => {
    // Above x ≈ 4 the double p = Φ(x) sits within a few ulps of 1, so x itself is not recoverable to 1e-12 from
    // p: one ulp of p moves x by ulp(p)/φ(x). The tolerance is 1e-12 plus that conditioning floor (it is
    // negligible for x ≤ 0, where Φ(x) is small and carries full relative precision).
    const ulp1 = 2 ** -53
    for (let i = 0; i <= 1600; i++) {
      const x = -8 + i / 100
      const p = normCdf(x)
      const phi = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI)
      const floor = 2 * Math.max(ulp1 * p, p >= 0.5 ? ulp1 : 0) / phi
      expect(Math.abs(normInv(p) - x), `x=${x}`).toBeLessThanOrEqual(1e-12 + floor)
    }
  })
  it("the upper tail agrees: 1 − normCdf(x) = normCdf(−x) to one rounding on [0, 8]", () => {
    // normCdf(x) for x > 0 is 1 − tail, rounded once; 1 − that is exact (Sterbenz), so it is within ½ ulp of 1.
    for (let i = 0; i <= 800; i++) {
      const x = i / 100
      expect(Math.abs((1 - normCdf(x)) - normCdf(-x)), `x=${x}`).toBeLessThanOrEqual(2 ** -54)
    }
  })
  it("normInv(0.975) = 1.959963984540054", () => {
    // 1.959963984540054 is the textbook value to 16 digits, not a double result: Φ⁻¹ of the double 0.975 is
    // 1.95996398454005386 (mpmath), whose nearest double is 1.9599639845400538. AS241 returns
    // 1.9599639845400536, 1.1 ulp (1.3e-16 relative) below the truth and 2 ulps below the literal; toBeCloseTo
    // with 15 digits allows |Δ| < 5e-16.
    expect(normInv(0.975)).toBeCloseTo(1.959963984540054, 15)
    expect(normInv(0.5)).toBe(0)
    expect(normInv(0)).toBeNaN()
    expect(normInv(1)).toBeNaN()
    expect(normCdf(0)).toBe(0.5)
  })
})

describe("chi-square", () => {
  it("chiSquareSf(16.919, 9) = 0.05 to 1e-4; chiSquareSf(0, k) = 1", () => {
    expect(Math.abs(chiSquareSf(16.919, 9) - 0.05)).toBeLessThan(1e-4)
    for (const k of [1, 2, 3, 9, 30]) expect(chiSquareSf(0, k)).toBe(1)
    // Both branches (series for x < a+1, continued fraction above) against closed forms.
    for (const x of [0.3, 1.5, 4, 12]) expect(Math.abs(chiSquareSf(x, 2) - Math.exp(-x / 2))).toBeLessThan(1e-14)
    expect(Math.abs(lnGamma(5) - Math.log(24))).toBeLessThan(1e-13)
    expect(Math.abs(lnGamma(0.5) - Math.log(Math.sqrt(Math.PI)))).toBeLessThan(1e-13)
  })
})

describe("sobolPoints", () => {
  it("sobolPoints(12, 9, 'calib/1') are in [0,1)¹², distinct, and equal the golden fixture", () => {
    const pts = sobolPoints(12, 9, "calib/1")
    expect(pts).toHaveLength(9)
    for (const p of pts) {
      expect(p).toHaveLength(12)
      for (const v of p) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThan(1) }
    }
    expect(new Set(pts.map(p => p.join(","))).size).toBe(9)
    const golden = JSON.parse(readFileSync(GOLDEN, "utf8")) as Golden
    expect(pts.map(p => p.map(bits))).toEqual(golden.sobol["calib/1"])
  })
  it("n = 0 gives no points", () => {
    expect(sobolPoints(3, 0, "s")).toEqual([])
  })
  it("rejects dims outside 1..16 and non-integer n", () => {
    expect(() => sobolPoints(17, 1, "s")).toThrow(RangeError)
    expect(() => sobolPoints(0, 1, "s")).toThrow(RangeError)
    expect(() => sobolPoints(2, 1.5, "s")).toThrow(RangeError)
  })
})

// ---------------------------------------------------------------------------------------------------------------
// Golden values: fixed inputs, outputs stored as float64 bits. DETMATH_WRITE=1 regenerates the file.

type Golden = {
  unary: Record<string, [number, string][]>
  chiSquareSf: [number, number, string][]
  hashSeed: [number, string, number][]
  createRng: [number, number[]][]
  sobol: Record<string, string[][]>
}

const UNARY = { dexp, dlog, normCdf, normInv, lnGamma } as const
const UNARY_INPUTS: Record<keyof typeof UNARY, number[]> = {
  dexp: [-745, -700, -30.5, -1, -1e-10, 0, 1e-10, 0.5, 1, 2.302585092994046, 88.7, 700, 709.7],
  dlog: [5e-324, 1e-300, 1e-10, 0.1, 0.5, 0.7071067811865476, 0.9999999, 1, 1.0000001, 2, 10, 1e10, 1.7e308],
  normCdf: [-38, -8, -5, -2, -1.959963984540054, -1, -0.1, 0, 0.1, 1, 2, 5, 8],
  normInv: [1e-300, 1e-20, 1e-5, 0.001, 0.025, 0.1, 0.3, 0.5, 0.7, 0.9, 0.975, 0.999, 1 - 1e-12],
  lnGamma: [1e-8, 0.1, 0.5, 1, 1.5, 2, 2.5, 4.5, 10, 50.5, 171, 1e5],
}
const CHI_INPUTS: [number, number][] = [
  [0, 1], [0.5, 1], [3.841458820694124, 1], [1, 2], [16.919, 9], [5, 9], [30, 9], [100, 50], [0.01, 30], [200, 3],
]
const SEED_INPUTS: [number, string][] = [[0, ""], [0, "calib/1"], [1, "gp-fit/1"], [4294967295, "x"], [12345, "longer salt"]]

function compute(): Golden {
  return {
    unary: Object.fromEntries(Object.entries(UNARY_INPUTS).map(([f, xs]) =>
      [f, xs.map(x => [x, bits(UNARY[f as keyof typeof UNARY](x))] as [number, string])])),
    chiSquareSf: CHI_INPUTS.map(([x, k]) => [x, k, bits(chiSquareSf(x, k))]),
    hashSeed: SEED_INPUTS.map(([s, salt]) => [s, salt, hashSeed(s, salt)]),
    createRng: [0, 1, 0x9e3779b9].map(s => { const r = createRng(s); return [s, Array.from({ length: 4 }, () => r())] }),
    sobol: Object.fromEntries(["calib/1", "gp-fit/1"].map(salt =>
      [salt, sobolPoints(salt === "calib/1" ? 12 : 16, 9, salt).map(p => p.map(bits))])),
  }
}

describe("golden fixture", () => {
  it("every function's outputs on the fixture inputs equal golden.json bit for bit", () => {
    const now = compute()
    if (process.env.DETMATH_WRITE === "1") writeFileSync(GOLDEN, JSON.stringify(now, null, 2) + "\n")
    expect(now).toEqual(JSON.parse(readFileSync(GOLDEN, "utf8")))
  })
})
