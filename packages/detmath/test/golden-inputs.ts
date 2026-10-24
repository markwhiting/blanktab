// The detmath golden fixture's inputs and how its outputs are computed (float64 bits), shared by detmath.test.ts (Node)
// and the e2e detmath probe page (app/detmath-probe.html: Chromium's main thread and a worker), so both compare the same
// values against fixtures/golden.json. Browser-safe: no Node API.
import { chiSquareSf, createRng, dexp, dlog, hashSeed, lnGamma, normCdf, normInv, sobolPoints } from "../src/index"

/** The IEEE-754 bits of a double as 16 hex digits: exact for every value, including -0, NaN and ±Infinity. */
export function bits(x: number): string {
  const v = new DataView(new ArrayBuffer(8))
  v.setFloat64(0, x)
  return v.getBigUint64(0).toString(16).padStart(16, "0")
}

export type Golden = {
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

export function compute(): Golden {
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
