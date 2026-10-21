import { describe, it, expect } from "vitest"
import { PRED_FOVEA } from "@blanktab/interface"
import { createCpuBackend, createRng, getDefaultBackend, gradCheck, tensor } from "@blanktab/tensor"
import { ACTION_DIM } from "../src/encode"
import { SHIFT_TAPS, createFoveaShift } from "../src/shift"

const FW = PRED_FOVEA.width, FH = PRED_FOVEA.height, FC = PRED_FOVEA.channels
const PF = FW * FH * FC
const R = (SHIFT_TAPS - 1) / 2
const rng = createRng(21)
/** Values in ±[0.2, 1.2]. */
const signed = (n: number) => Float32Array.from({ length: n }, () => (rng() < 0.5 ? -1 : 1) * (0.2 + rng()))
/** A known image: every HWC element distinct, in (0, 1). */
const image = () => Float32Array.from({ length: PF }, (_, i) => (i + 1) / (PF + 1))
const pulse = (j: number) => {
  const k = new Float32Array(SHIFT_TAPS)
  k[j] = 1
  return k
}

describe("fovea shift", () => {
  it("has 17 taps per axis: ±8 target pixels, the largest move (16 world px)", () => {
    expect(SHIFT_TAPS).toBe(17)
  })

  it("its kernels are softmaxes: non-negative, summing to 1, for any action and weights", async () => {
    const be = createCpuBackend()
    const shift = createFoveaShift(be)
    const w = tensor(signed(ACTION_DIM * SHIFT_TAPS).map((v) => 3 * v), [ACTION_DIM, SHIFT_TAPS], { backend: be })
    const zeroW = tensor(new Float32Array(ACTION_DIM * SHIFT_TAPS), [ACTION_DIM, SHIFT_TAPS], { backend: be })
    for (let trial = 0; trial < 8; trial++) {
      const a = new Float32Array(ACTION_DIM)
      for (const i of [trial, 25 + trial, 28 + trial, 40 + trial]) a[i] = 1
      for (const weights of [w, zeroW]) {
        const k = await shift.kernel(tensor(a, [1, ACTION_DIM], { backend: be }), weights).read()
        expect(k.length).toBe(SHIFT_TAPS)
        expect(k.every((v) => v >= 0)).toBe(true)
        expect(Math.abs(k.reduce((s, v) => s + v, 0) - 1)).toBeLessThan(1e-6)
      }
    }
    // At birth (zero weights) the centre prior keeps most of the mass in place: e^4 / (e^4 + 16) ≈ 0.773.
    const a0 = new Float32Array(ACTION_DIM)
    a0[3] = 1
    const k0 = await shift.kernel(tensor(a0, [1, ACTION_DIM], { backend: be }), zeroW).read()
    expect(k0[R]).toBeCloseTo(Math.exp(4) / (Math.exp(4) + 16), 6)
    shift.dispose()
  })

  it("replays the oracle exactly: a one-hot kernel shifts a known image by exactly k pixels, zero-filled", async () => {
    const be = createCpuBackend()
    const shift = createFoveaShift(be)
    const src = image()
    const b = tensor(src, [1, PF], { backend: be })
    // Every tap pair, which includes the world's moves of 1, 4 and 16 px (0.5, 2 and 8 target pixels).
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const kx = tensor(pulse(R + dx), [1, SHIFT_TAPS], { backend: be })
      const ky = tensor(pulse(R + dy), [1, SHIFT_TAPS], { backend: be })
      const got = await shift.translate(b, kx, ky).read()
      const want = new Float32Array(PF)
      for (let y = 0; y < FH; y++) for (let x = 0; x < FW; x++) for (let c = 0; c < FC; c++) {
        const ys = y + dy, xs = x + dx // tap j samples the source at offset j − R
        want[(y * FW + x) * FC + c] = ys >= 0 && ys < FH && xs >= 0 && xs < FW ? src[(ys * FW + xs) * FC + c]! : 0
      }
      expect(got).toEqual(want) // bit-exact
    }
    shift.dispose()
  })

  it("is a convex blend: a two-tap kernel mixes the two shifted copies by its weights", async () => {
    const be = createCpuBackend()
    const shift = createFoveaShift(be)
    const src = image()
    const kxv = new Float32Array(SHIFT_TAPS)
    kxv[R] = 0.25
    kxv[R + 1] = 0.75
    const got = await shift.translate(tensor(src, [1, PF], { backend: be }), tensor(kxv, [1, SHIFT_TAPS], { backend: be }), tensor(pulse(R), [1, SHIFT_TAPS], { backend: be })).read()
    let worst = 0
    for (let y = 0; y < FH; y++) for (let x = 0; x < FW; x++) for (let c = 0; c < FC; c++) {
      const at = (xx: number) => (xx < FW ? src[(y * FW + xx) * FC + c]! : 0)
      worst = Math.max(worst, Math.abs(got[(y * FW + x) * FC + c]! - (0.25 * at(x) + 0.75 * at(x + 1))))
    }
    expect(worst).toBeLessThan(1e-7)
    shift.dispose()
  })

  it("its term g(a)·(shift(base) − base) is exactly zero while the gate is zero (a newborn predicts copy-last)", async () => {
    const be = createCpuBackend()
    const shift = createFoveaShift(be)
    const a = new Float32Array(ACTION_DIM)
    for (const i of [3, 25, 28, 40]) a[i] = 1
    const t = shift.term(tensor(a, [1, ACTION_DIM], { backend: be }), tensor(image(), [1, PF], { backend: be }), {
      x: tensor(signed(ACTION_DIM * SHIFT_TAPS), [ACTION_DIM, SHIFT_TAPS], { backend: be }),
      y: tensor(signed(ACTION_DIM * SHIFT_TAPS), [ACTION_DIM, SHIFT_TAPS], { backend: be }),
      gate: tensor(new Float32Array(ACTION_DIM), [ACTION_DIM, 1], { backend: be }),
    })
    expect((await t.read()).every((v) => v === 0)).toBe(true)
    shift.dispose()
  })

  it("passes gradCheck on its term over the kernel logits, the gate, the action and the base", () => {
    const shift = createFoveaShift(getDefaultBackend())
    const err = gradCheck(
      (x, y, gate, a, b) => shift.term(a, b, { x, y, gate }),
      [
        { data: signed(ACTION_DIM * SHIFT_TAPS), shape: [ACTION_DIM, SHIFT_TAPS] },
        { data: signed(ACTION_DIM * SHIFT_TAPS), shape: [ACTION_DIM, SHIFT_TAPS] },
        { data: signed(ACTION_DIM), shape: [ACTION_DIM, 1] },
        { data: signed(ACTION_DIM).map((v) => 0.3 * v), shape: [1, ACTION_DIM] },
        { data: image(), shape: [1, PF] },
      ],
    )
    expect(err).toBeLessThan(5e-2)
    shift.dispose()
  }, 120_000)

  it("frees its constants on dispose and leaks nothing per call", () => {
    const be = createCpuBackend()
    const before = be.stats.liveBuffers
    const shift = createFoveaShift(be)
    const act = tensor(new Float32Array(ACTION_DIM), [1, ACTION_DIM], { backend: be })
    const b = tensor(image(), [1, PF], { backend: be })
    const p = {
      x: tensor(new Float32Array(ACTION_DIM * SHIFT_TAPS), [ACTION_DIM, SHIFT_TAPS], { backend: be }),
      y: tensor(new Float32Array(ACTION_DIM * SHIFT_TAPS), [ACTION_DIM, SHIFT_TAPS], { backend: be }),
      gate: tensor(new Float32Array(ACTION_DIM), [ACTION_DIM, 1], { backend: be }),
    }
    const live = be.stats.liveBuffers
    const t = shift.term(act, b, p) // all constants: no graph node, caller-owned result only
    expect(be.stats.liveBuffers).toBe(live + 1)
    t.dispose()
    for (const x of [act, b, p.x, p.y, p.gate]) x.dispose()
    shift.dispose()
    expect(be.stats.liveBuffers).toBe(before)
  })
})
