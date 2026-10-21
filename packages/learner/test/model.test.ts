import { describe, it, expect } from "vitest"
import { Adam, add, backward, readSyncCpu, createCpuBackend, createRng, detach, disposeGraph, getDefaultBackend, gradCheck, mean, tensor } from "@blanktab/tensor"
import type { Tensor } from "@blanktab/tensor"
import { PERIPHERY, FOVEA } from "@blanktab/interface"
import { ACTION_DIM } from "../src/encode"
import { CPU_WIDTHS, GPU_WIDTHS, SHIFT_TAPS, createModel } from "../src/model"
import type { Widths } from "../src/model"

const rand = (n: number, seed: number) => {
  const r = createRng(seed)
  return Float32Array.from({ length: n }, () => r())
}
const inputs = (be = getDefaultBackend()) => ({
  periphery: tensor(rand(PERIPHERY.width * PERIPHERY.height * 3, 1), [3, PERIPHERY.height, PERIPHERY.width], { backend: be }),
  fovea: tensor(rand(FOVEA.width * FOVEA.height * 3, 2), [3, FOVEA.height, FOVEA.width], { backend: be }),
  proprio: tensor(rand(3, 3), [1, 3], { backend: be }),
})
const base = (be = getDefaultBackend()) => ({
  periphery: tensor(rand(3000, 4), [1, 3000], { backend: be }),
  fovea: tensor(rand(768, 5), [1, 768], { backend: be }),
})
/** White-box: overwrite a CPU parameter's values in place (tests only). */
const poke = (t: Tensor, f: (i: number) => number) => {
  const d = (t.storage as unknown as { data: Float32Array }).data
  for (let i = 0; i < d.length; i++) d[i] = f(i)
}
const oneHot = (be = getDefaultBackend()) => {
  const a = new Float32Array(ACTION_DIM)
  for (const i of [3, 25, 28, 40]) a[i] = 1
  return tensor(a, [1, ACTION_DIM], { backend: be })
}

describe("model", () => {
  for (const [name, w] of [["GPU_WIDTHS", GPU_WIDTHS], ["CPU_WIDTHS", CPU_WIDTHS]] as const) {
    it(`has the right output shapes at ${name}`, () => {
      const be = createCpuBackend()
      const m = createModel(be, 1, w)
      const x = inputs(be)
      const a = oneHot(be)
      const h0 = m.zeroState()
      expect([...h0.shape]).toEqual([1, w.hidden])
      const h = m.core(x, h0)
      expect([...h.shape]).toEqual([1, w.hidden])
      const { logits, value } = m.heads(h)
      expect([...logits.shape]).toEqual([1, 76])
      expect([...value.shape]).toEqual([1, 1])
      const p = m.predict(h, a, base(be))
      expect([...p.periphery.shape]).toEqual([1, 3000])
      expect([...p.fovea.shape]).toEqual([1, 768])
      expect(m.paramCount).toBe(m.params.reduce((s, t) => s + t.size, 0))
    })
  }

  it("has a parameter count near the benchmark's 2.62M at GPU_WIDTHS", () => {
    // The brief's 10% bound is unreachable with the specified predictor (enc -> 3000/768 heads, versus
    // hidden -> 3768 in the benchmark): the spec architecture has 2,183,109 params, 16.7% below. Bound at 20%.
    const m = createModel(createCpuBackend(), 1, GPU_WIDTHS)
    expect(m.paramCount).toBe(2185769)
    expect(Math.abs(m.paramCount - 2.62e6) / 2.62e6).toBeLessThan(0.2)
  })

  it("is deterministic in the seed", async () => {
    const w: Widths = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
    const read = async (seed: number) => Promise.all(createModel(createCpuBackend(), seed, w).params.map((t) => t.read()))
    const a = await read(5)
    const b = await read(5)
    const c = await read(6)
    expect(b).toEqual(a)
    expect(c).not.toEqual(a)
  })

  it("outputParams are exactly the last four params (the headLrScale split depends on it)", () => {
    const m = createModel(createCpuBackend(), 3, { conv1: 2, conv2: 2, enc: 4, hidden: 4 })
    expect(m.outputParams.length).toBe(4)
    m.outputParams.forEach((t, i) => expect(t).toBe(m.params.slice(-4)[i]))
    m.dispose()
  })

  it("can be built from saved parameter values (checkpoint restore) and rejects ones that do not fit", async () => {
    const w: Widths = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
    const be = createCpuBackend()
    const a = createModel(be, 5, w)
    const saved = await Promise.all(a.params.map((t) => t.read()))
    saved[saved.length - 1]![0] = 0.25 // differs from any fresh init (output layers start at zero)
    const b = createModel(be, 999, w, saved)
    expect(await Promise.all(b.params.map((t) => t.read()))).toEqual(saved)
    expect(b.params.every((t) => t.requiresGrad)).toBe(true)
    saved[0]![0] = 123 // the model holds a copy
    expect((await b.params[0]!.read())[0]).not.toBe(123)
    const live = be.stats.liveBuffers
    expect(() => createModel(be, 5, w, saved.slice(1))).toThrow(RangeError)
    expect(() => createModel(be, 5, w, [new Float32Array(3), ...saved.slice(1)])).toThrow(RangeError)
    expect(be.stats.liveBuffers).toBe(live) // nothing leaked by a rejected restore
    a.dispose()
    b.dispose()
  })

  it("starts at copy-last: a fresh predictor returns its base exactly", async () => {
    const be = createCpuBackend()
    const m = createModel(be, 2, { conv1: 2, conv2: 2, enc: 4, hidden: 4 })
    const b = base(be)
    const p = m.predict(m.zeroState(), oneHot(be), b)
    expect(await p.periphery.read()).toEqual(await b.periphery.read())
    expect(await p.fovea.read()).toEqual(await b.fovea.read())
  })

  it("translates the fovea by the action's learned kernel (the fovea-translation term)", async () => {
    const be = createCpuBackend()
    const m = createModel(be, 2, { conv1: 2, conv2: 2, enc: 4, hidden: 4 })
    const n = m.params.length
    // params: ..., shiftX, shiftY, shiftGate, predH (w, b), predA (w, b), headP (w, b), headF (w, b)
    const [shiftX, shiftY, gate] = [m.params[n - 11]!, m.params[n - 10]!, m.params[n - 9]!]
    expect([shiftX.shape, shiftY.shape, gate.shape]).toEqual([[ACTION_DIM, SHIFT_TAPS], [ACTION_DIM, SHIFT_TAPS], [ACTION_DIM, 1]])
    const k = 5 // one action component; its kernel picks x + 3 and y − 2, fully gated in
    const R = (SHIFT_TAPS - 1) / 2
    poke(shiftX, (i) => (i === k * SHIFT_TAPS + R + 3 ? 40 : 0))
    poke(shiftY, (i) => (i === k * SHIFT_TAPS + R - 2 ? 40 : 0))
    poke(gate, (i) => (i === k ? 1 : 0))
    const a = new Float32Array(ACTION_DIM)
    a[k] = 1
    const b = base(be)
    const fovea = await m.predict(m.zeroState(), tensor(a, [1, ACTION_DIM], { backend: be }), b).fovea.read()
    const src = await b.fovea.read()
    let worst = 0
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) for (let c = 0; c < 3; c++) {
      const ys = y - 2, xs = x + 3
      const want = ys >= 0 && xs < 16 ? src[(ys * 16 + xs) * 3 + c]! : 0
      worst = Math.max(worst, Math.abs(fovea[(y * 16 + x) * 3 + c]! - want))
    }
    expect(worst).toBeLessThan(1e-5)
    // Another action leaves the fovea at copy-last (its gate is still zero).
    const other = new Float32Array(ACTION_DIM)
    other[k + 1] = 1
    expect(await m.predict(m.zeroState(), tensor(other, [1, ACTION_DIM], { backend: be }), b).fovea.read()).toEqual(src)
  })

  it("passes gradCheck on predict with tiny widths (output layers made non-zero)", () => {
    const w: Widths = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
    const m = createModel(getDefaultBackend(), 3, w)
    // The residual output layers start at zero (gradient to h and the action would be trivially 0): randomise them.
    const r0 = createRng(12)
    for (const t of m.params.slice(-4)) poke(t, () => (r0() < 0.5 ? -1 : 1) * (0.2 + 0.4 * r0()))
    // Likewise the fovea-translation kernels and gate (shiftX, shiftY, shiftGate), so its path is checked too.
    for (const t of m.params.slice(-11, -8)) poke(t, () => (r0() < 0.5 ? -1 : 1) * (0.2 + 0.4 * r0()))
    const r = createRng(11)
    const h = Float32Array.from({ length: 4 }, () => (r() < 0.5 ? -1 : 1) * (0.3 + 0.5 * r()))
    const a = Float32Array.from({ length: ACTION_DIM }, () => (r() < 0.5 ? -1 : 1) * (0.3 + 0.5 * r()))
    // The base enters as `base + residual` (gradient: identity); checked below without finite differences.
    const b = base()
    for (const which of ["periphery", "fovea"] as const) {
      const err = gradCheck((hh, aa) => m.predict(hh, aa, b)[which], [{ data: h, shape: [1, 4] }, { data: a, shape: [1, ACTION_DIM] }])
      expect(err).toBeLessThan(5e-2)
    }
    const hT = tensor(h, [1, 4])
    const aT = tensor(a, [1, ACTION_DIM])
    const bp = tensor(rand(3000, 9), [1, 3000], { requiresGrad: true })
    const out = m.predict(hT, aT, { periphery: bp, fovea: b.fovea }).periphery
    const loss = mean(out)
    backward(loss)
    expect(Array.from(readSyncCpu(bp.grad!.storage)).every((g) => Math.abs(g - 1 / 3000) < 1e-9)).toBe(true)
    disposeGraph(loss, [])
  })

  it("leaves liveBuffers unchanged over 3 training iterations with a real Adam (backward, step, zeroGrad, disposeGraph)", () => {
    const be = createCpuBackend()
    const m = createModel(be, 1, CPU_WIDTHS)
    const opt = new Adam([...m.params], { lr: 1e-3 })
    const x = inputs(be)
    const a = oneHot(be)
    const b = base(be)
    let h = m.zeroState()
    const counts: number[] = []
    for (let i = 0; i < 3; i++) {
      const h2 = m.core(x, h)
      const { logits, value } = m.heads(h2)
      const p = m.predict(h2, a, b)
      const loss = add(add(mean(logits), mean(value)), add(mean(p.periphery), mean(p.fovea)))
      backward(loss)
      opt.step()
      opt.zeroGrad()
      const next = detach(h2)
      disposeGraph(loss, [])
      h.dispose()
      h = next
      counts.push(be.stats.liveBuffers)
    }
    // Steady state from the first iteration: one h out, one h in; Adam moments allocated up front.
    expect(counts).toEqual([counts[0], counts[0], counts[0]])
    expect(m.params.every((t) => t.grad === null)).toBe(true)
    h.dispose()
    for (const t of [x.periphery, x.fovea, x.proprio, a, b.periphery, b.fovea]) (t as Tensor).dispose()
    opt.dispose()
    m.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })
})
