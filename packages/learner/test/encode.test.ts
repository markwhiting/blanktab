import { describe, it, expect } from "vitest"
import { emptyObservation, NOOP_ACTION, KEYS } from "@blanktab/interface"
import { ACTION_DIM, obsToInputs, targetsFromObs, actionOneHot, copyLastBaseline } from "../src/encode"

describe("encode", () => {
  it("has the expected shapes and value ranges", () => {
    const o = emptyObservation()
    o.periphery.fill(255)
    o.fovea.fill(51)
    o.cursor = { x: 0.25, y: 0.75 }
    o.buttonHeld = true
    const i = obsToInputs(o)
    expect(i.periphery.length).toBe(3 * 50 * 80)
    expect(i.fovea.length).toBe(3 * 32 * 32)
    expect(Array.from(i.proprio)).toEqual([0.25, 0.75, 1])
    expect(i.periphery.every((v) => v === 1)).toBe(true)
    expect(i.fovea[0]).toBeCloseTo(0.2)
    const t = targetsFromObs(o)
    expect(t.periphery.length).toBe(40 * 25 * 3)
    expect(t.fovea.length).toBe(16 * 16 * 3)
    expect(ACTION_DIM).toBe(76)
    expect(KEYS.length).toBe(45)
  })

  it("converts HWC to CHW in RGB order", () => {
    const o = emptyObservation()
    for (let p = 0; p < 80 * 50; p++) o.periphery[p * 3] = 255
    const i = obsToInputs(o)
    expect(i.periphery[0]).toBe(1) // R plane
    expect(i.periphery[50 * 80]).toBe(0) // G plane
    expect(i.periphery[2 * 50 * 80]).toBe(0) // B plane
    // single pixel at (x=3,y=2) green
    const o2 = emptyObservation()
    o2.periphery[(2 * 80 + 3) * 3 + 1] = 255
    const j = obsToInputs(o2)
    expect(j.periphery[50 * 80 + 2 * 80 + 3]).toBe(1)
    expect(j.periphery.reduce((a, b) => a + b, 0)).toBe(1)
  })

  it("box-averages 2x2 exactly, in HWC", () => {
    const o = emptyObservation()
    const set = (x: number, y: number, c: number, v: number) => { o.periphery[(y * 80 + x) * 3 + c] = v }
    // target cell (1,0): source x 2..3, y 0..1, channel 1
    set(2, 0, 1, 255); set(3, 0, 1, 0); set(2, 1, 1, 51); set(3, 1, 1, 0)
    const t = targetsFromObs(o)
    expect(t.periphery[(0 * 40 + 1) * 3 + 1]).toBeCloseTo((255 + 51) / 4 / 255, 6)
    expect(t.periphery[(0 * 40 + 0) * 3 + 1]).toBe(0)
    const f = emptyObservation()
    f.fovea[((3 * 32) + 5) * 3 + 2] = 255 // x=5,y=3 -> cell (2,1)
    const tf = targetsFromObs(f)
    expect(tf.fovea[(1 * 16 + 2) * 3 + 2]).toBeCloseTo(0.25, 6)
  })

  it("one-hot encodes actions", () => {
    const n = actionOneHot(NOOP_ACTION)
    expect(n.length).toBe(ACTION_DIM)
    expect(Array.from(n).flatMap((v, k) => (v === 1 ? [k] : []))).toEqual([0, 25, 28, 31])
    const a = actionOneHot({ move: 24, button: 2, scroll: 2, key: 44 })
    expect(Array.from(a).flatMap((v, k) => (v === 1 ? [k] : []))).toEqual([24, 27, 30, 75])
    expect(a.reduce((s, v) => s + v, 0)).toBe(4)
  })

  it("copyLastBaseline is MSE over both targets combined", () => {
    const o = emptyObservation()
    const t = targetsFromObs(o)
    expect(copyLastBaseline(t, t)).toBe(0)
    const p = targetsFromObs(o)
    p.periphery.fill(1)
    const n = 40 * 25 * 3
    const m = 16 * 16 * 3
    expect(copyLastBaseline(p, t)).toBeCloseTo(n / (n + m), 6)
  })
})
