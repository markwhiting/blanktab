import { describe, it, expect } from "vitest"
import { emptyObservation, isValidAction } from "@blanktab/interface"
import { createFlailer } from "../src/index"

const run = (seed: number, p: number, n = 500) => {
  const l = createFlailer({ seed, moveProbability: p })
  const o = emptyObservation()
  return Array.from({ length: n }, () => l.step(o))
}

describe("flailer", () => {
  it("is deterministic for a seed and differs across seeds", () => {
    expect(run(7, 0.5)).toEqual(run(7, 0.5))
    expect(run(7, 0.5)).not.toEqual(run(8, 0.5))
  })
  it("always returns valid actions, never scrolls or types", () => {
    for (const a of run(3, 0.7, 2000)) {
      expect(isValidAction(a)).toBe(true)
      expect(a.scroll).toBe(0)
      expect(a.key).toBe(0)
    }
  })
  it("with moveProbability 0 only returns move 0", () => {
    const acts = run(5, 0, 1000)
    expect(acts.every(a => a.move === 0)).toBe(true)
    expect(acts.some(a => a.button !== 0)).toBe(true)
  })
  it("moves roughly moveProbability of the time", () => {
    const frac = run(9, 0.3, 4000).filter(a => a.move !== 0).length / 4000
    expect(frac).toBeGreaterThan(0.25)
    expect(frac).toBeLessThan(0.35)
  })
})
