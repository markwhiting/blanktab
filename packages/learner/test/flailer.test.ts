import { describe, it, expect } from "vitest"
import { emptyObservation, isValidAction } from "@blanktab/interface"
import { createFlailer } from "../src/index"

const run = async (seed: number, p: number, n = 500) => {
  const l = createFlailer({ seed, moveProbability: p })
  const o = emptyObservation()
  const out = []
  for (let i = 0; i < n; i++) out.push(await l.step(o))
  return out
}

describe("flailer", () => {
  it("step returns a Promise that resolves to a valid action", async () => {
    const p = createFlailer({ seed: 1, moveProbability: 0.5 }).step(emptyObservation())
    expect(p).toBeInstanceOf(Promise)
    expect(isValidAction(await p)).toBe(true)
  })
  it("is deterministic for a seed and differs across seeds", async () => {
    expect(await run(7, 0.5)).toEqual(await run(7, 0.5))
    expect(await run(7, 0.5)).not.toEqual(await run(8, 0.5))
  })
  it("always returns valid actions, never scrolls or types", async () => {
    for (const a of await run(3, 0.7, 2000)) {
      expect(isValidAction(a)).toBe(true)
      expect(a.scroll).toBe(0)
      expect(a.key).toBe(0)
    }
  })
  it("with moveProbability 0 only returns move 0", async () => {
    const acts = await run(5, 0, 1000)
    expect(acts.every(a => a.move === 0)).toBe(true)
    expect(acts.some(a => a.button !== 0)).toBe(true)
  })
  it("moves roughly moveProbability of the time", async () => {
    const frac = (await run(9, 0.3, 4000)).filter(a => a.move !== 0).length / 4000
    expect(frac).toBeGreaterThan(0.25)
    expect(frac).toBeLessThan(0.35)
  })
})
