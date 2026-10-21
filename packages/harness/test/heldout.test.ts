import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { isValidAction } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, createLearner, type Widths } from "@blanktab/learner"
import { createWorld } from "@blanktab/world"
import { HELD_OUT_TRANSITIONS, buildHeldOut } from "../src/heldout"

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const canvas = () => createCanvas(320, 200)
const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i])

describe("buildHeldOut (decision 0015 §3)", { timeout: 60_000 }, () => {
  it("records 64 transitions (65 observations) of uniformly random, valid actions", () => {
    const set = buildHeldOut(7, canvas())
    expect(HELD_OUT_TRANSITIONS).toBe(64)
    expect(set.actions).toHaveLength(64)
    expect(set.obs).toHaveLength(65)
    expect(set.actions.every(isValidAction)).toBe(true)
    // Uniform over each factor: across 64 draws, many distinct moves, and buttons, scrolls and keys besides "none".
    expect(new Set(set.actions.map(a => a.move)).size).toBeGreaterThan(10)
    for (const k of ["button", "scroll", "key"] as const) expect(set.actions.some(a => a[k] !== 0), k).toBe(true)
    // The world responds: the trajectory is not one frame repeated.
    expect(set.obs.some(o => !same(o.periphery, set.obs[0]!.periphery))).toBe(true)
  })
  it("is a pure function of the learner's seed, in a world seeded hashSeed(seed, \"heldout\"), not the seed's own world", () => {
    const a = buildHeldOut(7, canvas()), b = buildHeldOut(7, canvas()), c = buildHeldOut(8, canvas())
    expect(b.actions).toEqual(a.actions)
    expect(b.obs.every((o, i) => same(o.periphery, a.obs[i]!.periphery) && same(o.fovea, a.obs[i]!.fovea))).toBe(true)
    expect(c.actions).not.toEqual(a.actions)
    const own = createWorld({ seed: 7, canvas: canvas() }).observe()
    expect(same(own.periphery, a.obs[0]!.periphery)).toBe(false)
  })
  it("is what createLearner takes: a newborn scores copy-last on it at its first sleep", async () => {
    const be = createCpuBackend()
    const l = await createLearner({ backend: be, seed: 7, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: buildHeldOut(7, canvas()) })
    await l.sleep()
    const s = l.stats()
    expect(s.heldOutBaseline).toBeGreaterThan(0)
    expect(Math.abs(s.heldOutLoss / s.heldOutBaseline - 1)).toBeLessThan(1e-5)
    l.dispose()
  })
})
