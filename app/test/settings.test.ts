import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { DEFAULT_SETTINGS } from "@blanktab/learner"
import { DEFAULT_MAX_STEPS_PER_SEC } from "@blanktab/harness"
import { birthSettings } from "../src/settings"

describe("birthSettings", () => {
  it("is the learner's defaults when the knowledge picks nothing (offline)", () => {
    expect(birthSettings({})).toEqual({ learnerSettings: { ...DEFAULT_SETTINGS }, maxStepsPerSec: DEFAULT_MAX_STEPS_PER_SEC })
  })
  it("takes each picked learner setting and maxStepsPerSec, rounds the integer ones, and ignores the rest (moveProbability)", () => {
    const b = birthSettings({ lr: 0.002, entropyWeight: 0.02, curiosityWeight: 1.5, bufferSize: 1234.6, replayPerSleep: 16.4, headLrScale: 0.05, exploreEpsilon: 0.15, residualDecay: 1e-3, maxStepsPerSec: 30, moveProbability: 0.7 })
    expect(b).toEqual({
      learnerSettings: { lr: 0.002, entropyWeight: 0.02, curiosityWeight: 1.5, bufferSize: 1235, replayPerSleep: 16, headLrScale: 0.05, exploreEpsilon: 0.15, residualDecay: 1e-3 },
      maxStepsPerSec: 30,
    })
  })
  it("never rounds an integer setting down to 0", () => {
    expect(birthSettings({ bufferSize: 0.2, replayPerSleep: 0.4 }).learnerSettings).toMatchObject({ bufferSize: 1, replayPerSleep: 1 })
  })
})

describe("knowledge/settings-space.json", () => {
  const space = JSON.parse(readFileSync(new URL("../../knowledge/settings-space.json", import.meta.url), "utf8")) as { settings: { name: string; min: number; max: number }[] }
  const range = (name: string) => { const s = space.settings.find(x => x.name === name); return s && [s.min, s.max] }
  it("holds the plan's ranges for the learner settings and the speed cap, and keeps moveProbability", () => {
    // Plan 04a.2 Task 1 (controller ruling): narrowed to the measured neighbourhood of the defaults, so the picker
    // cannot sample untested corners.
    expect(range("lr")).toEqual([4e-4, 6e-4])
    expect(range("entropyWeight")).toEqual([0.001, 0.05])
    expect(range("curiosityWeight")).toEqual([0.5, 2])
    expect(range("bufferSize")).toEqual([500, 4000])
    expect(range("replayPerSleep")).toEqual([16, 128])
    expect(range("headLrScale")).toEqual([0.08, 0.12])
    expect(range("exploreEpsilon")).toEqual([0.08, 0.12])
    expect(range("residualDecay")).toEqual([7e-4, 1.5e-3])
    expect(range("maxStepsPerSec")).toEqual([30, 30])
    expect(range("moveProbability")).toEqual([0.1, 1])
  })
  it("contains the learner's defaults in every range it gives a learner setting (the measured point is pickable)", () => {
    for (const [k, v] of Object.entries(DEFAULT_SETTINGS)) {
      const r = range(k)
      if (r) expect(v >= r[0]! && v <= r[1]!, `${k} = ${v} outside [${r.join(", ")}]`).toBe(true)
    }
  })
})
