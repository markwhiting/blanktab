import { describe, it, expect } from "vitest"
import type { CheckupResult } from "@blanktab/harness/log"
import { MILESTONE_NAMES, appendCheckups, caregiverText, checkupStatus, milestoneBars, milestoneText } from "../src/milestones"

const result = (level: number, passed: string[]): CheckupResult => ({ level, steps: level * 10, passed, measures: {}, evalCU: 0.1 })

describe("milestones", () => {
  it("names the six milestones in plain lower-case words", () => {
    expect(MILESTONE_NAMES).toEqual({
      M1: "predicts its world", M2: "moves with purpose", M3: "prefers things that respond",
      M4: "uncovers hidden content", M5: "explores places efficiently", M6: "uses the taskbar",
    })
  })

  it("milestoneText: none → 'no checkup yet (the first is at 8 CU)'; latest result → level, n of 6, and the passed names", () => {
    expect(milestoneText([])).toBe("no checkup yet (the first is at 8 CU)")
    expect(milestoneText([result(8, []), result(512, ["M1"])])).toBe("Latest checkup at 512 CU: passed 1 of 6 (predicts its world).")
    expect(milestoneText([result(16, [])])).toBe("Latest checkup at 16 CU: passed 0 of 6.")
    expect(milestoneText([result(2048, ["M1", "M6"])])).toBe("Latest checkup at 2,048 CU: passed 2 of 6 (predicts its world, uses the taskbar).")
  })

  it("milestoneBars: one bar per level in level order, height score/6 of the box, a zero score drawn as a 1 px baseline", () => {
    const bars = milestoneBars([result(8, []), result(16, ["M1", "M2", "M3"]), result(32, ["M1", "M2", "M3", "M4", "M5", "M6"])], 100, 64, 2)
    expect(bars.map(b => b.score)).toEqual([0, 3, 6])
    const inner = 64 - 2 * 2
    expect(bars[0]!.h).toBe(1)
    expect(bars[1]!.h).toBeCloseTo(inner / 2)
    expect(bars[2]!.h).toBeCloseTo(inner)
    for (const b of bars) {
      expect(b.y + b.h).toBeCloseTo(64 - 2) // standing on the baseline
      expect(b.x).toBeGreaterThanOrEqual(2)
      expect(b.x + b.w).toBeLessThanOrEqual(100 - 2)
      expect(b.w).toBeGreaterThan(0)
    }
    expect(bars[0]!.x).toBeLessThan(bars[1]!.x)
    expect(bars[1]!.x).toBeLessThan(bars[2]!.x)
    expect(milestoneBars([], 100, 64)).toEqual([])
    // Drawn in level order whatever order they come in.
    expect(milestoneBars([result(16, ["M1"]), result(8, [])], 100, 64).map(b => b.score)).toEqual([0, 1])
  })

  it("appendCheckups keeps only results ahead in level", () => {
    const a = appendCheckups([], [result(8, []), result(16, ["M1"])])
    expect(a.map(r => r.level)).toEqual([8, 16])
    // A repeat (a reload's tail, an event the page already has) and an older level are dropped.
    const b = appendCheckups(a, [result(16, ["M1", "M2"]), result(8, ["M1"]), result(32, ["M2"])])
    expect(b.map(r => r.level)).toEqual([8, 16, 32])
    expect(b[1]!.passed).toEqual(["M1"])
    expect(a.map(r => r.level)).toEqual([8, 16]) // not mutated
  })

  it("checkupStatus says the checkup's level and that its learning is paused", () => {
    expect(checkupStatus(512)).toBe("having a checkup at 512 CU (its learning is paused)")
    expect(checkupStatus(1024)).toBe("having a checkup at 1,024 CU (its learning is paused)")
  })

  it("caregiverText: carries, the last room, and safety-net mode", () => {
    expect(caregiverText({ carries: 0, to: null, safetyNet: false })).toBe("has not carried it yet")
    expect(caregiverText({ carries: 12, to: "toys", safetyNet: false })).toBe("has carried it 12 times; last to the toys room")
    expect(caregiverText({ carries: 1, to: "hideseek", safetyNet: false })).toBe("has carried it once; last to the hide-and-seek room")
    expect(caregiverText({ carries: 3, to: null, safetyNet: false })).toBe("has carried it 3 times")
    expect(caregiverText({ carries: 12, to: "places", safetyNet: true })).toBe(
      "has carried it 12 times; last to the places room; it now uses the taskbar itself, so the caregiver only steps in when it is stuck")
  })
})
