import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import type { CheckupResult } from "@blanktab/harness/log"
import { MILESTONE_NAMES, appendCheckups, caregiverText, checkupStatus, createCarryTally, milestoneBars, milestoneText } from "../src/milestones"

const result = (level: number, passed: string[]): CheckupResult => ({ level, steps: level * 10, passed, measures: {}, evalCU: 0.1 })

describe("milestones", () => {
  it("names the six milestones in plain lower-case words", () => {
    expect(MILESTONE_NAMES).toEqual({
      M1: "predicts its world", M2: "moves with purpose", M3: "prefers things that respond",
      M4: "uncovers hidden content", M5: "explores places efficiently", M6: "uses the taskbar",
    })
  })

  it("the page's static milestone text (before any script runs) is milestoneText's for no checkup", () => {
    const html = readFileSync(join(__dirname, "../index.html"), "utf8")
    const m = html.match(/<p data-testid="milestone-text"[^>]*>([^<]*)<\/p>/)
    expect(m?.[1]).toBe(milestoneText([]))
  })
  it("milestoneText: none → 'no checkup yet (the first comes soon after it has lived 8 CU)'; latest result → level, n of 6, and the passed names", () => {
    expect(milestoneText([])).toBe("no checkup yet (the first comes soon after it has lived 8 CU)")
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

describe("createCarryTally", () => {
  const file = (seq: number, carries: number, safetyNet = false) => ({ seq, carries, safetyNet })
  it("counts the files' carries plus the carries since the latest file, and the last room", () => {
    const t = createCarryTally([file(0, 2), file(1, 3)], 2)
    expect(t.line()).toEqual({ carries: 5, to: null, safetyNet: false })
    t.carried("toys")
    t.carried("places")
    expect(t.line()).toEqual({ carries: 7, to: "places", safetyNet: false })
  })
  it("a logged file's own count replaces the carries seen for it (it may hold more: a reload's checkpoint carried some over)", () => {
    const t = createCarryTally([], 0)
    t.carried("toys")
    t.logged(0)
    t.carried("crib") // belongs to file 1
    expect(t.line().carries).toBe(2)
    t.file(file(0, 4, true)) // file 0 also held 3 carries made before a reload
    expect(t.line()).toEqual({ carries: 5, to: "crib", safetyNet: true })
  })
  it("overlapping or repeated file reads never count twice, and out-of-order ones change nothing", () => {
    const t = createCarryTally([], 0)
    t.carried("toys"); t.logged(0)
    t.carried("toys"); t.logged(1)
    t.file(file(1, 1)); t.file(file(0, 1)); t.file(file(1, 1)); t.file(file(0, 1))
    expect(t.line().carries).toBe(2)
    t.logged(0) // a late, older logged: the next file is still 2
    t.carried("crib")
    t.file(file(1, 1))
    expect(t.line().carries).toBe(3)
  })
  it("never goes negative or below the files' counts; a file with fewer carries than were seen (a restore undid some) wins", () => {
    const t = createCarryTally([], 0)
    t.carried("toys"); t.carried("toys"); t.carried("toys")
    t.logged(0)
    t.file(file(0, 1))
    expect(t.line().carries).toBe(1)
    expect(createCarryTally([], 0).line()).toEqual({ carries: 0, to: null, safetyNet: false })
  })
  it("keeps the files it has seen for the next run (the queue may delete them); safety net from the latest file", () => {
    const t = createCarryTally([file(0, 2), file(3, 1, true), file(1, 1)], 4)
    expect(t.line().safetyNet).toBe(true)
    const next = createCarryTally([...t.files(), file(3, 1, true)], 4)
    expect(next.line().carries).toBe(4)
  })
})
