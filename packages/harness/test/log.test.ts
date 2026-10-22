import { describe, it, expect } from "vitest"
import { MAX_CHECKUPS_PER_FILE, MILESTONE_IDS, MILESTONE_SUITE_VERSION, isLogLevel, levelAtOrBelow, logThresholds, nextCrossing, parseLogFile, parseSessionRow, type CheckupResult } from "../src/log"
import { mkFile } from "./helpers"

const ID = "lrn_0123456789abcdef"
const IDS = MILESTONE_IDS[MILESTONE_SUITE_VERSION]!
const measures = (over: Record<string, number | null> = {}) => ({ ...Object.fromEntries(IDS.map(id => [id, 0.5])), ...over })
const checkup = (level: number, over: Partial<CheckupResult> = {}): CheckupResult =>
  ({ level, steps: level, passed: ["M1", "M3"], measures: measures({ M2: null }), evalCU: 0.01, ...over })
const v2 = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  ...mkFile(ID, 0), schemaVersion: 2, stepsTotal: 100_000, cu: 300, condition: "A",
  widths: { conv1: 16, conv2: 32, enc: 128, hidden: 256 }, curve: [], skippedUpdates: 0, restores: 0, lostCU: 0, evalCU: 0.5,
  milestoneSuiteVersion: 1, checkups: [checkup(64), checkup(128, { error: true })],
  caregiver: { carries: 5, to: { left: 2, right: 3 }, safetyNet: false }, ...over,
})
const ok = (o: unknown) => parseLogFile(o).ok

describe("log schedule helpers", () => {
  it("isLogLevel and levelAtOrBelow agree with logThresholds", () => {
    const levels = new Set<number>()
    for (const t of logThresholds()) { if (t > 20_000) break; levels.add(t) }
    for (let n = 0; n <= 5000; n++) expect(isLogLevel(n), String(n)).toBe(levels.has(n))
    expect(isLogLevel(12)).toBe(false)
    expect(isLogLevel(0.5)).toBe(false)
    const at = (cu: number) => { let best: number | null = null; for (const t of logThresholds()) { if (t > cu) break; best = t } return best }
    for (const cu of [0, 7.99, 8, 15.9, 16, 511, 512, 1000, 1023.9, 1024, 1500, 2048, 5000.5]) expect(levelAtOrBelow(cu), String(cu)).toBe(at(cu))
    expect(nextCrossing(8)).toBe(16)
  })
})

describe("parseLogFile: checkups, milestoneSuiteVersion, caregiver", () => {
  it("a v2 file with checkups, milestoneSuiteVersion and caregiver parses and round-trips; each object has strict keys", () => {
    const f = v2()
    const r = parseLogFile(f)
    expect(r.ok).toBe(true)
    if (r.ok) expect(JSON.parse(JSON.stringify(r.value))).toEqual(f)
    const withExtra = (mut: (o: any) => void) => { const o = JSON.parse(JSON.stringify(f)); mut(o); return o } // eslint-disable-line @typescript-eslint/no-explicit-any
    expect(ok(withExtra(o => { o.checkups[0].extra = 1 }))).toBe(false)
    expect(ok(withExtra(o => { o.caregiver.extra = 1 }))).toBe(false)
  })
  it("rejects bad checkups", () => {
    const bad = (checkups: unknown, over: Record<string, unknown> = {}) => parseLogFile(v2({ checkups, ...over }))
    const cases: [string, ReturnType<typeof bad>][] = [
      ["level off the schedule", bad([checkup(12)])],
      ["level above cu", bad([checkup(512)])],
      ["levels equal", bad([checkup(64), checkup(64)])],
      ["levels decreasing", bad([checkup(128), checkup(64)])],
      ["steps > stepsTotal", bad([checkup(64, { steps: 100_001 })])],
      ["unknown id", bad([checkup(64, { passed: ["M9"] })])],
      ["duplicate id", bad([checkup(64, { passed: ["M1", "M1"] })])],
      ["out of suite order", bad([checkup(64, { passed: ["M3", "M1"] })])],
      ["measures key outside suite", bad([checkup(64, { measures: measures({ M7: 1 }) })])],
      ["measures key missing", bad([checkup(64, { measures: { M1: 1 } })])],
      ["non-finite measure", bad([checkup(64, { measures: measures({ M1: NaN }) })])],
      ["string measure", bad([checkup(64, { measures: measures({ M1: "1" as unknown as number }) })])],
      ["error: false", bad([{ ...checkup(64), error: false }])],
      ["negative evalCU", bad([checkup(64, { evalCU: -1 })])],
      ["9 checkups", bad(Array.from({ length: MAX_CHECKUPS_PER_FILE + 1 }, (_, i) => checkup(8 * 2 ** (i % 6)))), ],
      ["not an array", bad({})],
      ["no milestoneSuiteVersion", parseLogFile((() => { const o = v2(); delete o.milestoneSuiteVersion; return o })())],
      ["unknown suite version 2", bad([checkup(64)], { milestoneSuiteVersion: 2 })],
    ]
    for (const [name, r] of cases) expect(r.ok, name).toBe(false)
    // 9 genuinely-increasing levels on a big file fails on count alone
    const many = [8, 16, 32, 64, 128, 256, 512, 1024, 2048].map(l => checkup(l))
    const r = bad(many, { cu: 5000 })
    expect(r.ok).toBe(false)
    expect(bad(many.slice(0, 8), { cu: 5000 }).ok).toBe(true)
  })
  it("rejects a bad caregiver summary", () => {
    const cg = (c: unknown) => ok(v2({ caregiver: c }))
    expect(cg({ carries: 2, to: { a: 2 }, safetyNet: true })).toBe(true)
    expect(cg({ carries: 0, to: {}, safetyNet: false })).toBe(true)
    expect(cg({ carries: 1.5, to: { a: 1.5 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: -1, to: { a: -1 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: 1, to: { A: 1 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: 1, to: { abcdefghijklmnopq: 1 }, safetyNet: false })).toBe(false)
    const many = Object.fromEntries("abcdefghijklmnopq".split("").map(k => [k, 1]))
    expect(cg({ carries: 17, to: many, safetyNet: false })).toBe(false)
    expect(cg({ carries: 3, to: { a: 1, b: 1 }, safetyNet: false })).toBe(false)
    expect(cg({ carries: 1, to: { a: 1 }, safetyNet: "no" })).toBe(false)
    expect(cg(null)).toBe(false)
  })
  it("files without the new fields (old v2, v1) still parse", () => {
    const o = v2(); delete o.checkups; delete o.milestoneSuiteVersion; delete o.caregiver; delete o.evalCU
    const r = parseLogFile(o)
    expect(r.ok).toBe(true)
    if (r.ok) expect(Object.keys(r.value)).not.toContain("checkups")
    expect(ok(mkFile(ID, 0))).toBe(true)
    expect(ok({ ...mkFile(ID, 0), checkups: [] })).toBe(false) // v1 has no such field
  })
  it("checkups are billed inside the file's evalCU: non-empty checkups need evalCU, and their sum fits in it (1e-9 relative)", () => {
    const noEval = v2(); delete noEval.evalCU
    expect(parseLogFile(noEval)).toEqual({ ok: false, error: "checkups need the file's evalCU" })
    const empty = v2({ checkups: [] }); delete empty.evalCU
    expect(ok(empty)).toBe(true) // no checkups: nothing to bill
    expect(ok(v2({ evalCU: 0.02 }))).toBe(true) // exactly the sum (0.01 + 0.01)
    expect(ok(v2({ evalCU: 0.019 }))).toBe(false)
    expect(ok(v2({ evalCU: 0, checkups: [checkup(64, { evalCU: 0 })] }))).toBe(true)
    // Float sums: 0.1 + 0.2 > 0.3 by one ulp, within the tolerance; a real excess of 1e-6 relative is not.
    const tenths = [checkup(64, { evalCU: 0.1 }), checkup(128, { evalCU: 0.2 })]
    expect(ok(v2({ evalCU: 0.3, checkups: tenths }))).toBe(true)
    expect(ok(v2({ evalCU: 0.3 * (1 - 1e-6), checkups: tenths }))).toBe(false)
  })
})

describe("parseSessionRow checkupScore", () => {
  const row = { learnerId: ID, settings: {}, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, stepsTotal: 1, sessionsSeenAtBirth: 0, final: false }
  it("keeps a good checkupScore and drops a malformed one to null", () => {
    expect(parseSessionRow({ ...row, checkupScore: { level: 64, score: 3 } })!.checkupScore).toEqual({ level: 64, score: 3 })
    for (const bad of [{ level: "64", score: 3 }, { level: 64 }, { level: -1, score: 1 }, { level: 64, score: 1.5 }, 3, [], "x"]) {
      expect(parseSessionRow({ ...row, checkupScore: bad })!.checkupScore, JSON.stringify(bad)).toBeNull()
    }
    expect(parseSessionRow(row)!.checkupScore).toBeNull()
  })
})
