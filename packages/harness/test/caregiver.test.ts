import { describe, it, expect } from "vitest"
import {
  CAREGIVER_DEFAULTS, CAREGIVER_RANGES, clampCaregiverSettings, createCaregiver, learningProgress, type CaregiverSettings,
} from "../src/caregiver"

const mk = (over: Partial<CaregiverSettings> = {}, seed = 7) => createCaregiver({ settings: { ...CAREGIVER_DEFAULTS, ...over }, seed })

describe("learningProgress", () => {
  it("falling loss > 0, flat = 0, rising < 0; a zero slow EMA does not divide by zero", () => {
    expect(learningProgress({ fast: 0.5, slow: 1 })).toBeCloseTo(0.5)
    expect(learningProgress({ fast: 1, slow: 1 })).toBe(0)
    expect(learningProgress({ fast: 2, slow: 1 })).toBeLessThan(0)
    expect(Number.isFinite(learningProgress({ fast: 0, slow: 0 }))).toBe(true)
    expect(learningProgress({ fast: 0, slow: 0 })).toBe(0)
    expect(Number.isFinite(learningProgress({ fast: 0, slow: 1e-9 }))).toBe(true)
  })
})

describe("observe", () => {
  it("updates only that room; NaN is ignored; the first loss sets both EMAs", () => {
    const c = mk()
    c.observe("a", 2)
    expect(c.state().rooms).toEqual({ a: { fast: 2, slow: 2 } })
    c.observe("b", 4)
    c.observe("a", 1)
    const s = c.state().rooms
    expect(s.b).toEqual({ fast: 4, slow: 4 })
    expect(s.a!.fast).toBeCloseTo(0.9 * 2 + 0.1 * 1)
    expect(s.a!.slow).toBeCloseTo(0.99 * 2 + 0.01 * 1)
    const before = JSON.stringify(c.state())
    c.observe("a", NaN)
    c.observe("c", NaN)
    c.observe("a", Infinity)
    c.observe("a", -Infinity)
    c.observe("c", Infinity)
    expect(JSON.stringify(c.state())).toBe(before)
  })
})

describe("decide timing", () => {
  it("decides after a sleep and every caregiverPeriod steps, otherwise null", () => {
    const c = mk({ caregiverPeriod: 50, caregiverEpsilon: 1 }) // epsilon 1: uniform pick, so a move is likely
    c.observe("a", 1)
    const rooms = ["a", "b", "c", "d"]
    const decided: number[] = []
    for (let i = 1; i <= 200; i++) {
      const before = c.state().sinceDecision
      c.decide("a", rooms, false)
      if (c.state().sinceDecision === 0 && before !== 0) decided.push(i)
    }
    expect(decided).toEqual([50, 100, 150, 200])
    // after a sleep: decides at once (sinceDecision resets)
    c.decide("a", rooms, false)
    expect(c.state().sinceDecision).toBe(1)
    c.decide("a", rooms, true)
    expect(c.state().sinceDecision).toBe(0)
  })
  it("returns null between decisions", () => {
    const c = mk({ caregiverPeriod: 50 })
    for (let i = 1; i < 50; i++) expect(c.decide("a", ["a", "b"], false)).toBeNull()
  })
})

describe("pick distribution", () => {
  it("pick frequencies over 10,000 decisions are within 0.02 of (1 − ε)·softmax(LP/τ) + ε/N; unvisited rooms get the max visited LP", () => {
    const settings = { ...CAREGIVER_DEFAULTS, caregiverTau: 0.3, caregiverEpsilon: 0.1 }
    const c = createCaregiver({ settings, seed: 12345 })
    for (let i = 0; i < 40; i++) c.observe("a", 1) // flat: LP 0
    for (let i = 0; i < 40; i++) c.observe("b", 1)
    for (let i = 0; i < 40; i++) c.observe("b", 0.5) // falling: LP > 0
    c.observe("c", 1)
    for (let i = 0; i < 10; i++) c.observe("c", 1.5) // rising: LP < 0
    const rooms = ["a", "b", "c", "d"] // d is unvisited
    const lp = c.state().rooms
    const lpB = learningProgress(lp.b!)
    expect(lpB).toBeGreaterThan(0)
    expect(learningProgress(lp.c!)).toBeLessThan(0)
    const lps = [learningProgress(lp.a!), lpB, learningProgress(lp.c!), lpB] // d = max visited
    const w = lps.map(x => Math.exp((x - lpB) / settings.caregiverTau))
    const z = w.reduce((s, x) => s + x, 0)
    const expected = w.map(x => (1 - settings.caregiverEpsilon) * (x / z) + settings.caregiverEpsilon / rooms.length)
    const counts = new Map<string, number>()
    const N = 10_000
    for (let i = 0; i < N; i++) {
      const to = c.decide("elsewhere", rooms, true)!
      counts.set(to, (counts.get(to) ?? 0) + 1)
    }
    rooms.forEach((r, i) => expect(Math.abs((counts.get(r) ?? 0) / N - expected[i]!), r).toBeLessThan(0.02))
  })
  it("with nothing visited, every room is equally likely", () => {
    const c = mk({ caregiverEpsilon: 0 }, 3)
    const counts: Record<string, number> = {}
    for (let i = 0; i < 3000; i++) { const t = c.decide("x", ["a", "b", "c"], true)!; counts[t] = (counts[t] ?? 0) + 1 }
    for (const r of ["a", "b", "c"]) expect(Math.abs(counts[r]! / 3000 - 1 / 3)).toBeLessThan(0.04)
  })
})

describe("no move", () => {
  it("a pick of the current room returns null and is not counted", () => {
    const c = mk({ caregiverEpsilon: 0 })
    for (let i = 0; i < 100; i++) expect(c.decide("only", ["only"], true)).toBeNull()
    expect(c.state().carries).toBe(0)
    expect(c.takeSummary()).toEqual({ carries: 0, to: {}, safetyNet: false })
  })
})

describe("safety net", () => {
  it("two consecutive M6 passes switch it on; a miss between resets the streak; it then carries only after stuckSteps stuck steps, and resets the count", () => {
    const c = mk({ caregiverStuckSteps: 100, caregiverThreshold: 0.01, caregiverPeriod: 50, caregiverEpsilon: 1 })
    c.onCheckup(["M1", "M6"])
    c.onCheckup(["M1"])
    c.onCheckup(["M6"])
    expect(c.state().safetyNet).toBe(false)
    c.onCheckup(["M6"])
    expect(c.state().safetyNet).toBe(true)
    c.onCheckup([]) // sticky
    expect(c.state().safetyNet).toBe(true)
    c.observe("a", 1)
    c.observe("b", 1)
    const rooms = ["a", "b", "c", "d", "e"]
    // never decides on a sleep or the period while safety net is on and not stuck long enough
    for (let i = 1; i < 100; i++) expect(c.decide("a", rooms, i % 10 === 0)).toBeNull()
    expect(c.state().stuckFor).toBe(99)
    // step 100: stuck long enough -> it decides; from "z" (not in rooms) every pick is a real move
    expect(c.decide("z", rooms, false)).not.toBeNull()
    expect(c.state().stuckFor).toBe(0)
    expect(c.state().carries).toBe(1)
    // and then waits another full stuckSteps
    for (let i = 1; i < 100; i++) expect(c.decide("a", rooms, true)).toBeNull()
    expect(c.state().stuckFor).toBe(99)
  })
  it("a room that is learning clears the stuck count", () => {
    const c = mk({ caregiverStuckSteps: 100 })
    c.onCheckup(["M6"]); c.onCheckup(["M6"])
    c.observe("a", 1)
    for (let i = 0; i < 50; i++) c.decide("a", ["a", "b"], false)
    expect(c.state().stuckFor).toBe(50)
    for (let i = 0; i < 30; i++) c.observe("a", 0.2) // falling loss: LP high
    c.decide("a", ["a", "b"], false)
    expect(c.state().stuckFor).toBe(0)
  })
})

describe("state", () => {
  it("survives structuredClone: a restored caregiver makes the same next 1,000 decisions", () => {
    const rooms = ["a", "b", "c"]
    const c = mk({ caregiverPeriod: 5 }, 99)
    for (let i = 0; i < 30; i++) { c.observe(rooms[i % 3]!, 1 / (1 + i)); c.decide("a", rooms, false) }
    c.onCheckup(["M6"])
    const copy = createCaregiver({ state: structuredClone(c.state()) })
    const run = (g: typeof c) => {
      const out: (string | null)[] = []
      for (let i = 0; i < 1000; i++) { g.observe(rooms[i % 3]!, 1 / (1 + (i % 17))); out.push(g.decide(rooms[i % 2]!, rooms, i % 40 === 0)) }
      return out
    }
    expect(run(copy)).toEqual(run(c))
    expect(copy.state()).toEqual(c.state())
  })
  it("a restored state is isolated from the original", () => {
    const c = mk()
    c.observe("a", 1)
    const st = c.state()
    const d = createCaregiver({ state: st })
    d.observe("a", 5)
    expect(c.state().rooms.a).toEqual({ fast: 1, slow: 1 })
    expect(st.rooms.a).toEqual({ fast: 1, slow: 1 })
  })
})

describe("robustness", () => {
  it("clamps settings on creation and on restore (tau 0 cannot divide by zero)", () => {
    const c = createCaregiver({ settings: { ...CAREGIVER_DEFAULTS, caregiverTau: 0, caregiverPeriod: NaN, caregiverEpsilon: 9 }, seed: 1 })
    expect(c.state().settings).toMatchObject({ caregiverTau: 0.001, caregiverPeriod: 500, caregiverEpsilon: 1 })
    c.observe("a", 1); for (let i = 0; i < 20; i++) c.observe("a", 0.1)
    expect(["a", "b"]).toContain(c.decide("z", ["a", "b"], true))
    const st = c.state(); st.settings.caregiverTau = 0
    expect(createCaregiver({ state: st }).state().settings.caregiverTau).toBe(0.001)
  })
  it("rejects a restored state with a non-finite average or a bad rng", () => {
    const st = mk().state()
    expect(() => createCaregiver({ state: { ...st, rooms: { a: { fast: NaN, slow: 1 } } } })).toThrow(RangeError)
    expect(() => createCaregiver({ state: { ...st, rooms: { a: { fast: 1, slow: Infinity } } } })).toThrow(RangeError)
    for (const rng of [-1, 1.5, 2 ** 32, NaN]) expect(() => createCaregiver({ state: { ...st, rng } })).toThrow(RangeError)
  })
})

describe("takeSummary", () => {
  it("counts carries per room since the last take, then resets", () => {
    const c = mk({ caregiverEpsilon: 1 })
    let n = 0
    const to: Record<string, number> = {}
    while (n < 20) { const t = c.decide("x", ["a", "b"], true); if (t) { n++; to[t] = (to[t] ?? 0) + 1 } }
    expect(c.takeSummary()).toEqual({ carries: 20, to, safetyNet: false })
    expect(c.takeSummary()).toEqual({ carries: 0, to: {}, safetyNet: false })
    c.decide("x", ["a"], true)
    expect(c.takeSummary()).toEqual({ carries: 1, to: { a: 1 }, safetyNet: false })
  })
})

describe("clampCaregiverSettings", () => {
  it("clamps into CAREGIVER_RANGES, rounds integers, and defaults non-finite values", () => {
    expect(clampCaregiverSettings({})).toEqual(CAREGIVER_DEFAULTS)
    const lo = clampCaregiverSettings({ caregiverPeriod: -5, caregiverTau: 0, caregiverEpsilon: -1, caregiverThreshold: -1, caregiverStuckSteps: 0 })
    expect(lo).toEqual({ caregiverPeriod: 50, caregiverTau: 0.001, caregiverEpsilon: 0, caregiverThreshold: 0, caregiverStuckSteps: 100 })
    const hi = clampCaregiverSettings({ caregiverPeriod: 1e9, caregiverTau: 99, caregiverEpsilon: 5, caregiverThreshold: 5, caregiverStuckSteps: 1e9 })
    expect(hi).toEqual({ caregiverPeriod: 10_000, caregiverTau: 10, caregiverEpsilon: 1, caregiverThreshold: 0.5, caregiverStuckSteps: 100_000 })
    expect(clampCaregiverSettings({ caregiverPeriod: 500.6, caregiverStuckSteps: 2000.4 })).toMatchObject({ caregiverPeriod: 501, caregiverStuckSteps: 2000 })
    for (const bad of [NaN, Infinity, -Infinity, "7", null, undefined])
      expect(clampCaregiverSettings({ caregiverPeriod: bad, caregiverTau: bad, caregiverEpsilon: bad, caregiverThreshold: bad, caregiverStuckSteps: bad })).toEqual(CAREGIVER_DEFAULTS)
    expect(Object.keys(CAREGIVER_RANGES).sort()).toEqual(Object.keys(CAREGIVER_DEFAULTS).sort())
  })
})
