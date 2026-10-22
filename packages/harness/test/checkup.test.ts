import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD, isValidAction, type Action } from "@blanktab/interface"
import { DEFAULT_SETTINGS, createLearner, type FrozenRun, type FrozenRunOptions, type LearnerStats, type Widths } from "@blanktab/learner"
import { createCpuBackend, createRng } from "@blanktab/tensor"
import { createWorld, hashSeed } from "@blanktab/world"
import { uniformAction } from "../src/heldout"
import { MILESTONE_IDS, MILESTONE_SUITE_VERSION } from "../src/log"
import { MILESTONES_V1, YIELD_EVERY, type Milestone } from "../src/milestones"
import { checkupSeed, runCheckup, type CheckupDeps } from "../src/checkup"

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const canvas = () => createCanvas(WORLD.width, WORLD.height)
const SUITE_IDS = MILESTONE_IDS[MILESTONE_SUITE_VERSION]!

/** A stub learner: frozen runs act uniformly at random from their seed; each step bills 0.001 evalCU. */
function stubLearner() {
  const stats = { evalCU: 0.5, cu: 3 } as LearnerStats
  const log = { openRuns: 0, opened: [] as FrozenRunOptions[], disposed: 0 }
  return {
    log,
    learner: {
      stats: () => ({ ...stats }),
      frozenRun(o: FrozenRunOptions): FrozenRun {
        if (log.openRuns > 0) throw new Error("stub: a frozen run is already open")
        log.openRuns++
        log.opened.push(o)
        const rnd = createRng(o.seed)
        let closed = false
        return {
          step: async () => { stats.evalCU += 0.001; return uniformAction(rnd) },
          result: () => ({ predLoss: NaN, baselineLoss: NaN, transitions: 0 }),
          dispose() { if (!closed) { closed = true; log.openRuns--; log.disposed++ } },
        }
      },
    },
  }
}

/** The v1 suite at small budgets (same worlds, measures and pass rules). */
const small = (budget = 20): Milestone[] => MILESTONES_V1.map(m => ({ ...m, budgetSteps: budget }))
const base = (over: Partial<CheckupDeps> & Pick<CheckupDeps, "learner">): CheckupDeps =>
  ({ learnerId: "abc", level: 8, steps: 1234, canvas, suite: small(), ...over })

describe("checkups (decision 0016)", { timeout: 60_000 }, () => {
  it("checkupSeed is deterministic and differs across learnerId, level and milestone", () => {
    const s = checkupSeed("abc", 8, "M1")
    expect(checkupSeed("abc", 8, "M1")).toBe(s)
    expect(s).toBe(hashSeed(0, "abc/8/M1"))
    const others = [checkupSeed("abd", 8, "M1"), checkupSeed("abc", 16, "M1"), checkupSeed("abc", 8, "M2")]
    expect(new Set([s, ...others]).size).toBe(4)
  })

  it("each milestone runs in its own fresh world; the main world's tick, state() and trace are untouched", async () => {
    const mk = () => createWorld({ seed: 5, canvas: canvas() })
    const main = mk(), twin = mk()
    const rnd = createRng(9), rndTwin = createRng(9)
    for (let i = 0; i < 30; i++) { main.act(uniformAction(rnd)); twin.act(uniformAction(rndTwin)) }
    const seeds: [string, number][] = []
    const suite = small().map(m => ({ ...m, makeVariant: (seed: number, c: Parameters<Milestone["makeVariant"]>[1]) => { seeds.push([m.id, seed]); return m.makeVariant(seed, c) } }))
    const { learner, log } = stubLearner()
    const out = await runCheckup(base({ learner, suite }))
    expect(out.status).toBe("done")
    expect(seeds).toEqual(SUITE_IDS.map(id => [id, checkupSeed("abc", 8, id)]))
    expect(log.opened.map(o => o.seed)).toEqual(SUITE_IDS.map(id => hashSeed(checkupSeed("abc", 8, id), "policy")))
    expect(log.opened.map(o => o.measureLoss)).toEqual(SUITE_IDS.map(id => id === "M1"))
    expect(main.tick).toBe(twin.tick)
    expect(main.state()).toEqual(twin.state())
    expect(main.drainTrace()).toEqual(twin.drainTrace())
  })

  it("result: level, steps, passed in suite order, a measure per id, evalCU = the learner's evalCU delta", async () => {
    const { learner } = stubLearner()
    // Force passes: M3 and M1 always pass, M2 never; the order of `passed` follows the suite.
    const suite = small(10).map(m => (m.id === "M3" || m.id === "M1" ? { ...m, passes: () => true } : m.id === "M2" ? { ...m, passes: () => false } : m))
    const out = await runCheckup(base({ learner, suite, level: 16, steps: 777 }))
    if (out.status !== "done") throw new Error("abandoned")
    const r = out.result
    expect(r.level).toBe(16)
    expect(r.steps).toBe(777)
    expect(r.passed.slice(0, 2)).toEqual(["M1", "M3"])
    expect(r.passed).toEqual(SUITE_IDS.filter(id => r.passed.includes(id)))
    expect(Object.keys(r.measures)).toEqual([...SUITE_IDS])
    for (const v of Object.values(r.measures)) expect(v === null || Number.isFinite(v)).toBe(true)
    expect(r.measures.M1).toBeNull() // the stub measures no loss: NaN → null
    expect(r.evalCU).toBeCloseTo(6 * 10 * 0.001, 12)
    expect(r.error).toBeUndefined()
  })

  it("a milestone that throws is not passed, measure null, error true; the others still run; every frozen run is disposed", async () => {
    const { learner, log } = stubLearner()
    const suite = small(10).map(m => (m.id === "M3" ? { ...m, measure: () => { throw new Error("boom") }, passes: () => true } : m.id === "M4" ? { ...m, passes: () => true } : m))
    const out = await runCheckup(base({ learner, suite }))
    if (out.status !== "done") throw new Error("abandoned")
    expect(out.result.error).toBe(true)
    expect(out.result.measures.M3).toBeNull()
    expect(out.result.passed).not.toContain("M3")
    expect(out.result.passed).toContain("M4") // a later milestone still ran
    expect(Object.keys(out.result.measures)).toEqual([...SUITE_IDS])
    expect(log.opened).toHaveLength(6)
    expect(log.openRuns).toBe(0)
    expect(log.disposed).toBe(6)
  })

  it("a recoverable error is rethrown after disposing the open run", async () => {
    const { learner, log } = stubLearner()
    class Lost extends Error {}
    const suite = small(10).map(m => (m.id === "M2" ? { ...m, makeVariant: (s: number, c: Parameters<Milestone["makeVariant"]>[1]) => { const w = m.makeVariant(s, c); let n = 0; return { ...w, tick: w.tick, observe: () => { if (++n === 5) throw new Lost("device lost"); return w.observe() } } as typeof w } } : m))
    await expect(runCheckup(base({ learner, suite, recoverable: e => e instanceof Lost }))).rejects.toBeInstanceOf(Lost)
    expect(log.opened).toHaveLength(2)
    expect(log.openRuns).toBe(0)
  })

  it("shouldStop() true before a milestone: abandoned, no result, nothing left open", async () => {
    const { learner, log } = stubLearner()
    let calls = 0
    const out = await runCheckup(base({ learner, shouldStop: () => ++calls > 3 }))
    expect(out).toEqual({ status: "abandoned" })
    expect(log.opened).toHaveLength(3)
    expect(log.openRuns).toBe(0)
    // And a halt during the last milestone abandons the checkup too.
    const s2 = stubLearner()
    let started = 0
    const suite = small(10).map(m => ({ ...m, makeVariant: (s: number, c: Parameters<Milestone["makeVariant"]>[1]) => { started++; return m.makeVariant(s, c) } }))
    const out2 = await runCheckup(base({ learner: s2.learner, suite, shouldStop: () => started === 6 }))
    expect(out2).toEqual({ status: "abandoned" })
    expect(s2.log.openRuns).toBe(0)
  })

  it("yields at least every YIELD_EVERY steps", async () => {
    const { learner } = stubLearner()
    const budget = 120
    let yields = 0
    const out = await runCheckup(base({ learner, suite: small(budget), yieldNow: async () => { yields++ } }))
    expect(out.status).toBe("done")
    expect(yields).toBeGreaterThanOrEqual(6 * Math.floor(budget / YIELD_EVERY))
    expect(yields).toBe(6 * budget) // passed straight through to recordRun: awaited every step
  })

  it("with a real CPU learner: save() equal before and after except stats.evalCU; a newborn passes nothing; cu unchanged", async () => {
    // A newborn is not a uniform-random policy, and newborns do pass M3–M6 by luck now and then (measured on 25:
    // see the task-4 report). This fixture (seed 21, id "nb-21") is one that passes nothing; M1 never passes (ratio 1).
    const l = await createLearner({ backend: createCpuBackend(), seed: 21, widths: TINY, settings: DEFAULT_SETTINGS })
    const before = await l.save()
    const t0 = Date.now()
    const out = await runCheckup({ learnerId: "nb-21", level: 8, steps: 0, learner: l, canvas })
    const ms = Date.now() - t0
    const after = await l.save()
    if (out.status !== "done") throw new Error("abandoned")
    expect({ ...after, stats: { ...after.stats, evalCU: 0 } }).toEqual({ ...before, stats: { ...before.stats, evalCU: 0 } })
    expect(after.stats.cu).toBe(before.stats.cu)
    expect(after.stats.evalCU).toBeGreaterThan(before.stats.evalCU)
    expect(out.result.evalCU).toBeCloseTo(after.stats.evalCU - before.stats.evalCU, 12)
    expect(out.result.passed).toEqual([])
    expect(out.result.error).toBeUndefined()
    expect(out.result.measures.M1).toBe(1) // a newborn predicts copy-last exactly
    expect(Object.keys(out.result.measures)).toEqual([...SUITE_IDS])
    // The learner is usable afterwards (no frozen run left open).
    const a: Action = await l.step(createWorld({ seed: 1, canvas: canvas() }).observe())
    expect(isValidAction(a)).toBe(true)
    console.info(`full v1 checkup, real CPU learner (TINY): ${ms} ms; measures ${JSON.stringify(out.result.measures)}`)
    l.dispose()
  })
})
