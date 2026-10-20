import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { NOOP_ACTION, emptyObservation, decodeMove, PERIPHERY, type Action, type Observation, type WorldView } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, clampSettings, createLearner, type LearnerSettings, type LearnerState, type LearnerStats, type Widths } from "@blanktab/learner"
import { openQueue } from "../src/queue"
import { loadCheckpoint, saveCheckpoint } from "../src/checkpoint"
import { parseLogFile, type LogFile, type LogFileV2 } from "../src/log"
import { runSession, createPacer, type Clock, type LearnerInit, type SessionConfig, type SessionDeps, type SessionLearner } from "../src/session"
import { mkFile } from "./helpers"

const ID = "lrn_0123456789abcdef"
const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const SETTINGS: LearnerSettings = { ...DEFAULT_SETTINGS, lr: 0.5, replayPerSleep: 8 } // lr is clamped to 1e-2

const cfgFor = (over: Partial<SessionConfig> = {}): SessionConfig => ({
  learnerId: ID, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 3,
  hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 1 },
  seed: 7, widths: TINY, learnerSettings: SETTINGS, ...over,
})

/** Time stands still unless something sleeps (or a step takes time): pacing is exact and costs no wall time. */
function fakeClock(): Clock & { t: number; sleeps: number[] } {
  const c = { t: 0, sleeps: [] as number[], now: () => c.t, sleep: async (ms: number) => { c.sleeps.push(ms); c.t += ms } }
  return c
}

/** A plain frame with a bright patch at a position the actions move: a WorldView producing a synthetic stream. */
function stubWorld(): WorldView & { acts: number } {
  let x = 40, y = 25
  const w = {
    acts: 0,
    observe(): Observation {
      const o = emptyObservation()
      o.periphery.fill(30)
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const px = Math.min(PERIPHERY.width - 1, Math.max(0, x + dx)), py = Math.min(PERIPHERY.height - 1, Math.max(0, y + dy))
        o.periphery.fill(220, (py * PERIPHERY.width + px) * 3, (py * PERIPHERY.width + px) * 3 + 3)
      }
      o.fovea.fill(x % 7 * 30)
      o.cursor = { x: x / (PERIPHERY.width - 1), y: y / (PERIPHERY.height - 1) }
      return o
    },
    act(a: Action) {
      w.acts++
      const { dx, dy } = decodeMove(a.move)
      x = Math.min(PERIPHERY.width - 1, Math.max(0, x + Math.sign(dx)))
      y = Math.min(PERIPHERY.height - 1, Math.max(0, y + Math.sign(dy)))
    },
  }
  return w
}

interface StubState { seed: number; widths: Widths; settings: LearnerSettings; stats: LearnerStats }

/**
 * Flailer-like TrainableLearner: no-op actions, `cuPerStep` CU per step and `sleepCu` per sleep, and losses that are
 * a pure function of the step count (NaN before step `nanUntil`). `skip(steps)` makes that step's update a skipped one.
 */
function stubFactory(o: { cuPerStep: number; sleepCu?: number; nanUntil?: number; skip?: (steps: number) => boolean; saves?: number[]; created?: LearnerInit[] }) {
  return async (init: LearnerInit): Promise<SessionLearner> => {
    o.created?.push(init)
    const s0 = "state" in init ? (init.state as unknown as StubState) : { seed: init.seed, widths: init.widths, settings: init.settings, stats: { steps: 0, cu: 0, predLoss: Number.NaN, baselineLoss: Number.NaN, skippedUpdates: 0, consecutiveSkips: 0, sleeps: 0 } }
    const st: StubState = structuredClone(s0)
    const loss = () => {
      const n = st.stats.steps
      st.stats.predLoss = n < (o.nanUntil ?? 2) ? Number.NaN : 0.5 / (1 + n / 100)
      st.stats.baselineLoss = n < (o.nanUntil ?? 2) ? Number.NaN : 0.4
    }
    return {
      step: async () => {
        st.stats.steps++
        st.stats.cu += o.cuPerStep
        if (o.skip?.(st.stats.steps)) { st.stats.skippedUpdates++; st.stats.consecutiveSkips++ } else st.stats.consecutiveSkips = 0
        loss()
        return NOOP_ACTION
      },
      sleep: async () => { st.stats.sleeps++; st.stats.cu += o.sleepCu ?? 0 },
      save: async () => { o.saves?.push(st.stats.consecutiveSkips); return structuredClone(st) as unknown as LearnerState },
      stats: () => ({ ...st.stats }),
      dispose: () => {},
    }
  }
}

const depsFor = (idb: IDBFactory, over: Partial<SessionDeps> & Pick<SessionDeps, "createLearner">): SessionDeps => ({
  world: stubWorld(), idb, now: () => new Date(1_700_000_000_000), clock: fakeClock(), ...over,
})
const logsOf = async (idb: IDBFactory) => (await (await openQueue(idb)).pending()).filter(f => f.learnerId === ID) as LogFileV2[]
const strip = (f: LogFile) => ({ ...f, writtenAt: "" })

function expectContinuous(logs: LogFileV2[]) {
  expect(logs.map(f => f.seq)).toEqual(logs.map((_, i) => i))
  for (const f of logs) {
    const p = parseLogFile(JSON.parse(JSON.stringify(f)))
    expect(p.ok, p.ok ? "" : p.error).toBe(true)
    expect(f.schemaVersion).toBe(2)
  }
  for (let i = 1; i < logs.length; i++) {
    expect(logs[i]!.cu).toBeGreaterThan(logs[i - 1]!.cu)
    expect(logs[i]!.stepsTotal).toBeGreaterThan(logs[i - 1]!.stepsTotal)
  }
  const curve = logs.flatMap(f => f.curve)
  for (let i = 1; i < curve.length; i++) expect(curve[i]!.steps).toBeGreaterThan(curve[i - 1]!.steps)
  for (const f of logs) for (const p of f.curve) expect(p.steps).toBeLessThanOrEqual(f.stepsTotal)
  expect(logs.filter(f => f.final).map(f => f.seq)).toEqual([logs.length - 1])
}

describe("session v2", { timeout: 60_000 }, () => {
  it("writes log files exactly at the 2^k CU crossings plus the final one, all v2-valid", async () => {
    const idb = new IDBFactory()
    const r = await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16 }) }))
    const logs = await logsOf(idb)
    expect(logs.map(f => f.cu)).toEqual([1, 2, 4, 8, 16, 32, 64])
    expect(logs.map(f => f.stepsTotal)).toEqual([16, 32, 64, 128, 256, 512, 1024])
    expect(logs.map(f => f.final)).toEqual([false, false, false, false, false, false, true])
    expectContinuous(logs)
    expect(r).toMatchObject({ steps: 1024, seq: 7, resumedFrom: null })
    const f = logs[3]!
    expect(f).toMatchObject({
      learnerId: ID, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 3, condition: "A", widths: TINY, skippedUpdates: 0,
      hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 1 }, writtenAt: new Date(1_700_000_000_000).toISOString(),
    })
    // settings: the learner settings as the learner uses them (lr clamped), the speed cap and the seed.
    expect(f.settings).toEqual({ ...clampSettings(SETTINGS), maxStepsPerSec: 30, seed: 7 })
    expect(f.settings.lr).toBe(1e-2)
    // The curve: a point at every sleep (every 50 steps) and one at the end of life, each in the file that follows it.
    expect(logs.flatMap(l => l.curve.map(p => p.steps))).toEqual([...Array.from({ length: 20 }, (_, i) => 50 * (i + 1)), 1024])
    expect(logs[6]!.curve.map(p => p.steps)).toEqual([550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1024])
    expect(logs[6]!.curve[0]).toEqual({ cu: 550 / 16, steps: 550, predLoss: 0.5 / 6.5, baselineLoss: 0.4 })
  })

  it("the default life is 1024 CU with a sleep every 250 steps: at most 11 + 1 files", async () => {
    const idb = new IDBFactory()
    const created: LearnerInit[] = []
    const r = await runSession(cfgFor(), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 0.25, created }) }))
    const logs = await logsOf(idb)
    expect(logs.length).toBeLessThanOrEqual(12)
    expect(logs.map(f => f.cu)).toEqual([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    expect(logs.at(-1)).toMatchObject({ final: true, cu: 1024 })
    expect(logs.flatMap(l => l.curve).length).toBe(Math.floor(r.steps / 250) + 1)
    expectContinuous(logs)
    expect(created).toEqual([{ seed: 7, widths: TINY, settings: clampSettings(SETTINGS) }])
  })

  it("CU a sleep adds counts too: a crossing at a sleep is logged with that sleep's curve point", async () => {
    const idb = new IDBFactory()
    await runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 10 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 0, sleepCu: 1 }) }))
    const logs = await logsOf(idb)
    expect(logs.map(f => [f.stepsTotal, f.cu, f.final])).toEqual([[10, 1, false], [20, 2, false], [40, 4, true]])
    expect(logs.map(f => f.curve.map(p => p.steps))).toEqual([[10], [20], [30, 40]])
  })

  it("never writes NaN: curve points are omitted until both losses are finite", async () => {
    const idb = new IDBFactory()
    await runSession(cfgFor({ lifetimeCU: 8, sleepInterval: 10 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16, nanUntil: 35 }) }))
    const logs = await logsOf(idb)
    expect(logs.flatMap(l => l.curve.map(p => p.steps)).slice(0, 2)).toEqual([40, 50])
    expect(logs[0]!.curve).toEqual([]) // seq 0 at step 16: nothing finite yet
    for (const f of logs) expect(JSON.stringify(f)).not.toMatch(/null|NaN/)
  })

  it("a log file's curve is thinned to at most 64 points, keeping the newest; so is the checkpoint's tail", async () => {
    const idb = new IDBFactory()
    const tails: number[] = []
    await runSession(cfgFor({ lifetimeCU: 2, sleepInterval: 1 }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 / 1024 }),
      onTick: async ({ steps }) => { if (steps % 97 === 0) tails.push((await loadCheckpoint(idb, ID))?.curveTail.length ?? 0) },
    }))
    const logs = await logsOf(idb)
    expect(logs.map(f => f.stepsTotal)).toEqual([1024, 2048])
    for (const f of logs) {
      expect(f.curve.length).toBeGreaterThan(16)
      expect(f.curve.length).toBeLessThanOrEqual(64)
      expect(f.curve.at(-1)!.steps).toBe(f.stepsTotal)
    }
    expect(Math.max(...tails)).toBeLessThanOrEqual(64)
    expectContinuous(logs)
  })

  describe("Review Focus 2: reload mid-life resumes from the checkpoint", () => {
    const run = (idb: IDBFactory, stopAt?: number, ticks: number[] = []) => runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50 }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 / 16 }),
      onTick: ({ steps }) => { ticks.push(steps); if (steps === stopAt) throw new Error("reload") },
    }))

    for (const stopAt of [10, 49, 50, 333, 513, 700, 1023]) {
      it(`interrupted at step ${stopAt}: steps, CU and seq continue with no duplicate seq, the curve continues, files = an uninterrupted life`, async () => {
        const whole = new IDBFactory()
        await run(whole)
        const idb = new IDBFactory()
        await expect(run(idb, stopAt)).rejects.toThrow("reload")
        const before = await logsOf(idb)
        const cp = await loadCheckpoint(idb, ID)
        expect(cp).toBeDefined()
        expect(cp!.seq).toBe(before.length) // the checkpoint is never behind the latest log file
        // The latest of: birth, the last sleep before the interrupted step, the last log file (onTick runs before them).
        expect(cp!.steps).toBe(Math.max(50 * Math.floor((stopAt - 1) / 50), before.at(-1)?.stepsTotal ?? 0))
        const ticks: number[] = []
        const r = await run(idb, undefined, ticks)
        expect(r.resumedFrom).toBe(cp!.steps)
        expect(ticks[0]).toBe(cp!.steps + 1)
        const logs = await logsOf(idb)
        expectContinuous(logs)
        expect(logs.slice(0, before.length)).toEqual(before) // nothing rewritten
        expect(logs.map(strip)).toEqual((await logsOf(whole)).map(strip))
        expect(await loadCheckpoint(idb, ID)).toBeUndefined()
      })
    }

    it("a real tiny CPU learner in a synthetic world: interrupted, resumed, logs continuous and valid", async () => {
      const idb = new IDBFactory()
      const cfg = cfgFor({ lifetimeCU: 4, sleepInterval: 50, learnerSettings: { ...DEFAULT_SETTINGS, replayPerSleep: 8, bufferSize: 200 } })
      const make = (init: LearnerInit) => createLearner({ backend: createCpuBackend(), ...init })
      let stoppedAt = -1
      await expect(runSession(cfg, depsFor(idb, {
        createLearner: make,
        onTick: ({ steps, cu }) => { if (cu > 1.5 && steps % 50 === 20) { stoppedAt = steps; throw new Error("reload") } },
      }))).rejects.toThrow("reload")
      const before = await logsOf(idb)
      expect(before.map(f => f.cu)).toEqual([expect.any(Number)]) // the 1 CU file
      const cp = (await loadCheckpoint(idb, ID))!
      expect(cp.steps).toBe(stoppedAt - 20)
      expect(cp.learnerState.stats.steps).toBe(cp.steps)
      expect(cp.learnerState.stats.cu).toBe(cp.cu)
      const ticks: number[] = []
      const r = await runSession(cfg, depsFor(idb, { createLearner: make, onTick: ({ steps }) => { ticks.push(steps) } }))
      expect(ticks[0]).toBe(cp.steps + 1)
      expect(r.resumedFrom).toBe(cp.steps)
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.slice(0, 1)).toEqual(before)
      expect(logs.length).toBe(3) // 1 CU, 2 CU, final at 4 CU
      expect(logs[0]!.cu).toBeGreaterThanOrEqual(1)
      expect(logs[1]!.cu).toBeGreaterThanOrEqual(2)
      expect(logs[2]!.cu).toBeGreaterThanOrEqual(4)
      const curve = logs.flatMap(f => f.curve)
      expect(curve.length).toBeGreaterThan(5)
      for (const p of curve) expect(Number.isFinite(p.predLoss) && Number.isFinite(p.baselineLoss)).toBe(true)
      expect(logs[2]!.widths).toEqual(TINY)
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })
  })

  it("deletes the checkpoint once the final log is queued; a finished learner does not run again", async () => {
    const idb = new IDBFactory()
    let seen = false
    await runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50 }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 / 16 }),
      onTick: async () => { seen ||= (await loadCheckpoint(idb, ID)) !== undefined },
    }))
    expect(seen).toBe(true)
    expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    // An orphan checkpoint (say a crash between two writes in older code) is cleared, and nothing runs.
    await saveCheckpoint(idb, { learnerId: ID, savedAt: "x", steps: 1, cu: 0, seq: 9, curveTail: [], learnerState: {} as LearnerState })
    const ticks: number[] = []
    const r = await runSession(cfgFor({ lifetimeCU: 4 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 }), onTick: ({ steps }) => { ticks.push(steps) } }))
    expect(ticks).toEqual([])
    expect(r.alreadyFinal).toBe(true)
    expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    expect((await logsOf(idb)).length).toBe(3)
  })

  it("refuses to resume a learner with log files but no checkpoint, or a checkpoint of another birth", async () => {
    const idb = new IDBFactory()
    await (await openQueue(idb)).put(mkFile(ID, 0))
    await expect(runSession(cfgFor(), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 }) }))).rejects.toThrow(/no checkpoint/)
    for (const over of [{ seed: 8 }, { widths: { ...TINY, hidden: 32 } }, { learnerSettings: { ...SETTINGS, bufferSize: 99 } }]) {
      const fresh = new IDBFactory()
      await expect(runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50 }), depsFor(fresh, {
        createLearner: stubFactory({ cuPerStep: 1 / 16 }), onTick: ({ steps }) => { if (steps === 60) throw new Error("reload") },
      }))).rejects.toThrow("reload")
      await expect(runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50, ...over }), depsFor(fresh, { createLearner: stubFactory({ cuPerStep: 1 / 16 }) }))).rejects.toThrow(/checkpoint/)
    }
  })

  describe("skipped updates", () => {
    it("never saves a checkpoint while consecutiveSkips > 0; a crossing during skips is logged once they clear", async () => {
      const idb = new IDBFactory()
      const saves: number[] = []
      await runSession(cfgFor({ lifetimeCU: 8, sleepInterval: 10 }), depsFor(idb, {
        // Skips over steps 14-21 (the 1 CU crossing is step 16) and over the sleep at step 40.
        createLearner: stubFactory({ cuPerStep: 1 / 16, saves, skip: n => (n >= 14 && n <= 21) || (n >= 38 && n <= 41) }),
      }))
      expect(saves.length).toBeGreaterThan(5)
      expect(saves.every(s => s === 0)).toBe(true)
      const logs = await logsOf(idb)
      expect(logs[0]).toMatchObject({ seq: 0, stepsTotal: 22, cu: 22 / 16, skippedUpdates: 8 })
      expect(logs[1]).toMatchObject({ seq: 1, stepsTotal: 32 })
      expect(logs.at(-1)).toMatchObject({ final: true, skippedUpdates: 12 })
      expectContinuous(logs)
    })

    it("after 20 consecutive skipped updates the learner is restored from the last checkpoint and lives on", async () => {
      const idb = new IDBFactory()
      const created: LearnerInit[] = []
      const events: unknown[] = []
      let poisoned = true
      await runSession(cfgFor({ lifetimeCU: 16, sleepInterval: 50 }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 16, created, skip: n => poisoned && n >= 120 }),
        onEvent: e => { events.push(e); if (e.type === "restored") poisoned = false },
      }))
      expect(events).toEqual([{ type: "restored", fromSteps: 139, toSteps: 100 }])
      expect(created.length).toBe(2)
      expect("state" in created[1]! && (created[1].state as unknown as StubState).stats.steps).toBe(100)
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.map(f => f.cu)).toEqual([1, 2, 4, 8, 16])
      expect(logs.at(-1)!.skippedUpdates).toBe(0) // the restored learner never skipped
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("a learner that keeps failing after restores stops with an error instead of looping", async () => {
      const idb = new IDBFactory()
      await expect(runSession(cfgFor({ lifetimeCU: 16, sleepInterval: 50 }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 16, skip: n => n >= 60 }),
      }))).rejects.toThrow(/restore/)
      const logs = await logsOf(idb) // whatever was written is valid
      expect(logs.map(f => f.seq)).toEqual([0, 1]) // the 4 CU crossing (step 64) falls in the skips: never logged
      for (const f of logs) expect(parseLogFile(f).ok).toBe(true)
    })
  })

  describe("speed cap", () => {
    it("holds the loop to maxStepsPerSec (fake clock): consecutive steps start at least 1/maxStepsPerSec apart", async () => {
      const idb = new IDBFactory()
      const clock = fakeClock()
      const at: number[] = []
      await runSession(cfgFor({ lifetimeCU: 300 / 16, sleepInterval: 50, maxStepsPerSec: 20 }), depsFor(idb, {
        clock, createLearner: stubFactory({ cuPerStep: 1 / 16 }), onTick: () => { at.push(clock.now()) },
      }))
      expect(at.length).toBe(300)
      for (let i = 1; i < at.length; i++) expect(at[i]! - at[i - 1]!).toBeGreaterThanOrEqual(50 - 1e-9)
      expect(at.at(-1)! - at[0]!).toBeCloseTo(299 * 50, 6)
      expect((await logsOf(idb)).at(-1)!.settings.maxStepsPerSec).toBe(20)
    })

    it("adds no wait when a step already took longer than the interval", async () => {
      const clock = fakeClock()
      const pace = createPacer(30, clock)
      await pace()
      for (let i = 0; i < 5; i++) { clock.t += 100; await pace() } // each step took 100 ms > 33 ms
      expect(clock.sleeps.slice(1)).toEqual([0, 0, 0, 0, 0])
      clock.t += 10
      await pace()
      expect(clock.sleeps.at(-1)).toBeCloseTo(1000 / 30 - 10, 9)
    })
  })

  it("rejects bad configuration", async () => {
    for (const over of [{ lifetimeCU: 0 }, { lifetimeCU: Number.NaN }, { sleepInterval: 0 }, { sleepInterval: 2.5 }, { maxStepsPerSec: 0 }, { maxStepsPerSec: Number.POSITIVE_INFINITY }]) {
      await expect(runSession(cfgFor(over), depsFor(new IDBFactory(), { createLearner: stubFactory({ cuPerStep: 1 }) })), JSON.stringify(over)).rejects.toThrow(RangeError)
    }
  })
})
