import { describe, it, expect, vi } from "vitest"
import { IDBFactory, IDBObjectStore as FakeObjectStore } from "fake-indexeddb"
import { NOOP_ACTION, emptyObservation, decodeMove, PERIPHERY, WORLD, type Action, type Observation, type WorldView } from "@blanktab/interface"
import { createCpuBackend, createRng } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, clampSettings, createLearner, type FrozenRun, type FrozenRunOptions, type LearnerSettings, type LearnerState, type LearnerStats, type Widths } from "@blanktab/learner"
import { openQueue } from "../src/queue"
import { loadCheckpoint, readCheckpointRecord, saveCheckpoint } from "../src/checkpoint"
import { MAX_CHECKUPS_PER_FILE, MAX_RESTORES, MILESTONE_IDS, MILESTONE_SUITE_VERSION, levelAtOrBelow, parseLogFile, type LogFile, type LogFileV2 } from "../src/log"
import { createWorld, hashSeed, type TraceEvent } from "@blanktab/world"
import { uniformAction } from "../src/heldout"
import { MILESTONES_V1, type Milestone } from "../src/milestones"
import { CAREGIVER_DEFAULTS, clampCaregiverSettings } from "../src/caregiver"
import { validateSessionPr } from "../../../tools/gate/validate"
import { createCanvas } from "@napi-rs/canvas"
import { OFFLINE_KNOWLEDGE_SHA, type Manifest } from "../src/knowledge"
import { buildHeldOut } from "../src/heldout"
import { DEFAULT_CHECKPOINT_EVERY_MS, curvePoint, DEFAULT_MAX_STEPS_PER_SEC, FIRST_LOG_CU, LAST_DOUBLING_CU, SessionInbox, logThresholds, nextCrossing, runSession, createPacer, type Clock, type LearnerInit, type SessionConfig, type SessionDeps, type SessionEvent, type SessionLearner } from "../src/session"
import { mkFile } from "./helpers"
import { pre04cCheckpointError } from "./pre04c-checkpoint"

// Plan 04b: a spy on every caregiver the session makes (each delegates to a real one), in one call log with the events.
const cgSpy = vi.hoisted(() => ({ log: [] as string[], inits: [] as unknown[] }))
vi.mock("../src/caregiver", async importOriginal => {
  const m = await importOriginal<typeof import("../src/caregiver")>()
  return {
    ...m,
    createCaregiver: (init: Parameters<typeof m.createCaregiver>[0]) => {
      cgSpy.inits.push(structuredClone(init))
      const c = m.createCaregiver(init)
      const spy = <K extends keyof typeof c>(k: K): (typeof c)[K] => ((...a: unknown[]) => {
        cgSpy.log.push(k)
        return (c[k] as (...x: unknown[]) => unknown)(...a)
      }) as (typeof c)[K]
      return { observe: spy("observe"), decide: spy("decide"), onCheckup: spy("onCheckup"), takeSummary: spy("takeSummary"), state: spy("state"), noteLost: spy("noteLost") }
    },
  }
})

const ID = "lrn_0123456789abcdef"
const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const CPU_HALF: Widths = { conv1: 8, conv2: 16, enc: 64, hidden: 128 }
const SETTINGS: LearnerSettings = { ...DEFAULT_SETTINGS, lr: 0.5, replayPerSleep: 8 } // lr is clamped to 1e-2

const cfgFor = (over: Partial<SessionConfig> = {}): SessionConfig => ({
  learnerId: ID, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 3,
  hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 1 },
  // A checkpoint at every clean sleep unless a test says otherwise (the throttle has its own tests).
  seed: 7, widths: TINY, learnerSettings: SETTINGS, checkpointEveryMs: 0, ...over,
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
interface StubOpts { cuPerStep: number; sleepCu?: number; nanUntil?: number; skip?: (steps: number) => boolean; saves?: number[]; savedSteps?: number[]; created?: LearnerInit[]; throwAt?: { steps: number; error: Error }; knowledge?: [number, string][]; onSave?: (steps: number) => void; evalCu?: number; heldOutLoss?: (steps: number) => number; frozen?: { evalPerStep: number } }
function stubFactory(o: StubOpts) {
  return async (init: LearnerInit): Promise<SessionLearner> => {
    o.created?.push(init)
    const s0 = "state" in init ? (init.state as unknown as StubState) : { seed: init.seed, widths: init.widths, settings: init.settings, stats: { steps: 0, cu: 0, predLoss: Number.NaN, baselineLoss: Number.NaN, skippedUpdates: 0, consecutiveSkips: 0, sleeps: 0, evalCU: 0, heldOutLoss: Number.NaN, heldOutBaseline: Number.NaN, lastLoss: Number.NaN } }
    const st: StubState = structuredClone(s0)
    const loss = () => {
      const n = st.stats.steps
      st.stats.predLoss = n < (o.nanUntil ?? 2) ? Number.NaN : 0.5 / (1 + n / 100)
      st.stats.baselineLoss = n < (o.nanUntil ?? 2) ? Number.NaN : 0.4
      if (o.frozen) st.stats.lastLoss = st.stats.predLoss // the caregiver's signal (Plan 04b)
    }
    return {
      step: async () => {
        if (o.throwAt && st.stats.steps + 1 === o.throwAt.steps) throw o.throwAt.error
        st.stats.steps++
        st.stats.cu += o.cuPerStep
        if (o.skip?.(st.stats.steps)) { st.stats.skippedUpdates++; st.stats.consecutiveSkips++ } else st.stats.consecutiveSkips = 0
        loss()
        return NOOP_ACTION
      },
      sleep: async () => {
        st.stats.sleeps++
        st.stats.cu += o.sleepCu ?? 0
        // With `evalCu`, a learner with a held-out set: each sleep scores it (a pure function of the step count), billed to evalCU.
        if (o.evalCu !== undefined) {
          st.stats.evalCU += o.evalCu
          st.stats.heldOutLoss = o.heldOutLoss ? o.heldOutLoss(st.stats.steps) : 0.4 / (1 + st.stats.steps / 1000)
          st.stats.heldOutBaseline = 0.4
        }
      },
      save: async () => { o.onSave?.(st.stats.steps); o.saves?.push(st.stats.consecutiveSkips); o.savedSteps?.push(st.stats.steps); return structuredClone(st) as unknown as LearnerState },
      stats: () => ({ ...st.stats }),
      dispose: () => {},
      ...(o.knowledge ? { onKnowledge: (m: Manifest) => void o.knowledge!.push([st.stats.steps, m.knowledgeSha]) } : {}),
      // With `frozen`, a frozen run (Plan 04b checkups): uniform actions from its own seed, `evalPerStep` evalCU a step, nothing else changes.
      ...(o.frozen ? { frozenRun: (fo: FrozenRunOptions): FrozenRun => {
        const rnd = createRng(fo.seed)
        return {
          step: async () => { st.stats.evalCU += o.frozen!.evalPerStep; return uniformAction(rnd) },
          result: () => ({ predLoss: Number.NaN, baselineLoss: Number.NaN, transitions: 0 }),
          dispose: () => {},
        }
      } } : {}),
    }
  }
}

/** What the page posts on a knowledge refresh: the manifest only (its knowledgeSha is what the logs record). */
const knowledgeWith = (knowledgeSha: string): Manifest => ({ knowledgeSha, codeSha: "dev", learnerStateVersion: 1 })

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

  describe("the held-out check (decision 0015 §3)", () => {
    it("curvePoint: a sleep's point, with the held-out score when there is one; null while a loss is NaN", () => {
      const st = { cu: 2, predLoss: 0.1, baselineLoss: 0.2, heldOutLoss: 0.3, heldOutBaseline: 0.4 }
      expect(curvePoint(st, 50)).toEqual({ cu: 2, steps: 50, predLoss: 0.1, baselineLoss: 0.2, heldOut: { loss: 0.3, baseline: 0.4 } })
      expect(curvePoint({ ...st, heldOutLoss: Number.NaN, heldOutBaseline: Number.NaN }, 50)).toEqual({ cu: 2, steps: 50, predLoss: 0.1, baselineLoss: 0.2 })
      expect(curvePoint({ ...st, heldOutLoss: Number.POSITIVE_INFINITY }, 50)).toEqual({ cu: 2, steps: 50, predLoss: 0.1, baselineLoss: 0.2 })
      expect(curvePoint({ ...st, predLoss: Number.NaN }, 50)).toBeNull()
      expect(curvePoint({ ...st, baselineLoss: Number.NaN }, 50)).toBeNull()
    })

    it("each curve point carries the held-out score and each file its evalCU, which is not in cu", async () => {
      const idb = new IDBFactory()
      await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16, evalCu: 0.25 }) }))
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.map(f => f.cu)).toEqual([8, 16, 32, 64]) // as without the check
      expect(logs.map(f => f.evalCU)).toEqual(logs.map(f => 0.25 * Math.floor(f.stepsTotal / 50)))
      // Only a sleep's point carries a score (its own): the end-of-life point, between sleeps, has none.
      for (const f of logs) for (const p of f.curve) {
        if (p.steps % 50 === 0) expect(p.heldOut).toEqual({ loss: 0.4 / (1 + p.steps / 1000), baseline: 0.4 })
        else expect("heldOut" in p, `point at step ${p.steps}`).toBe(false)
      }
      const end = logs.at(-1)!.curve.at(-1)!
      expect(end.steps % 50).not.toBe(0) // this life ends between sleeps: the case above is exercised
    })

    it("a diverged learner's huge (finite) held-out loss is written and the file still parses: unfavourable results are kept", async () => {
      const huge = { cu: 1, predLoss: 0.1, baselineLoss: 0.2, heldOutLoss: 1e30, heldOutBaseline: 2.5 }
      expect(curvePoint(huge, 10)!.heldOut).toEqual({ loss: 1e30, baseline: 2.5 })
      expect(curvePoint({ ...huge, heldOutLoss: -1 }, 10)!.heldOut).toBeUndefined() // never a value the parser refuses
      expect(curvePoint({ ...huge, predLoss: -1 }, 10)).toBeNull()
      const idb = new IDBFactory()
      await runSession(cfgFor({ lifetimeCU: 16, sleepInterval: 50 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16, evalCu: 0.25, heldOutLoss: n => 1e6 * n }) }))
      const logs = await logsOf(idb)
      expectContinuous(logs) // every file parses
      expect(logs.flatMap(f => f.curve).some(p => (p.heldOut?.loss ?? 0) > 1e8)).toBe(true)
    })

    it("a failed life whose checkpoint has no evalCU keeps the previous file's (never from a number to absent)", async () => {
      const idb = new IDBFactory()
      const cfg = cfgFor({ lifetimeCU: 128, sleepInterval: 10 })
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, evalCu: 0.5 }), onTick: ({ steps }) => { if (steps === 25) throw new Error("reload") } }))).rejects.toThrow("reload")
      const cp = (await loadCheckpoint(idb, ID))!
      const prevEval = (await logsOf(idb)).at(-1)!.evalCU!
      expect(prevEval).toBeGreaterThan(0)
      const st = structuredClone(cp.learnerState) as unknown as { stats: Record<string, unknown> }
      delete st.stats.evalCU // a checkpoint whose learner state has none (an older or stub state)
      await saveCheckpoint(idb, { ...cp, learnerState: st })
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, throwAt: { steps: cp.steps + 3, error: new Error("boom") } }) }))).rejects.toThrow("boom")
      const last = (await logsOf(idb)).at(-1)!
      expect(last.final).toBe(true)
      expect(last.evalCU).toBe(prevEval)
    })

    it("a learner without a held-out set writes points without heldOut and evalCU 0", async () => {
      const idb = new IDBFactory()
      await runSession(cfgFor({ lifetimeCU: 16, sleepInterval: 50 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16 }) }))
      const logs = await logsOf(idb)
      expect(logs.map(f => f.evalCU)).toEqual([0, 0])
      for (const f of logs) for (const p of f.curve) expect("heldOut" in p).toBe(false)
    })

    it("a failed life's final file keeps the evalCU of the checkpoint it is built from (never below the last file's)", async () => {
      const idb = new IDBFactory()
      const cfg = cfgFor({ lifetimeCU: 128, sleepInterval: 10 })
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, evalCu: 0.5, throwAt: { steps: 60, error: new Error("boom") } }) }))).rejects.toThrow("boom")
      const logs = await logsOf(idb)
      expectContinuous(logs)
      const last = logs.at(-1)!
      expect(last.final).toBe(true)
      expect(last.evalCU).toBe(0.5 * Math.floor(last.stepsTotal / 10))
      expect(last.evalCU!).toBeGreaterThanOrEqual(logs.at(-2)!.evalCU!)
    })

    it("a real CPU learner with the real held-out set: points carry heldOut, files evalCU, all valid; cu excludes the scoring", async () => {
      const run = async (withSet: boolean) => {
        const idb = new IDBFactory()
        const make = (init: LearnerInit) => createLearner({
          backend: createCpuBackend(), ...init, ...(withSet && !("state" in init) ? { heldOut: buildHeldOut(init.seed, createCanvas(WORLD.width, WORLD.height)) } : {}),
        })
        const world = createWorld({ seed: 7, canvas: createCanvas(WORLD.width, WORLD.height) })
        await runSession(cfgFor({ lifetimeCU: 0.2, sleepInterval: 10 }), depsFor(idb, { createLearner: make, world }))
        return logsOf(idb)
      }
      const [withSet, without] = [await run(true), await run(false)]
      expectContinuous(withSet)
      const f = withSet.at(-1)!
      expect(f.curve.length).toBeGreaterThan(2)
      for (const p of f.curve) {
        if (p.steps % 10 !== 0) { expect("heldOut" in p).toBe(false); continue } // the end of the life, between sleeps
        expect(p.heldOut!.baseline).toBeGreaterThan(0)
        expect(Number.isFinite(p.heldOut!.loss)).toBe(true)
      }
      expect(f.evalCU!).toBeGreaterThan(0)
      // The scoring is not billed to training: the same life in steps and CU as without the check.
      expect(withSet.map(x => [x.stepsTotal, x.cu])).toEqual(without.map(x => [x.stepsTotal, x.cu]))
      expect(without.at(-1)!.evalCU).toBe(0)
    })
  })
  it("the log schedule: 8 ... 512 by doubling, then every 1024, forever", () => {
    expect(FIRST_LOG_CU).toBe(8)
    const first: number[] = []
    for (const t of logThresholds()) { first.push(t); if (first.length === 12) break }
    expect(first).toEqual([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120])
    expect([0, 0.5, 7.99, 8, 15, 16, 511, 512, 1000, 1024, 1535, 2048, 10_000].map(nextCrossing)).toEqual([8, 8, 8, 16, 16, 32, 512, 1024, 1024, 2048, 2048, 3072, 10_240])
    // nextCrossing is the schedule's next value above cu.
    const all = logThresholds()[Symbol.iterator]()
    const seen: number[] = [0]
    for (let i = 0; i < 40; i++) seen.push(all.next().value as number)
    for (let i = 1; i < seen.length; i++) expect(nextCrossing(seen[i - 1]!)).toBe(seen[i])
  })

  it("writes log files exactly at the 2^k CU crossings from 8 CU plus the final one, all v2-valid", async () => {
    const idb = new IDBFactory()
    const events: SessionEvent[] = []
    const r = await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16 }), onEvent: e => events.push(e) }))
    const logs = await logsOf(idb)
    expect(logs.map(f => f.cu)).toEqual([8, 16, 32, 64])
    // Each file is announced once it is queued (the page flushes the outbox on the final one).
    expect(events).toEqual(logs.map(f => ({ type: "logged", seq: f.seq, steps: f.stepsTotal, cu: f.cu, final: f.final })))
    expect(logs.map(f => f.stepsTotal)).toEqual([128, 256, 512, 1024])
    expect(logs.map(f => f.final)).toEqual([false, false, false, true])
    expectContinuous(logs)
    expect(r).toMatchObject({ steps: 1024, seq: 4, resumedFrom: null })
    const f = logs[1]!
    expect(f).toMatchObject({
      learnerId: ID, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 3, condition: "A", widths: TINY, skippedUpdates: 0,
      hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 1 }, writtenAt: new Date(1_700_000_000_000).toISOString(),
    })
    // settings: the learner settings as the learner uses them (lr clamped), the speed cap and the seed.
    expect(f.settings).toEqual({ ...clampSettings(SETTINGS), maxStepsPerSec: 30, seed: 7 })
    expect(f.settings.lr).toBe(1e-2)
    expect(f.settings.gamma).toBe(0.9) // Plan 05a: gamma is a learner setting, logged with the rest
    // The curve: a point at every sleep (every 50 steps) and one at the end of life, each in the file that follows it.
    expect(logs.flatMap(l => l.curve.map(p => p.steps))).toEqual([...Array.from({ length: 20 }, (_, i) => 50 * (i + 1)), 1024])
    expect(logs[0]!.curve.map(p => p.steps)).toEqual([50, 100])
    expect(logs[3]!.curve.map(p => p.steps)).toEqual([550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1024])
    expect(logs[3]!.curve[0]).toEqual({ cu: 550 / 16, steps: 550, predLoss: 0.5 / 6.5, baselineLoss: 0.4 })
  })

  it("without a lifetime the life never ends: no final log, however long it runs", async () => {
    const idb = new IDBFactory()
    await expect(runSession(cfgFor(), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 }), onTick: ({ cu }) => { if (cu >= 1100) throw new Error("tab closed") },
    }))).rejects.toThrow("tab closed")
    const logs = await logsOf(idb)
    expect(logs.map(f => f.cu)).toEqual([8, 16, 32, 64, 128, 256, 512, 1024])
    expect(logs.some(f => f.final)).toBe(false)
    expect(await loadCheckpoint(idb, ID)).toBeDefined()
  })

  it("a learner adding 10,240 CU writes exactly 7 + 10 = 17 files (8 ... 512, then 1024 ... 10240 by 1024), none final, each under 1 MB", async () => {
    const idb = new IDBFactory()
    await expect(runSession(cfgFor(), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 8 }), onTick: ({ cu }) => { if (cu > 10_240) throw new Error("tab closed") },
    }))).rejects.toThrow("tab closed")
    const logs = await logsOf(idb)
    expect(logs).toHaveLength(17)
    expect(logs.map(f => f.cu)).toEqual([8, 16, 32, 64, 128, 256, 512, ...Array.from({ length: 10 }, (_, i) => 1024 * (i + 1))])
    expect(logs.some(f => f.final)).toBe(false)
    expect(logs.map(f => f.seq)).toEqual(logs.map((_, i) => i))
    for (const f of logs) {
      expect(parseLogFile(JSON.parse(JSON.stringify(f))).ok).toBe(true)
      expect(new TextEncoder().encode(JSON.stringify(f)).length).toBeLessThanOrEqual(1_000_000)
    }
  })

  it("at the speed cap (30 steps/s) no hour holds more than 10 logs past the first 7 early ones", () => {
    // The speed cap in CU/s: DEFAULT_MAX_STEPS_PER_SEC steps/s at the most CU a step costs at the widest widths (0.08, which gives the 2.4 CU/s the schedule was sized for).
    const MAX_CU_PER_STEP = 0.08
    const CU_PER_SEC = DEFAULT_MAX_STEPS_PER_SEC * MAX_CU_PER_STEP
    const times: number[] = []
    for (const t of logThresholds()) { if (t > 200_000) break; if (t > LAST_DOUBLING_CU) times.push(t / CU_PER_SEC) }
    let most = 0
    for (let i = 0; i < times.length; i++) most = Math.max(most, times.filter(x => x >= times[i]! && x < times[i]! + 3600).length)
    expect(most).toBeLessThanOrEqual(10)
    expect(most).toBe(9) // 8640 CU an hour / 1024
  })

  it("a test lifetime still ends in a final log", async () => {
    const idb = new IDBFactory()
    const r = await runSession(cfgFor({ lifetimeCU: 1100 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 4 }) }))
    const logs = await logsOf(idb)
    expect(logs.map(f => f.cu)).toEqual([8, 16, 32, 64, 128, 256, 512, 1024, 1100])
    expect(logs.at(-1)).toMatchObject({ final: true, cu: 1100 })
    expect(r.cu).toBe(1100)
    expectContinuous(logs)
    expect(await loadCheckpoint(idb, ID)).toBeUndefined()
  })

  it("CU a sleep adds counts too: a crossing at a sleep is logged with that sleep's curve point", async () => {
    const idb = new IDBFactory()
    await runSession(cfgFor({ lifetimeCU: 32, sleepInterval: 10 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 0, sleepCu: 8 }) }))
    const logs = await logsOf(idb)
    expect(logs.map(f => [f.stepsTotal, f.cu, f.final])).toEqual([[10, 8, false], [20, 16, false], [40, 32, true]])
    expect(logs.map(f => f.curve.map(p => p.steps))).toEqual([[10], [20], [30, 40]])
  })

  it("never writes NaN: curve points are omitted until both losses are finite", async () => {
    const idb = new IDBFactory()
    await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 10 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, nanUntil: 35 }) }))
    const logs = await logsOf(idb)
    expect(logs.flatMap(l => l.curve.map(p => p.steps)).slice(0, 2)).toEqual([40, 50])
    expect(logs[0]!.curve).toEqual([]) // seq 0 at step 16 (8 CU): nothing finite yet
    for (const f of logs) expect(JSON.stringify(f)).not.toMatch(/null|NaN/)
  })

  it("a log file's curve is thinned to at most 64 points, keeping the newest; so is the checkpoint's tail", async () => {
    const idb = new IDBFactory()
    const tails: number[] = []
    await runSession(cfgFor({ lifetimeCU: 16, sleepInterval: 1 }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 8 / 1024 }),
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
        const events: SessionEvent[] = []
        const r = await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50 }), depsFor(idb, {
          createLearner: stubFactory({ cuPerStep: 1 / 16 }), onTick: ({ steps }) => { ticks.push(steps) }, onEvent: e => events.push(e),
        }))
        expect(r.resumedFrom).toBe(cp!.steps)
        // The page redraws the curve from the event: the checkpoint's place in the life and its curve since the last log.
        expect(events[0]).toEqual({ type: "resumed", steps: cp!.steps, cu: cp!.cu, curveTail: cp!.curveTail, checkupTail: [] })
        expect(ticks[0]).toBe(cp!.steps + 1)
        const logs = await logsOf(idb)
        expectContinuous(logs)
        expect(logs.slice(0, before.length)).toEqual(before) // nothing rewritten
        expect(logs.map(strip)).toEqual((await logsOf(whole)).map(strip))
        expect(await loadCheckpoint(idb, ID)).toBeUndefined()
      })
    }

    it("a real CPU learner in a synthetic world: interrupted, resumed, logs continuous and valid", async () => {
      const idb = new IDBFactory()
      // The CPU fallback widths, not TINY: many more FLOPs per op, so the 8 CU crossing comes in a few hundred steps.
      const cfg = cfgFor({ lifetimeCU: 10, sleepInterval: 50, widths: CPU_HALF, learnerSettings: { ...DEFAULT_SETTINGS, replayPerSleep: 8, bufferSize: 200 } })
      const make = (init: LearnerInit) => createLearner({ backend: createCpuBackend(), ...init })
      let stoppedAt = -1
      await expect(runSession(cfg, depsFor(idb, {
        createLearner: make,
        onTick: ({ steps, cu }) => { if (cu > 8.5 && steps % 50 === 20) { stoppedAt = steps; throw new Error("reload") } },
      }))).rejects.toThrow("reload")
      const before = await logsOf(idb)
      expect(before.map(f => f.cu)).toEqual([expect.any(Number)]) // the 8 CU file
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
      expect(logs.length).toBe(2) // 8 CU, final at 10 CU
      expect(logs[0]!.cu).toBeGreaterThanOrEqual(8)
      expect(logs[1]!.cu).toBeGreaterThanOrEqual(10)
      const curve = logs.flatMap(f => f.curve)
      expect(curve.length).toBeGreaterThan(5)
      for (const p of curve) expect(Number.isFinite(p.predLoss) && Number.isFinite(p.baselineLoss)).toBe(true)
      expect(logs[1]!.widths).toEqual(CPU_HALF)
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })
  })

  it("resumes from a checkpoint record already read (the sweep's adopted one) without reading it again", async () => {
    const run = (idb: IDBFactory, over: Partial<SessionDeps> = {}, stopAt?: number) => runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50 }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 / 16 }), onTick: ({ steps }) => { if (steps === stopAt) throw new Error("reload") }, ...over,
    }))
    const whole = new IDBFactory()
    await run(whole)
    const idb = new IDBFactory()
    await expect(run(idb, {}, 300)).rejects.toThrow("reload")
    const record = await readCheckpointRecord(idb, ID)
    const reads: unknown[] = []
    const get = FakeObjectStore.prototype.get
    FakeObjectStore.prototype.get = function (this: IDBObjectStore, key: IDBValidKey | IDBKeyRange) {
      if (this.name === "checkpoints") reads.push(key)
      return get.call(this, key)
    } as typeof get
    let r: Awaited<ReturnType<typeof runSession>>
    try {
      r = await run(idb, { checkpointRecord: record })
    } finally {
      FakeObjectStore.prototype.get = get
    }
    expect(reads).toEqual([])
    expect(r.resumedFrom).toBe(256) // the 16 CU crossing
    expect((await logsOf(idb)).map(strip)).toEqual((await logsOf(whole)).map(strip))
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
    await saveCheckpoint(idb, { learnerId: ID, savedAt: "x", steps: 1, cu: 0, seq: 9, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0, learnerState: {} as LearnerState })
    const ticks: number[] = []
    const r = await runSession(cfgFor({ lifetimeCU: 4 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 }), onTick: ({ steps }) => { ticks.push(steps) } }))
    expect(ticks).toEqual([])
    expect(r.alreadyFinal).toBe(true)
    expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    expect((await logsOf(idb)).length).toBe(1) // a 4 CU life: only its final file
  })

  it("refuses to resume a learner with log files but no checkpoint, or a checkpoint out of step with its log files", async () => {
    const idb = new IDBFactory()
    await (await openQueue(idb)).put(mkFile(ID, 0))
    await expect(runSession(cfgFor(), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 }) }))).rejects.toThrow(/no checkpoint/)
    for (const seq of [0, 2, 5]) { // only seq 1 (= latest + 1) is consistent
      await saveCheckpoint(idb, { learnerId: ID, savedAt: "x", steps: 1, cu: 0, seq, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0, learnerState: { seed: 7, widths: TINY, settings: clampSettings(SETTINGS) } as unknown as LearnerState })
      await expect(runSession(cfgFor(), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 }) })), `seq ${seq}`).rejects.toThrow(/seq/)
    }
  })

  it("resumed before its first log file, a learner keeps the knowledgeSha and sessionsSeenAtBirth it was born with", async () => {
    const idb = new IDBFactory()
    const run = (over: Partial<SessionConfig>, stopAt?: number) => runSession(cfgFor({ lifetimeCU: 32, sleepInterval: 5, ...over }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 / 2 }), onTick: ({ steps }) => { if (steps === stopAt) throw new Error("reload") },
    }))
    await expect(run({}, 10)).rejects.toThrow("reload") // the first log file would be at step 16 (8 CU)
    expect(await logsOf(idb)).toEqual([])
    expect(await loadCheckpoint(idb, ID)).toMatchObject({ knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 3 })
    await run({ knowledgeSha: "e".repeat(64), sessionsSeenAtBirth: 9 }) // the knowledge changed before the reload
    const logs = await logsOf(idb)
    expect(logs.length).toBeGreaterThan(1)
    for (const f of logs) expect(f).toMatchObject({ knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 3 })
  })

  describe("knowledge refreshed during a life (inbox.setKnowledge)", () => {
    // 1/2 CU a step: logs at steps 16 (8 CU), 32 (16 CU), 64 (32 CU) and the final one at 128 (64 CU).
    const live = (idb: IDBFactory, inbox: SessionInbox, o: { at?: Record<number, Manifest>; stopAt?: number; knowledge?: [number, string][] } = {}) =>
      runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 5 }), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 1 / 2, ...(o.knowledge ? { knowledge: o.knowledge } : {}) }),
        onTick: ({ steps }) => {
          const k = o.at?.[steps]
          if (k) inbox.setKnowledge(k)
          if (steps === o.stopAt) throw new Error("reload")
        },
      }))

    it("a new knowledgeSha: a log under the old one first (with its checkpoint), then the new one in every later file; the learner is handed the knowledge", async () => {
      const idb = new IDBFactory()
      const handed: [number, string][] = []
      await live(idb, new SessionInbox(), { at: { 20: knowledgeWith("e".repeat(64)) }, knowledge: handed })
      const logs = await logsOf(idb)
      expect(logs.map(f => [f.stepsTotal, f.knowledgeSha[0]])).toEqual([[16, "f"], [20, "f"], [32, "e"], [64, "e"], [128, "e"]])
      expect(logs.every(f => f.sessionsSeenAtBirth === 3)).toBe(true) // still a birth fact
      expect(handed).toEqual([[20, "e".repeat(64)]]) // before step 21, once
      expectContinuous(logs)
    })

    it("offline knowledge (a failed refresh) never replaces the knowledge in force", async () => {
      const idb = new IDBFactory()
      await live(idb, new SessionInbox(), { at: { 20: knowledgeWith("e".repeat(64)), 40: knowledgeWith(OFFLINE_KNOWLEDGE_SHA) } })
      expect((await logsOf(idb)).map(f => f.knowledgeSha[0])).toEqual(["f", "f", "e", "e", "e"])
    })

    it("the same knowledgeSha again (new content, same priors) writes no boundary log; the learner is still handed it", async () => {
      const idb = new IDBFactory()
      const handed: [number, string][] = []
      await live(idb, new SessionInbox(), { at: { 20: knowledgeWith("f".repeat(64)) }, knowledge: handed })
      expect((await logsOf(idb)).map(f => f.stepsTotal)).toEqual([16, 32, 64, 128])
      expect(handed).toEqual([[20, "f".repeat(64)]])
    })

    it("no log file holds curve points from two knowledge versions", async () => {
      const idb = new IDBFactory()
      // Changes at steps 23 and 41 (off the crossings at 16, 32 and 64; sleeps every 5 steps).
      await live(idb, new SessionInbox(), { at: { 23: knowledgeWith("e".repeat(64)), 41: knowledgeWith("d".repeat(64)) } })
      const logs = await logsOf(idb)
      expect(logs.map(f => [f.stepsTotal, f.knowledgeSha[0]])).toEqual([[16, "f"], [23, "f"], [32, "e"], [41, "e"], [64, "d"], [128, "d"]])
      const switchedAt = { f: 0, e: 23, d: 41 } as Record<string, number>
      const until = { f: 23, e: 41, d: Infinity } as Record<string, number>
      for (const f of logs) {
        const k = f.knowledgeSha[0]!
        for (const p of f.curve) {
          expect(p.steps, `${k} @ ${f.stepsTotal}`).toBeGreaterThan(switchedAt[k]!)
          expect(p.steps, `${k} @ ${f.stepsTotal}`).toBeLessThanOrEqual(until[k]!)
        }
      }
      expectContinuous(logs)
    })

    it("a change that comes while skipped updates are outstanding waits for them to clear (the boundary log needs a checkpoint)", async () => {
      const idb = new IDBFactory()
      const handed: [number, string][] = []
      const inbox = new SessionInbox()
      await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 5 }), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 1 / 2, knowledge: handed, skip: n => n >= 20 && n <= 24 }),
        onTick: ({ steps }) => { if (steps === 20) inbox.setKnowledge(knowledgeWith("e".repeat(64))) },
      }))
      const logs = await logsOf(idb)
      expect(logs.map(f => [f.stepsTotal, f.knowledgeSha[0]])).toEqual([[16, "f"], [25, "f"], [32, "e"], [64, "e"], [128, "e"]])
      expect(handed).toEqual([[25, "e".repeat(64)]])
    })

    it("is kept by the checkpoint: a reload before the next log file still writes it", async () => {
      const idb = new IDBFactory()
      await expect(live(idb, new SessionInbox(), { at: { 20: knowledgeWith("e".repeat(64)) }, stopAt: 25 })).rejects.toThrow("reload")
      expect(await loadCheckpoint(idb, ID)).toMatchObject({ steps: 20, seq: 2, knowledgeSha: "e".repeat(64) }) // saved with the boundary log at step 20
      await live(idb, new SessionInbox())
      expect((await logsOf(idb)).map(f => f.knowledgeSha[0])).toEqual(["f", "f", "e", "e", "e"])
    })

    it("knowledge given before a resumed life starts: the points the checkpoint carries are logged under its own sha first", async () => {
      const idb = new IDBFactory()
      await expect(live(idb, new SessionInbox(), { stopAt: 12 })).rejects.toThrow("reload")
      expect(await loadCheckpoint(idb, ID)).toMatchObject({ steps: 10, knowledgeSha: "f".repeat(64) }) // the sleep at step 10, two points in its tail
      const inbox = new SessionInbox()
      inbox.setKnowledge(knowledgeWith("e".repeat(64)))
      await live(idb, inbox)
      const logs = await logsOf(idb)
      expect(logs.map(f => [f.stepsTotal, f.knowledgeSha[0]])).toEqual([[11, "f"], [16, "e"], [32, "e"], [64, "e"], [128, "e"]])
      expect(logs[0]!.curve.map(p => p.steps)).toEqual([5, 10])
      expectContinuous(logs)
    })
  })

  it("a resumed learner keeps its own birth facts (seed, widths, settings): the config's are for newborns only", async () => {
    const run = (idb: IDBFactory, over: Partial<SessionConfig>, stopAt?: number) => runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50, ...over }), depsFor(idb, {
      createLearner: stubFactory({ cuPerStep: 1 / 16 }), onTick: ({ steps }) => { if (steps === stopAt) throw new Error("reload") },
    }))
    const whole = new IDBFactory()
    await run(whole, {})
    for (const over of [{ seed: 8 }, { widths: { ...TINY, hidden: 32 } }, { learnerSettings: { ...SETTINGS, bufferSize: 99 } }]) {
      const idb = new IDBFactory()
      await expect(run(idb, {}, 60)).rejects.toThrow("reload")
      await run(idb, over)
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.map(strip)).toEqual((await logsOf(whole)).map(strip))
      expect(logs.at(-1)).toMatchObject({ widths: TINY, settings: { seed: 7, bufferSize: SETTINGS.bufferSize } })
    }
  })

  it("a checkpoint whose learnerState is missing or malformed is a RangeError", async () => {
    const good = { seed: 7, widths: TINY, settings: clampSettings(SETTINGS) }
    for (const learnerState of [null, 3, {}, { ...good, seed: -1 }, { ...good, widths: { ...TINY, enc: 0 } }, { ...good, widths: null }, { ...good, settings: null }, { ...good, settings: { ...good.settings, lr: Number.NaN } }]) {
      const idb = new IDBFactory()
      await saveCheckpoint(idb, { learnerId: ID, savedAt: "x", steps: 1, cu: 0, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0, learnerState: learnerState as unknown as LearnerState })
      await expect(runSession(cfgFor(), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 }) })), JSON.stringify(learnerState)).rejects.toThrow(RangeError)
    }
  })

  it("drains the world trace after every act, so it never grows", async () => {
    const world = createWorld({ seed: 1, canvas: createCanvas(WORLD.width, WORLD.height) })
    const log: string[] = []
    const counted = {
      observe: () => world.observe(),
      act: (a: Action) => { log.push("act"); world.act(a) },
      drainTrace: () => { log.push("drain"); return world.drainTrace() },
    }
    await runSession(cfgFor({ lifetimeCU: 300 / 16, sleepInterval: 50 }), depsFor(new IDBFactory(), { world: counted, createLearner: stubFactory({ cuPerStep: 1 / 16 }) }))
    expect(log).toEqual(Array.from({ length: 300 }, () => ["act", "drain"]).flat())
    expect(world.drainTrace()).toEqual([])
  })

  it("two learners sharing one database: no shared (learnerId, seq), each life continuous", async () => {
    const idb = new IDBFactory()
    const B = "lrn_fedcba9876543210"
    const cfg = { lifetimeCU: 16, sleepInterval: 50 }
    await Promise.all([
      runSession(cfgFor(cfg), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 16 }) })),
      runSession(cfgFor({ ...cfg, learnerId: B, seed: 9 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 8 }) })),
    ])
    const all = (await (await openQueue(idb)).pending()) as LogFileV2[]
    expect(new Set(all.map(f => `${f.learnerId}/${f.seq}`)).size).toBe(all.length)
    const a = all.filter(f => f.learnerId === ID), b = all.filter(f => f.learnerId === B)
    expect(a.map(f => f.cu)).toEqual([8, 16])
    expect(b.map(f => f.cu)).toEqual([8, 16])
    expect(b.map(f => f.settings.seed)).toEqual([9, 9])
    expectContinuous(a)
    expectContinuous(b)
    expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    expect(await loadCheckpoint(idb, B)).toBeUndefined()
  })

  describe("checkpoint throttle (checkpointEveryMs, default 300 s on the injected clock)", () => {
    // 10 steps/s on the fake clock: step n starts at (n - 1) * 100 ms; a sleep every 50 steps; crossings (8, 16, 32 CU)
    // at steps 1024, 2048, 4096; the end of life (64 CU) at step 8192.
    const throttled = (idb: IDBFactory, o: { skip?: (n: number) => boolean; stopAt?: number; savedSteps?: number[]; checkpointEveryMs?: number } = {}) =>
      runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: o.checkpointEveryMs }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 8 / 1024, ...(o.skip ? { skip: o.skip } : {}), ...(o.savedSteps ? { savedSteps: o.savedSteps } : {}) }),
        onTick: ({ steps }) => { if (steps === o.stopAt) throw new Error("reload") },
      }))

    it("the default is 300 s", () => {
      expect(DEFAULT_CHECKPOINT_EVERY_MS).toBe(300_000)
    })

    it("no sleep checkpoint until 300 s after the last save; crossings save with their log regardless", async () => {
      const savedSteps: number[] = []
      await throttled(new IDBFactory(), { savedSteps })
      // birth; the crossings at t = 102.3, 204.7 and 409.5 s (each < 300 s after the save before it); then the first
      // sleep 300 s after the last: step 7100 (t = 709.9 s); the 64 CU crossing is the end of life (the final log removes it).
      expect(savedSteps).toEqual([0, 1024, 2048, 4096, 7100])
    })

    it("a save that falls due while skips are outstanding is taken at the first clean sleep", async () => {
      const savedSteps: number[] = []
      await throttled(new IDBFactory(), { savedSteps, skip: n => n >= 7095 && n <= 7105 })
      expect(savedSteps).toEqual([0, 1024, 2048, 4096, 7150])
    })

    it("checkpointEveryMs 0 saves at every clean sleep; a negative or non-finite value is a RangeError", async () => {
      const savedSteps: number[] = []
      await throttled(new IDBFactory(), { savedSteps, checkpointEveryMs: 0 })
      expect(savedSteps.length).toBe(1 + 163 + 3) // birth, every sleep (50..8150), the 8, 16 and 32 CU crossings
      for (const bad of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
        await expect(throttled(new IDBFactory(), { checkpointEveryMs: bad })).rejects.toThrow(RangeError)
      }
    })

    it("a forced checkpoint (inbox.checkpointNow) is saved at once inside the throttle window, with no log file; a reload resumes from it", async () => {
      const idb = new IDBFactory()
      const inbox = new SessionInbox()
      const savedSteps: number[] = []
      let answer: Promise<boolean> | undefined
      await expect(runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024, savedSteps }),
        onTick: async ({ steps }) => {
          if (steps === 1500) answer = inbox.checkpointNow() // 50 s after the save at the 8 CU crossing (step 1024): well inside the 300 s throttle
          if (steps === 1510) throw new Error("reload")
        },
      }))).rejects.toThrow("reload")
      expect(await answer).toBe(true)
      expect(savedSteps).toEqual([0, 1024, 1500]) // saved off the sleep schedule (1500 is a sleep, but the throttle had 250 s to go)
      expect((await logsOf(idb)).map(f => f.stepsTotal)).toEqual([1024]) // the 8 CU file only: the forced save wrote no log
      const cp = (await loadCheckpoint(idb, ID))!
      expect(cp).toMatchObject({ steps: 1500, seq: 1 })
      const resumed: SessionEvent[] = []
      await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 8 / 1024 }), onEvent: e => resumed.push(e),
      }))
      expect(resumed[0]).toMatchObject({ type: "resumed", steps: 1500 })
      expectContinuous(await logsOf(idb))
    })

    it("a forced checkpoint off a sleep step is saved too", async () => {
      const inbox = new SessionInbox()
      const savedSteps: number[] = []
      let answer: Promise<boolean> | undefined
      await runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024, savedSteps }),
        onTick: ({ steps }) => { if (steps === 333) answer = inbox.checkpointNow() },
      }))
      expect(await answer).toBe(true)
      expect(savedSteps).toEqual([0, 333])
    })

    it("with skipped updates outstanding a forced checkpoint is not saved: answered false, and a later request is saved", async () => {
      const inbox = new SessionInbox()
      const savedSteps: number[] = []
      const answers: Promise<boolean>[] = []
      await runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024, savedSteps, skip: n => n >= 95 && n <= 105 }),
        onTick: ({ steps }) => { if (steps === 100 || steps === 120) answers.push(inbox.checkpointNow()) },
      }))
      expect(await Promise.all(answers)).toEqual([false, true])
      expect(savedSteps).toEqual([0, 120]) // nothing saved during the skips
    })

    it("requests are taken before the save: one made during the save waits for (and is answered by) the next step's", async () => {
      const inbox = new SessionInbox()
      const savedSteps: number[] = []
      let first: Promise<boolean> | undefined, during: Promise<boolean> | undefined
      const settled: string[] = []
      await runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox,
        createLearner: stubFactory({ cuPerStep: 8 / 1024, savedSteps, onSave: steps => {
          if (steps === 333) { during = inbox.checkpointNow(); void during.then(() => settled.push("during")) }
        } }),
        onTick: ({ steps }) => { if (steps === 333) { first = inbox.checkpointNow(); void first.then(() => settled.push("first")) } },
      }))
      expect([await first, await during]).toEqual([true, true])
      expect(savedSteps).toEqual([0, 333, 334]) // the request made during the save at 333 got a save of its own
      expect(settled).toEqual(["first", "during"])
    })

    it("halt: the life stops right after a forced save, at that step, with no final log; the next run resumes there", async () => {
      const idb = new IDBFactory()
      const inbox = new SessionInbox()
      const ticks: number[] = []
      let answer: Promise<boolean> | undefined
      const r = await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024 }),
        onTick: ({ steps }) => { ticks.push(steps); if (steps === 1500) answer = inbox.checkpointNow({ halt: true }) },
      }))
      expect(await answer).toBe(true)
      expect(r).toMatchObject({ steps: 1500, seq: 2, halted: true, alreadyFinal: false })
      expect(ticks.at(-1)).toBe(1500) // not one step more
      // The halting save writes a log with it (one transaction): every point lived on this code is logged on it.
      expect((await logsOf(idb)).map(f => [f.stepsTotal, f.final])).toEqual([[1024, false], [1500, false]])
      expect(await loadCheckpoint(idb, ID)).toMatchObject({ steps: 1500, seq: 2, curveTail: [] })
      const events: SessionEvent[] = []
      const again = await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024 }), onEvent: e => events.push(e),
      }))
      expect(events[0]).toMatchObject({ type: "resumed", steps: 1500 })
      expect(again.halted).toBe(false)
      expectContinuous(await logsOf(idb))
    })

    it("after a halt, every later request (one the page timed out on, or a retry) is answered true at once: the halted save is current", async () => {
      const inbox = new SessionInbox()
      await runSession(cfgFor({ sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024 }),
        onTick: ({ steps }) => { if (steps === 300) void inbox.checkpointNow({ halt: true }) },
      }))
      expect(inbox.halted).toBe(true)
      const answers = await Promise.all([inbox.checkpointNow(), inbox.checkpointNow({ halt: true })])
      expect(answers).toEqual([true, true])
    })

    it("a request made during the halting save is answered true by the halt (no run is left to answer it)", async () => {
      const inbox = new SessionInbox()
      let during: Promise<boolean> | undefined
      await runSession(cfgFor({ sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024, onSave: steps => { if (steps === 300) during = inbox.checkpointNow({ halt: true }) } }),
        onTick: ({ steps }) => { if (steps === 300) void inbox.checkpointNow({ halt: true }) },
      }))
      expect(await during).toBe(true)
    })

    it("new code after a halt: logs before the reload carry the old codeSha, after it the new one; none holds points from both", async () => {
      const idb = new IDBFactory()
      const inbox = new SessionInbox()
      const cfg = (codeSha: string) => cfgFor({ codeSha, lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined })
      const OLD = "a".repeat(40), NEW = "b".repeat(40)
      await runSession(cfg(OLD), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024 }),
        onTick: ({ steps }) => { if (steps === 1700) void inbox.checkpointNow({ halt: true }) },
      }))
      await runSession(cfg(NEW), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 8 / 1024 }) }))
      const logs = await logsOf(idb)
      expect(logs.map(f => [f.stepsTotal, f.codeSha[0]])).toEqual([[1024, "a"], [1700, "a"], [2048, "b"], [4096, "b"], [8192, "b"]])
      for (const f of logs) for (const p of f.curve) expect(f.codeSha === OLD ? p.steps <= 1700 : p.steps > 1700).toBe(true)
      expect(logs[1]!.curve.map(p => p.steps)).toEqual([1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700])
      expectContinuous(logs)
    })

    it("halt with skips outstanding: not saved (false) and the life goes on; a later halt request stops it", async () => {
      const inbox = new SessionInbox()
      const answers: Promise<boolean>[] = []
      const r = await runSession(cfgFor({ lifetimeCU: 4, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024, skip: n => n >= 95 && n <= 105 }),
        onTick: ({ steps }) => { if (steps === 100 || steps === 120) answers.push(inbox.checkpointNow({ halt: true })) },
      }))
      expect(await Promise.all(answers)).toEqual([false, true])
      expect(r).toMatchObject({ steps: 120, halted: true })
    })

    it("a request still waiting when the life ends is answered false", async () => {
      const inbox = new SessionInbox()
      let answer: Promise<boolean> | undefined
      // The request comes at the step the test lifetime ends: the final log removes the checkpoint instead.
      await runSession(cfgFor({ lifetimeCU: 1, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(new IDBFactory(), {
        inbox, createLearner: stubFactory({ cuPerStep: 1 / 4 }), onTick: ({ steps }) => { if (steps === 4) answer = inbox.checkpointNow() },
      }))
      expect(await answer).toBe(false)
      // And one made after the life is over (no session runs it) is answered by the next life to run with this inbox.
      const late = inbox.checkpointNow()
      await runSession(cfgFor({ learnerId: "lrn_00000000000000aa", lifetimeCU: 1, sleepInterval: 1 }), depsFor(new IDBFactory(), { inbox, createLearner: stubFactory({ cuPerStep: 1 / 4 }) }))
      expect(await late).toBe(true)
    })

    for (const stopAt of [1000, 2100, 4200, 7200, 8000]) {
      it(`throttled and interrupted at step ${stopAt}: the resumed life's files equal an uninterrupted life's`, async () => {
        const whole = new IDBFactory()
        await throttled(whole)
        const idb = new IDBFactory()
        await expect(throttled(idb, { stopAt })).rejects.toThrow("reload")
        const cp = (await loadCheckpoint(idb, ID))!
        expect(cp.steps).toBe([0, 1024, 2048, 4096, 7100].filter(s => s < stopAt).at(-1))
        await throttled(idb)
        const logs = await logsOf(idb)
        expectContinuous(logs)
        expect(logs.map(strip)).toEqual((await logsOf(whole)).map(strip))
      })
    }
  })

  describe("skipped updates", () => {
    it("never saves a checkpoint while consecutiveSkips > 0; a crossing during skips is logged once they clear", async () => {
      const idb = new IDBFactory()
      const saves: number[] = []
      await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 10 }), depsFor(idb, {
        // Skips over steps 14-21 (the 8 CU crossing is step 16) and over the sleep at step 40.
        createLearner: stubFactory({ cuPerStep: 1 / 2, saves, skip: n => (n >= 14 && n <= 21) || (n >= 38 && n <= 41) }),
      }))
      expect(saves.length).toBeGreaterThan(5)
      expect(saves.every(s => s === 0)).toBe(true)
      const logs = await logsOf(idb)
      expect(logs[0]).toMatchObject({ seq: 0, stepsTotal: 22, cu: 22 / 2, skippedUpdates: 8 })
      expect(logs[1]).toMatchObject({ seq: 1, stepsTotal: 32 })
      expect(logs.at(-1)).toMatchObject({ final: true, skippedUpdates: 12 })
      expectContinuous(logs)
    })

    it("after 20 consecutive skipped updates the learner is restored from the last checkpoint and lives on", async () => {
      const idb = new IDBFactory()
      const created: LearnerInit[] = []
      const events: unknown[] = []
      let poisoned = true
      await runSession(cfgFor({ lifetimeCU: 128, sleepInterval: 50 }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 2, created, skip: n => poisoned && n >= 120 }),
        onEvent: e => { events.push(e); if (e.type === "restored") poisoned = false },
      }))
      expect(events.filter(e => (e as SessionEvent).type === "restored")).toEqual([{ type: "restored", fromSteps: 139, toSteps: 100, restores: 1, lostCU: 39 / 2 }])
      expect(created.length).toBe(2)
      expect("state" in created[1]! && (created[1].state as unknown as StubState).stats.steps).toBe(100)
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.map(f => f.cu)).toEqual([8, 16, 32, 64, 128])
      // Restores stay on the record: the 20 skips and the 39 steps of CU thrown away are still counted.
      expect(logs.map(f => [f.restores, f.lostCU, f.skippedUpdates])).toEqual([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 39 / 2, 20], [1, 39 / 2, 20]])
      for (const f of logs) expect(f.cu).toBe(f.stepsTotal / 2) // cu is the life the learner lived, lostCU is extra
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("every checkpoint records the code that saved it; a life that fails on other code is ended under the checkpoint's codeSha", async () => {
      const OLD = "a".repeat(40), NEW = "b".repeat(40)
      const idb = new IDBFactory()
      await expect(runSession(cfgFor({ codeSha: OLD, lifetimeCU: 128, sleepInterval: 10 }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 2 }), onTick: ({ steps }) => { if (steps === 25) throw new Error("reload") },
      }))).rejects.toThrow("reload")
      const cp = (await loadCheckpoint(idb, ID))!
      expect(cp.codeSha).toBe(OLD)
      // New code that cannot resume it (e.g. another learner state version): the final log is the old code's life.
      await expect(runSession(cfgFor({ codeSha: NEW, lifetimeCU: 128, sleepInterval: 10 }), depsFor(idb, {
        createLearner: async () => { throw new RangeError("LearnerState: version 1, expected 2") },
      }))).rejects.toThrow("version 1")
      const last = (await logsOf(idb)).at(-1)!
      expect(last).toMatchObject({ final: true, codeSha: OLD, stepsTotal: cp.steps })
    })

    it("a checkpoint from before codeSha was recorded falls back to the latest file's, then the config's", async () => {
      const OLD = "a".repeat(40), NEW = "b".repeat(40)
      for (const withFile of [true, false]) {
        const idb = new IDBFactory()
        await expect(runSession(cfgFor({ codeSha: OLD, lifetimeCU: 128, sleepInterval: 10 }), depsFor(idb, {
          createLearner: stubFactory({ cuPerStep: withFile ? 1 / 2 : 1 / 16 }), onTick: ({ steps }) => { if (steps === 25) throw new Error("reload") },
        }))).rejects.toThrow("reload")
        const cp = (await loadCheckpoint(idb, ID))!
        const { codeSha: _c, ...old } = cp
        void _c
        await saveCheckpoint(idb, old as typeof cp)
        expect((await logsOf(idb)).length > 0).toBe(withFile)
        await expect(runSession(cfgFor({ codeSha: NEW, lifetimeCU: 128, sleepInterval: 10 }), depsFor(idb, {
          createLearner: async () => { throw new RangeError("no") },
        }))).rejects.toThrow("no")
        expect((await logsOf(idb)).at(-1)!.codeSha, `with a file: ${withFile}`).toBe(withFile ? OLD : NEW)
      }
    })

    it("a reload right after a restore still has the restore on the record", async () => {
      const idb = new IDBFactory()
      let poisoned = true, justRestored = false
      const deps = () => depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 2, skip: n => poisoned && n >= 120 }),
        onEvent: e => { if (e.type === "restored") { poisoned = false; justRestored = true } },
        onTick: () => { if (justRestored) { justRestored = false; throw new Error("reload") } },
      })
      await expect(runSession(cfgFor({ lifetimeCU: 128, sleepInterval: 50 }), deps())).rejects.toThrow("reload")
      expect(await loadCheckpoint(idb, ID)).toMatchObject({ steps: 100, restores: 1, lostCU: 39 / 2, lostSkips: 20 })
      await runSession(cfgFor({ lifetimeCU: 128, sleepInterval: 50 }), deps())
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.at(-1)).toMatchObject({ restores: 1, lostCU: 39 / 2, skippedUpdates: 20 })
    })

    it("a learner that keeps failing after restores stops with an error instead of looping, its life ended on the record", async () => {
      const idb = new IDBFactory()
      const events: SessionEvent[] = []
      await expect(runSession(cfgFor({ lifetimeCU: 128, sleepInterval: 50 }), depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 2, skip: n => n >= 60 }), onEvent: e => events.push(e),
      }))).rejects.toThrow(/restore/)
      const logs = await logsOf(idb) // whatever was written is valid
      // The 32 CU crossing (step 64) falls in the skips: never logged. The failed life's final file follows.
      expect(logs.map(f => f.seq)).toEqual([0, 1, 2])
      expectContinuous(logs)
      // From its checkpoint (the sleep at step 50): 3 restores, each throwing away steps 51-79, then the fourth try's.
      expect(logs[2]).toMatchObject({ final: true, stepsTotal: 50, cu: 25, restores: 3, lostCU: 4 * 29 / 2, skippedUpdates: 80, curve: [{ steps: 50 }] })
      expect(events.at(-1)).toEqual({ type: "logged", seq: 2, steps: 50, cu: 25, final: true })
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
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

  describe("a failed life ends on the record: a final log file, and the checkpoint removed with it (invariant 7)", () => {
    const cfg = cfgFor({ lifetimeCU: 128, sleepInterval: 50 })
    /** Lives to `stopAt` (a reload): checkpoints at the 8 and 16 CU crossings (steps 16, 32) and at the sleeps. */
    const interrupted = async (idb: IDBFactory, stopAt: number) => {
      await expect(runSession(cfg, depsFor(idb, {
        createLearner: stubFactory({ cuPerStep: 1 / 2 }), onTick: ({ steps }) => { if (steps === stopAt) throw new Error("reload") },
      }))).rejects.toThrow("reload")
      return (await loadCheckpoint(idb, ID))!
    }
    const valid = (logs: LogFileV2[]) => { for (const f of logs) expect(parseLogFile(JSON.parse(JSON.stringify(f))).ok).toBe(true) }

    it("the learner throwing: the final file is the checkpoint's place in the life; the CU since it is lostCU", async () => {
      const idb = new IDBFactory()
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, throwAt: { steps: 60, error: new Error("boom") } }) }))).rejects.toThrow("boom")
      const logs = await logsOf(idb)
      expectContinuous(logs)
      expect(logs.at(-1)).toMatchObject({ seq: 2, final: true, stepsTotal: 50, cu: 25, lostCU: (59 - 50) / 2, restores: 0, widths: TINY, settings: { seed: 7 } })
      expect(logs.at(-1)!.curve.map(p => p.steps)).toEqual([50])
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it(`${MAX_RESTORES} restores: the most a log file may record`, async () => {
      const idb = new IDBFactory()
      const cp = await interrupted(idb, 60)
      await saveCheckpoint(idb, { ...cp, restores: MAX_RESTORES })
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, skip: () => true }) }))).rejects.toThrow(new RegExp(`${MAX_RESTORES} restores`))
      const logs = await logsOf(idb)
      expectContinuous(logs)
      valid(logs)
      expect(logs.at(-1)).toMatchObject({ seq: cp.seq, final: true, stepsTotal: 50, restores: MAX_RESTORES, lostCU: 10, skippedUpdates: 20 })
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("a real learner's checkpoint saved before Plan 05a (settings without gamma) resumes at γ 0.9 and logs and checkpoints it", async () => {
      const idb = new IDBFactory()
      const make = (init: LearnerInit) => createLearner({ backend: createCpuBackend(), ...init })
      const real = cfgFor({ lifetimeCU: 10, sleepInterval: 20, learnerSettings: { ...DEFAULT_SETTINGS, replayPerSleep: 2, bufferSize: 50 } })
      await expect(runSession(real, depsFor(idb, { createLearner: make, onTick: ({ steps }) => { if (steps === 45) throw new Error("reload") } }))).rejects.toThrow("reload")
      const cp = (await loadCheckpoint(idb, ID))!
      const { gamma: _g, ...older } = cp.learnerState.settings
      void _g
      await saveCheckpoint(idb, { ...cp, learnerState: { ...cp.learnerState, settings: older as LearnerSettings } })
      const later: { steps: number; gamma: number | undefined }[] = []
      await runSession(real, depsFor(idb, { createLearner: make, onTick: async () => {
        const c = await loadCheckpoint(idb, ID)
        if (c && c.steps > cp.steps) later.push({ steps: c.steps, gamma: c.learnerState.settings.gamma })
      } }))
      const logs = await logsOf(idb)
      valid(logs)
      expect(logs.at(-1)).toMatchObject({ final: true })
      expect(logs.at(-1)!.stepsTotal).toBeGreaterThan(cp.steps)
      for (const f of logs) expect(f.settings.gamma).toBe(0.9)
      // The resumed learner's own checkpoints now carry gamma.
      expect(later.length).toBeGreaterThan(0)
      expect(later.every((c) => c.gamma === 0.9)).toBe(true)
    })

    it("a checkpoint the learner refuses (a learner state version mismatch, on a real learner)", async () => {
      const idb = new IDBFactory()
      const make = (init: LearnerInit) => createLearner({ backend: createCpuBackend(), ...init })
      const real = cfgFor({ lifetimeCU: 10, sleepInterval: 20, learnerSettings: { ...DEFAULT_SETTINGS, replayPerSleep: 2, bufferSize: 50 } })
      await expect(runSession(real, depsFor(idb, { createLearner: make, onTick: ({ steps }) => { if (steps === 45) throw new Error("reload") } }))).rejects.toThrow("reload")
      const cp = (await loadCheckpoint(idb, ID))!
      expect(cp.steps).toBe(40)
      await saveCheckpoint(idb, { ...cp, learnerState: { ...cp.learnerState, version: 99 as 1 } })
      await expect(runSession(real, depsFor(idb, { createLearner: make }))).rejects.toThrow(/version 99/)
      const logs = await logsOf(idb)
      expect(logs).toHaveLength(1)
      valid(logs)
      expect(logs[0]).toMatchObject({ seq: 0, final: true, stepsTotal: 40, cu: cp.cu, curve: cp.curveTail, widths: TINY, lostCU: 0 })
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("a corrupt checkpoint record: the final file repeats the latest log file's counters, at the next seq", async () => {
      const idb = new IDBFactory()
      const cp = await interrupted(idb, 60)
      const before = await logsOf(idb)
      await saveCheckpoint(idb, { ...cp, steps: -1 })
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2 }) }))).rejects.toThrow(RangeError)
      const logs = await logsOf(idb)
      expect(logs.slice(0, -1)).toEqual(before)
      valid(logs)
      expect(logs.at(-1)).toEqual({ ...before.at(-1)!, seq: 2, final: true, curve: [], writtenAt: expect.any(String) })
      expect(await loadCheckpoint(idb, ID).catch(() => "still there")).toBeUndefined()
    })

    it("a corrupt checkpoint and no log file yet: no file (nothing to repeat), the checkpoint is removed", async () => {
      const idb = new IDBFactory()
      const cp = await interrupted(idb, 12)
      expect(await logsOf(idb)).toEqual([])
      await saveCheckpoint(idb, { ...cp, learnerState: null as unknown as LearnerState })
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2 }) }))).rejects.toThrow(RangeError)
      expect(await logsOf(idb)).toEqual([])
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("a full disk (QuotaExceededError on a checkpoint write): the final file still goes in, freeing the checkpoint", async () => {
      const idb = new IDBFactory()
      const put = FakeObjectStore.prototype.put
      let full = false
      FakeObjectStore.prototype.put = function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
        if (full && this.name === "checkpoints") throw new DOMException("the disk is full", "QuotaExceededError")
        return put.call(this, value, key)
      } as typeof put
      try {
        await expect(runSession(cfg, depsFor(idb, {
          createLearner: stubFactory({ cuPerStep: 1 / 2 }), onTick: ({ steps }) => { full = steps >= 40 },
        }))).rejects.toMatchObject({ name: "QuotaExceededError" })
      } finally {
        FakeObjectStore.prototype.put = put
      }
      const logs = await logsOf(idb)
      valid(logs)
      expect(logs.map(f => f.seq)).toEqual([0, 1, 2])
      // The sleep at step 50 could not save: the life ends at its checkpoint of the 16 CU crossing (step 32).
      expect(logs.at(-1)).toMatchObject({ seq: 2, final: true, stepsTotal: 32, cu: 16, lostCU: (50 - 32) / 2 })
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("a newborn that fails before its first step leaves no file and no checkpoint", async () => {
      const idb = new IDBFactory()
      await expect(runSession(cfg, depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, throwAt: { steps: 1, error: new Error("no") } }) }))).rejects.toThrow("no")
      expect(await logsOf(idb)).toEqual([])
      expect(await loadCheckpoint(idb, ID)).toBeUndefined()
    })

    it("a failure before the first log file: the final file is seq 0 from the checkpoint", async () => {
      const idb = new IDBFactory()
      await expect(runSession(cfgFor({ lifetimeCU: 128, sleepInterval: 5 }), depsFor(idb, { createLearner: stubFactory({ cuPerStep: 1 / 2, throwAt: { steps: 13, error: new Error("boom") } }) }))).rejects.toThrow("boom")
      const logs = await logsOf(idb)
      valid(logs)
      expect(logs).toEqual([expect.objectContaining({ seq: 0, final: true, stepsTotal: 10, cu: 5, lostCU: 1, sessionsSeenAtBirth: 3, knowledgeSha: "f".repeat(64) })])
    })

    it("errors the caller recovers from (recoverable), and its own callbacks' errors (a reload), leave the life as it was", async () => {
      for (const over of [
        { createLearner: stubFactory({ cuPerStep: 1 / 2, throwAt: { steps: 70, error: new Error("GPU lost") } }), recoverable: (e: unknown) => (e as Error).message === "GPU lost" },
        { createLearner: stubFactory({ cuPerStep: 1 / 2 }), onTick: ({ steps }: { steps: number }) => { if (steps === 70) throw new Error("GPU lost") } },
        { createLearner: stubFactory({ cuPerStep: 1 / 2 }), onEvent: (e: SessionEvent) => { if (e.type === "logged" && e.seq === 1) throw new Error("GPU lost") } },
      ]) {
        const idb = new IDBFactory()
        await expect(runSession(cfg, depsFor(idb, over))).rejects.toThrow("GPU lost")
        expect((await logsOf(idb)).some(f => f.final)).toBe(false)
        expect(await loadCheckpoint(idb, ID)).toBeDefined()
      }
    })
  })

  describe("growing up (Plan 04b)", { timeout: 120_000 }, () => {
    const canvas = () => createCanvas(WORLD.width, WORLD.height)
    /**
     * The v1 suite at a 5-step budget (same worlds, measures and pass rules). M5's reference (16 random clickers at its
     * own budget, ~0.4 s) is a pure function of the seed: memoized, so the many lives here reuse it.
     */
    const refs = new Map<number, number>()
    const SUITE: Milestone[] = MILESTONES_V1.map(m => ({
      ...m, budgetSteps: 5,
      ...(m.reference ? { reference: (seed: number, c: Parameters<NonNullable<Milestone["reference"]>>[1]) => {
        if (!refs.has(seed)) refs.set(seed, m.reference!(seed, c))
        return refs.get(seed)!
      } } : {}),
    }))
    /** Epsilon 1: every decision picks a room uniformly, so carries are frequent. */
    const CARE = { ...CAREGIVER_DEFAULTS, caregiverPeriod: 50, caregiverEpsilon: 1 }
    type GrowOpts = { cfg?: Partial<SessionConfig>; deps?: Partial<SessionDeps>; stub?: Partial<StubOpts>; checkups?: false; suite?: Milestone[] }
    /** 2 CU a step and a sleep every 5 steps by default, in a real world; checkups on unless `checkups: false`. */
    const grow = (idb: IDBFactory, o: GrowOpts = {}) => runSession(
      cfgFor({ lifetimeCU: 40, sleepInterval: 5, ...o.cfg }),
      depsFor(idb, {
        world: createWorld({ seed: 7, canvas: canvas() }),
        createLearner: stubFactory({ cuPerStep: 2, frozen: { evalPerStep: 0.001 }, ...o.stub }),
        ...(o.checkups === false ? {} : { checkups: { canvas, suite: o.suite ?? SUITE } }),
        ...o.deps,
      }),
    )
    const results = (logs: LogFileV2[]) => logs.flatMap(f => f.checkups ?? [])
    const levelsOf = (logs: LogFileV2[]) => results(logs).map(c => c.level)
    /** Writer/validator agreement: the files as written parse, and the gate accepts them as one PR. */
    const gateAccepts = async (logs: LogFileV2[]) => {
      for (const f of logs) {
        const p = parseLogFile(JSON.parse(JSON.stringify(f)))
        expect(p.ok, p.ok ? "" : p.error).toBe(true)
      }
      const v = await validateSessionPr(logs.map(f => ({ path: `data/runs/${f.learnerId}/${f.seq}.json`, content: JSON.stringify(f, null, 2) })), [])
      expect(v.errors).toEqual([])
      expect(v.ok).toBe(true)
    }
    const checkupEvents = (events: SessionEvent[]) => events.flatMap(e =>
      e.type === "checkupStarted" ? [["start", e.level]] : e.type === "checkup" ? [["done", e.result.level, e.result.steps]] : [])

    it("checkups run at the first clean sleep after each crossing, once per level; crossings within one sleep interval give one checkup, at the highest", async () => {
      const events: SessionEvent[] = []
      await grow(new IDBFactory(), { cfg: { lifetimeCU: 140, sleepInterval: 10 }, deps: { onEvent: e => events.push(e) } })
      // 2 CU a step, a sleep every 10: at the first sleep (20 CU) 8 and 16 are both due; only 16 is checked.
      expect(checkupEvents(events)).toEqual([
        ["start", 16], ["done", 16, 10], ["start", 32], ["done", 32, 20], ["start", 64], ["done", 64, 40], ["start", 128], ["done", 128, 70],
      ])
    })

    it("results go into the next log file (the same file when the crossing and the sleep share a step), with milestoneSuiteVersion 1; every file parses and the validator accepts the sequence", async () => {
      // Sleeps every 10 steps: each result waits for the next file. Every 8: the crossings fall on sleeps, same file.
      for (const [sleepInterval, expected] of [
        [10, [[4, []], [8, []], [16, [16]], [32, [32]], [64, [64]], [70, [128]]]],
        [8, [[4, []], [8, [16]], [16, [32]], [32, [64]], [64, [128]], [70, []]]],
      ] as const) {
        const idb = new IDBFactory()
        await grow(idb, { cfg: { lifetimeCU: 140, sleepInterval } })
        const logs = await logsOf(idb)
        expect(logs.map(f => [f.stepsTotal, (f.checkups ?? []).map(c => c.level)]), `sleepInterval ${sleepInterval}`).toEqual(expected)
        for (const f of logs) expect(f.milestoneSuiteVersion).toBe(MILESTONE_SUITE_VERSION)
        expect(MILESTONE_SUITE_VERSION).toBe(1)
        for (const c of results(logs)) expect(c.passed.every(id => typeof id === "string") && Object.keys(c.measures).length === 6).toBe(true)
        await gateAccepts(logs)
      }
    })

    it("checkups are billed to evalCU only", async () => {
      const life = async (checkups: boolean) => {
        const idb = new IDBFactory()
        await grow(idb, { cfg: { lifetimeCU: 140, sleepInterval: 10 }, stub: { evalCu: 0.25 }, ...(checkups ? {} : { checkups: false as const }) })
        return logsOf(idb)
      }
      const [withC, without] = [await life(true), await life(false)]
      expect(withC.map(f => [f.stepsTotal, f.cu])).toEqual(without.map(f => [f.stepsTotal, f.cu]))
      expect(results(withC)).toHaveLength(4)
      let billed = 0
      for (const [i, f] of withC.entries()) {
        for (const c of f.checkups!) { expect(c.evalCU).toBeCloseTo(6 * 5 * 0.001, 12); billed += c.evalCU } // 6 milestones x 5 frozen steps
        expect(f.evalCU! - without[i]!.evalCU!).toBeCloseTo(billed, 12)
      }
      expect(without.every(f => !("checkups" in f) && !("milestoneSuiteVersion" in f))).toBe(true)
    })

    it("no checkup while consecutiveSkips > 0; it runs at the next clean sleep", async () => {
      const events: SessionEvent[] = []
      // Skips over steps 18-22: the sleep at step 20 (32 CU due) is not clean.
      await grow(new IDBFactory(), { cfg: { lifetimeCU: 140, sleepInterval: 10 }, stub: { skip: n => n >= 18 && n <= 22 }, deps: { onEvent: e => events.push(e) } })
      expect(checkupEvents(events).filter(e => e[0] === "done")).toEqual([["done", 16, 10], ["done", 32, 30], ["done", 64, 40], ["done", 128, 70]])
    })

    it("the caregiver does not act or observe during a checkup", async () => {
      cgSpy.log.length = 0
      await grow(new IDBFactory(), { cfg: { lifetimeCU: 140, sleepInterval: 10, caregiver: CARE }, deps: { onEvent: e => cgSpy.log.push(`event:${e.type}`) } })
      const log = cgSpy.log
      const starts = log.flatMap((x, i) => (x === "event:checkupStarted" ? [i] : []))
      expect(starts).toHaveLength(4)
      for (const s of starts) {
        const end = log.indexOf("event:checkup", s)
        expect(end).toBeGreaterThan(s)
        expect(log.slice(s + 1, end)).toEqual([])
        expect(log[end + 1]).toBe("onCheckup") // told the result once it is in
      }
      // Non-vacuous: it decided every step and watched every one within a room (all but the first and each after a carry).
      const carries = log.filter(x => x === "event:carried").length
      expect(log.filter(x => x === "decide").length).toBe(70)
      expect(log.filter(x => x === "observe").length).toBe(70 - 1 - carries)
    })

    it("the caregiver carries through world.bringToFront: focus events with source caregiver, counted in the log's caregiver summary; its settings are in the log's settings", async () => {
      const idb = new IDBFactory()
      const world = createWorld({ seed: 7, canvas: canvas() })
      const trace: TraceEvent[] = []
      const traced = {
        observe: () => world.observe(), act: (a: Action) => world.act(a), state: () => world.state(), rooms: () => world.rooms(),
        bringToFront: (k: Parameters<typeof world.bringToFront>[0]) => world.bringToFront(k),
        drainTrace: () => { const t = world.drainTrace(); trace.push(...t); return t },
      }
      const events: SessionEvent[] = []
      await grow(idb, { checkups: false, cfg: { lifetimeCU: 64, sleepInterval: 5, caregiver: CARE }, stub: { cuPerStep: 0.125 }, deps: { world: traced, onEvent: e => events.push(e) } })
      trace.push(...world.drainTrace()) // a carry at the last step
      const carried = events.flatMap(e => (e.type === "carried" ? [e] : []))
      const focus = trace.filter(e => e.kind === "focus" && e.source === "caregiver")
      const logs = await logsOf(idb)
      expect(carried.length).toBeGreaterThan(3)
      expect(focus.map(e => e.room)).toEqual(carried.map(e => e.to))
      expect(logs.reduce((a, f) => a + f.caregiver!.carries, 0)).toBe(carried.length)
      const to: Record<string, number> = {}
      for (const f of logs) for (const [k, n] of Object.entries(f.caregiver!.to)) to[k] = (to[k] ?? 0) + n
      const byRoom: Record<string, number> = {}
      for (const e of carried) byRoom[e.to] = (byRoom[e.to] ?? 0) + 1
      expect(to).toEqual(byRoom)
      // Carries only on the period (50 steps), not at the sleeps in between (decision 0016).
      for (const e of carried) expect(e.steps % 50).toBe(0)
      for (const f of logs) expect(f.settings).toMatchObject(clampCaregiverSettings(CARE))
      await gateAccepts(logs)
    })

    it("the caregiver observes a loss only when the previous frame was seen in the same room: not the first transition after a carry, nor a run's first", async () => {
      cgSpy.log.length = 0
      const world = createWorld({ seed: 7, canvas: canvas() })
      const rooms: string[] = [] // the room of each step's frame, as the session reads it
      const watched = { observe: () => { rooms.push(world.state().room); return world.observe() }, state: () => world.state(), rooms: () => world.rooms(), bringToFront: (k: Parameters<typeof world.bringToFront>[0]) => world.bringToFront(k), act: (a: Action) => world.act(a), drainTrace: () => world.drainTrace() }
      await grow(new IDBFactory(), { checkups: false, cfg: { lifetimeCU: 64, sleepInterval: 5, caregiver: CARE }, stub: { cuPerStep: 0.125 }, deps: { world: watched, onEvent: e => cgSpy.log.push(`event:${e.type}`) } })
      // Per step: [observe] decide [event:carried].
      const perStep: { observed: boolean; carried: boolean }[] = []
      let observed = false
      for (const x of cgSpy.log) {
        if (x === "observe") observed = true
        else if (x === "decide") { perStep.push({ observed, carried: false }); observed = false }
        else if (x === "event:carried") perStep.at(-1)!.carried = true
      }
      expect(perStep).toHaveLength(512)
      expect(perStep[0]!.observed).toBe(false)
      const afterCarry = perStep.flatMap((p, i) => (p.carried && i + 2 < perStep.length ? [i] : []))
      expect(afterCarry.length).toBeGreaterThan(3)
      for (const i of afterCarry) {
        expect(rooms[i + 1], `step ${i + 2}`).not.toBe(rooms[i])
        expect(perStep[i + 1]!.observed, `step ${i + 2}: across the carry`).toBe(false)
        expect(perStep[i + 2]!.observed, `step ${i + 3}: within the new room`).toBe(true)
      }
      // And in general: observed exactly when the frame before was in the same room.
      perStep.forEach((p, i) => expect(p.observed, `step ${i + 1}`).toBe(i > 0 && rooms[i] === rooms[i - 1]))
    })

    it("a failed life's final file carries its checkpoint's checkup tail; one rebuilt from the latest file does not repeat its checkups", async () => {
      // 2 CU a step, a sleep every 10: the 16 CU checkup at step 10 waits for the file at step 16; the learner fails at 13.
      const idb = new IDBFactory()
      await expect(grow(idb, { cfg: { lifetimeCU: 140, sleepInterval: 10, caregiver: CARE }, stub: { throwAt: { steps: 13, error: new Error("boom") } } })).rejects.toThrow("boom")
      const logs = await logsOf(idb)
      expect(logs.at(-1)).toMatchObject({ final: true, stepsTotal: 10, milestoneSuiteVersion: 1 })
      expect(levelsOf(logs)).toEqual([16])
      expect(logs.at(-1)!.caregiver).toBeDefined()
      expect(logs.at(-1)!.settings).toMatchObject(clampCaregiverSettings(CARE))
      await gateAccepts(logs)
      // A corrupt checkpoint after the 32 CU file: the final file repeats that file's counters, not its checkups.
      const idb2 = new IDBFactory()
      await expect(grow(idb2, { cfg: { lifetimeCU: 140, sleepInterval: 10 }, deps: { onTick: ({ steps }) => { if (steps === 33) throw new Error("reload") } } })).rejects.toThrow("reload")
      expect(levelsOf(await logsOf(idb2))).toEqual([16, 32])
      const c = (await loadCheckpoint(idb2, ID))!
      await saveCheckpoint(idb2, { ...c, steps: -1 })
      await expect(grow(idb2, { cfg: { lifetimeCU: 140, sleepInterval: 10 } })).rejects.toThrow(RangeError)
      const after = await logsOf(idb2)
      expect(after.at(-1)!.final).toBe(true)
      expect(after.at(-1)!.checkups).toBeUndefined()
      expect(levelsOf(after)).toEqual([16, 32])
      await gateAccepts(after)
    })

    it("a finished checkup is checkpointed at once, inside the throttle window (a reload never checks its level again)", async () => {
      const savedSteps: number[] = []
      // The default 300 s throttle: no sleep would save. Crossings 8, 16, 32 CU at steps 2, 4, 8; checkups at 5 and 10 (the end).
      await grow(new IDBFactory(), { cfg: { checkpointEveryMs: undefined }, stub: { cuPerStep: 4, savedSteps } })
      expect(savedSteps).toEqual([0, 2, 4, 5, 8])
    })

    describe("Review Focus 3: reload at every step across two checkups", () => {
      // 4 CU a step, a sleep every 5: checkups 16 (step 5) and 32 (step 10, the end of the life).
      const run = (idb: IDBFactory, o: { stopAt?: number; stopAtCheckup?: number } = {}) => {
        let started = 0
        return grow(idb, { stub: { cuPerStep: 4 }, deps: {
          onTick: ({ steps }) => { if (steps === o.stopAt) throw new Error("reload") },
          onEvent: e => { if (e.type === "checkupStarted" && ++started === o.stopAtCheckup) throw new Error("reload") },
        } })
      }
      const cases = [...Array.from({ length: 10 }, (_, i) => ({ stopAt: i + 1 })), ...[1, 2].map(k => ({ stopAtCheckup: k }))]
      it("the levels in the logs equal an uninterrupted life's, none twice", async () => {
        const whole = new IDBFactory()
        await run(whole)
        const wholeLogs = await logsOf(whole)
        expect(levelsOf(wholeLogs)).toEqual([16, 32])
        for (const c of cases) {
          const idb = new IDBFactory()
          await expect(run(idb, c)).rejects.toThrow("reload")
          await run(idb)
          const logs = await logsOf(idb)
          expect(levelsOf(logs), JSON.stringify(c)).toEqual([16, 32])
          expect(logs.map(strip), JSON.stringify(c)).toEqual(wholeLogs.map(strip))
          await gateAccepts(logs)
        }
      })
    })

    it("a halt asked during a checkup abandons it; the halting save keeps the old lastLevel; after resume the level is checked exactly once", async () => {
      const idb = new IDBFactory()
      const inbox = new SessionInbox()
      let level = 0
      let answer: Promise<boolean> | undefined
      // The halt is asked from inside M3 of the 16 CU checkup (step 10).
      const suite = SUITE.map(m => (m.id === "M3" ? { ...m, measure: (...a: Parameters<Milestone["measure"]>) => {
        if (level === 16 && !answer) answer = inbox.checkpointNow({ halt: true })
        return m.measure(...a)
      } } : m))
      const events: SessionEvent[] = []
      const r = await grow(idb, { suite, deps: { inbox, onEvent: e => { events.push(e); if (e.type === "checkupStarted") level = e.level } } })
      expect(await answer).toBe(true)
      expect(r).toMatchObject({ halted: true, steps: 10 })
      expect(checkupEvents(events)).toEqual([["start", 8], ["done", 8, 5], ["start", 16]])
      const cp = (await loadCheckpoint(idb, ID))!
      expect(cp.steps).toBe(10)
      expect(cp.checkups).toEqual({ lastLevel: 8, tail: [] }) // no suiteVersion: an empty tail (decision 0017, rollback)
      expect(levelsOf(await logsOf(idb))).toEqual([8])
      const again: SessionEvent[] = []
      await grow(idb, { suite, deps: { onEvent: e => again.push(e) } })
      expect(again[0]).toMatchObject({ type: "resumed", steps: 10, checkupTail: [] })
      expect(checkupEvents(again)).toEqual([["start", 16], ["done", 16, 15], ["start", 32], ["done", 32, 20]])
      const logs = await logsOf(idb)
      expect(levelsOf(logs)).toEqual([8, 16, 32])
      await gateAccepts(logs)
    })

    it("a restore after 20 skips rolls back lastLevel, tail and caregiver to the checkpoint's, with no duplicate level in the logs", async () => {
      const idb = new IDBFactory()
      let poisoned = true, justRestored = false
      let after: Awaited<ReturnType<typeof loadCheckpoint>>
      cgSpy.inits.length = 0
      // 0.5 CU a step, a sleep every 10: checkups 8 (step 20), 16 (40), 32 (70). Skips from 75 restore at 94 to the checkpoint at 70.
      await grow(idb, {
        cfg: { lifetimeCU: 64, sleepInterval: 10, caregiver: CARE }, stub: { cuPerStep: 0.5, skip: n => poisoned && n >= 75 },
        deps: {
          onEvent: e => { if (e.type === "restored") { poisoned = false; justRestored = true } },
          onTick: async () => { if (justRestored) { justRestored = false; after = await loadCheckpoint(idb, ID) } },
        },
      })
      expect(after!.steps).toBe(70)
      expect(cgSpy.inits).toHaveLength(2)
      expect(cgSpy.inits[1]).toEqual({ state: after!.caregiver })
      const logs = await logsOf(idb)
      expect(levelsOf(logs)).toEqual([8, 16, 32])
      expect(logs.at(-1)).toMatchObject({ final: true, restores: 1 })
      await gateAccepts(logs)
    })

    /** Every carry the caregiver made (the carried events) is on the record once: in a file's carries or its lostCarries. */
    const carriesOnRecord = (logs: LogFileV2[]) => logs.reduce((a, f) => a + (f.caregiver?.carries ?? 0) + (f.caregiver?.lostCarries ?? 0), 0)
    /** 0.5 CU a step, a sleep every 10, a carry decision every 50 steps: skips from step 95 restore at 114 to the checkpoint at 90. */
    const restoreLife = (idb: IDBFactory, events: SessionEvent[], o: { onRestored?: () => Promise<void>; poisoned?: boolean } = {}) => {
      let poisoned = o.poisoned ?? true, justRestored = false
      return grow(idb, {
        cfg: { lifetimeCU: 64, sleepInterval: 10, caregiver: CARE }, stub: { cuPerStep: 0.5, skip: n => poisoned && n >= 95 },
        deps: {
          onEvent: e => { events.push(e); if (e.type === "restored") { poisoned = false; justRestored = true } },
          onTick: async () => { if (justRestored) { justRestored = false; await o.onRestored?.() } },
        },
      })
    }
    /** The carries thrown away by the restore: those after its checkpoint's step and before the restored event. */
    const thrownAway = (events: SessionEvent[]) => {
      const r = events.findIndex(e => e.type === "restored")
      const back = (events[r] as Extract<SessionEvent, { type: "restored" }>).toSteps
      return events.slice(0, r).filter(e => e.type === "carried" && e.steps > back).length
    }

    it("a restore after 20 skips puts the carries since the checkpoint into the next file's lostCarries; the sum of carries and lostCarries over the files equals every carried event", async () => {
      const idb = new IDBFactory()
      const events: SessionEvent[] = []
      await restoreLife(idb, events)
      const lost = thrownAway(events)
      expect(lost).toBeGreaterThan(0) // non-vacuous
      const logs = await logsOf(idb)
      expect(logs.reduce((a, f) => a + (f.caregiver!.lostCarries ?? 0), 0)).toBe(lost)
      expect(logs.filter(f => f.caregiver!.lostCarries !== undefined)).toHaveLength(1)
      expect(carriesOnRecord(logs)).toBe(events.filter(e => e.type === "carried").length)
      await gateAccepts(logs)
    })

    it("a reload right after a restore keeps lostCarries", async () => {
      const idb = new IDBFactory()
      const events: SessionEvent[] = []
      let after: Awaited<ReturnType<typeof loadCheckpoint>>
      await expect(restoreLife(idb, events, { onRestored: async () => { after = await loadCheckpoint(idb, ID); throw new Error("reload") } })).rejects.toThrow("reload")
      const lost = thrownAway(events)
      expect(lost).toBeGreaterThan(0)
      expect(after!.steps).toBe(90)
      expect(after!.caregiver!.lost).toBe(lost) // the restore's own save holds the restored caregiver with its count
      await restoreLife(idb, events, { poisoned: false }) // resumed from that save
      const logs = await logsOf(idb)
      expect(logs.reduce((a, f) => a + (f.caregiver!.lostCarries ?? 0), 0)).toBe(lost)
      expect(carriesOnRecord(logs)).toBe(events.filter(e => e.type === "carried").length)
      await gateAccepts(logs)
    })

    it("a failed life's final file carries the lost carries", async () => {
      // 0.125 CU a step, the default 300 s throttle: saves only at the crossings (8, 16, 32 CU at steps 64, 128, 256). The
      // learner fails at step 255: the carries after step 128 (here one, at step 250) are thrown away with the life.
      const idb = new IDBFactory()
      const events: SessionEvent[] = []
      await expect(grow(idb, {
        checkups: false, cfg: { lifetimeCU: 64, sleepInterval: 5, caregiver: CARE, checkpointEveryMs: undefined },
        stub: { cuPerStep: 0.125, throwAt: { steps: 255, error: new Error("boom") } }, deps: { onEvent: e => events.push(e) },
      })).rejects.toThrow("boom")
      const lost = events.filter(e => e.type === "carried" && e.steps > 128).length
      expect(lost).toBeGreaterThan(0)
      const logs = await logsOf(idb)
      expect(logs.at(-1)).toMatchObject({ final: true, stepsTotal: 128, caregiver: { carries: 0, lostCarries: lost } })
      expect(carriesOnRecord(logs)).toBe(events.filter(e => e.type === "carried").length)
      await gateAccepts(logs)
    })

    describe("a save or log write that fails after the caregiver's summary was taken (fix round 1)", () => {
      // 0.125 CU a step, the default 300 s throttle: saves only at the crossings (8, 16, 32 CU at steps 64, 128, 256).
      const careLife = (idb: IDBFactory, events: SessionEvent[], stub: Partial<StubOpts> = {}) => grow(idb, {
        checkups: false, cfg: { lifetimeCU: 64, sleepInterval: 5, caregiver: CARE, checkpointEveryMs: undefined },
        stub: { cuPerStep: 0.125, ...stub }, deps: { onEvent: e => events.push(e) },
      })
      it("learner.save throws (not recoverable) during a crossing's save: the carries since the checkpoint are lost carries, none vanish", async () => {
        const idb = new IDBFactory()
        const events: SessionEvent[] = []
        // The 32 CU file at step 256 takes the caregiver's summary, then its checkpoint's learner.save throws.
        const errors = vi.spyOn(console, "error").mockImplementation(() => {})
        await expect(careLife(idb, events, { onSave: steps => { if (steps === 256) throw new Error("save failed") } })).rejects.toThrow("save failed")
        expect(errors).not.toHaveBeenCalled() // the final file was written
        errors.mockRestore()
        const lost = events.filter(e => e.type === "carried" && e.steps > 128).length
        expect(lost).toBeGreaterThan(0)
        const logs = await logsOf(idb)
        expect(logs.at(-1)).toMatchObject({ final: true, stepsTotal: 128, caregiver: { lostCarries: lost } })
        expect(carriesOnRecord(logs)).toBe(events.filter(e => e.type === "carried").length)
        await gateAccepts(logs)
      })
      it("queue.put of the final log throws: the failed life's final file holds the carries since the checkpoint as lost carries", async () => {
        const idb = new IDBFactory()
        const events: SessionEvent[] = []
        const add = FakeObjectStore.prototype.add
        let failed = false
        FakeObjectStore.prototype.add = function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
          if (!failed && this.name === "outbox" && (value as { file: { final: boolean; stepsTotal: number } }).file.stepsTotal === 512) {
            failed = true
            throw new DOMException("the disk is full", "QuotaExceededError")
          }
          return add.call(this, value, key)
        } as typeof add
        try {
          await expect(careLife(idb, events)).rejects.toMatchObject({ name: "QuotaExceededError" })
        } finally {
          FakeObjectStore.prototype.add = add
        }
        expect(failed).toBe(true)
        const lost = events.filter(e => e.type === "carried" && e.steps > 256).length
        expect(lost).toBeGreaterThan(0)
        const logs = await logsOf(idb)
        expect(logs.at(-1)).toMatchObject({ final: true, stepsTotal: 256, caregiver: { carries: 0, lostCarries: lost } })
        expect(carriesOnRecord(logs)).toBe(events.filter(e => e.type === "carried").length)
        await gateAccepts(logs)
      })
      it("a learner too broken to report its stats: the carries since the checkpoint are still lost carries", async () => {
        const idb = new IDBFactory()
        const events: SessionEvent[] = []
        const inner = stubFactory({ cuPerStep: 0.125 })
        let broken = false
        const createLearner = async (init: LearnerInit): Promise<SessionLearner> => {
          const l = await inner(init)
          return { ...l, stats: () => { if (broken) throw new Error("stats broken"); return l.stats() }, step: async o => {
            if (l.stats().steps === 254) { broken = true; throw new Error("boom") }
            return l.step(o)
          } }
        }
        await expect(grow(idb, {
          checkups: false, cfg: { lifetimeCU: 64, sleepInterval: 5, caregiver: CARE, checkpointEveryMs: undefined },
          deps: { createLearner, onEvent: e => events.push(e) },
        })).rejects.toThrow("boom")
        const lost = events.filter(e => e.type === "carried" && e.steps > 128).length
        expect(lost).toBeGreaterThan(0)
        const logs = await logsOf(idb)
        expect(logs.at(-1)).toMatchObject({ final: true, stepsTotal: 128, caregiver: { lostCarries: lost } })
        expect(carriesOnRecord(logs)).toBe(events.filter(e => e.type === "carried").length)
      })
    })

    it("a checkpoint whose tail lacks the learner's evalCU is never written: the save fails, the life ends on its last good checkpoint", async () => {
      // 2 CU a step, a sleep every 5: the 8 CU file and its checkpoint at step 4; the 8 CU checkup at step 5, whose save
      // would hold a tail with a learner state without evalCU.
      const inner = stubFactory({ cuPerStep: 2, frozen: { evalPerStep: 0.001 } })
      const createLearner = async (init: LearnerInit): Promise<SessionLearner> => {
        const l = await inner(init)
        return { ...l, save: async () => { const st = (await l.save()) as unknown as StubState; delete (st.stats as Partial<LearnerStats>).evalCU; return st as unknown as LearnerState } }
      }
      const idb = new IDBFactory()
      const saved: unknown[] = []
      const put = FakeObjectStore.prototype.put
      FakeObjectStore.prototype.put = function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
        if (this.name === "checkpoints") saved.push(value)
        return put.call(this, value, key)
      } as typeof put
      try {
        await expect(grow(idb, { deps: { createLearner } })).rejects.toThrow("checkups.tail without the learner's evalCU")
      } finally {
        FakeObjectStore.prototype.put = put
      }
      expect(saved.length).toBeGreaterThan(0)
      for (const c of saved as { checkups?: { tail: unknown[] } }[]) expect(c.checkups!.tail).toEqual([]) // none with a tail was stored
      const logs = await logsOf(idb)
      expect(logs.at(-1)).toMatchObject({ final: true, stepsTotal: 4 })
      await gateAccepts(logs)
    })

    it("a failed life's final log that cannot be written is reported on console.error, not silently dropped", async () => {
      const idb = new IDBFactory()
      const add = FakeObjectStore.prototype.add
      FakeObjectStore.prototype.add = function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
        if (this.name === "outbox" && (value as { file: { final: boolean } }).file.final) throw new DOMException("the disk is full", "QuotaExceededError")
        return add.call(this, value, key)
      } as typeof add
      const errors = vi.spyOn(console, "error").mockImplementation(() => {})
      try {
        await expect(grow(idb, { stub: { throwAt: { steps: 7, error: new Error("boom") } } })).rejects.toThrow("boom")
        expect(errors).toHaveBeenCalledTimes(1)
        expect(String(errors.mock.calls[0]![0])).toContain("final log was not written")
        expect(errors.mock.calls[0]![1]).toMatchObject({ name: "QuotaExceededError" })
      } finally {
        FakeObjectStore.prototype.add = add
        errors.mockRestore()
      }
    })

    it("no checkup runs while the tail holds MAX_CHECKUPS_PER_FILE; the level runs after the next file", async () => {
      // 100 CU a step, a sleep every 5. A first run without checkups stops at step 22 (its checkpoint: step 21, 2100 CU).
      const idb = new IDBFactory()
      const stub = { cuPerStep: 100 }
      await expect(grow(idb, { checkups: false, cfg: { lifetimeCU: 3200 }, stub, deps: { onTick: ({ steps }) => { if (steps === 22) throw new Error("reload") } } })).rejects.toThrow("reload")
      // Its checkpoint now holds a full tail (levels 8 ... 1024): level 2048 is due at the next sleep (step 25).
      const c = (await loadCheckpoint(idb, ID))!
      expect(c.steps).toBe(21)
      const levels = [8, 16, 32, 64, 128, 256, 512, 1024]
      expect(levels).toHaveLength(MAX_CHECKUPS_PER_FILE)
      const ids = MILESTONE_IDS[MILESTONE_SUITE_VERSION]!
      const tail = levels.map(level => ({ level, steps: 1, passed: [], measures: Object.fromEntries(ids.map(id => [id, null])), evalCU: 0 }))
      await saveCheckpoint(idb, { ...c, checkups: { lastLevel: 1024, tail, suiteVersion: MILESTONE_SUITE_VERSION } })
      // New knowledge at step 27 writes a file (the boundary) that takes the tail; the sleep at step 30 then checks 2048.
      const inbox = new SessionInbox()
      const events: SessionEvent[] = []
      await grow(idb, { cfg: { lifetimeCU: 3200 }, stub, deps: {
        inbox, onEvent: e => events.push(e), onTick: ({ steps }) => { if (steps === 27) inbox.setKnowledge(knowledgeWith("e".repeat(64))) },
      } })
      const order = events.flatMap(e => (e.type === "logged" ? [`logged ${e.steps}`] : e.type === "checkupStarted" ? [`checkup ${e.level}`] : []))
      expect(order).toEqual(["logged 27", "checkup 2048", "logged 31", "logged 32"])
      const logs = await logsOf(idb)
      expect(logs.map(f => (f.checkups ?? []).map(x => x.level)).filter(l => l.length > 0)).toEqual([levels, [2048]])
      await gateAccepts(logs)
    })

    it("after a checkup event, a checkpoint holding its level is saved before the next learner step", async () => {
      // The default 300 s throttle: only crossings and finished checkups save. 2 CU a step, a sleep every 5, 70 steps.
      const idb = new IDBFactory()
      let due: number | null = null
      const checked: number[] = []
      const inner = stubFactory({ cuPerStep: 2, frozen: { evalPerStep: 0.001 } })
      const createLearner = async (init: LearnerInit): Promise<SessionLearner> => {
        const l = await inner(init)
        return { ...l, step: async o => {
          if (due !== null) {
            const c = (await loadCheckpoint(idb, ID))!
            expect(c.checkups!.lastLevel, `before the step after the ${due} CU checkup`).toBe(due)
            checked.push(due)
            due = null
          }
          return l.step(o)
        } }
      }
      await grow(idb, { cfg: { lifetimeCU: 140, sleepInterval: 5, checkpointEveryMs: undefined }, deps: {
        createLearner, onEvent: e => { if (e.type === "checkup") due = e.result.level },
      } })
      expect(checked).toEqual([8, 16, 32, 64, 128])
      expect(levelsOf(await logsOf(idb))).toEqual([8, 16, 32, 64, 128])
    })

    it("a recoverable error while saving right after a checkup: the run resumes from the older checkpoint, the level reruns once, and the logs hold it once", async () => {
      class Lost extends Error {}
      // 2 CU a step, a sleep every 5: the 8 CU file and its checkpoint at step 4; the 8 CU checkup at step 5, then its save throws.
      let thrown = false
      const onSave = (steps: number) => { if (steps === 5 && !thrown) { thrown = true; throw new Lost("device lost") } }
      const idb = new IDBFactory()
      const events: SessionEvent[] = []
      const deps = { recoverable: (e: unknown) => e instanceof Lost, onEvent: (e: SessionEvent) => events.push(e) }
      await expect(grow(idb, { stub: { onSave }, deps })).rejects.toBeInstanceOf(Lost)
      const c = (await loadCheckpoint(idb, ID))!
      expect(c.steps).toBe(4)
      expect(c.checkups).toMatchObject({ lastLevel: 0, tail: [] })
      await grow(idb, { stub: { onSave }, deps })
      expect(checkupEvents(events).filter(e => e[1] === 8)).toEqual([["start", 8], ["done", 8, 5], ["start", 8], ["done", 8, 5]])
      const logs = await logsOf(idb)
      expect(levelsOf(logs)).toEqual([8, 16, 32])
      await gateAccepts(logs)
    })

    it("a checkup whose milestone throws: the life goes on, the result has error true, the next log is written", async () => {
      const idb = new IDBFactory()
      const suite = SUITE.map(m => (m.id === "M2" ? { ...m, measure: () => { throw new Error("bad milestone") } } : m))
      await grow(idb, { suite })
      const logs = await logsOf(idb)
      expect(levelsOf(logs)).toEqual([8, 16, 32])
      for (const c of results(logs)) {
        expect(c.error).toBe(true)
        expect(c.measures.M2).toBeNull()
        expect(c.passed).not.toContain("M2")
      }
      expect(logs.at(-1)!.final).toBe(true)
      await gateAccepts(logs)
    })

    it("a recoverable error inside a checkup propagates (resume from checkpoint) and the level reruns once", async () => {
      class Lost extends Error {}
      let m2 = 0
      const suite = SUITE.map(m => (m.id === "M2" ? { ...m, measure: (...a: Parameters<Milestone["measure"]>) => {
        if (m2++ === 1) throw new Lost("device lost") // in the 16 CU checkup
        return m.measure(...a)
      } } : m))
      const idb = new IDBFactory()
      const events: SessionEvent[] = []
      const deps = { recoverable: (e: unknown) => e instanceof Lost, onEvent: (e: SessionEvent) => events.push(e) }
      await expect(grow(idb, { suite, deps })).rejects.toBeInstanceOf(Lost)
      expect((await logsOf(idb)).some(f => f.final)).toBe(false)
      expect((await loadCheckpoint(idb, ID))!.checkups!.lastLevel).toBe(8)
      await grow(idb, { suite, deps })
      expect(checkupEvents(events).filter(e => e[1] === 16).map(e => e[0])).toEqual(["start", "start", "done"])
      const logs = await logsOf(idb)
      expect(levelsOf(logs)).toEqual([8, 16, 32])
      expect(results(logs).some(c => c.error)).toBe(false) // it was not recorded as a milestone error
      await gateAccepts(logs)
    })

    it("every checkpoint a 04c life writes (caregiver, world, checkups, tails empty and not) loads on pre-04c code (decision 0017, rollback)", async () => {
      const idb = new IDBFactory()
      const records: Record<string, unknown>[] = []
      await grow(idb, {
        cfg: { lifetimeCU: 140, sleepInterval: 10, caregiver: CARE },
        deps: { onTick: async () => { records.push((await readCheckpointRecord(idb, ID)) as Record<string, unknown>) } },
      })
      const seen = records.filter(r => r !== undefined)
      for (const r of seen) expect(pre04cCheckpointError(r, ID), `steps ${String(r.steps)}`).toBeNull()
      const tails = seen.map(r => (r.checkups as { tail: unknown[] }).tail.length)
      expect(tails.some(n => n === 0) && tails.some(n => n > 0)).toBe(true)
      expect(seen.every(r => "caregiver" in r && "world" in r)).toBe(true)
      expect(seen.some(r => "suiteVersion" in (r.checkups as object))).toBe(false)
      // The frozen validator is the old one: it rejects what 04c added only when it carries information.
      const one = seen.find(r => (r.checkups as { tail: unknown[] }).tail.length > 0)!
      expect(pre04cCheckpointError({ ...one, checkups: { ...(one.checkups as object), suiteVersion: 1 } }, ID)).toMatch(/checkups must be/)
      expect(pre04cCheckpointError({ ...one, caregiver: { ...(one.caregiver as object), lost: 1 } }, ID)).toMatch(/not a caregiver state/)
    })

    it("a pre-04b checkpoint (no checkups or caregiver fields) resumes: lastLevel 0, caregiver from cfg", async () => {
      const idb = new IDBFactory()
      await expect(grow(idb, { checkups: false, deps: { onTick: ({ steps }) => { if (steps === 12) throw new Error("reload") } } })).rejects.toThrow("reload")
      const old = (await readCheckpointRecord(idb, ID)) as Record<string, unknown>
      expect(old.steps).toBe(10)
      expect("checkups" in old || "caregiver" in old).toBe(false)
      cgSpy.inits.length = 0
      const events: SessionEvent[] = []
      await grow(idb, { cfg: { caregiver: CARE }, deps: { onEvent: e => events.push(e) } })
      expect(events[0]).toMatchObject({ type: "resumed", steps: 10, checkupTail: [] })
      // lastLevel 0: the first clean sleep (step 15, 30 CU) checks the highest level due, 16.
      expect(levelAtOrBelow(30)).toBe(16)
      expect(checkupEvents(events)).toEqual([["start", 16], ["done", 16, 15], ["start", 32], ["done", 32, 20]])
      expect(cgSpy.inits).toEqual([{ settings: clampCaregiverSettings(CARE), seed: hashSeed(7, "caregiver") }])
      await gateAccepts(await logsOf(idb))
    })

    /** A learner factory that records every call made to the learner it makes (its frozen runs' too) and every init. */
    const spyFactory = (inner: (init: LearnerInit) => Promise<SessionLearner>) => {
      const calls: [string, unknown[]][] = []
      const inits: LearnerInit[] = []
      const createLearner = async (init: LearnerInit): Promise<SessionLearner> => {
        inits.push(init)
        const l = await inner(init)
        const wrap = <A extends unknown[], R>(name: string, f: (...a: A) => R) => (...a: A): R => { calls.push([name, a]); return f(...a) }
        return {
          step: wrap("step", l.step), sleep: wrap("sleep", l.sleep), save: wrap("save", l.save), stats: wrap("stats", l.stats), dispose: wrap("dispose", l.dispose),
          ...(l.onKnowledge ? { onKnowledge: wrap("onKnowledge", l.onKnowledge) } : {}),
          ...(l.frozenRun ? { frozenRun: wrap("frozenRun", (fo: FrozenRunOptions): FrozenRun => {
            const r = l.frozenRun!(fo)
            return { step: wrap("frozen.step", r.step), result: wrap("frozen.result", r.result), dispose: wrap("frozen.dispose", r.dispose) }
          }) } : {}),
        }
      }
      return { calls, inits, createLearner }
    }
    /** The Privileged wall (AGENTS.md invariant 2): what the learner was handed holds nothing privileged. */
    const expectWall = (calls: [string, unknown[]][], inits: LearnerInit[]) => {
      const observations = calls.filter(([n]) => n === "step" || n === "frozen.step")
      expect(observations.length).toBeGreaterThan(0)
      const strings: string[] = []
      const noStrings = (v: unknown, path: string) => {
        if (typeof v === "string") strings.push(path)
        else if (v && typeof v === "object" && !ArrayBuffer.isView(v)) for (const [k, x] of Object.entries(v)) noStrings(x, `${path}.${k}`)
      }
      for (const [n, args] of observations) {
        expect(args).toHaveLength(1)
        const o = args[0] as Record<string, unknown>
        expect(Object.keys(o).sort(), n).toEqual(["buttonHeld", "cursor", "fovea", "periphery"])
        expect(o.fovea).toBeInstanceOf(Uint8Array)
        expect(o.periphery).toBeInstanceOf(Uint8Array)
        noStrings(o, n)
      }
      expect(strings).toEqual([])
      for (const i of inits) for (const k of Object.keys(i)) expect(["seed", "widths", "settings", "heldOut", "state"]).toContain(k)
      for (const [, args] of calls.filter(([n]) => n === "onKnowledge")) {
        expect(args).toHaveLength(1)
        for (const k of Object.keys(args[0] as object)) expect(["knowledgeSha", "codeSha", "learnerStateVersion"]).toContain(k)
        expect((args[0] as Manifest).knowledgeSha).toMatch(/^[0-9a-f]{64}$/)
      }
      // No TraceEvent (kind, room, source), PrivilegedState (room, hovered), CheckupResult (passed), CaregiverState
      // (m6Streak) or Checkpoint.world (world, room).
      const FORBIDDEN = ["kind", "room", "source", "hovered", "passed", "m6Streak", "world"]
      const found: string[] = []
      const seen = new Set<object>()
      const scan = (v: unknown, path: string) => {
        if (!v || typeof v !== "object" || ArrayBuffer.isView(v) || seen.has(v)) return
        seen.add(v)
        for (const [k, x] of Object.entries(v)) {
          if (FORBIDDEN.includes(k)) found.push(`${path}.${k}`)
          scan(x, `${path}.${k}`)
        }
      }
      calls.forEach(([n, args], i) => scan(args, `${n}#${i}`))
      inits.forEach((x, i) => scan(x, `init#${i}`))
      expect(found).toEqual([])
    }

    it("Privileged wall: the learner gets observations, birth facts, its own state and manifests, nothing caregiver- or checkup-derived", async () => {
      const idb = new IDBFactory()
      const { calls, inits, createLearner: spyLearner } = spyFactory(stubFactory({ cuPerStep: 2, frozen: { evalPerStep: 0.001 }, knowledge: [] }))
      const inbox = new SessionInbox()
      const events: SessionEvent[] = []
      const deps = (stopAt?: number) => ({
        createLearner: spyLearner, inbox, onEvent: (e: SessionEvent) => events.push(e),
        onTick: ({ steps }: { steps: number }) => { if (steps === 3) inbox.setKnowledge(knowledgeWith("e".repeat(64))); if (steps === stopAt) throw new Error("reload") },
      })
      // 120 steps: long enough for the caregiver's period (50 steps).
      await expect(grow(idb, { cfg: { lifetimeCU: 240, caregiver: CARE }, deps: deps(12) })).rejects.toThrow("reload")
      await grow(idb, { cfg: { lifetimeCU: 240, caregiver: CARE }, deps: deps() })
      // Non-vacuous: checkups ran, the caregiver carried, knowledge was handed over, and a resume created from state.
      expect(events.some(e => e.type === "checkup")).toBe(true)
      expect(events.some(e => e.type === "carried")).toBe(true)
      expect(calls.some(([n]) => n === "onKnowledge")).toBe(true)
      expect(inits.some(i => "state" in i)).toBe(true)
      expect(calls.filter(([n]) => n === "frozen.step").length).toBeGreaterThan(0)
      expectWall(calls, inits)
    })

    describe("world continuity: a resumed or restored learner wakes up in its checkpoint's room (Plan 04c)", () => {
      /** A real world whose drained trace is kept, and the room each frame was seen in (read as the session observes). */
      const tracedWorld = () => {
        const world = createWorld({ seed: 7, canvas: canvas() })
        const trace: TraceEvent[][] = [] // one entry per drain (one per step)
        const rooms: string[] = []
        const w = {
          observe: () => { rooms.push(world.state().room); return world.observe() }, act: (a: Action) => world.act(a),
          state: () => world.state(), rooms: () => world.rooms(),
          bringToFront: (k: Parameters<typeof world.bringToFront>[0], s?: Parameters<typeof world.bringToFront>[1]) => world.bringToFront(k, s),
          drainTrace: () => { const t = world.drainTrace(); trace.push(t); return t },
        }
        return { world, w, trace, rooms }
      }
      // 2 CU a step, a sleep every 5, a carry decision every 50 steps, no checkups. The first life is stopped (a reload)
      // at step 57: its checkpoint is the one at step 55, after the carry at step 50.
      const LIFE = { lifetimeCU: 240, sleepInterval: 5, caregiver: CARE }
      const firstLife = async (idb: IDBFactory, events: SessionEvent[]) => {
        const t = tracedWorld()
        await expect(grow(idb, { checkups: false, cfg: LIFE, deps: { world: t.w, onEvent: e => events.push(e), onTick: ({ steps }) => { if (steps === 57) throw new Error("reload") } } })).rejects.toThrow("reload")
        const c = (await loadCheckpoint(idb, ID))!
        expect(c.steps).toBe(55)
        return { ...t, cp: c }
      }

      it("a resumed life starts in its checkpoint's room", async () => {
        const idb = new IDBFactory()
        const events: SessionEvent[] = []
        const first = await firstLife(idb, events)
        const carried = events.flatMap(e => (e.type === "carried" ? [e] : []))
        expect(carried.map(e => e.steps)).toEqual([50])
        expect(first.world.state().room).toBe(carried[0]!.to)
        expect(carried[0]!.to).not.toBe("crib") // non-vacuous: a fresh world starts in the crib
        expect(first.cp.world).toEqual({ room: carried[0]!.to })
        // Resume on a fresh world (in the crib): before the first step, it is in the checkpoint's room.
        const second = tracedWorld()
        expect(second.world.state().room).toBe("crib")
        let atFirstStep: string | undefined
        const inner = stubFactory({ cuPerStep: 2, frozen: { evalPerStep: 0.001 } })
        const createLearner = async (init: LearnerInit): Promise<SessionLearner> => {
          const l = await inner(init)
          return { ...l, step: async o => { atFirstStep ??= second.world.state().room; return l.step(o) } }
        }
        await grow(idb, { checkups: false, cfg: LIFE, deps: { world: second.w, createLearner } })
        expect(atFirstStep).toBe(carried[0]!.to)
        expect(second.rooms[0]).toBe(carried[0]!.to)
      })

      it("the restore is not a carry and not a learner switch", async () => {
        const idb = new IDBFactory()
        const events: SessionEvent[] = []
        const first = await firstLife(idb, events)
        const room = first.cp.world!.room
        const { calls, inits, createLearner } = spyFactory(stubFactory({ cuPerStep: 2, frozen: { evalPerStep: 0.001 } }))
        const second = tracedWorld()
        const resumed: SessionEvent[] = []
        await grow(idb, { checkups: false, cfg: LIFE, deps: { world: second.w, createLearner, onEvent: e => { events.push(e); resumed.push(e) } } })
        // No carried event for it: the resumed run's carries are the caregiver's own, on its period (step 100).
        expect(resumed.filter(e => e.type === "carried").map(e => (e as Extract<SessionEvent, { type: "carried" }>).steps)).toEqual([100])
        // The files' carries and to are the carried events, once each: the restored room is in neither.
        const carried = events.flatMap(e => (e.type === "carried" ? [e] : []))
        const logs = await logsOf(idb)
        expect(logs.reduce((a, f) => a + f.caregiver!.carries + (f.caregiver!.lostCarries ?? 0), 0)).toBe(carried.length)
        const to: Record<string, number> = {}
        for (const f of logs) for (const [k, n] of Object.entries(f.caregiver!.to)) to[k] = (to[k] ?? 0) + n
        const byRoom: Record<string, number> = {}
        for (const e of carried) byRoom[e.to] = (byRoom[e.to] ?? 0) + 1
        expect(to).toEqual(byRoom)
        // The next file (at step 64) holds the step-50 carry only, as it would without the reload.
        const next = logs.find(f => f.stepsTotal === 64)!
        expect(next.caregiver).toMatchObject({ carries: 1, to: { [room]: 1 } })
        // The first drained trace of the resumed run has the restore as a focus with source harness, and no other focus.
        const focus = second.trace.flat().filter(e => e.kind === "focus")
        expect(second.trace[0]!.filter(e => e.kind === "focus")).toEqual([{ tick: 0, room, kind: "focus", source: "harness" }])
        expect(focus.filter(e => e.source === "harness")).toHaveLength(1)
        expect(focus.filter(e => e.source === "learner")).toHaveLength(0) // M6 counts these only
        expect(focus.filter(e => e.source === "caregiver").map(e => e.room)).toEqual(resumed.flatMap(e => (e.type === "carried" ? [e.to] : [])))
        // The learner was handed nothing but what it always is.
        expect(inits.some(i => "state" in i)).toBe(true)
        expectWall(calls, inits)
        await gateAccepts(logs)
      })

      it("a checkpoint without world (pre-04c) resumes where the world starts", async () => {
        const idb = new IDBFactory()
        const first = await firstLife(idb, [])
        const { world: _w, ...old } = first.cp
        void _w
        expect(first.cp.world!.room).not.toBe("crib")
        await saveCheckpoint(idb, old)
        expect("world" in ((await readCheckpointRecord(idb, ID)) as object)).toBe(false)
        const second = tracedWorld()
        await grow(idb, { checkups: false, cfg: LIFE, deps: { world: second.w } })
        expect(second.rooms[0]).toBe("crib")
        expect(second.trace.flat().filter(e => e.kind === "focus" && e.source === "harness")).toEqual([])
        await gateAccepts(await logsOf(idb))
      })

      it("a checkpoint's room this world does not have is skipped, not an error", async () => {
        const idb = new IDBFactory()
        const first = await firstLife(idb, [])
        const room = first.cp.world!.room
        const kinds = createWorld({ seed: 7, canvas: canvas() }).rooms().filter(k => k !== room)
        const world = createWorld({ seed: 7, canvas: canvas(), rooms: kinds })
        const seen: string[] = []
        await grow(idb, { checkups: false, cfg: LIFE, deps: { world, onTick: () => void seen.push(world.state().room) } })
        expect(seen[0]).toBe(kinds[0])
      })

      it("an in-life restore brings the checkpoint's room back", async () => {
        // 0.5 CU a step, a sleep every 10, carries decided at steps 50 and 100: skips from step 95 restore at 114 to the
        // checkpoint at 90. Before the restore the test moves the world to another room than the checkpoint's.
        const idb = new IDBFactory()
        const t = tracedWorld()
        let poisoned = true, justRestored = false
        let back: Awaited<ReturnType<typeof loadCheckpoint>>
        let before: string | undefined, after: string | undefined
        const events: SessionEvent[] = []
        await grow(idb, {
          checkups: false,
          cfg: { lifetimeCU: 64, sleepInterval: 10, caregiver: CARE }, stub: { cuPerStep: 0.5, skip: n => poisoned && n >= 95 },
          deps: {
            world: t.w,
            onEvent: e => { events.push(e); if (e.type === "restored") { poisoned = false; justRestored = true } },
            onTick: async ({ steps }) => {
              if (steps === 113 && poisoned) { // the first time only: after the restore, step 113 comes again
                back = await loadCheckpoint(idb, ID)
                const other = t.world.rooms().find(k => k !== back!.world!.room)!
                t.world.bringToFront(other)
                before = other
              }
              if (justRestored) { justRestored = false; after = t.rooms.at(-1) }
            },
          },
        })
        expect(back!.steps).toBe(90)
        expect(before).not.toBe(back!.world!.room)
        expect(events.some(e => e.type === "restored")).toBe(true)
        // The first frame after the restore was seen in the checkpoint's room.
        expect(after).toBe(back!.world!.room)
        const harness = t.trace.flat().filter(e => e.kind === "focus" && e.source === "harness")
        expect(harness.map(e => e.room)).toEqual([back!.world!.room])
        // Not a carry: the files' carries and lostCarries are the carried events, once each.
        const logs = await logsOf(idb)
        expect(logs.reduce((a, f) => a + f.caregiver!.carries + (f.caregiver!.lostCarries ?? 0), 0)).toBe(events.filter(e => e.type === "carried").length)
        await gateAccepts(logs)
      })
    })
  })
})
