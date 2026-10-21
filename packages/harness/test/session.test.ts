import { describe, it, expect } from "vitest"
import { IDBFactory, IDBObjectStore as FakeObjectStore } from "fake-indexeddb"
import { NOOP_ACTION, emptyObservation, decodeMove, PERIPHERY, WORLD, type Action, type Observation, type WorldView } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, clampSettings, createLearner, type LearnerSettings, type LearnerState, type LearnerStats, type Widths } from "@blanktab/learner"
import { openQueue } from "../src/queue"
import { loadCheckpoint, readCheckpointRecord, saveCheckpoint } from "../src/checkpoint"
import { MAX_RESTORES, parseLogFile, type LogFile, type LogFileV2 } from "../src/log"
import { createWorld } from "@blanktab/world"
import { createCanvas } from "@napi-rs/canvas"
import { OFFLINE_KNOWLEDGE_SHA, type Manifest } from "../src/knowledge"
import { DEFAULT_CHECKPOINT_EVERY_MS, DEFAULT_MAX_STEPS_PER_SEC, FIRST_LOG_CU, LAST_DOUBLING_CU, SessionInbox, logThresholds, nextCrossing, runSession, createPacer, type Clock, type LearnerInit, type SessionConfig, type SessionDeps, type SessionEvent, type SessionLearner } from "../src/session"
import { mkFile } from "./helpers"

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
function stubFactory(o: { cuPerStep: number; sleepCu?: number; nanUntil?: number; skip?: (steps: number) => boolean; saves?: number[]; savedSteps?: number[]; created?: LearnerInit[]; throwAt?: { steps: number; error: Error }; knowledge?: [number, string][]; onSave?: (steps: number) => void }) {
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
        if (o.throwAt && st.stats.steps + 1 === o.throwAt.steps) throw o.throwAt.error
        st.stats.steps++
        st.stats.cu += o.cuPerStep
        if (o.skip?.(st.stats.steps)) { st.stats.skippedUpdates++; st.stats.consecutiveSkips++ } else st.stats.consecutiveSkips = 0
        loss()
        return NOOP_ACTION
      },
      sleep: async () => { st.stats.sleeps++; st.stats.cu += o.sleepCu ?? 0 },
      save: async () => { o.onSave?.(st.stats.steps); o.saves?.push(st.stats.consecutiveSkips); o.savedSteps?.push(st.stats.steps); return structuredClone(st) as unknown as LearnerState },
      stats: () => ({ ...st.stats }),
      dispose: () => {},
      ...(o.knowledge ? { onKnowledge: (m: Manifest) => void o.knowledge!.push([st.stats.steps, m.knowledgeSha]) } : {}),
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
        expect(events[0]).toEqual({ type: "resumed", steps: cp!.steps, cu: cp!.cu, curveTail: cp!.curveTail })
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

    it("the new knowledgeSha is in the next log file (and every later one); the learner is handed the knowledge", async () => {
      const idb = new IDBFactory()
      const handed: [number, string][] = []
      await live(idb, new SessionInbox(), { at: { 20: knowledgeWith("e".repeat(64)) }, knowledge: handed })
      const logs = await logsOf(idb)
      expect(logs.map(f => [f.stepsTotal, f.knowledgeSha[0]])).toEqual([[16, "f"], [32, "e"], [64, "e"], [128, "e"]])
      expect(logs.every(f => f.sessionsSeenAtBirth === 3)).toBe(true) // still a birth fact
      expect(handed).toEqual([[20, "e".repeat(64)]]) // before step 21, once
      expectContinuous(logs)
    })

    it("offline knowledge (a failed refresh) never replaces the knowledge in force", async () => {
      const idb = new IDBFactory()
      await live(idb, new SessionInbox(), { at: { 20: knowledgeWith("e".repeat(64)), 40: knowledgeWith(OFFLINE_KNOWLEDGE_SHA) } })
      expect((await logsOf(idb)).map(f => f.knowledgeSha[0])).toEqual(["f", "e", "e", "e"])
    })

    it("is kept by the checkpoint: a reload before the next log file still writes it", async () => {
      const idb = new IDBFactory()
      await expect(live(idb, new SessionInbox(), { at: { 20: knowledgeWith("e".repeat(64)) }, stopAt: 25 })).rejects.toThrow("reload")
      expect(await loadCheckpoint(idb, ID)).toMatchObject({ steps: 20, knowledgeSha: "e".repeat(64) }) // the sleep at step 20, just after the knowledge came
      await live(idb, new SessionInbox())
      expect((await logsOf(idb)).map(f => f.knowledgeSha[0])).toEqual(["f", "e", "e", "e"])
    })

    it("knowledge given before the life starts applies from its first step (after the checkpoint's own is read)", async () => {
      const idb = new IDBFactory()
      await expect(live(idb, new SessionInbox(), { stopAt: 10 })).rejects.toThrow("reload")
      const inbox = new SessionInbox()
      inbox.setKnowledge(knowledgeWith("e".repeat(64)))
      await live(idb, inbox)
      expect((await logsOf(idb)).map(f => f.knowledgeSha[0])).toEqual(["e", "e", "e", "e"])
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
      expect(r).toMatchObject({ steps: 1500, seq: 1, halted: true, alreadyFinal: false })
      expect(ticks.at(-1)).toBe(1500) // not one step more
      expect((await logsOf(idb)).map(f => [f.stepsTotal, f.final])).toEqual([[1024, false]])
      expect(await loadCheckpoint(idb, ID)).toMatchObject({ steps: 1500, seq: 1 })
      const events: SessionEvent[] = []
      const again = await runSession(cfgFor({ lifetimeCU: 64, sleepInterval: 50, maxStepsPerSec: 10, checkpointEveryMs: undefined }), depsFor(idb, {
        inbox, createLearner: stubFactory({ cuPerStep: 8 / 1024 }), onEvent: e => events.push(e),
      }))
      expect(events[0]).toMatchObject({ type: "resumed", steps: 1500 })
      expect(again.halted).toBe(false)
      expectContinuous(await logsOf(idb))
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
})
