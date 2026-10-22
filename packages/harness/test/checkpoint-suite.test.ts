// The milestone suite moves on (here: a mocked suite 2, of seven milestones) while a checkpoint still holds a tail
// recorded under suite 1 (Plan 04c, decision 0017). The checkpoint and the session read the mocked current suite.
import { describe, it, expect, vi } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { NOOP_ACTION, emptyObservation, type WorldView } from "@blanktab/interface"
import { DEFAULT_SETTINGS, type LearnerState, type LearnerStats } from "@blanktab/learner"
import { loadCheckpoint, readCheckpointRecord, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { MILESTONE_SUITE_VERSION, parseLogFile, type CheckupResult, type LogFileV2 } from "../src/log"
import { openQueue } from "../src/queue"
import { runSession, type SessionEvent, type SessionLearner } from "../src/session"

vi.mock("../src/log", async orig => ({
  ...await orig<typeof import("../src/log")>(),
  MILESTONE_SUITE_VERSION: 2,
  MILESTONE_IDS: { 1: ["M1", "M2", "M3", "M4", "M5", "M6"], 2: ["M1", "M2", "M3", "M4", "M5", "M6", "M7"] },
}))

const ID = "lrn_0123456789abcdef"
const WIDTHS = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const result = (level: number): CheckupResult =>
  ({ level, steps: 4, passed: ["M1"], measures: { M1: 0.5, M2: null, M3: 1, M4: 0, M5: 1, M6: 0 }, evalCU: 0 })
const stats = (steps: number, cu: number): LearnerStats => ({
  steps, cu, predLoss: 0.5, baselineLoss: 0.4, skippedUpdates: 0, consecutiveSkips: 0, sleeps: 0, evalCU: 0,
  heldOutLoss: Number.NaN, heldOutBaseline: Number.NaN, lastLoss: Number.NaN,
})
const checkpoint = (over: Partial<Checkpoint> = {}): Checkpoint => ({
  learnerId: ID, savedAt: new Date(0).toISOString(), steps: 10, cu: 20, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0,
  knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 0, codeSha: "dev",
  learnerState: { seed: 7, widths: WIDTHS, settings: DEFAULT_SETTINGS, stats: stats(10, 20) } as unknown as LearnerState,
  ...over,
})

describe("a checkpoint's tail keeps its suite version", () => {
  it("the current suite here is 2", () => expect(MILESTONE_SUITE_VERSION).toBe(2))

  it("a tail recorded under suite 1 validates against suite 1 after the current suite moves on", async () => {
    const idb = new IDBFactory()
    for (const suiteVersion of [undefined, 1]) {
      const checkups = { lastLevel: 16, tail: [result(8), result(16)], ...(suiteVersion ? { suiteVersion } : {}) }
      await saveCheckpoint(idb, checkpoint({ checkups }))
      expect((await loadCheckpoint(idb, ID))!.checkups).toEqual(checkups)
      // Suite 1 is the rule for it, not the current suite: M7 is not one of suite 1's milestones.
      const seven = { ...checkups, tail: [{ ...result(8), measures: { ...result(8).measures, M7: 1 } }] }
      await saveCheckpoint(idb, checkpoint({ checkups: seven }))
      await expect(loadCheckpoint(idb, ID)).rejects.toThrow(RangeError)
    }
    await saveCheckpoint(idb, checkpoint({ checkups: { lastLevel: 16, tail: [], suiteVersion: 3 } }))
    await expect(loadCheckpoint(idb, ID)).rejects.toThrow(RangeError)
  })

  it("a resumed suite-1 tail goes into the next file under milestoneSuiteVersion 1; no checkup runs before that file", async () => {
    const idb = new IDBFactory()
    // Resumed at step 10, 20 CU, 2 CU a step, a sleep every 5: level 16 is due at the sleep at step 15 (30 CU), but the
    // tail is of another suite. The 32 CU file at step 16 takes it; the sleep at step 20 then checks level 32.
    await saveCheckpoint(idb, checkpoint({ checkups: { lastLevel: 8, tail: [result(8)] } }))
    const createLearner = async (init: unknown): Promise<SessionLearner> => {
      const st = structuredClone((init as { state: { stats: LearnerStats } }).state)
      return {
        step: async () => { st.stats.steps++; st.stats.cu += 2; return NOOP_ACTION },
        sleep: async () => {}, stats: () => ({ ...st.stats }), dispose: () => {},
        save: async () => structuredClone(st) as unknown as LearnerState,
        frozenRun: () => ({ step: async () => NOOP_ACTION, result: () => ({ predLoss: Number.NaN, baselineLoss: Number.NaN, transitions: 0 }), dispose: () => {} }),
      }
    }
    const world: WorldView = { observe: () => emptyObservation(), act: () => {} }
    const events: SessionEvent[] = []
    let afterFile: Checkpoint | undefined
    const saved: Checkpoint[] = []
    await runSession(
      { learnerId: ID, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 0, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
        seed: 7, widths: WIDTHS, learnerSettings: DEFAULT_SETTINGS, lifetimeCU: 42, sleepInterval: 5, checkpointEveryMs: 0 },
      {
        world, idb, createLearner, now: () => new Date(0), clock: { now: () => 0, sleep: async () => {} },
        checkups: { canvas: () => { throw new Error("no canvas needed") }, suite: [] },
        onEvent: e => events.push(e),
        onTick: async ({ steps }) => {
          // The raw record: log.ts's own parseCheckups does not see the mocked suite 2, so a suite-2 tail would not load here.
          const c = (await readCheckpointRecord(idb, ID)) as Checkpoint | undefined
          if (c) saved.push(c)
          if (steps === 17) afterFile = await loadCheckpoint(idb, ID)
        },
      },
    )
    const order = events.flatMap(e => (e.type === "logged" ? [`logged ${e.steps}`] : e.type === "checkupStarted" ? [`checkup ${e.level}`] : []))
    expect(order).toEqual(["logged 16", "checkup 32", "logged 21"])
    const logs = (await (await openQueue(idb)).pending()) as LogFileV2[]
    expect(logs[0]).toMatchObject({ stepsTotal: 16, milestoneSuiteVersion: 1, checkups: [result(8)] })
    // It parses under the real (unmocked) parser: suite 1 is its suite.
    const p = parseLogFile(JSON.parse(JSON.stringify(logs[0])))
    expect(p.ok, p.ok ? "" : p.error).toBe(true)
    // After that file the checkpoint is of the current suite.
    expect(afterFile!.checkups).toEqual({ lastLevel: 8, tail: [] }) // an empty tail carries no suiteVersion
    // A non-empty tail of suite 2 does carry it; the resumed suite-1 tail did not (absent means suite 1).
    expect(saved.filter(c => c.checkups!.tail.length > 0).map(c => [c.steps, c.checkups!.suiteVersion])).toEqual([
      ...saved.filter(c => c.steps < 16).map(c => [c.steps, undefined]),
      ...saved.filter(c => c.steps >= 20).map(c => [c.steps, 2]),
    ])
    expect(saved.some(c => c.steps >= 20)).toBe(true)
    expect(logs[1]).toMatchObject({ final: true, milestoneSuiteVersion: 2 })
    expect(logs[1]!.checkups!.map(c => c.level)).toEqual([32])
  })
})
