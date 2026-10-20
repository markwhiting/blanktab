import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD } from "@blanktab/interface"
import { createWorld } from "@blanktab/world"
import { createFlailer } from "@blanktab/learner"
import { openQueue, type Queue } from "../src/queue"
import { runSession, resumePoint, type SessionConfig } from "../src/session"
import { getOrCreateLearnerId } from "../src/identity"
import { parseLogFile } from "../src/log"
import { stubStorage } from "./helpers"

const cfgFor = (learnerId: string, over: Partial<SessionConfig> = {}): SessionConfig => ({
  learnerId, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 0,
  settings: { moveProbability: 0.5, note: "x" }, hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 1 },
  lifetimeSteps: 3000, logEverySteps: 500, startSeq: 0, startSteps: 0, ...over,
})
const deps = (queue: Queue, onTick?: (s: { steps: number }) => void) => ({
  world: createWorld({ seed: 1, canvas: createCanvas(WORLD.width, WORLD.height) }),
  learner: createFlailer({ seed: 2, moveProbability: 0.5 }),
  queue, now: () => new Date(1_700_000_000_000), ...(onTick ? { onTick } : {}),
})

describe("session", { timeout: 60_000 }, () => {
  it("writes 6 files: seq 0-4 every 500 steps, seq 5 final at 3000", async () => {
    const q = await openQueue(new IDBFactory())
    const id = getOrCreateLearnerId(stubStorage())
    await runSession(cfgFor(id), deps(q))
    const p = await q.pending()
    expect(p.map(f => f.seq)).toEqual([0, 1, 2, 3, 4, 5])
    expect(p.map(f => f.stepsTotal)).toEqual([500, 1000, 1500, 2000, 2500, 3000])
    expect(p.map(f => f.final)).toEqual([false, false, false, false, false, true])
    for (const f of p) expect(parseLogFile(JSON.parse(JSON.stringify(f))).ok).toBe(true)
  })

  it("reload: same id, resume continues at seq 2 with no duplicate", async () => {
    const q = await openQueue(new IDBFactory())
    const storage = stubStorage()
    const id = getOrCreateLearnerId(storage)
    await expect(runSession(cfgFor(id), deps(q, s => { if (s.steps === 1200) throw new Error("interrupted") }))).rejects.toThrow("interrupted")
    expect((await q.pending()).map(f => f.seq)).toEqual([0, 1])
    expect(getOrCreateLearnerId(storage)).toBe(id)
    const r = await resumePoint(q, id)
    expect(r).toEqual({ startSeq: 2, startSteps: 1000 })
    await runSession(cfgFor(id, r), deps(q))
    const p = await q.pending()
    expect(p.map(f => f.seq)).toEqual([0, 1, 2, 3, 4, 5])
    expect(p.at(-1)).toMatchObject({ final: true, stepsTotal: 3000 })
  })

  it("resumePoint is {0,0} for an unknown learner and survives markSent", async () => {
    const q = await openQueue(new IDBFactory())
    const id = "lrn_0123456789abcdef"
    expect(await resumePoint(q, id)).toEqual({ startSeq: 0, startSteps: 0 })
    await runSession(cfgFor(id, { lifetimeSteps: 1000 }), deps(q))
    await q.markSent(id, 0); await q.markSent(id, 1)
    expect(await q.pending()).toEqual([])
    expect(await resumePoint(q, id)).toEqual({ startSeq: 2, startSteps: 1000 })
  })

  it("two tabs: distinct ids, no shared (learnerId, seq) in one queue", async () => {
    const q = await openQueue(new IDBFactory())
    const a = getOrCreateLearnerId(stubStorage()), b = getOrCreateLearnerId(stubStorage())
    expect(a).not.toBe(b)
    await Promise.all([runSession(cfgFor(a), deps(q)), runSession(cfgFor(b), deps(q))])
    const p = await q.pending()
    expect(p).toHaveLength(12)
    expect(new Set(p.map(f => `${f.learnerId}/${f.seq}`)).size).toBe(12)
    for (const f of p) expect(parseLogFile(f).ok).toBe(true)
  })

  it("lifetime not a multiple of logEvery: seq 0 at 500, final seq 1 at 600", async () => {
    const q = await openQueue(new IDBFactory())
    const id = "lrn_0123456789abcdef"
    await runSession(cfgFor(id, { lifetimeSteps: 600, logEverySteps: 500 }), deps(q))
    const p = await q.pending()
    expect(p.map(f => [f.seq, f.stepsTotal, f.final])).toEqual([[0, 500, false], [1, 600, true]])
  })

  it("rejects non-positive-integer logEverySteps", async () => {
    const q = await openQueue(new IDBFactory())
    await expect(runSession(cfgFor("lrn_0123456789abcdef", { logEverySteps: 0 }), deps(q))).rejects.toThrow()
  })

  it("awaits pace after every step when given one", async () => {
    const q = await openQueue(new IDBFactory())
    const paced: number[] = []
    await runSession(cfgFor("lrn_0123456789abcdef", { lifetimeSteps: 120, logEverySteps: 100 }), { ...deps(q), pace: async (steps: number) => { paced.push(steps) } })
    expect(paced).toEqual(Array.from({ length: 120 }, (_, i) => i + 1))
    expect((await q.pending()).map(f => f.seq)).toEqual([0, 1])
  })
})
