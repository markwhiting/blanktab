import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { LEARNER_STATE_VERSION } from "@blanktab/learner"
import { checkpointIds, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { openQueue } from "../src/queue"
import { MAX_ORPHANS, ORPHAN_MAX_AGE_MS, sweepCheckpoints, type TryLock } from "../src/orphans"
import { adoptLearnerId, storedLearnerId } from "../src/identity"
import { mkFile, stubStorage } from "./helpers"

const NOW = new Date("2026-10-20T12:00:00.000Z")
const HOUR = 60 * 60 * 1000
const id = (n: number) => `lrn_${n.toString(16).padStart(16, "0")}`

const cp = (learnerId: string, ageMs: number, over: Partial<Checkpoint<unknown>> = {}): Checkpoint<unknown> => ({
  learnerId, savedAt: new Date(NOW.getTime() - ageMs).toISOString(), steps: 100, cu: 1, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0,
  learnerState: { version: LEARNER_STATE_VERSION }, ...over,
})

/** Web Locks as other tabs see them: `held` are the learners alive in some tab. */
function locks(heldByOthers: string[] = []) {
  const held = new Set(heldByOthers)
  const tryLock: TryLock = async learnerId => {
    if (held.has(learnerId)) return null
    held.add(learnerId)
    return () => { held.delete(learnerId) }
  }
  return { tryLock, held }
}

describe("orphaned checkpoints", () => {
  it("six closed tabs and more: the newest eligible orphan is adopted, at most 3 orphans remain, live ones untouched", async () => {
    const idb = new IDBFactory()
    // Six tabs closed 1..6 hours ago (tab 1 the most recent).
    for (let n = 1; n <= 6; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    await saveCheckpoint(idb, cp(id(7), ORPHAN_MAX_AGE_MS)) // closed 24 h ago: stale
    await saveCheckpoint(idb, cp(id(8), 10 * 60 * 1000, { steps: -1 })) // malformed
    await saveCheckpoint(idb, cp(id(9), 10 * 60 * 1000, { learnerState: { version: LEARNER_STATE_VERSION + 1 } })) // other code
    await saveCheckpoint(idb, cp(id(10), 10 * 60 * 1000)) // its final log is queued already
    await (await openQueue(idb)).put(mkFile(id(10), 0, { final: true }))
    // Alive in other tabs (their locks held), however old or odd: never touched.
    await saveCheckpoint(idb, cp(id(11), 5 * 60 * 1000))
    await saveCheckpoint(idb, cp(id(12), 30 * HOUR, { steps: -1 }))
    const l = locks([id(11), id(12)])

    const r = await sweepCheckpoints({ idb, tryLock: l.tryLock, now: () => NOW, adopt: true })
    expect(r.adopted?.learnerId).toBe(id(1))
    expect(r.verdicts).toEqual({
      [id(1)]: "adopted", [id(2)]: "kept", [id(3)]: "kept", [id(4)]: "surplus", [id(5)]: "surplus", [id(6)]: "surplus",
      [id(7)]: "stale", [id(8)]: "invalid", [id(9)]: "version", [id(10)]: "final", [id(11)]: "locked", [id(12)]: "locked",
    })
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3), id(11), id(12)].sort())
    const orphans = (await checkpointIds(idb)).filter(i => i !== id(11) && i !== id(12))
    expect(orphans.length).toBeLessThanOrEqual(MAX_ORPHANS)
    // This tab holds the adopted learner's lock (for its life); every other lock it took is released.
    expect([...l.held].sort()).toEqual([id(1), id(11), id(12)].sort())
    r.adopted!.release()
    expect(l.held.has(id(1))).toBe(false)
  })

  it("without adopting (this tab has a learner) it still bounds the orphans, and skips this tab's own learner", async () => {
    const idb = new IDBFactory()
    for (let n = 1; n <= 5; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    await saveCheckpoint(idb, cp(id(20), 40 * HOUR)) // this tab's (a reload): its lock is about to be taken
    const l = locks()
    const r = await sweepCheckpoints({ idb, tryLock: l.tryLock, now: () => NOW, adopt: false, exclude: [id(20)] })
    expect(r.adopted).toBeNull()
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3), id(20)].sort())
    expect(l.held.size).toBe(0)
  })

  it("an orphan another tab adopts meanwhile is not adopted twice", async () => {
    const idb = new IDBFactory()
    await saveCheckpoint(idb, cp(id(1), HOUR))
    const l = locks()
    const [a, b] = await Promise.all([1, 2].map(() => sweepCheckpoints({ idb, tryLock: l.tryLock, now: () => NOW, adopt: true })))
    expect([a!.adopted?.learnerId, b!.adopted?.learnerId].filter(Boolean)).toEqual([id(1)])
    expect(await checkpointIds(idb)).toEqual([id(1)])
  })

  it("nothing to adopt: null, and nothing deleted", async () => {
    const idb = new IDBFactory()
    const r = await sweepCheckpoints({ idb, tryLock: locks().tryLock, now: () => NOW, adopt: true })
    expect(r).toEqual({ adopted: null, verdicts: {} })
  })

  it("an adopted learnerId becomes this tab's", () => {
    const s = stubStorage()
    expect(storedLearnerId(s)).toBeNull()
    adoptLearnerId(s, id(1))
    expect(storedLearnerId(s)).toBe(id(1))
    expect(() => adoptLearnerId(s, "nope")).toThrow(RangeError)
    s.setItem("blanktab:learnerId", "garbage")
    expect(storedLearnerId(s)).toBeNull()
  })
})
