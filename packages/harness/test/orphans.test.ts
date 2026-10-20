import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { LEARNER_STATE_VERSION } from "@blanktab/learner"
import { checkpointIds, readCheckpointRecord, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { CHECKPOINTS, CHECKPOINT_META, openDb } from "../src/db"
import { openQueue } from "../src/queue"
import { MAX_ORPHANS, ORPHAN_MAX_AGE_MS, learnerLockName, sweepCheckpoints, waitLock, webLock } from "../src/orphans"
import { adoptLearnerId, storedLearnerId } from "../src/identity"
import { mkFile, stubStorage } from "./helpers"

const NOW = new Date("2026-10-20T12:00:00.000Z")
const HOUR = 60 * 60 * 1000
const id = (n: number) => `lrn_${n.toString(16).padStart(16, "0")}`

const cp = (learnerId: string, ageMs: number, over: Partial<Checkpoint<unknown>> = {}): Checkpoint<unknown> => ({
  learnerId, savedAt: new Date(NOW.getTime() - ageMs).toISOString(), steps: 100, cu: 1, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0,
  learnerState: { version: LEARNER_STATE_VERSION }, ...over,
})

/**
 * Web Locks as the browser keeps them, for every tab of the origin: exclusive locks, ifAvailable, waiting requests
 * (with an abort signal) and query(). `hold(id)` is another tab living that learner.
 */
class FakeLocks {
  readonly held = new Set<string>()
  readonly waiting = new Map<string, (() => void)[]>()
  failRequests = false
  async request(name: string, options: LockOptions, cb: (lock: Lock | null) => unknown): Promise<unknown> {
    if (this.failRequests) throw new DOMException("no locks here", "SecurityError")
    if (this.held.has(name)) {
      if (options.ifAvailable) return cb(null)
      await new Promise<void>((res, rej) => {
        const q = this.waiting.get(name) ?? []
        this.waiting.set(name, q)
        q.push(res)
        options.signal?.addEventListener("abort", () => {
          q.splice(q.indexOf(res), 1)
          rej(options.signal!.reason)
        })
      })
    }
    this.held.add(name)
    try {
      return await cb({ name, mode: "exclusive" } as Lock)
    } finally {
      this.held.delete(name)
      this.waiting.get(name)?.shift()?.()
    }
  }
  async query(): Promise<LockManagerSnapshot> {
    return {
      held: [...this.held].map(name => ({ name, mode: "exclusive" as const })),
      pending: [...this.waiting].flatMap(([name, q]) => q.map(() => ({ name, mode: "exclusive" as const }))),
    }
  }
  /** Another tab takes the learner's lock for its life; resolves to its release. */
  hold(learnerId: string): Promise<() => void> {
    return new Promise(granted => void this.request(learnerLockName(learnerId), {}, () => new Promise<void>(r => granted(r))))
  }
  /** Locks still held, as learnerIds. */
  ids(): string[] {
    return [...this.held].map(n => n.replace("blanktab:learner:", "")).sort()
  }
}
const locks = async (heldByOthers: string[] = []) => {
  const l = new FakeLocks()
  for (const i of heldByOthers) await l.hold(i)
  return { l, tryLock: webLock(l) }
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
    const { l, tryLock } = await locks([id(11), id(12)])
    // A checkpoint written by hand (no meta record): not one this code wrote, so not one to resume.
    const db = await openDb(idb)
    await new Promise(r => { const tx = db.transaction(CHECKPOINTS, "readwrite"); tx.objectStore(CHECKPOINTS).put(cp(id(13), HOUR / 2)); tx.oncomplete = r })

    const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })
    expect(r.adopted?.learnerId).toBe(id(1))
    expect(r.verdicts).toEqual({
      [id(1)]: "adopted", [id(2)]: "kept", [id(3)]: "kept", [id(4)]: "surplus", [id(5)]: "surplus", [id(6)]: "surplus",
      [id(7)]: "stale", [id(8)]: "invalid", [id(9)]: "version", [id(10)]: "final", [id(11)]: "locked", [id(12)]: "locked", [id(13)]: "invalid",
    })
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3), id(11), id(12)].sort())
    const orphans = (await checkpointIds(idb)).filter(i => i !== id(11) && i !== id(12))
    expect(orphans.length).toBeLessThanOrEqual(MAX_ORPHANS)
    // This tab holds the adopted learner's lock (for its life); every other lock it took is released.
    expect(l.ids()).toEqual([id(1), id(11), id(12)].sort())
    r.adopted!.release()
    await Promise.resolve()
    expect(l.ids()).toEqual([id(11), id(12)].sort())
  })

  it("without adopting (this tab has a learner) it still bounds the orphans, and skips this tab's own learner", async () => {
    const idb = new IDBFactory()
    for (let n = 1; n <= 5; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    await saveCheckpoint(idb, cp(id(20), 40 * HOUR)) // this tab's (a reload): its lock is about to be taken
    const { l, tryLock } = await locks()
    const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: false, exclude: [id(20)] })
    expect(r.adopted).toBeNull()
    expect((await checkpointIds(idb)).sort()).toEqual([id(1), id(2), id(3), id(20)].sort())
    await Promise.resolve()
    expect(l.ids()).toEqual([])
  })

  it("an orphan another tab adopts meanwhile is not adopted twice", async () => {
    const idb = new IDBFactory()
    await saveCheckpoint(idb, cp(id(1), HOUR))
    const { tryLock } = await locks()
    const [a, b] = await Promise.all([1, 2].map(() => sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })))
    expect([a!.adopted?.learnerId, b!.adopted?.learnerId].filter(Boolean)).toEqual([id(1)])
    expect(await checkpointIds(idb)).toEqual([id(1)])
  })

  it("nothing to adopt: null, and nothing deleted", async () => {
    const idb = new IDBFactory()
    const r = await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true })
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

  describe("two tabs start at once: one reloading (its own learner X in its sessionStorage), one new", () => {
    const setupX = async () => {
      const idb = new IDBFactory()
      await saveCheckpoint(idb, cp(id(1), 60 * 1000)) // X: the newest checkpoint, its lock free during the reload
      await saveCheckpoint(idb, cp(id(2), 2 * HOUR)) // a real orphan
      return { idb, ...(await locks()) }
    }

    it("the reloading tab asks first: the new tab's sweep leaves X alone and adopts the real orphan", async () => {
      const { idb, l, tryLock } = await setupX()
      const own = waitLock(l, id(1), 1_000)
      const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })
      expect((await own)).not.toBeNull()
      expect(r.adopted?.learnerId).toBe(id(2))
      expect(r.verdicts[id(1)]).toBe("locked")
      expect(await readCheckpointRecord(idb, id(1))).toBeDefined()
    })

    it("the new tab's sweep takes X's lock first: the reloading tab waits, the sweep sees it waiting and lets X go", async () => {
      const { idb, l } = await setupX()
      let own: Promise<unknown> | undefined
      // The reloading tab asks for X right after the sweep took X's lock (the worst moment).
      const tryLock = async (learnerId: string) => {
        const held = await webLock(l)(learnerId)
        if (learnerId === id(1)) own = waitLock(l, id(1), 1_000)
        return held
      }
      const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: true })
      expect(r.verdicts[id(1)]).toBe("wanted")
      expect(r.adopted?.learnerId).toBe(id(2))
      expect(await own).not.toBeNull() // the reloading tab keeps its learner
      expect(await readCheckpointRecord(idb, id(1))).toBeDefined()
    })

    it("nor is X deleted as surplus or stale while its tab waits for it", async () => {
      const idb = new IDBFactory()
      await saveCheckpoint(idb, cp(id(1), 30 * HOUR)) // stale by age, but its tab is reloading right now
      const l = new FakeLocks()
      let own: Promise<unknown> | undefined
      const tryLock = async (learnerId: string) => {
        const held = await webLock(l)(learnerId)
        own ??= waitLock(l, id(1), 1_000)
        return held
      }
      const r = await sweepCheckpoints({ idb, tryLock, now: () => NOW, adopt: false })
      expect(r.verdicts[id(1)]).toBe("wanted")
      expect(await own).not.toBeNull()
      expect(await readCheckpointRecord(idb, id(1))).toBeDefined()
    })
  })

  it("waitLock gives up after its timeout when another tab lives the learner (a duplicated tab)", async () => {
    const { l } = await locks([id(1)])
    expect(await waitLock(l, id(1), 50)).toBeNull()
    expect(l.waiting.get(learnerLockName(id(1)))).toEqual([])
  })

  it("a lock request the browser rejects is 'not acquired', not a hang", async () => {
    const l = new FakeLocks()
    l.failRequests = true
    expect(await webLock(l)(id(1))).toBeNull()
    expect(await waitLock(l, id(1), 10_000)).toBeNull()
  })

  it("the sweep judges checkpoints by their meta records, and reads a full record only to adopt it", async () => {
    const idb = new IDBFactory()
    for (let n = 1; n <= 5; n++) await saveCheckpoint(idb, cp(id(n), n * HOUR))
    const metas = await (async () => { const db = await openDb(idb); return new Promise<unknown[]>(r => { const q = db.transaction(CHECKPOINT_META).objectStore(CHECKPOINT_META).getAll(); q.onsuccess = () => r(q.result) }) })()
    expect(metas).toHaveLength(5)
    expect(metas[0]).toEqual({ learnerId: id(1), savedAt: cp(id(1), HOUR).savedAt, version: LEARNER_STATE_VERSION, steps: 100, seq: 0 })
    const reads: string[] = []
    const { IDBObjectStore: Store } = await import("fake-indexeddb")
    const get = Store.prototype.get
    Store.prototype.get = function (this: IDBObjectStore, key: IDBValidKey | IDBKeyRange) {
      if (this.name === CHECKPOINTS) reads.push(String(key))
      return get.call(this, key)
    } as typeof get
    try {
      await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: false })
      expect(reads).toEqual([])
      await sweepCheckpoints({ idb, tryLock: (await locks()).tryLock, now: () => NOW, adopt: true })
      expect(reads).toEqual([id(1)])
    } finally {
      Store.prototype.get = get
    }
  })
})
