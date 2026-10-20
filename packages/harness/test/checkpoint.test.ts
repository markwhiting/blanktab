import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { deleteCheckpoint, loadCheckpoint, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { openQueue } from "../src/queue"
import { openDb } from "../src/db"
import { mkFile } from "./helpers"

const A = "lrn_aaaaaaaaaaaaaaaa", B = "lrn_bbbbbbbbbbbbbbbb"
const cp = (learnerId: string, steps: number): Checkpoint<{ w: Float32Array }> => ({
  learnerId, savedAt: new Date(0).toISOString(), steps, cu: steps / 100, seq: 1, curveTail: [{ cu: 0.5, steps: 50, predLoss: 0.1, baselineLoss: 0.2 }], restores: 0, lostCU: 0, lostSkips: 0,
  learnerState: { w: Float32Array.of(1, 2, 3) },
})

describe("checkpoints", () => {
  it("keeps only the latest checkpoint per learnerId, structured-clone exact, until deleted", async () => {
    const idb = new IDBFactory()
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    await saveCheckpoint(idb, cp(A, 100))
    await saveCheckpoint(idb, cp(A, 250))
    await saveCheckpoint(idb, cp(B, 7))
    const a = await loadCheckpoint<{ w: Float32Array }>(idb, A)
    expect(a).toEqual(cp(A, 250))
    expect(a!.learnerState.w).toBeInstanceOf(Float32Array)
    expect((await loadCheckpoint(idb, B))?.steps).toBe(7)
    await deleteCheckpoint(idb, A)
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    expect((await loadCheckpoint(idb, B))?.steps).toBe(7)
    await deleteCheckpoint(idb, A) // deleting nothing is fine
  })

  it("a log and a checkpoint change are queued in one transaction: both happen or neither", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0), { save: cp(A, 500) })
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
    expect((await loadCheckpoint(idb, A))?.steps).toBe(500)
    // A duplicate seq aborts the transaction: the checkpoint is not replaced either.
    await expect(q.put(mkFile(A, 0, { stepsTotal: 1 }), { save: cp(A, 999) })).rejects.toThrow()
    expect((await loadCheckpoint(idb, A))?.steps).toBe(500)
    await q.put(mkFile(A, 1, { final: true }), { remove: A })
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    expect((await q.pending()).map(f => f.seq)).toEqual([0, 1])
  })

  it("rejects a checkpoint for another learner than the log it travels with", async () => {
    const q = await openQueue(new IDBFactory())
    await expect(q.put(mkFile(A, 0), { save: cp(B, 1) })).rejects.toThrow(/learner/)
    await expect(q.put(mkFile(A, 0), { remove: B })).rejects.toThrow(/learner/)
    expect(await q.pending()).toEqual([])
  })

  it("rejects a malformed checkpoint record on load", async () => {
    const idb = new IDBFactory()
    for (const bad of [{ steps: -1 }, { seq: 1.5 }, { cu: Number.NaN }, { curveTail: null }, { savedAt: 3 }, { restores: -1 }, { restores: undefined }, { lostCU: Number.NaN }, { lostSkips: 0.5 }]) {
      await saveCheckpoint(idb, { ...cp(A, 1), ...bad } as unknown as Checkpoint)
      await expect(loadCheckpoint(idb, A)).rejects.toThrow(/checkpoint/)
    }
  })

  it("upgrades a v1 database (outbox only): queued logs survive and the checkpoint store appears", async () => {
    const idb = new IDBFactory()
    await new Promise<void>((res, rej) => {
      const req = idb.open("blanktab", 1)
      req.onupgradeneeded = () => req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
      req.onerror = () => rej(req.error)
      req.onsuccess = () => {
        const db = req.result
        const tx = db.transaction("outbox", "readwrite")
        tx.objectStore("outbox").add({ learnerId: A, seq: 0, sent: false, file: mkFile(A, 0) })
        tx.oncomplete = () => { db.close(); res() }
        tx.onerror = () => rej(tx.error)
      }
    })
    const q = await openQueue(idb)
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
    await saveCheckpoint(idb, cp(A, 3))
    expect((await loadCheckpoint(idb, A))?.steps).toBe(3)
  })
})

describe("the database connection", () => {
  it("an upgrade blocked by an old tab's open connection fails after a timeout with a message the page can show", async () => {
    const idb = new IDBFactory()
    const old = await new Promise<IDBDatabase>((res, rej) => {
      const req = idb.open("blanktab", 1) // old code: no versionchange handler, never closes
      req.onupgradeneeded = () => req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
      req.onsuccess = () => res(req.result)
      req.onerror = () => rej(req.error)
    })
    await expect(openDb(idb, 50)).rejects.toThrow("Close other learner tabs to update")
    old.close() // the old tab goes away: the next attempt upgrades
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0))
    expect((await q.pending()).length).toBe(1)
  })

  it("steps aside for a version change (here: the database being deleted) and reopens on the next call", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    await q.put(mkFile(A, 0))
    await saveCheckpoint(idb, cp(A, 1))
    await new Promise<void>((res, rej) => { const r = idb.deleteDatabase("blanktab"); r.onsuccess = () => res(); r.onerror = () => rej(r.error); r.onblocked = () => rej(new Error("blocked")) })
    await q.put(mkFile(A, 1)) // the same Queue object keeps working
    expect((await q.pending()).map(f => f.seq)).toEqual([1])
    expect(await loadCheckpoint(idb, A)).toBeUndefined()
    await saveCheckpoint(idb, cp(A, 2))
    expect((await loadCheckpoint(idb, A))?.steps).toBe(2)
  })
})
