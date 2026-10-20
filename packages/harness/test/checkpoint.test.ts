import { describe, it, expect } from "vitest"
import { IDBFactory, IDBObjectStore as FakeObjectStore } from "fake-indexeddb"
import { deleteCheckpoint, hasCheckpoint, loadCheckpoint, saveCheckpoint, type Checkpoint } from "../src/checkpoint"
import { openQueue } from "../src/queue"
import { openDb } from "../src/db"
import { mkFile } from "./helpers"
import { emptyObservation, PERIPHERY, decodeMove, type Action, type Observation } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, createLearner, type LearnerState } from "@blanktab/learner"
import { CHECKPOINTS, result } from "../src/db"
import { packLearnerState, unpackLearnerState } from "../src/pack"

const A = "lrn_aaaaaaaaaaaaaaaa", B = "lrn_bbbbbbbbbbbbbbbb"
const cp = (learnerId: string, steps: number): Checkpoint<{ w: Float32Array }> => ({
  learnerId, savedAt: new Date(0).toISOString(), steps, cu: steps / 100, seq: 1, curveTail: [{ cu: 0.5, steps: 50, predLoss: 0.1, baselineLoss: 0.2 }], restores: 0, lostCU: 0, lostSkips: 0,
  learnerState: { w: Float32Array.of(1, 2, 3) },
})

describe("checkpoints", () => {
  it("hasCheckpoint says whether a learner has one, without reading it", async () => {
    const idb = new IDBFactory()
    expect(await hasCheckpoint(idb, A)).toBe(false)
    await saveCheckpoint(idb, cp(A, 100))
    expect(await hasCheckpoint(idb, A)).toBe(true)
    expect(await hasCheckpoint(idb, B)).toBe(false)
    await deleteCheckpoint(idb, A)
    expect(await hasCheckpoint(idb, A)).toBe(false)
  })
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

  it("a checkpoint write that throws at once (a value that cannot be stored) does not let the log commit alone", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    const put = FakeObjectStore.prototype.put
    FakeObjectStore.prototype.put = function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
      if (this.name === "checkpoints") throw new DOMException("the disk is full", "QuotaExceededError")
      return put.call(this, value, key)
    } as typeof put
    try {
      await expect(q.put(mkFile(A, 0), { save: cp(A, 500) })).rejects.toMatchObject({ name: "QuotaExceededError" })
    } finally {
      FakeObjectStore.prototype.put = put
    }
    expect(await q.pending()).toEqual([])
    await q.put(mkFile(A, 0), { save: cp(A, 500) }) // and the seq is still free
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
  })

  it("rejects a checkpoint for another learner than the log it travels with", async () => {
    const q = await openQueue(new IDBFactory())
    await expect(q.put(mkFile(A, 0), { save: cp(B, 1) })).rejects.toThrow(/learner/)
    await expect(q.put(mkFile(A, 0), { remove: B })).rejects.toThrow(/learner/)
    expect(await q.pending()).toEqual([])
  })

  it("rejects a malformed checkpoint record on load", async () => {
    const idb = new IDBFactory()
    for (const bad of [{ steps: -1 }, { seq: 1.5 }, { cu: Number.NaN }, { curveTail: null }, { savedAt: 3 }, { restores: -1 }, { restores: undefined }, { lostCU: Number.NaN }, { lostSkips: 0.5 }, { knowledgeSha: "nope" }, { sessionsSeenAtBirth: -2 }]) {
      await saveCheckpoint(idb, { ...cp(A, 1), ...bad } as unknown as Checkpoint)
      await expect(loadCheckpoint(idb, A)).rejects.toThrow(/checkpoint/)
    }
    // Birth facts are optional (checkpoints written before they were added) but checked when present.
    await saveCheckpoint(idb, { ...cp(A, 1), knowledgeSha: "a".repeat(64), sessionsSeenAtBirth: 2 })
    expect(await loadCheckpoint(idb, A)).toMatchObject({ knowledgeSha: "a".repeat(64), sessionsSeenAtBirth: 2 })
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

/** A tiny stream whose frames depend on the actions (a bright patch the cursor moves). */
function stream() {
  let x = 40, y = 25, t = 0
  return {
    observe(): Observation {
      const o = emptyObservation()
      o.periphery.fill(30 + (t++ % 5))
      o.periphery.fill(220, (y * PERIPHERY.width + x) * 3, (y * PERIPHERY.width + x) * 3 + 3)
      o.fovea.fill(x % 7 * 30)
      o.cursor = { x: x / (PERIPHERY.width - 1), y: y / (PERIPHERY.height - 1) }
      o.buttonHeld = t % 3 === 0
      return o
    },
    act(a: Action) {
      const { dx, dy } = decodeMove(a.move)
      x = Math.min(PERIPHERY.width - 1, Math.max(0, x + Math.sign(dx)))
      y = Math.min(PERIPHERY.height - 1, Math.max(0, y + Math.sign(dy)))
    },
  }
}
const TINY = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
const rawRecord = async (idb: IDBFactory, id: string) =>
  result((await openDb(idb)).transaction(CHECKPOINTS).objectStore(CHECKPOINTS).get(id) as IDBRequest<Checkpoint<Record<string, unknown>>>)
const learnerCp = (learnerState: LearnerState): Checkpoint => ({ ...cp(A, learnerState.stats.steps), seq: 0, learnerState })

describe("checkpoint frames (write volume)", { timeout: 60_000 }, () => {
  async function lived(steps: number) {
    const world = stream()
    const l = await createLearner({ backend: createCpuBackend(), seed: 3, widths: TINY, settings: { ...DEFAULT_SETTINGS, bufferSize: 100, replayPerSleep: 4 } })
    for (let i = 0; i < steps; i++) {
      world.act(await l.step(world.observe()))
      if (i % 25 === 24) await l.sleep()
    }
    return { l, world }
  }

  it("stores the replay frames gzip-compressed, once per distinct observation, and loads the state back exactly", async () => {
    const { l } = await lived(120)
    const state = await l.save()
    l.dispose()
    const idb = new IDBFactory()
    await saveCheckpoint(idb, learnerCp(state))
    const stored = (await rawRecord(idb, A))!.learnerState
    const frames = stored.frames as { codec: string; count: number; data: Uint8Array }
    expect(frames.codec).toBe("gzip")
    expect(frames.count).toBe(state.replay.length + 1) // the transitions share their frames: n + 1 observations
    expect(frames.data.byteLength).toBeLessThan((frames.count * (state.replay[0]!.obs.periphery.length + state.replay[0]!.obs.fovea.length)) / 10)
    const back = (await loadCheckpoint(idb, A))!.learnerState
    expect(back).toEqual(state)
    for (let i = 1; i < back.replay.length; i++) expect(back.replay[i]!.obs).toBe(back.replay[i - 1]!.next)
    expect(back.last!.obs).toBe(back.replay.at(-1)!.next)
  })

  it("a learner resumed from a stored (packed) checkpoint goes on bit-identically to one resumed from its unstored state", async () => {
    const { l, world } = await lived(60)
    const state = await l.save()
    l.dispose()
    const idb = new IDBFactory()
    await (await openQueue(idb)).put({ ...mkFile(A, 0) }, { save: learnerCp(state) }) // the log + checkpoint path packs too
    expect(((await rawRecord(idb, A))!.learnerState.frames as { codec: string }).codec).toBe("gzip")
    const fromDb = await createLearner({ backend: createCpuBackend(), state: (await loadCheckpoint(idb, A))!.learnerState })
    const direct = await createLearner({ backend: createCpuBackend(), state })
    for (let i = 0; i < 60; i++) {
      const o = world.observe()
      const [a, b] = [await fromDb.step(o), await direct.step(o)]
      expect(a).toEqual(b)
      world.act(a)
      if (i % 25 === 24) { await fromDb.sleep(); await direct.sleep() }
    }
    expect(await fromDb.save()).toEqual(await direct.save())
    fromDb.dispose()
    direct.dispose()
  })

  it("without CompressionStream the frames are stored raw, and still load exactly", async () => {
    const { l } = await lived(30)
    const state = await l.save()
    l.dispose()
    const saved = globalThis.CompressionStream
    try {
      Object.defineProperty(globalThis, "CompressionStream", { value: undefined, configurable: true, writable: true })
      const idb = new IDBFactory()
      await saveCheckpoint(idb, learnerCp(state))
      expect(((await rawRecord(idb, A))!.learnerState.frames as { codec: string }).codec).toBe("raw")
      expect((await loadCheckpoint(idb, A))!.learnerState).toEqual(state)
    } finally {
      Object.defineProperty(globalThis, "CompressionStream", { value: saved, configurable: true, writable: true })
    }
  })

  it("checkpoints written before packing (raw learner states) still load; anything that is not a learner state is stored as is", async () => {
    const { l } = await lived(10)
    const state = await l.save()
    l.dispose()
    expect(await unpackLearnerState(state)).toBe(state)
    for (const odd of [{}, null, { replay: [{ obs: 1 }], last: null }, { w: Float32Array.of(1) }]) expect(await packLearnerState(odd)).toBe(odd)
    expect(await unpackLearnerState(await packLearnerState(state))).toEqual(state)
  })

  it("a corrupt packed frame buffer is a RangeError on load", async () => {
    const { l } = await lived(30)
    const state = await l.save()
    l.dispose()
    const idb = new IDBFactory()
    await saveCheckpoint(idb, learnerCp(state))
    const rec = (await rawRecord(idb, A))!
    const frames = rec.learnerState.frames as { data: Uint8Array }
    const tx = (await openDb(idb)).transaction(CHECKPOINTS, "readwrite")
    tx.objectStore(CHECKPOINTS).put({ ...rec, learnerState: { ...rec.learnerState, frames: { ...frames, data: frames.data.slice(0, 20) } } })
    await new Promise(r => { tx.oncomplete = r })
    await expect(loadCheckpoint(idb, A)).rejects.toThrow(RangeError)
  })
})
