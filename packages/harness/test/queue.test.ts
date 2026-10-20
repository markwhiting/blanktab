import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { MAX_UNSENT, openQueue } from "../src/queue"
import { saveCheckpoint } from "../src/checkpoint"
import { parseLogFile } from "../src/log"
import { mkFile } from "./helpers"

const A = "lrn_aaaaaaaaaaaaaaaa", B = "lrn_bbbbbbbbbbbbbbbb"

describe("queue", () => {
  it("orders pending by learnerId then seq", async () => {
    const q = await openQueue(new IDBFactory())
    for (const f of [mkFile(B, 0), mkFile(A, 1), mkFile(A, 0), mkFile(A, 10), mkFile(A, 2)]) await q.put(f)
    expect((await q.pending()).map(f => `${f.learnerId.slice(4, 5)}${f.seq}`)).toEqual(["a0", "a1", "a2", "a10", "b0"])
  })
  it("markSent removes from pending but keeps the record for latest()", async () => {
    const q = await openQueue(new IDBFactory())
    await q.put(mkFile(A, 0)); await q.put(mkFile(A, 1))
    await q.markSent(A, 1)
    expect((await q.pending()).map(f => f.seq)).toEqual([0])
    expect((await q.latest(A))?.seq).toBe(1)
    expect(await q.latest(B)).toBeUndefined()
  })
  it("files() lists one learner's files, sent or not, by seq", async () => {
    const q = await openQueue(new IDBFactory())
    for (const f of [mkFile(A, 2), mkFile(B, 0), mkFile(A, 0), mkFile(A, 10)]) await q.put(f)
    await q.markSent(A, 0)
    expect((await q.files(A)).map(f => f.seq)).toEqual([0, 2, 10])
    expect(await q.files("lrn_cccccccccccccccc")).toEqual([])
  })
  it("persists across reopen and round-trips valid files", async () => {
    const idb = new IDBFactory()
    await (await openQueue(idb)).put(mkFile(A, 0))
    const p = await (await openQueue(idb)).pending()
    expect(p).toHaveLength(1)
    expect(parseLogFile(p[0]).ok).toBe(true)
  })
  it("put never overwrites an existing (learnerId, seq)", async () => {
    const q = await openQueue(new IDBFactory())
    await q.put(mkFile(A, 0))
    await expect(q.put(mkFile(A, 0, { stepsTotal: 1 }))).rejects.toThrow()
    expect((await q.pending())[0]!.stepsTotal).toBe(500)
  })
  it("prune deletes the sent files of learners whose final file was sent, keeping that final file", async () => {
    const C = "lrn_cccccccccccccccc"
    const q = await openQueue(new IDBFactory())
    for (const f of [mkFile(A, 0), mkFile(A, 1), mkFile(A, 2, { final: true }), mkFile(B, 0), mkFile(B, 1, { final: true }), mkFile(C, 0), mkFile(C, 1)]) await q.put(f)
    for (const [id, seq] of [[A, 0], [A, 1], [A, 2], [B, 0], [C, 0], [C, 1]] as const) await q.markSent(id, seq)
    expect(await q.prune()).toBe(2)
    expect((await q.files(A)).map(f => f.seq)).toEqual([2]) // A's life is over and on GitHub
    expect((await q.latest(A))?.final).toBe(true) // ...and still known to be over
    expect((await q.files(B)).map(f => f.seq)).toEqual([0, 1]) // B's final file is not sent yet
    expect((await q.files(C)).map(f => f.seq)).toEqual([0, 1]) // C is still living
    expect(await q.prune()).toBe(0)
  })
  it("capUnsent keeps at most max unsent files: the oldest learners' go first, never this tab's or one with a checkpoint", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    const L = (n: number) => `lrn_${n.toString(16).padStart(16, "0")}`
    const at = (n: number) => new Date(Date.UTC(2026, 9, 1) + n * 3_600_000).toISOString()
    // Six finished lives of 3 files each, written an hour apart (L1 oldest), none sent (no token).
    for (let n = 1; n <= 6; n++) for (let s = 0; s < 3; s++) await q.put(mkFile(L(n), s, { writtenAt: at(n), final: s === 2 }))
    await saveCheckpoint(idb, { learnerId: L(2), savedAt: at(2), steps: 1, cu: 0, seq: 3, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0, learnerState: {} })
    expect(await q.capUnsent(10, [L(1)])).toBe(9) // L1 is this tab's, L2 has a checkpoint: L3, L4 and L5 go
    const pending = await q.pending()
    expect(pending.length).toBeLessThanOrEqual(10)
    expect([...new Set(pending.map(f => f.learnerId))]).toEqual([L(1), L(2), L(6)])
    for (const n of [3, 4, 5]) {
      expect((await q.files(L(n))).map(f => f.seq)).toEqual([2]) // the final file stays, as sent
      expect((await q.latest(L(n)))?.final).toBe(true)
    }
    expect(await q.capUnsent(10, [L(1)])).toBe(0)
    expect(MAX_UNSENT).toBe(200)
  })
})
