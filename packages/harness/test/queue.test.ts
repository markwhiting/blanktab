import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { MAX_UNSENT, openQueue } from "../src/queue"
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
  it("prune deletes every learner's sent (or dropped) non-final files except its newest file; unsent files stay", async () => {
    const C = "lrn_cccccccccccccccc", D = "lrn_dddddddddddddddd"
    const q = await openQueue(new IDBFactory())
    for (const f of [mkFile(A, 0), mkFile(A, 1), mkFile(A, 2, { final: true }), mkFile(B, 0), mkFile(B, 1, { final: true }), mkFile(C, 0), mkFile(C, 1), mkFile(C, 2), mkFile(D, 0), mkFile(D, 1)]) await q.put(f)
    for (const [id, seq] of [[A, 0], [A, 1], [A, 2], [B, 0], [C, 0], [C, 1], [C, 2], [D, 0]] as const) await q.markSent(id, seq)
    expect(await q.prune()).toBe(6)
    expect((await q.files(A)).map(f => f.seq)).toEqual([2]) // a finished life: its final file stays, so it is known to be over
    expect((await q.latest(A))?.final).toBe(true)
    expect((await q.files(B)).map(f => f.seq)).toEqual([1]) // B's final file is not sent yet; its sent seq 0 goes
    expect((await q.files(C)).map(f => f.seq)).toEqual([2]) // a living learner, all sent: its newest stays (latest() resumes numbering)
    expect((await q.latest(C))?.seq).toBe(2)
    expect((await q.files(D)).map(f => f.seq)).toEqual([1]) // seq 1 unsent: kept; sent seq 0 goes
    expect(await q.prune()).toBe(0)
  })
  it("markSent records the PR a file went in; dropPr marks that PR's files dropped (sent, never resent) and prune removes them", async () => {
    const q = await openQueue(new IDBFactory())
    for (let s = 0; s < 5; s++) await q.put(mkFile(A, s))
    await q.markSent(A, 0, 7); await q.markSent(A, 1, 7); await q.markSent(A, 2, 8)
    expect(await q.dropPr(A, 7)).toBe(2)
    expect(await q.dropPr(A, 7)).toBe(0) // once
    expect(await q.dropPr(B, 8)).toBe(0) // another learner's PR number
    expect((await q.pending()).map(f => f.seq)).toEqual([3, 4])
    await q.prune()
    expect((await q.files(A)).map(f => f.seq)).toEqual([3, 4]) // the newest (4) is unsent: every sent file goes
  })
  it("capUnsent keeps at most max unsent files per learner, dropping its oldest (this tab's learner too); never its newest or a final", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    for (let s = 0; s < 12; s++) await q.put(mkFile(A, s))
    for (let s = 0; s < 3; s++) await q.put(mkFile(B, s, { final: s === 2 }))
    for (let s = 0; s < 6; s++) await q.put(mkFile("lrn_cccccccccccccccc", s, { final: s === 5 }))
    await q.markSent(A, 0)
    expect(await q.capUnsent(4)).toBe(7 + 2) // A: 11 unsent -> 4 (seq 1-7 go); C: 6 -> 4 (seq 0-1 go); B has 3
    expect((await q.files(A)).map(f => f.seq)).toEqual([0, 8, 9, 10, 11]) // seq 0 was sent: not counted, not dropped by the cap
    expect((await q.files(B)).map(f => f.seq)).toEqual([0, 1, 2])
    expect((await q.files("lrn_cccccccccccccccc")).map(f => f.seq)).toEqual([2, 3, 4, 5])
    expect((await q.latest("lrn_cccccccccccccccc"))?.final).toBe(true)
    expect(await q.capUnsent(4)).toBe(0)
    expect(await q.capUnsent(1)).toBe(3 + 2 + 3)
    expect((await q.files(A)).map(f => f.seq)).toEqual([0, 11])
    expect(MAX_UNSENT).toBe(200)
  })
  it("contribution state: closed-PR streak and local-only, per learner, kept across reopen; a local-only learner's files are never pending", async () => {
    const idb = new IDBFactory()
    const q = await openQueue(idb)
    expect(await q.contribution(A)).toEqual({ learnerId: A, lastPr: 0, closedInARow: 0, localOnly: false })
    for (let s = 0; s < 3; s++) { await q.put(mkFile(A, s)); await q.put(mkFile(B, s)) }
    await q.setContribution({ learnerId: A, lastPr: 9, closedInARow: 3, localOnly: true })
    const again = await openQueue(idb)
    expect(await again.contribution(A)).toEqual({ learnerId: A, lastPr: 9, closedInARow: 3, localOnly: true })
    expect(await again.localOnly()).toEqual([A])
    expect([...new Set((await again.pending()).map(f => f.learnerId))]).toEqual([B])
    expect((await again.files(A)).map(f => f.seq)).toEqual([0, 1, 2]) // still kept (and capped) for the page's curve
    expect((await again.latest(A))?.seq).toBe(2)
  })
})
