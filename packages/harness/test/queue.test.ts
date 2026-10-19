import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { openQueue } from "../src/queue"
import { parseLogFile, type LogFile } from "../src/log"

export const mkFile = (learnerId: string, seq: number, over: Partial<LogFile> = {}): LogFile => ({
  schemaVersion: 1, learnerId, seq, codeSha: "dev", knowledgeSha: "0".repeat(64),
  writtenAt: new Date(0).toISOString(), stepsTotal: (seq + 1) * 500, settings: {},
  hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: false, ...over,
})
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
})
