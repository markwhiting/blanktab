import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { buildSummary } from "../summary.ts"
import type { LogFile, LogFileV1, LogFileV2 } from "@blanktab/harness/log"

const A = "lrn_aaaaaaaaaaaaaaaa"
const B = "lrn_bbbbbbbbbbbbbbbb"
const log = (id: string, seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId: id, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: "2026-10-19T12:00:00.000Z", stepsTotal: 100 * (seq + 1), settings: { moveProbability: 0.5 },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})

let dir: string
beforeEach(() => { dir = mkdtempSync(join(tmpdir(), "summary-")) })
afterEach(() => { rmSync(dir, { recursive: true, force: true }) })
const put = (l: LogFile) => {
  mkdirSync(join(dir, l.learnerId), { recursive: true })
  writeFileSync(join(dir, l.learnerId, `${l.seq}.json`), JSON.stringify(l))
}

describe("buildSummary", () => {
  it("one row per learner from its highest-seq file", () => {
    for (let s = 0; s <= 5; s++) put(log(A, s, { final: s === 5, sessionsSeenAtBirth: 3 }))
    put(log(B, 0)); put(log(B, 1, { settings: { moveProbability: 0.9 } }))
    writeFileSync(join(dir, B, "heartbeat.json"), JSON.stringify({ learnerId: B, writtenAt: "2026-10-19T12:00:00.000Z" }))
    const s = buildSummary(dir)
    expect(s.schemaVersion).toBe(1)
    expect(s.sessions).toHaveLength(2)
    const a = s.sessions.find(r => r.learnerId === A)!
    const b = s.sessions.find(r => r.learnerId === B)!
    expect(a).toMatchObject({ stepsTotal: 600, final: true, sessionsSeenAtBirth: 3, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64) })
    expect(b).toMatchObject({ stepsTotal: 200, final: false, settings: { moveProbability: 0.9 } })
  })
  it("records lastWrittenAt from the highest-seq file", () => {
    put(log(A, 0, { writtenAt: "2026-10-19T12:00:00.000Z" })); put(log(A, 1, { writtenAt: "2026-10-19T12:05:00.000Z" }))
    expect(buildSummary(dir).sessions[0]!.lastWrittenAt).toBe("2026-10-19T12:05:00.000Z")
  })
  it("sorts numerically (seq 10 beats seq 9) and rows by learnerId", () => {
    for (let s = 0; s <= 10; s++) put(log(B, s))
    put(log(A, 0))
    const s = buildSummary(dir)
    expect(s.sessions.map(r => r.learnerId)).toEqual([A, B])
    expect(s.sessions[1]!.stepsTotal).toBe(1100)
  })
  it("skips unparseable files and non-learner dirs; missing dir gives empty", () => {
    put(log(A, 0))
    mkdirSync(join(dir, "junk")); writeFileSync(join(dir, "junk", "0.json"), "{}")
    mkdirSync(join(dir, B)); writeFileSync(join(dir, B, "0.json"), "not json")
    expect(buildSummary(dir).sessions.map(r => r.learnerId)).toEqual([A])
    expect(buildSummary(join(dir, "nope")).sessions).toEqual([])
  })
  it("rows gain cu and lossRatio (the last curve point's predLoss / baselineLoss); null for v1 or without a point", () => {
    const v2 = (id: string, seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => ({
      ...log(id, seq), schemaVersion: 2, cu: 2 ** seq, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 },
      curve: [{ cu: 2 ** seq, steps: 100 * (seq + 1), predLoss: 0.02, baselineLoss: 0.025 }], skippedUpdates: 0, restores: 0, lostCU: 0, ...over,
    })
    put(v2(A, 0)); put(v2(A, 1, { final: true, curve: [{ cu: 1.5, steps: 150, predLoss: 0.03, baselineLoss: 0.04 }, { cu: 2, steps: 200, predLoss: 0.018, baselineLoss: 0.024 }] }))
    put(v2(B, 0, { curve: [{ cu: 1, steps: 100, predLoss: 0.05, baselineLoss: 0.04 }] })); put(v2(B, 1, { curve: [] })) // falls back to an earlier file's point
    const C = "lrn_cccccccccccccccc"
    put(log(C, 0))
    const D = "lrn_dddddddddddddddd"
    put(v2(D, 0, { curve: [] }))
    const rows = Object.fromEntries(buildSummary(dir).sessions.map(r => [r.learnerId, r]))
    expect(rows[A]).toMatchObject({ cu: 2, final: true })
    expect(rows[A]!.lossRatio).toBeCloseTo(0.75, 12)
    expect(rows[B]).toMatchObject({ cu: 2 })
    expect(rows[B]!.lossRatio).toBeCloseTo(1.25, 12)
    expect(rows[C]).toMatchObject({ cu: null, lossRatio: null })
    expect(rows[D]).toMatchObject({ cu: 1, lossRatio: null })
  })
  it("a zero baseline gives lossRatio null, not Infinity", () => {
    put({ ...log(A, 0), schemaVersion: 2, cu: 1, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 }, curve: [{ cu: 1, steps: 100, predLoss: 0.1, baselineLoss: 0 }], skippedUpdates: 0, restores: 0, lostCU: 0 })
    expect(buildSummary(dir).sessions[0]!.lossRatio).toBeNull()
    expect(JSON.parse(JSON.stringify(buildSummary(dir))).sessions[0].lossRatio).toBeNull()
  })
  it("rows gain heldOutRatio: the latest curve point with a held-out score, loss / baseline; null without one", () => {
    const v2 = (id: string, seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => ({
      ...log(id, seq), schemaVersion: 2, cu: 2 ** seq, condition: "A", widths: { conv1: 8, conv2: 16, enc: 64, hidden: 128 },
      curve: [{ cu: 2 ** seq, steps: 100 * (seq + 1), predLoss: 0.02, baselineLoss: 0.025 }], skippedUpdates: 0, restores: 0, lostCU: 0, ...over,
    })
    const p = (cu: number, steps: number, heldOut?: { loss: number; baseline: number }) => ({ cu, steps, predLoss: 0.01, baselineLoss: 0.04, ...(heldOut ? { heldOut } : {}) })
    put(v2(A, 0, { curve: [p(1, 100, { loss: 0.5, baseline: 0.4 })] }))
    put(v2(A, 1, { curve: [p(1.5, 150, { loss: 0.3, baseline: 0.4 }), p(2, 200)] })) // the newest point has none: the latest one that has
    put(v2(B, 0, { curve: [p(1, 100, { loss: 0.2, baseline: 0.4 })] })); put(v2(B, 1, { curve: [] }))
    const C = "lrn_cccccccccccccccc"
    put(v2(C, 0)) // an older v2 learner: no held-out scores at all
    const D = "lrn_dddddddddddddddd"
    put(v2(D, 0, { curve: [p(1, 100, { loss: 0.2, baseline: 0 })] })) // a zero baseline
    const rows = Object.fromEntries(buildSummary(dir).sessions.map(r => [r.learnerId, r]))
    expect(rows[A]!.heldOutRatio).toBeCloseTo(0.75, 12)
    expect(rows[A]!.lossRatio).toBeCloseTo(0.25, 12) // the own-stream ratio stays as it was
    expect(rows[B]!.heldOutRatio).toBeCloseTo(0.5, 12)
    expect(rows[C]!.heldOutRatio).toBeNull()
    expect(rows[D]!.heldOutRatio).toBeNull()
  })
})
