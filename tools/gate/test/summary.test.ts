import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { buildSummary } from "../summary.ts"
import type { LogFile } from "@blanktab/harness/log"

const A = "lrn_aaaaaaaaaaaaaaaa"
const B = "lrn_bbbbbbbbbbbbbbbb"
const log = (id: string, seq: number, over: Partial<LogFile> = {}): LogFile => ({
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
})
