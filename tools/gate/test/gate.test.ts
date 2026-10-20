import { describe, it, expect } from "vitest"
import { decide, heartbeatFromFiles, type OpenPr } from "../gate.ts"
import type { PrFile } from "../validate.ts"
import type { LogFile } from "@blanktab/harness/log"

const ID = "lrn_0123456789abcdef"
const NOW = new Date("2026-10-21T12:00:00.000Z")
const hoursAgo = (h: number) => new Date(NOW.getTime() - h * 3_600_000)
const log = (seq: number, over: Partial<LogFile> = {}): LogFile => ({
  schemaVersion: 1, learnerId: ID, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: "2026-10-19T12:00:00.000Z", stepsTotal: 100 * (seq + 1), settings: { lr: 0.1 },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})
const file = (l: LogFile): PrFile => ({ path: `data/runs/${ID}/${l.seq}.json`, content: JSON.stringify(l) })
const hbFile = (at: Date): PrFile => ({ path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: at.toISOString() }) })
const pr = (over: Partial<OpenPr> = {}): OpenPr => ({ number: 1, headRef: "run/x", baseRef: "main", updatedAt: null, files: [file(log(0))], previous: [], heartbeatAt: null, ...over })

describe("decide", () => {
  it("valid + final -> merge", () => {
    const [d] = decide([pr({ files: [file(log(0)), file(log(1, { final: true }))], heartbeatAt: hoursAgo(1) })], NOW)
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("valid, no final, heartbeat 47 h old -> wait", () => {
    expect(decide([pr({ heartbeatAt: hoursAgo(47) })], NOW)[0]!.action).toBe("wait")
  })
  it("heartbeat 49 h old -> merge", () => {
    expect(decide([pr({ heartbeatAt: hoursAgo(49) })], NOW)[0]!.action).toBe("merge")
  })
  it("no heartbeat and no final -> wait", () => {
    expect(decide([pr({ heartbeatAt: null })], NOW)[0]!.action).toBe("wait")
  })
  it("reads the heartbeat time from heartbeat.json in the PR files", () => {
    expect(decide([pr({ files: [file(log(0)), hbFile(hoursAgo(49))] })], NOW)[0]!.action).toBe("merge")
    expect(decide([pr({ files: [file(log(0)), hbFile(hoursAgo(1))] })], NOW)[0]!.action).toBe("wait")
  })
  it("heartbeatFromFiles ignores malformed heartbeats", () => {
    expect(heartbeatFromFiles([{ path: `data/runs/${ID}/heartbeat.json`, content: "{nope" }], NOW)).toBeNull()
    expect(heartbeatFromFiles([file(log(0))], NOW)).toBeNull()
    expect(heartbeatFromFiles([hbFile(hoursAgo(2))], NOW)?.toISOString()).toBe(hoursAgo(2).toISOString())
  })
  it("invalid -> close with the validator error in the reason", () => {
    const [d] = decide([pr({ files: [file(log(1))] })], NOW)
    expect(d!.action).toBe("close")
    expect(d!.reason).toContain("expected seq 0 but got 1")
  })
  it("code path -> label-needs-human", () => {
    const [d] = decide([pr({ files: [file(log(0)), { path: "app/src/main.ts", content: "x" }] })], NOW)
    expect(d!.action).toBe("label-needs-human")
    expect(d!.reason).toContain("app/src/main.ts")
  })
  it("closes PRs whose base is not main, even if otherwise mergeable", () => {
    const [d] = decide([pr({ baseRef: "release", files: [file(log(0, { final: true }))] })], NOW)
    expect(d).toMatchObject({ action: "close", reason: "Session PRs must target main." })
  })
  it("without heartbeat.json falls back to the PR updatedAt", () => {
    expect(decide([pr({ updatedAt: hoursAgo(49) })], NOW)[0]!.action).toBe("merge")
    expect(decide([pr({ updatedAt: hoursAgo(1) })], NOW)[0]!.action).toBe("wait")
  })
  it("a future-dated heartbeat is ignored; updatedAt decides liveness", () => {
    const future = new Date(NOW.getTime() + 400 * 24 * 3_600_000)
    const files = [file(log(0)), hbFile(future)]
    expect(decide([pr({ files, updatedAt: hoursAgo(49) })], NOW)[0]!.action).toBe("merge")
    expect(decide([pr({ files, updatedAt: hoursAgo(1) })], NOW)[0]!.action).toBe("wait")
    expect(heartbeatFromFiles([hbFile(future)], NOW)).toBeNull()
  })
})
