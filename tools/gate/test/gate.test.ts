import { describe, it, expect } from "vitest"
import { decide, activityFromFiles, type OpenPr } from "../gate.ts"
import type { PrFile } from "../validate.ts"
import type { LogFile, LogFileV1 } from "@blanktab/harness/log"

const ID = "lrn_0123456789abcdef"
const NOW = new Date("2026-10-21T12:00:00.000Z")
const hoursAgo = (h: number) => new Date(NOW.getTime() - h * 3_600_000)
const log = (seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId: ID, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: hoursAgo(1).toISOString(), stepsTotal: 100 * (seq + 1), settings: { lr: 0.1 },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})
const file = (l: LogFile): PrFile => ({ path: `data/runs/${ID}/${l.seq}.json`, content: JSON.stringify(l) })
const written = (h: number) => ({ writtenAt: hoursAgo(h).toISOString() })
const pr = (over: Partial<OpenPr> = {}): OpenPr => ({ number: 1, headRef: "run/x", baseRef: "main", updatedAt: null, createdAt: hoursAgo(2), files: [file(log(0))], previous: [], ...over })

describe("decide", () => {
  it("valid + final -> merge", () => {
    const [d] = decide([pr({ files: [file(log(0)), file(log(1, { final: true }))] })], NOW)
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("liveness is the newest log's writtenAt: 47 h -> wait, 49 h -> merge", () => {
    expect(decide([pr({ files: [file(log(0, written(47)))] })], NOW)[0]!.action).toBe("wait")
    expect(decide([pr({ files: [file(log(0, written(49)))] })], NOW)[0]!.action).toBe("merge")
  })
  it("the newest log decides, not the first", () => {
    expect(decide([pr({ files: [file(log(0, written(100))), file(log(1, written(2)))] })], NOW)[0]!.action).toBe("wait")
    expect(decide([pr({ files: [file(log(0, written(100))), file(log(1, written(60)))] })], NOW)[0]!.action).toBe("merge")
  })
  it("activityFromFiles: newest valid log writtenAt; ignores non-logs, malformed and future-dated logs", () => {
    expect(activityFromFiles([file(log(0, written(5))), file(log(1, written(3)))], NOW)?.toISOString()).toBe(hoursAgo(3).toISOString())
    expect(activityFromFiles([{ path: `data/runs/${ID}/0.json`, content: "{nope" }], NOW)).toBeNull()
    expect(activityFromFiles([{ path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: hoursAgo(1).toISOString() }) }], NOW)).toBeNull()
    expect(activityFromFiles([file(log(0, written(-24)))], NOW)).toBeNull()
    expect(activityFromFiles([file(log(0, written(-24))), file(log(1, written(4)))], NOW)?.toISOString()).toBe(hoursAgo(4).toISOString())
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
  it("with only future-dated logs falls back to the PR updatedAt", () => {
    const files = [file(log(0, written(-400 * 24)))]
    expect(decide([pr({ files, updatedAt: hoursAgo(49) })], NOW)[0]!.action).toBe("merge")
    expect(decide([pr({ files, updatedAt: hoursAgo(1) })], NOW)[0]!.action).toBe("wait")
    expect(decide([pr({ files, updatedAt: null })], NOW)[0]!.action).toBe("wait")
  })
  describe("max PR age (7 days from createdAt)", () => {
    const DAY = 24
    it("an active, valid, non-final PR open > 7 days is merged now", () => {
      const [d] = decide([pr({ createdAt: hoursAgo(7 * DAY + 1), files: [file(log(0, written(1)))] })], NOW)
      expect(d).toMatchObject({ action: "merge" })
      expect(d!.reason).toMatch(/7 days/)
    })
    it("exactly 7 days, still active -> wait", () => {
      expect(decide([pr({ createdAt: hoursAgo(7 * DAY), files: [file(log(0, written(1)))] })], NOW)[0]!.action).toBe("wait")
    })
    it("an invalid PR open > 7 days is closed", () => {
      expect(decide([pr({ createdAt: hoursAgo(8 * DAY), files: [file(log(3, written(1)))] })], NOW)[0]!.action).toBe("close")
    })
    it("a future-dated log does not keep an old PR open", () => {
      expect(decide([pr({ createdAt: hoursAgo(8 * DAY), updatedAt: hoursAgo(1), files: [file(log(0, written(-DAY)))] })], NOW)[0]!.action).toBe("merge")
    })
  })
  describe("development builds", () => {
    it("any log with codeSha dev -> close, even when final", () => {
      const [d] = decide([pr({ files: [file(log(0)), file(log(1, { codeSha: "dev", final: true }))] })], NOW)
      expect(d).toEqual({ pr: 1, action: "close", reason: "Development builds don't contribute." })
    })
    it("dev build that is also old or stale -> close, never merge", () => {
      expect(decide([pr({ createdAt: hoursAgo(10 * 24), files: [file(log(0, { codeSha: "dev", ...written(100) }))] })], NOW)[0]).toMatchObject({ action: "close", reason: "Development builds don't contribute." })
    })
  })
})
