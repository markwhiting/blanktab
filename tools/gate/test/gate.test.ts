import { describe, it, expect } from "vitest"
import { act, decide, type GateOps, type OpenPr } from "../gate.ts"
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
const pr = (over: Partial<OpenPr> = {}): OpenPr => ({ number: 1, headRef: "run/x", baseRef: "main", files: [file(log(0))], previous: [], ...over })

describe("decide", () => {
  it("valid + final -> merge", () => {
    const [d] = decide([pr({ files: [file(log(0)), file(log(1, { final: true }))] })])
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("a valid PR that is not final is merged, however fresh its logs", () => {
    const [d] = decide([pr({ files: [file(log(0, written(0.01)))] })])
    expect(d).toMatchObject({ pr: 1, action: "merge" })
  })
  it("a valid PR is merged however old or new it is; decide never waits", () => {
    for (const h of [0, 1, 47, 49, 24 * 8]) {
      expect(decide([pr({ files: [file(log(0, written(h)))] })])[0]!.action).toBe("merge")
    }
    expect(decide([pr({ files: [file(log(0, written(-400 * 24)))] })])[0]!.action).toBe("merge")
  })
  it("invalid -> close with the validator error in the reason", () => {
    const [d] = decide([pr({ files: [file(log(1))] })])
    expect(d!.action).toBe("close")
    expect(d!.reason).toContain("expected seq 0 but got 1")
  })
  it("code path -> label-needs-human", () => {
    const [d] = decide([pr({ files: [file(log(0)), { path: "app/src/main.ts", content: "x" }] })])
    expect(d!.action).toBe("label-needs-human")
    expect(d!.reason).toContain("app/src/main.ts")
  })
  it("closes PRs whose base is not main, even if otherwise mergeable", () => {
    const [d] = decide([pr({ baseRef: "release", files: [file(log(0, { final: true }))] })])
    expect(d).toMatchObject({ action: "close", reason: "Session PRs must target main." })
  })
  describe("development builds", () => {
    it("any log with codeSha dev -> close, even when final", () => {
      const [d] = decide([pr({ files: [file(log(0)), file(log(1, { codeSha: "dev", final: true }))] })])
      expect(d).toEqual({ pr: 1, action: "close", reason: "Development builds don't contribute." })
    })
    it("dev build that is also old -> close, never merge", () => {
      expect(decide([pr({ files: [file(log(0, { codeSha: "dev", ...written(100) }))] })])[0]).toMatchObject({ action: "close", reason: "Development builds don't contribute." })
    })
  })
})

describe("act", () => {
  const ops = () => {
    const calls: string[] = []
    const o: GateOps = {
      close: n => void calls.push(`close ${n}`), label: n => void calls.push(`label ${n}`),
      merge: (n, sha) => void calls.push(`merge ${n} ${sha}`), log: m => void calls.push(`log ${m}`), error: m => void calls.push(`error ${m}`),
    }
    return { o, calls }
  }
  it("merges a valid PR pinned to its validated head commit", () => {
    const { o, calls } = ops()
    const p = { ...pr(), headSha: "c".repeat(40), learnerId: ID }
    expect(act(decide([p]), [p], o)).toBe(1)
    expect(calls).toContain(`merge 1 ${"c".repeat(40)}`)
  })
  it("a PR loaded without a head commit is not merged (merge never sees undefined); it is logged and left for the next run", () => {
    const { o, calls } = ops()
    const p = { ...pr(), learnerId: ID } // no headSha
    expect(act(decide([p]), [p], o)).toBe(0)
    expect(calls.some(c => c.startsWith("merge"))).toBe(false)
    expect(calls.some(c => c.startsWith("error PR #1") && c.includes("no head commit"))).toBe(true)
  })
  it("merges at most one PR per learner per run", () => {
    const { o, calls } = ops()
    const a = { ...pr({ number: 1 }), headSha: "c".repeat(40), learnerId: ID }, b = { ...pr({ number: 2 }), headSha: "d".repeat(40), learnerId: ID }
    expect(act(decide([a, b]), [a, b], o)).toBe(1)
    expect(calls.filter(c => c.startsWith("merge"))).toEqual([`merge 1 ${"c".repeat(40)}`])
  })
})
