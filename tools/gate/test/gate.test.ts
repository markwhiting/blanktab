import { describe, it, expect } from "vitest"
import { act, decide, memoBound, type GateOps, type OpenPr } from "../gate.ts"
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
const pr = (over: Partial<OpenPr> = {}): OpenPr => ({ number: 1, title: `run: ${ID}`, headRef: `run/${ID}`, baseRef: "main", files: [file(log(0))], previous: [], author: "ann", boundAuthor: null, ...over })

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
    const [d] = decide([pr({ files: [file(log(1))], previous: [log(0), log(1)] })])
    expect(d!.action).toBe("close")
    expect(d!.reason).toContain("seq 1 is not above seq 1, the highest on main")
  })
  it("after a closed PR, the learner's next PR (a seq gap on main) is merged", () => {
    // seq 0-1 merged; seq 2-3 went in a PR the gate closed; the learner goes on from seq 4.
    const [d] = decide([pr({ files: [file(log(4)), file(log(5))], previous: [log(0), log(1)] })])
    expect(d).toMatchObject({ pr: 1, action: "merge" })
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

describe("decide: a learner belongs to the author of its first merged PR", () => {
  it("bound to the PR's author (any letter case): merged", () => {
    expect(decide([pr({ author: "ann", boundAuthor: "ann", previous: [log(0)], files: [file(log(1))] })])[0]).toMatchObject({ action: "merge" })
    expect(decide([pr({ author: "Ann", boundAuthor: "aNN", previous: [log(0)], files: [file(log(1))] })])[0]).toMatchObject({ action: "merge" })
  })
  it("bound to someone else: closed, saying whose it is", () => {
    const [d] = decide([pr({ author: "mallory", boundAuthor: "ann", previous: [log(0)], files: [file(log(1))] })])
    expect(d).toEqual({ pr: 1, action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
  it("bound, but the PR's author is unknown: closed", () => {
    expect(decide([pr({ author: undefined, boundAuthor: "ann", previous: [log(0)], files: [file(log(1))] })])[0]).toMatchObject({ action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
  it("not bound yet (nothing of it on main): anyone's valid first PR is merged", () => {
    expect(decide([pr({ author: "bob", boundAuthor: null })])[0]).toMatchObject({ action: "merge" })
  })
  it("the title must name the files' learner", () => {
    const [d] = decide([pr({ title: "run: lrn_fedcba9876543210" })])
    expect(d).toMatchObject({ action: "close" })
    expect(d!.reason).toContain(`run: ${ID}`)
    expect(decide([pr({ title: "run: x" })])[0]!.action).toBe("close")
  })
  it("two first PRs for one learner in one run, from two authors: one merges; next run the other is closed as someone else's", () => {
    const { o, calls } = (() => {
      const calls: string[] = []
      const o: GateOps = { close: n => void calls.push(`close ${n}`), label: () => {}, merge: n => void calls.push(`merge ${n}`), log: () => {}, error: () => {} }
      return { o, calls }
    })()
    const a = { ...pr({ number: 1, author: "ann" }), headSha: "c".repeat(40), learnerId: ID }
    const b = { ...pr({ number: 2, author: "bob" }), headSha: "d".repeat(40), learnerId: ID }
    expect(act(decide([a, b]), [a, b], o)).toBe(1)
    expect(calls).toEqual(["merge 1"])
    // Next run: PR 1's merge bound the learner to ann.
    const b2 = { ...b, boundAuthor: "ann", previous: [log(0)] }
    expect(decide([b2])[0]).toEqual({ pr: 2, action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
})

describe("oldest PR first (a later copy never wins an unbound learner's first merge)", () => {
  it("fed newest first, the older PR merges; next run the newer is closed as the older author's", () => {
    const calls: string[] = []
    const o: GateOps = { close: n => void calls.push(`close ${n}`), label: () => {}, merge: n => void calls.push(`merge ${n}`), log: () => {}, error: () => {} }
    const older = { ...pr({ number: 5, author: "ann" }), headSha: "c".repeat(40), learnerId: ID }
    const copy = { ...pr({ number: 9, author: "mallory" }), headSha: "d".repeat(40), learnerId: ID }
    expect(act(decide([copy, older]), [copy, older], o)).toBe(1)
    expect(calls).toEqual(["merge 5"])
    const next = { ...copy, boundAuthor: "ann", previous: [log(0)] }
    expect(decide([next])[0]).toEqual({ pr: 9, action: "close", reason: `Learner ${ID} belongs to @ann.` })
  })
})

describe("memoBound: one owner lookup per learner per run", () => {
  it("looks each learner up once, failures included", () => {
    const asked: string[] = []
    const bound = memoBound(id => { asked.push(id); if (id === "lrn_2222222222222222") throw new Error("gh down"); return id === ID ? "ann" : null })
    expect(bound(ID)).toBe("ann")
    expect(bound(ID)).toBe("ann")
    expect(bound("lrn_1111111111111111")).toBeNull()
    expect(bound("lrn_1111111111111111")).toBeNull()
    expect(() => bound("lrn_2222222222222222")).toThrow("gh down")
    expect(() => bound("lrn_2222222222222222")).toThrow("gh down")
    expect(asked).toEqual([ID, "lrn_1111111111111111", "lrn_2222222222222222"])
  })
})
