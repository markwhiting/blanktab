import { describe, it, expect } from "vitest"
import { validateSessionPr, type PrFile } from "../validate.ts"
import type { CurvePoint, LogFile, LogFileV1, LogFileV2 } from "@blanktab/harness/log"

const ID = "lrn_0123456789abcdef"
const log = (seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId: ID, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: "2026-10-19T12:00:00.000Z", stepsTotal: 100 * (seq + 1), settings: { lr: 0.1, mode: "a" },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})
const file = (l: LogFile, id = ID): PrFile => ({ path: `data/runs/${id}/${l.seq}.json`, content: JSON.stringify(l) })
const errs = (files: PrFile[], prev: LogFile[] = []) => validateSessionPr(files, prev)

const pt = (cu: number, steps: number, predLoss = 0.02, baselineLoss = 0.025): CurvePoint => ({ cu, steps, predLoss, baselineLoss })
/** A v2 file: CU doubles with seq (seq 0 at 1 CU), 400 steps per CU. */
const logV2 = (seq: number, over: Partial<LogFileV2> = {}): LogFileV2 => {
  const cu = 2 ** seq
  return {
    ...log(seq), schemaVersion: 2, stepsTotal: 400 * cu, cu, condition: "A", widths: { conv1: 16, conv2: 32, enc: 128, hidden: 256 },
    settings: { lr: 0.001, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 2000, replayPerSleep: 64, headLrScale: 0.1, maxStepsPerSec: 30, seed: 7 },
    curve: [pt(cu * 0.75, 300 * cu), pt(cu, 400 * cu)], skippedUpdates: 0, restores: 0, lostCU: 0, ...over,
  }
}

describe("validateSessionPr v2", () => {
  const v2raw = (mut: (o: Record<string, any>) => void, seq = 0): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
    const o = JSON.parse(JSON.stringify(logV2(seq))); mut(o)
    return { path: `data/runs/${ID}/${seq}.json`, content: JSON.stringify(o) }
  }
  const msg = (files: PrFile[], prev: LogFile[] = []) => errs(files, prev).errors.join("\n")
  it("accepts a valid v2 PR, alone and continuing from previous v2 (or v1) files", () => {
    expect(errs([file(logV2(0)), file(logV2(1)), file(logV2(2, { final: true }))])).toEqual({ ok: true, errors: [], kind: "log" })
    expect(errs([file(logV2(2))], [logV2(0), logV2(1)]).ok).toBe(true)
    expect(errs([file(logV2(0, { curve: [] }))]).ok).toBe(true)
  })
  it("v1 files still validate", () => {
    expect(errs([file(log(0)), file(log(1, { final: true }))]).ok).toBe(true)
  })
  it("rejects cu that decreases across seq, within the PR and against previous files", () => {
    expect(msg([file(logV2(0)), file(logV2(1, { cu: 0.5, curve: [] }))])).toContain("cu decreased")
    expect(msg([file(logV2(2, { cu: 1.5, curve: [] }))], [logV2(0), logV2(1)])).toContain("cu decreased")
    expect(errs([file(logV2(1, { cu: 1, curve: [] }))], [logV2(0)]).ok).toBe(true) // equal is not a decrease
  })
  it("restores, lostCU and skippedUpdates never decrease across seq (restores stay on the record)", () => {
    expect(errs([file(logV2(0, { restores: 1, lostCU: 0.5, skippedUpdates: 20 })), file(logV2(1, { restores: 2, lostCU: 0.5, skippedUpdates: 20 }))]).ok).toBe(true)
    expect(msg([file(logV2(0, { restores: 2 })), file(logV2(1, { restores: 1 }))])).toContain("restores decreased")
    expect(msg([file(logV2(1, { lostCU: 0.25 }))], [logV2(0, { lostCU: 0.5 })])).toContain("lostCU decreased")
    expect(msg([file(logV2(0, { skippedUpdates: 5 })), file(logV2(1, { skippedUpdates: 4 }))])).toContain("skippedUpdates decreased")
  })
  it("bounds restores (integer 0..1000) and lostCU (finite >= 0); both are required in v2", () => {
    expect(errs([file(logV2(0, { restores: 1000 }))]).ok).toBe(true)
    for (const r of [1001, -1, 1.5]) expect(msg([file(logV2(0, { restores: r }))]), String(r)).toContain("restores")
    for (const l of [-0.1]) expect(msg([file(logV2(0, { lostCU: l }))])).toContain("lostCU")
    expect(msg([v2raw(o => { delete o.restores })])).toContain("restores")
    expect(msg([v2raw(o => { delete o.lostCU })])).toContain("lostCU")
    const f = file(logV2(0, { lostCU: 0.123456 }))
    f.content = f.content.replace("0.123456", "1e999")
    expect(msg([f])).toContain("lostCU")
  })
  it("rejects a curve of 65 points; 64 is fine", () => {
    const curve = (n: number) => Array.from({ length: n }, (_, i) => pt(i / 100, i))
    expect(msg([file(logV2(0, { curve: curve(65) }))])).toContain("at most 64")
    expect(errs([file(logV2(0, { curve: curve(64) }))]).ok).toBe(true)
  })
  it("rejects a NaN, an infinity or a negative number in the curve", () => {
    for (const [bad, why] of [["NaN", "invalid JSON"], ["1e999", "curve[0]"], ["-1e999", "curve[0]"], ["-0.5", "curve[0]"], ["null", "curve[0]"], ["\"0.1\"", "curve[0]"]]) {
      const f = file(logV2(0, { curve: [pt(0.5, 200, 0.0123456)] }))
      f.content = f.content.replace("0.0123456", bad!)
      expect(msg([f]), bad).toContain(why)
    }
    // JSON.stringify turns NaN into null: still rejected.
    expect(msg([file(logV2(0, { curve: [pt(0.5, 200, Number.NaN)] }))])).toContain("curve[0]")
  })
  it("rejects malformed curve points and v2 fields", () => {
    expect(msg([v2raw(o => { o.curve[0].extra = 1 })])).toContain("unknown key")
    expect(msg([v2raw(o => { delete o.curve[0].steps })])).toContain("curve[0]")
    expect(msg([v2raw(o => { o.curve = {} })])).toContain("curve must be an array")
    expect(msg([v2raw(o => { o.curve = [pt(0.9, 360), pt(0.5, 380)] })])).toContain("goes back")
    expect(msg([v2raw(o => { o.curve = [pt(2, 360)] })])).toContain("ahead")
    expect(msg([v2raw(o => { o.curve = [pt(0.5, 401)] })])).toContain("ahead")
    expect(msg([v2raw(o => { o.cu = -1 })])).toContain("cu must be")
    expect(msg([v2raw(o => { delete o.cu })])).toContain("cu must be")
    expect(msg([v2raw(o => { o.condition = "B" })])).toContain("condition")
    expect(msg([v2raw(o => { o.widths.extra = 1 })])).toContain("unknown key")
    expect(msg([v2raw(o => { o.widths.hidden = 0 })])).toContain("widths.hidden")
    expect(msg([v2raw(o => { o.widths.enc = 1e9 })])).toContain("widths.enc")
    expect(msg([v2raw(o => { o.skippedUpdates = 1.5 })])).toContain("skippedUpdates")
    expect(msg([v2raw(o => { o.note = "spam" })])).toContain("unknown key")
  })
  it("v2 keys are not allowed on a v1 file", () => {
    expect(msg([withRawV1(o => { o.cu = 1 })])).toContain("unknown key")
  })
  it("the settings rules (keys, values, count) apply to v2", () => {
    expect(msg([file(logV2(0, { settings: { "a-b": 1 } }))])).toContain("settings key")
    expect(msg([file(logV2(0, { settings: Object.fromEntries(Array.from({ length: 51 }, (_, i) => [`k${i}`, i])) }))])).toContain("at most 50")
  })
})

const withRawV1 = (mut: (o: Record<string, any>) => void): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
  const o = JSON.parse(file(log(0)).content); mut(o)
  return { path: `data/runs/${ID}/0.json`, content: JSON.stringify(o) }
}

describe("validateSessionPr", () => {
  it("accepts seq 0-2", () => {
    const r = errs([file(log(0)), file(log(1)), file(log(2, { final: true }))])
    expect(r).toEqual({ ok: true, errors: [], kind: "log" })
  })
  it("heartbeat.json is no longer a session log path (like any other non-log path)", () => {
    const hb: PrFile = { path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: "2026-10-19T12:00:00.000Z" }) }
    const r = errs([file(log(0)), hb])
    expect(r.ok).toBe(false)
    expect(r.kind).toBe("needs-human")
    expect(r.errors.join("\n")).toContain(`data/runs/${ID}/heartbeat.json: path is not a session log path`)
    expect(r.errors.join("\n")).not.toContain("heartbeat.json)")
  })
  it("accepts continuing from previous", () => {
    expect(errs([file(log(2))], [log(0), log(1)]).ok).toBe(true)
  })
  it("rejects non-log paths as needs-human", () => {
    const r = errs([file(log(0)), { path: "app/src/main.ts", content: "x" }])
    expect(r.ok).toBe(false)
    expect(r.kind).toBe("needs-human")
    expect(r.errors.join("\n")).toContain("app/src/main.ts")
  })
  it("rejects a malformed learner dir as needs-human", () => {
    const r = errs([{ path: "data/runs/lrn_XYZ/0.json", content: "{}" }])
    expect(r.kind).toBe("needs-human")
  })
  it("rejects mixed learner ids", () => {
    const other = "lrn_fedcba9876543210"
    const r = errs([file(log(0)), file(log(1, { learnerId: other }), other)])
    expect(r.ok).toBe(false)
    expect(r.errors.join("\n")).toContain("same learnerId")
  })
  it("rejects path/learnerId mismatch", () => {
    const r = errs([file(log(0, { learnerId: "lrn_fedcba9876543210" }))])
    expect(r.errors.join("\n")).toContain("does not match learnerId")
  })
  const sizeErr = (content: string) => errs([{ path: `data/runs/${ID}/0.json`, content }]).errors.join("\n").includes("exceeds 1 MB")
  it("rejects content over 1 MB measured in UTF-8 bytes", () => {
    const content = "é".repeat(600_000) // 600k chars, 1.2M bytes
    expect(content.length).toBeLessThan(1_048_576)
    expect(sizeErr(content)).toBe(true)
  })
  it("size boundary: 1_048_576 bytes passes the size rule, 1_048_577 does not", () => {
    expect(sizeErr("a".repeat(1_048_576))).toBe(false)
    expect(sizeErr("a".repeat(1_048_577))).toBe(true)
    expect(sizeErr("é".repeat(524_288))).toBe(false) // exactly 1_048_576 bytes
    expect(sizeErr("é".repeat(524_288) + "a")).toBe(true)
  })
  it("rejects overlong or illegal settings keys", () => {
    expect(errs([file(log(0, { settings: { ["a".repeat(65)]: 1 } }))]).ok).toBe(false)
    expect(errs([file(log(0, { settings: { ["a".repeat(64)]: 1 } }))]).ok).toBe(true)
    for (const k of ["", "1abc", "a-b", "a b", "é"])
      expect(errs([file(log(0, { settings: { [k]: 1 } }))]).errors.join("\n")).toContain("settings key")
  })
  it("limits settings to 50 keys", () => {
    const mk = (n: number) => Object.fromEntries(Array.from({ length: n }, (_, i) => [`k${i}`, i]))
    expect(errs([file(log(0, { settings: mk(51) }))]).errors.join("\n")).toContain("at most 50")
    expect(errs([file(log(0, { settings: mk(50) }))]).ok).toBe(true)
  })
  it("rejects duplicate seq", () => {
    const r = errs([file(log(0)), { path: `data/runs/${ID}/0.json`, content: JSON.stringify(log(0)) }])
    expect(r.errors.join("\n")).toContain("duplicate seq 0")
  })
  it("seq must be strictly greater than the highest seq already on main (decision 0014: gaps are allowed, never a repeat)", () => {
    expect(errs([file(log(1))], [log(0), log(1)]).errors.join("\n")).toContain("seq 1 is not above seq 1, the highest on main")
    expect(errs([file(log(0))], [log(0), log(1)]).errors.join("\n")).toContain("seq 0 is not above seq 1, the highest on main")
    expect(errs([file(log(1)), file(log(2))], [log(0), log(1)]).errors.join("\n")).toContain("seq 1 is not above")
    // the order on main counts, not the order of `previous`
    expect(errs([file(log(2))], [log(3), log(0)]).ok).toBe(false)
  })
  it("a gap after main's highest seq is allowed: the files of a PR the gate closed are simply missing", () => {
    expect(errs([file(log(3))], [log(0), log(1)]).ok).toBe(true)
    expect(errs([file(log(5)), file(log(9))], [log(0), log(1)]).ok).toBe(true)
    expect(errs([file(log(1))]).ok).toBe(true) // nothing on main yet: the first PR (seq 0) was closed
    expect(errs([file(log(4)), file(log(7, { final: true }))]).ok).toBe(true)
  })
  it("a gap may not reach past the exact integers", () => {
    expect(errs([file(log(Number.MAX_SAFE_INTEGER))]).ok).toBe(true)
    const big = Number.MAX_SAFE_INTEGER + 1
    const f = { path: `data/runs/${ID}/${big}.json`, content: JSON.stringify(log(big)) }
    expect(errs([f]).errors.join("\n")).toContain("beyond the largest exact integer")
  })
  it("the other rules still hold across a gap: stepsTotal and the v2 counters never decrease, final is last", () => {
    expect(errs([file(log(5, { stepsTotal: 150 }))], [log(0), log(1)]).errors.join("\n")).toContain("stepsTotal decreased")
    expect(errs([file(logV2(6, { cu: 1, curve: [] }))], [logV2(0), logV2(1), logV2(2)]).errors.join("\n")).toContain("cu decreased")
    expect(errs([file(log(4, { final: true })), file(log(7))]).errors.join("\n")).toContain("highest seq")
    expect(errs([file(log(9))], [log(0), log(1, { final: true })]).ok).toBe(false)
  })
  it("rejects decreasing stepsTotal", () => {
    const r = errs([file(log(0)), file(log(1, { stepsTotal: 50 }))])
    expect(r.errors.join("\n")).toContain("stepsTotal decreased")
  })
  it("rejects stepsTotal below previous", () => {
    const r = errs([file(log(2, { stepsTotal: 5 }))], [log(0), log(1)])
    expect(r.errors.join("\n")).toContain("stepsTotal decreased")
  })
  it("rejects non-finite numbers", () => {
    const f = file(log(0))
    f.content = f.content.replace('"lr":0.1', '"lr":1e999')
    const r = errs([f])
    expect(r.ok).toBe(false)
    expect(r.errors.join("\n")).toContain("finite")
  })
  it("rejects two final files", () => {
    const r = errs([file(log(0, { final: true })), file(log(1, { final: true }))])
    expect(r.errors.join("\n")).toContain("at most one final")
  })
  it("rejects final that is not the highest seq", () => {
    const r = errs([file(log(0, { final: true })), file(log(1))])
    expect(r.errors.join("\n")).toContain("highest seq")
  })
  it("rejects invalid JSON and bad schema", () => {
    expect(errs([{ path: `data/runs/${ID}/0.json`, content: "{nope" }]).errors.join("\n")).toContain("invalid JSON")
    expect(errs([file(log(0, { schemaVersion: 3 as 1 }))]).errors.join("\n")).toContain("schemaVersion")
  })
  const withRaw = (mut: (o: Record<string, any>) => void): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
    const o = JSON.parse(file(log(0)).content); mut(o)
    return { path: `data/runs/${ID}/0.json`, content: JSON.stringify(o) }
  }
  it("rejects unknown top-level keys", () => {
    expect(errs([withRaw(o => { o.note = "spam" })]).errors.join("\n")).toContain("unknown key")
  })
  it("rejects unknown hostSig keys", () => {
    expect(errs([withRaw(o => { o.hostSig.extra = 1 })]).errors.join("\n")).toContain("unknown key")
  })
  it("rejects extra non-finite numbers", () => {
    const f = withRaw(o => { o.extra = 1 })
    f.content = f.content.replace('"extra":1', '"extra":1e999')
    const r = errs([f])
    expect(r.ok).toBe(false)
    expect(r.errors.join("\n")).toContain("unknown key")
  })
  it("rejects overlong settings strings", () => {
    expect(errs([file(log(0, { settings: { s: "a".repeat(201) } }))]).errors.join("\n")).toContain("200")
    expect(errs([file(log(0, { settings: { s: "a".repeat(200) } }))]).ok).toBe(true)
  })
  it("rejects payload-carrying writtenAt", () => {
    const evil = "Jan 1 2026 (" + "x".repeat(100_000) + ")"
    expect(Number.isNaN(Date.parse(evil))).toBe(false)
    expect(errs([file(log(0, { writtenAt: evil }))]).errors.join("\n")).toContain("writtenAt")
  })
  it("rejects writtenAt not in toISOString format", () => {
    for (const w of ["2026-10-19", "2026-10-19T12:00:00Z", "2026-10-19T12:00:00.000+00:00", "2026-13-45T99:99:99.000Z"])
      expect(errs([file(log(0, { writtenAt: w }))]).errors.join("\n")).toContain("writtenAt")
  })
  it("validates codeSha", () => {
    for (const c of ["a".repeat(41), "g".repeat(40), "A".repeat(40), ""])
      expect(errs([file(log(0, { codeSha: c }))]).errors.join("\n")).toContain("codeSha")
    expect(errs([file(log(0, { codeSha: "dev" }))]).ok).toBe(true)
  })
  it("validates knowledgeSha", () => {
    for (const c of ["0".repeat(63), "0".repeat(65), "z".repeat(64)])
      expect(errs([file(log(0, { knowledgeSha: c }))]).errors.join("\n")).toContain("knowledgeSha")
  })
  it("rejects an empty PR", () => {
    expect(errs([]).ok).toBe(false)
  })
  describe("previous-state hardening", () => {
    it("rejects a previous entry belonging to another learner", () => {
      const r = errs([file(log(1))], [log(0, { learnerId: "lrn_fedcba9876543210" })])
      expect(r.ok).toBe(false)
      expect(r.errors.join("\n")).toContain("previous")
    })
    it("rejects any new log when previous already has a final file", () => {
      const r = errs([file(log(2))], [log(0), log(1, { final: true })])
      expect(r.ok).toBe(false)
      expect(r.errors.join("\n")).toContain("final")
    })
    it("rejects a heartbeat-only PR after final", () => {
      const hb: PrFile = { path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: "2026-10-19T12:00:00.000Z" }) }
      expect(errs([hb], [log(0, { final: true })]).ok).toBe(false)
    })
  })
})

describe("validateSessionPr v2: the held-out check's optional fields (decision 0015 §3)", () => {
  const ho = (loss: number, baseline: number) => ({ heldOut: { loss, baseline } })
  const raw = (mut: (o: Record<string, any>) => void, seq = 0, over: Partial<LogFileV2> = {}): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
    const o = JSON.parse(JSON.stringify(logV2(seq, over))); mut(o)
    return { path: `data/runs/${ID}/${seq}.json`, content: JSON.stringify(o) }
  }
  const msg = (files: PrFile[], prev: LogFile[] = []) => errs(files, prev).errors.join("\n")
  it("accepts curve points with heldOut and files with evalCU, and old v2 files without either", () => {
    const f = logV2(0, { evalCU: 0.05, curve: [{ ...pt(0.75, 300), ...ho(0.02, 0.025) }, pt(1, 400)] })
    expect(errs([file(f)])).toEqual({ ok: true, errors: [], kind: "log" })
    expect(errs([file(logV2(0))]).ok).toBe(true)
    expect(errs([file(logV2(1, { evalCU: 0 }))], [logV2(0)]).ok).toBe(true) // a file before the field existed
  })
  it("heldOut is strict when present: exactly { loss, baseline }, each finite and >= 0 (no upper bound: a diverged learner's result is kept)", () => {
    const at = (h: unknown) => raw(o => { o.curve[0].heldOut = h })
    expect(errs([at({ loss: 1e30, baseline: 3 })]).ok).toBe(true)
    expect(errs([at({ loss: 0, baseline: 0 })]).ok).toBe(true)
    expect(msg([at({ loss: 0.1, baseline: 0.1, extra: 1 })])).toContain('unknown key "extra" in curve[0].heldOut')
    for (const bad of [{ loss: 0.1 }, { baseline: 0.1 }, { loss: -0.1, baseline: 0.1 }, { loss: 0.1, baseline: -0.1 },
      { loss: "0.1", baseline: 0.1 }, { loss: null, baseline: 0.1 }, null, [], 0.1]) {
      expect(msg([at(bad)]), JSON.stringify(bad)).toContain("curve[0].heldOut")
    }
    const f = file(logV2(0, { curve: [{ ...pt(1, 400), ...ho(0.123456, 0.2) }] }))
    f.content = f.content.replace("0.123456", "1e999")
    expect(msg([f])).toContain("curve[0].heldOut")
  })
  it("evalCU is finite and >= 0 when present, and never decreases across seq where both files have it", () => {
    for (const bad of [-1, "1", null]) expect(msg([raw(o => { o.evalCU = bad })]), String(bad)).toContain("evalCU")
    const f = file(logV2(0, { evalCU: 0.123456 }))
    f.content = f.content.replace("0.123456", "1e999")
    expect(msg([f])).toContain("evalCU")
    expect(msg([file(logV2(0, { evalCU: 2 })), file(logV2(1, { evalCU: 1 }))])).toContain("evalCU decreased")
    expect(msg([file(logV2(1, { evalCU: 1 }))], [logV2(0, { evalCU: 2 })])).toContain("evalCU decreased")
    expect(errs([file(logV2(0, { evalCU: 1 })), file(logV2(1, { evalCU: 1 }))]).ok).toBe(true)
  })
  it("within one learner, a file without evalCU after one with it is rejected (never from a number back to absent)", () => {
    expect(msg([file(logV2(0, { evalCU: 1 })), file(logV2(1))])).toContain("evalCU missing")
    expect(msg([file(logV2(1))], [logV2(0, { evalCU: 1 })])).toContain("evalCU missing")
    expect(errs([file(logV2(0)), file(logV2(1, { evalCU: 1 }))]).ok).toBe(true) // absent, then present: a newer page
    expect(errs([file(logV2(1))], [logV2(0)]).ok).toBe(true)
  })
})
