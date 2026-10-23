import { describe, it, expect } from "vitest"
import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { validateSessionPr, type PrFile, type ValidateOptions } from "../validate.ts"
import { parseLogFile, type CurvePoint, type LogFile, type LogFileV1, type LogFileV2, type MetaBlock } from "@blanktab/harness/log"
import { genomeSha, parseCentre, parseGenomeSpace, type Genome } from "@blanktab/meta"

const KNOWLEDGE = join(__dirname, "../../../knowledge")
const SPACE = parseGenomeSpace(JSON.parse(readFileSync(join(KNOWLEDGE, "genome-space.json"), "utf8")))
const CENTRE = parseCentre(JSON.parse(readFileSync(join(KNOWLEDGE, "genome.json"), "utf8")), SPACE)

const ID = "lrn_0123456789abcdef"
const log = (seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId: ID, seq, codeSha: "a".repeat(40), knowledgeSha: "0".repeat(64),
  writtenAt: "2026-10-19T12:00:00.000Z", stepsTotal: 100 * (seq + 1), settings: { lr: 0.1, mode: "a" },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})
const file = (l: LogFile, id = ID): PrFile => ({ path: `data/runs/${id}/${l.seq}.json`, content: JSON.stringify(l) })
const errs = (files: PrFile[], prev: LogFile[] = [], opts: ValidateOptions = { space: SPACE }) => validateSessionPr(files, prev, opts)

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
  const msg = async (files: PrFile[], prev: LogFile[] = []) => (await errs(files, prev)).errors.join("\n")
  it("accepts a valid v2 PR, alone and continuing from previous v2 (or v1) files", async () => {
    expect(await errs([file(logV2(0)), file(logV2(1)), file(logV2(2, { final: true }))])).toEqual({ ok: true, errors: [], kind: "log" })
    expect((await errs([file(logV2(2))], [logV2(0), logV2(1)])).ok).toBe(true)
    expect((await errs([file(logV2(0, { curve: [] }))])).ok).toBe(true)
  })
  it("v1 files still validate", async () => {
    expect((await errs([file(log(0)), file(log(1, { final: true }))])).ok).toBe(true)
  })
  it("rejects cu that decreases across seq, within the PR and against previous files", async () => {
    expect(await msg([file(logV2(0)), file(logV2(1, { cu: 0.5, curve: [] }))])).toContain("cu decreased")
    expect(await msg([file(logV2(2, { cu: 1.5, curve: [] }))], [logV2(0), logV2(1)])).toContain("cu decreased")
    expect((await errs([file(logV2(1, { cu: 1, curve: [] }))], [logV2(0)])).ok).toBe(true) // equal is not a decrease
  })
  it("restores, lostCU and skippedUpdates never decrease across seq (restores stay on the record)", async () => {
    expect((await errs([file(logV2(0, { restores: 1, lostCU: 0.5, skippedUpdates: 20 })), file(logV2(1, { restores: 2, lostCU: 0.5, skippedUpdates: 20 }))])).ok).toBe(true)
    expect(await msg([file(logV2(0, { restores: 2 })), file(logV2(1, { restores: 1 }))])).toContain("restores decreased")
    expect(await msg([file(logV2(1, { lostCU: 0.25 }))], [logV2(0, { lostCU: 0.5 })])).toContain("lostCU decreased")
    expect(await msg([file(logV2(0, { skippedUpdates: 5 })), file(logV2(1, { skippedUpdates: 4 }))])).toContain("skippedUpdates decreased")
  })
  it("bounds restores (integer 0..1000) and lostCU (finite >= 0); both are required in v2", async () => {
    expect((await errs([file(logV2(0, { restores: 1000 }))])).ok).toBe(true)
    for (const r of [1001, -1, 1.5]) expect(await msg([file(logV2(0, { restores: r }))]), String(r)).toContain("restores")
    for (const l of [-0.1]) expect(await msg([file(logV2(0, { lostCU: l }))])).toContain("lostCU")
    expect(await msg([v2raw(o => { delete o.restores })])).toContain("restores")
    expect(await msg([v2raw(o => { delete o.lostCU })])).toContain("lostCU")
    const f = file(logV2(0, { lostCU: 0.123456 }))
    f.content = f.content.replace("0.123456", "1e999")
    expect(await msg([f])).toContain("lostCU")
  })
  it("rejects a curve of 65 points; 64 is fine", async () => {
    const curve = (n: number) => Array.from({ length: n }, (_, i) => pt(i / 100, i))
    expect(await msg([file(logV2(0, { curve: curve(65) }))])).toContain("at most 64")
    expect((await errs([file(logV2(0, { curve: curve(64) }))])).ok).toBe(true)
  })
  it("rejects a NaN, an infinity or a negative number in the curve", async () => {
    for (const [bad, why] of [["NaN", "invalid JSON"], ["1e999", "curve[0]"], ["-1e999", "curve[0]"], ["-0.5", "curve[0]"], ["null", "curve[0]"], ["\"0.1\"", "curve[0]"]]) {
      const f = file(logV2(0, { curve: [pt(0.5, 200, 0.0123456)] }))
      f.content = f.content.replace("0.0123456", bad!)
      expect(await msg([f]), bad).toContain(why)
    }
    // JSON.stringify turns NaN into null: still rejected.
    expect(await msg([file(logV2(0, { curve: [pt(0.5, 200, Number.NaN)] }))])).toContain("curve[0]")
  })
  it("rejects malformed curve points and v2 fields", async () => {
    expect(await msg([v2raw(o => { o.curve[0].extra = 1 })])).toContain("unknown key")
    expect(await msg([v2raw(o => { delete o.curve[0].steps })])).toContain("curve[0]")
    expect(await msg([v2raw(o => { o.curve = {} })])).toContain("curve must be an array")
    expect(await msg([v2raw(o => { o.curve = [pt(0.9, 360), pt(0.5, 380)] })])).toContain("goes back")
    expect(await msg([v2raw(o => { o.curve = [pt(2, 360)] })])).toContain("ahead")
    expect(await msg([v2raw(o => { o.curve = [pt(0.5, 401)] })])).toContain("ahead")
    expect(await msg([v2raw(o => { o.cu = -1 })])).toContain("cu must be")
    expect(await msg([v2raw(o => { delete o.cu })])).toContain("cu must be")
    expect(await msg([v2raw(o => { o.condition = "B" })])).toContain("condition")
    expect(await msg([v2raw(o => { o.widths.extra = 1 })])).toContain("unknown key")
    expect(await msg([v2raw(o => { o.widths.hidden = 0 })])).toContain("widths.hidden")
    expect(await msg([v2raw(o => { o.widths.enc = 1e9 })])).toContain("widths.enc")
    expect(await msg([v2raw(o => { o.skippedUpdates = 1.5 })])).toContain("skippedUpdates")
    expect(await msg([v2raw(o => { o.note = "spam" })])).toContain("unknown key")
  })
  it("v2 keys are not allowed on a v1 file", async () => {
    expect(await msg([withRawV1(o => { o.cu = 1 })])).toContain("unknown key")
  })
  it("the settings rules (keys, values, count) apply to v2", async () => {
    expect(await msg([file(logV2(0, { settings: { "a-b": 1 } }))])).toContain("settings key")
    expect(await msg([file(logV2(0, { settings: Object.fromEntries(Array.from({ length: 51 }, (_, i) => [`k${i}`, i])) }))])).toContain("at most 50")
  })
})

const withRawV1 = (mut: (o: Record<string, any>) => void): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
  const o = JSON.parse(file(log(0)).content); mut(o)
  return { path: `data/runs/${ID}/0.json`, content: JSON.stringify(o) }
}

describe("validateSessionPr", () => {
  it("accepts seq 0-2", async () => {
    const r = await errs([file(log(0)), file(log(1)), file(log(2, { final: true }))])
    expect(r).toEqual({ ok: true, errors: [], kind: "log" })
  })
  it("heartbeat.json is no longer a session log path (like any other non-log path)", async () => {
    const hb: PrFile = { path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: "2026-10-19T12:00:00.000Z" }) }
    const r = await errs([file(log(0)), hb])
    expect(r.ok).toBe(false)
    expect(r.kind).toBe("needs-human")
    expect(r.errors.join("\n")).toContain(`data/runs/${ID}/heartbeat.json: path is not a session log path`)
    expect(r.errors.join("\n")).not.toContain("heartbeat.json)")
  })
  it("accepts continuing from previous", async () => {
    expect((await errs([file(log(2))], [log(0), log(1)])).ok).toBe(true)
  })
  it("rejects non-log paths as needs-human", async () => {
    const r = await errs([file(log(0)), { path: "app/src/main.ts", content: "x" }])
    expect(r.ok).toBe(false)
    expect(r.kind).toBe("needs-human")
    expect(r.errors.join("\n")).toContain("app/src/main.ts")
  })
  it("rejects a malformed learner dir as needs-human", async () => {
    const r = await errs([{ path: "data/runs/lrn_XYZ/0.json", content: "{}" }])
    expect(r.kind).toBe("needs-human")
  })
  it("rejects mixed learner ids", async () => {
    const other = "lrn_fedcba9876543210"
    const r = await errs([file(log(0)), file(log(1, { learnerId: other }), other)])
    expect(r.ok).toBe(false)
    expect(r.errors.join("\n")).toContain("same learnerId")
  })
  it("rejects path/learnerId mismatch", async () => {
    const r = await errs([file(log(0, { learnerId: "lrn_fedcba9876543210" }))])
    expect(r.errors.join("\n")).toContain("does not match learnerId")
  })
  const sizeErr = async (content: string) => (await errs([{ path: `data/runs/${ID}/0.json`, content }])).errors.join("\n").includes("exceeds 1 MB")
  it("rejects content over 1 MB measured in UTF-8 bytes", async () => {
    const content = "é".repeat(600_000) // 600k chars, 1.2M bytes
    expect(content.length).toBeLessThan(1_048_576)
    expect(await sizeErr(content)).toBe(true)
  })
  it("size boundary: 1_048_576 bytes passes the size rule, 1_048_577 does not", async () => {
    expect(await sizeErr("a".repeat(1_048_576))).toBe(false)
    expect(await sizeErr("a".repeat(1_048_577))).toBe(true)
    expect(await sizeErr("é".repeat(524_288))).toBe(false) // exactly 1_048_576 bytes
    expect(await sizeErr("é".repeat(524_288) + "a")).toBe(true)
  })
  it("rejects overlong or illegal settings keys", async () => {
    expect((await errs([file(log(0, { settings: { ["a".repeat(65)]: 1 } }))])).ok).toBe(false)
    expect((await errs([file(log(0, { settings: { ["a".repeat(64)]: 1 } }))])).ok).toBe(true)
    for (const k of ["", "1abc", "a-b", "a b", "é"])
      expect((await errs([file(log(0, { settings: { [k]: 1 } }))])).errors.join("\n")).toContain("settings key")
  })
  it("limits settings to 50 keys", async () => {
    const mk = (n: number) => Object.fromEntries(Array.from({ length: n }, (_, i) => [`k${i}`, i]))
    expect((await errs([file(log(0, { settings: mk(51) }))])).errors.join("\n")).toContain("at most 50")
    expect((await errs([file(log(0, { settings: mk(50) }))])).ok).toBe(true)
  })
  it("rejects duplicate seq", async () => {
    const r = await errs([file(log(0)), { path: `data/runs/${ID}/0.json`, content: JSON.stringify(log(0)) }])
    expect(r.errors.join("\n")).toContain("duplicate seq 0")
  })
  it("seq must be strictly greater than the highest seq already on main (decision 0014: gaps are allowed, never a repeat)", async () => {
    expect((await errs([file(log(1))], [log(0), log(1)])).errors.join("\n")).toContain("seq 1 is not above seq 1, the highest on main")
    expect((await errs([file(log(0))], [log(0), log(1)])).errors.join("\n")).toContain("seq 0 is not above seq 1, the highest on main")
    expect((await errs([file(log(1)), file(log(2))], [log(0), log(1)])).errors.join("\n")).toContain("seq 1 is not above")
    // the order on main counts, not the order of `previous`
    expect((await errs([file(log(2))], [log(3), log(0)])).ok).toBe(false)
  })
  it("a gap after main's highest seq is allowed: the files of a PR the gate closed are simply missing", async () => {
    expect((await errs([file(log(3))], [log(0), log(1)])).ok).toBe(true)
    expect((await errs([file(log(5)), file(log(9))], [log(0), log(1)])).ok).toBe(true)
    expect((await errs([file(log(1))])).ok).toBe(true) // nothing on main yet: the first PR (seq 0) was closed
    expect((await errs([file(log(4)), file(log(7, { final: true }))])).ok).toBe(true)
  })
  it("a gap may not reach past the exact integers", async () => {
    expect((await errs([file(log(Number.MAX_SAFE_INTEGER))])).ok).toBe(true)
    const big = Number.MAX_SAFE_INTEGER + 1
    const f = { path: `data/runs/${ID}/${big}.json`, content: JSON.stringify(log(big)) }
    expect((await errs([f])).errors.join("\n")).toContain("beyond the largest exact integer")
  })
  it("the other rules still hold across a gap: stepsTotal and the v2 counters never decrease, final is last", async () => {
    expect((await errs([file(log(5, { stepsTotal: 150 }))], [log(0), log(1)])).errors.join("\n")).toContain("stepsTotal decreased")
    expect((await errs([file(logV2(6, { cu: 1, curve: [] }))], [logV2(0), logV2(1), logV2(2)])).errors.join("\n")).toContain("cu decreased")
    expect((await errs([file(log(4, { final: true })), file(log(7))])).errors.join("\n")).toContain("highest seq")
    expect((await errs([file(log(9))], [log(0), log(1, { final: true })])).ok).toBe(false)
  })
  it("rejects decreasing stepsTotal", async () => {
    const r = await errs([file(log(0)), file(log(1, { stepsTotal: 50 }))])
    expect(r.errors.join("\n")).toContain("stepsTotal decreased")
  })
  it("rejects stepsTotal below previous", async () => {
    const r = await errs([file(log(2, { stepsTotal: 5 }))], [log(0), log(1)])
    expect(r.errors.join("\n")).toContain("stepsTotal decreased")
  })
  it("rejects non-finite numbers", async () => {
    const f = file(log(0))
    f.content = f.content.replace('"lr":0.1', '"lr":1e999')
    const r = await errs([f])
    expect(r.ok).toBe(false)
    expect(r.errors.join("\n")).toContain("finite")
  })
  it("rejects two final files", async () => {
    const r = await errs([file(log(0, { final: true })), file(log(1, { final: true }))])
    expect(r.errors.join("\n")).toContain("at most one final")
  })
  it("rejects final that is not the highest seq", async () => {
    const r = await errs([file(log(0, { final: true })), file(log(1))])
    expect(r.errors.join("\n")).toContain("highest seq")
  })
  it("rejects invalid JSON and bad schema", async () => {
    expect((await errs([{ path: `data/runs/${ID}/0.json`, content: "{nope" }])).errors.join("\n")).toContain("invalid JSON")
    expect((await errs([file(log(0, { schemaVersion: 3 as 1 }))])).errors.join("\n")).toContain("schemaVersion")
  })
  const withRaw = (mut: (o: Record<string, any>) => void): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
    const o = JSON.parse(file(log(0)).content); mut(o)
    return { path: `data/runs/${ID}/0.json`, content: JSON.stringify(o) }
  }
  it("rejects unknown top-level keys", async () => {
    expect((await errs([withRaw(o => { o.note = "spam" })])).errors.join("\n")).toContain("unknown key")
  })
  it("rejects unknown hostSig keys", async () => {
    expect((await errs([withRaw(o => { o.hostSig.extra = 1 })])).errors.join("\n")).toContain("unknown key")
  })
  it("rejects extra non-finite numbers", async () => {
    const f = withRaw(o => { o.extra = 1 })
    f.content = f.content.replace('"extra":1', '"extra":1e999')
    const r = await errs([f])
    expect(r.ok).toBe(false)
    expect(r.errors.join("\n")).toContain("unknown key")
  })
  it("rejects overlong settings strings", async () => {
    expect((await errs([file(log(0, { settings: { s: "a".repeat(201) } }))])).errors.join("\n")).toContain("200")
    expect((await errs([file(log(0, { settings: { s: "a".repeat(200) } }))])).ok).toBe(true)
  })
  it("rejects payload-carrying writtenAt", async () => {
    const evil = "Jan 1 2026 (" + "x".repeat(100_000) + ")"
    expect(Number.isNaN(Date.parse(evil))).toBe(false)
    expect((await errs([file(log(0, { writtenAt: evil }))])).errors.join("\n")).toContain("writtenAt")
  })
  it("rejects writtenAt not in toISOString format", async () => {
    for (const w of ["2026-10-19", "2026-10-19T12:00:00Z", "2026-10-19T12:00:00.000+00:00", "2026-13-45T99:99:99.000Z"])
      expect((await errs([file(log(0, { writtenAt: w }))])).errors.join("\n")).toContain("writtenAt")
  })
  it("validates codeSha", async () => {
    for (const c of ["a".repeat(41), "g".repeat(40), "A".repeat(40), ""])
      expect((await errs([file(log(0, { codeSha: c }))])).errors.join("\n")).toContain("codeSha")
    expect((await errs([file(log(0, { codeSha: "dev" }))])).ok).toBe(true)
  })
  it("validates knowledgeSha", async () => {
    for (const c of ["0".repeat(63), "0".repeat(65), "z".repeat(64)])
      expect((await errs([file(log(0, { knowledgeSha: c }))])).errors.join("\n")).toContain("knowledgeSha")
  })
  it("rejects an empty PR", async () => {
    expect((await errs([])).ok).toBe(false)
  })
  describe("previous-state hardening", () => {
    it("rejects a previous entry belonging to another learner", async () => {
      const r = await errs([file(log(1))], [log(0, { learnerId: "lrn_fedcba9876543210" })])
      expect(r.ok).toBe(false)
      expect(r.errors.join("\n")).toContain("previous")
    })
    it("rejects any new log when previous already has a final file", async () => {
      const r = await errs([file(log(2))], [log(0), log(1, { final: true })])
      expect(r.ok).toBe(false)
      expect(r.errors.join("\n")).toContain("final")
    })
    it("rejects a heartbeat-only PR after final", async () => {
      const hb: PrFile = { path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: "2026-10-19T12:00:00.000Z" }) }
      expect((await errs([hb], [log(0, { final: true })])).ok).toBe(false)
    })
  })
})

describe("validateSessionPr v2: the held-out check's optional fields (decision 0015 §3)", () => {
  const ho = (loss: number, baseline: number) => ({ heldOut: { loss, baseline } })
  const raw = (mut: (o: Record<string, any>) => void, seq = 0, over: Partial<LogFileV2> = {}): PrFile => { // eslint-disable-line @typescript-eslint/no-explicit-any
    const o = JSON.parse(JSON.stringify(logV2(seq, over))); mut(o)
    return { path: `data/runs/${ID}/${seq}.json`, content: JSON.stringify(o) }
  }
  const msg = async (files: PrFile[], prev: LogFile[] = []) => (await errs(files, prev)).errors.join("\n")
  it("accepts curve points with heldOut and files with evalCU, and old v2 files without either", async () => {
    const f = logV2(0, { evalCU: 0.05, curve: [{ ...pt(0.75, 300), ...ho(0.02, 0.025) }, pt(1, 400)] })
    expect(await errs([file(f)])).toEqual({ ok: true, errors: [], kind: "log" })
    expect((await errs([file(logV2(0))])).ok).toBe(true)
    expect((await errs([file(logV2(1, { evalCU: 0 }))], [logV2(0)])).ok).toBe(true) // a file before the field existed
  })
  it("heldOut is strict when present: exactly { loss, baseline }, each finite and >= 0 (no upper bound: a diverged learner's result is kept)", async () => {
    const at = (h: unknown) => raw(o => { o.curve[0].heldOut = h })
    expect((await errs([at({ loss: 1e30, baseline: 3 })])).ok).toBe(true)
    expect((await errs([at({ loss: 0, baseline: 0 })])).ok).toBe(true)
    expect(await msg([at({ loss: 0.1, baseline: 0.1, extra: 1 })])).toContain('unknown key "extra" in curve[0].heldOut')
    for (const bad of [{ loss: 0.1 }, { baseline: 0.1 }, { loss: -0.1, baseline: 0.1 }, { loss: 0.1, baseline: -0.1 },
      { loss: "0.1", baseline: 0.1 }, { loss: null, baseline: 0.1 }, null, [], 0.1]) {
      expect(await msg([at(bad)]), JSON.stringify(bad)).toContain("curve[0].heldOut")
    }
    const f = file(logV2(0, { curve: [{ ...pt(1, 400), ...ho(0.123456, 0.2) }] }))
    f.content = f.content.replace("0.123456", "1e999")
    expect(await msg([f])).toContain("curve[0].heldOut")
  })
  it("evalCU is finite and >= 0 when present, and never decreases across seq where both files have it", async () => {
    for (const bad of [-1, "1", null]) expect(await msg([raw(o => { o.evalCU = bad })]), String(bad)).toContain("evalCU")
    const f = file(logV2(0, { evalCU: 0.123456 }))
    f.content = f.content.replace("0.123456", "1e999")
    expect(await msg([f])).toContain("evalCU")
    expect(await msg([file(logV2(0, { evalCU: 2 })), file(logV2(1, { evalCU: 1 }))])).toContain("evalCU decreased")
    expect(await msg([file(logV2(1, { evalCU: 1 }))], [logV2(0, { evalCU: 2 })])).toContain("evalCU decreased")
    expect((await errs([file(logV2(0, { evalCU: 1 })), file(logV2(1, { evalCU: 1 }))])).ok).toBe(true)
  })
  it("within one learner, a file without evalCU after one with it is rejected (never from a number back to absent)", async () => {
    expect(await msg([file(logV2(0, { evalCU: 1 })), file(logV2(1))])).toContain("evalCU missing")
    expect(await msg([file(logV2(1))], [logV2(0, { evalCU: 1 })])).toContain("evalCU missing")
    expect((await errs([file(logV2(0)), file(logV2(1, { evalCU: 1 }))])).ok).toBe(true) // absent, then present: a newer page
    expect((await errs([file(logV2(1))], [logV2(0)])).ok).toBe(true)
  })
})
const joined = (r: { errors: string[] }) => r.errors.join("\n")
describe("validateSessionPr v2: checkups (decision 0016)", () => {
  const ck = (level: number) => ({ level, steps: 10, passed: ["M1"], measures: { M1: 1, M2: null, M3: 0, M4: 0, M5: 0, M6: 0 }, evalCU: 0.01 })
  const withCk = (seq: number, levels: number[], over: Partial<LogFileV2> = {}) =>
    logV2(seq, { cu: 1000 * (seq + 1), stepsTotal: 400_000 * (seq + 1), curve: [], milestoneSuiteVersion: 1, checkups: levels.map(ck), evalCU: seq + 1, ...over }) // checkups are billed inside evalCU's increase
  it("accepts increasing levels across the files on main and the PR", async () => {
    expect((await errs([file(withCk(1, [128, 256]))], [withCk(0, [32, 64])])).ok).toBe(true)
    expect((await errs([file(withCk(1, []))], [withCk(0, [64])])).ok).toBe(true)
  })
  it("checkup levels strictly increase across the files on main and the PR", async () => {
    expect(joined((await errs([file(withCk(1, [64]))], [withCk(0, [64])])))).toContain("checkup level 64 is not above 64")
    expect(joined((await errs([file(withCk(0, [64])), file(withCk(1, [64]))])))).toContain("checkup level 64 is not above 64")
    expect(joined((await errs([file(withCk(1, [32]))], [withCk(0, [64])])))).toContain("checkup level 32 is not above 64")
  })
  it("a file's checkups fit in its own evalCU increase over the file before (on main or in the PR), within 1e-9 relative", async () => {
    // Each checkup costs 0.01; withCk files rise by 1 evalCU a file.
    const over = (seq: number, ev: number, levels = [64, 128]) => withCk(seq, levels, { evalCU: ev })
    expect(joined((await errs([file(over(1, 1.015))], [over(0, 1, [32])])))).toContain("checkups' evalCU (0.02) exceeds this file's evalCU increase")
    expect(joined((await errs([file(over(0, 1, [32])), file(over(1, 1.015))])))).toContain("exceeds this file's evalCU increase")
    expect((await errs([file(over(1, 1.02))], [over(0, 1, [32])])).ok).toBe(true) // exactly its increase
    expect((await errs([file(over(1, 1 + 0.02 * (1 - 1e-12)))], [over(0, 1, [32])])).ok).toBe(true) // float noise
    // The file before is what counts, not the first: a big earlier increase cannot pay for a later file's checkups.
    expect(joined((await errs([file(over(1, 5, [32])), file(over(2, 5.01))], [over(0, 1, [])])))).toContain("exceeds this file's evalCU increase")
    // A previous file without evalCU: nothing to compare with (the parser still bounds checkups by the file's evalCU).
    expect((await errs([file(over(1, 0.02))], [logV2(0, { cu: 1000, stepsTotal: 400_000 })])).ok).toBe(true)
  })
})

describe("validateSessionPr v2: caregiver.lostCarries (Plan 04c)", () => {
  const cg = (seq: number, caregiver: unknown) => ({ ...logV2(seq), caregiver }) as LogFileV2
  it("a PR whose files carry lostCarries validates; a bad lostCarries is rejected through the parser", async () => {
    const good = [cg(0, { carries: 2, to: { crib: 2 }, safetyNet: false }), cg(1, { carries: 0, to: {}, safetyNet: false, lostCarries: 3 }), cg(2, { carries: 1, to: { toys: 1 }, safetyNet: false, lostCarries: 1 })]
    expect((await errs(good.map(f => file(f)))).errors).toEqual([])
    expect((await errs([file(good[1]!)], [good[0]!])).ok).toBe(true)
    for (const bad of [0, -1, 1.5, "2", null]) {
      const r = await errs([file(cg(0, { carries: 0, to: {}, safetyNet: false, lostCarries: bad }))])
      expect(r.ok, String(bad)).toBe(false)
      expect(joined(r)).toContain("lostCarries")
    }
  })
})

// Node-only (reads the repo), so it lives here under the root tsconfig, not in packages/harness (browser-typed).
describe("committed logs", () => {
  it("every committed log under data/runs still parses", async () => {
    const root = join(__dirname, "../../../data/runs")
    let n = 0
    for (const d of readdirSync(root).filter(d => d.startsWith("lrn_"))) {
      for (const f of readdirSync(join(root, d))) {
        if (!/^\d+\.json$/.test(f)) continue
        const r = parseLogFile(JSON.parse(readFileSync(join(root, d, f), "utf8")))
        expect(r.ok, `${d}/${f}: ${r.ok ? "" : r.error}`).toBe(true)
        n++
      }
    }
    expect(n).toBeGreaterThan(0)
  })
})

describe("validateSessionPr: the meta block (spec §7.3; Plan 05a)", () => {
  const metaOf = async (over: Partial<MetaBlock> = {}, genome: Genome = CENTRE, schema = SPACE.schemaVersion): Promise<MetaBlock> => ({
    v: 1, role: "search", method: "centre", genomeSchema: schema, genome, genomeSha: await genomeSha(schema, genome),
    drawSeed: null, posteriorSha: null, evalPanel: 3, evalSeed: null, confirmation: null, inheritance: { filters: "none" }, ...over,
  })
  const legacyOf = (genome: Genome, missing: string[] = []) => metaOf({ role: "legacy", method: "legacy", evalPanel: null, ...(missing.length ? { missing } : {}) }, genome)
  const outOfBounds = { ...CENTRE, entropyWeight: 0.001 } // today's picker spans 0.001–0.05; the v1 range is 0.0025–0.04
  const withMeta = (seq: number, meta: MetaBlock | undefined, over: Partial<LogFileV2> = {}) => logV2(seq, { ...(meta ? { meta } : {}), ...over })
  const msg = async (files: PrFile[], prev: LogFile[] = []) => (await errs(files, prev)).errors.join("\n")

  it("a search genome out of bounds is rejected; the same values under role legacy are accepted", async () => {
    expect((await errs([file(withMeta(0, await metaOf()))])).errors).toEqual([])
    expect(await msg([file(withMeta(0, await metaOf({}, outOfBounds)))])).toContain("meta.genome is outside the genome space: entropyWeight 0.001 is outside [0.0025, 0.04]")
    expect(await msg([file(withMeta(0, await metaOf({}, { ...CENTRE, latentWeight: 1 })))])).toContain("latentWeight is not implemented")
    // Review Focus 1: a forever life born before 05a keeps its old values; bounds are not applied to legacy.
    expect((await errs([file(withMeta(0, await legacyOf(outOfBounds, ["caregiverPeriod"])))])).errors).toEqual([])
    // Another genome schema: structure only (a newer or older space's bounds are not this one's).
    expect((await errs([file(withMeta(0, await metaOf({}, outOfBounds, SPACE.schemaVersion + 1)))])).errors).toEqual([])
    // No space to check against: never accepted unchecked (legacy needs none).
    expect((await errs([file(withMeta(0, await metaOf()))], [], {})).errors.join("\n")).toContain("no genome space")
    expect((await errs([file(withMeta(0, await legacyOf(outOfBounds)))], [], {})).ok).toBe(true)
  })

  it("a genomeSha that does not match its genome is rejected", async () => {
    const m = await metaOf()
    expect(await msg([file(withMeta(0, { ...m, genomeSha: "0".repeat(64) }))])).toContain("meta.genomeSha does not match its genome")
    const l = await legacyOf(outOfBounds)
    expect(await msg([file(withMeta(0, { ...l, genome: { ...outOfBounds, lr: 1e-3 } }))])).toContain("meta.genomeSha does not match its genome")
    // Key order never matters (genomeSha sorts keys).
    const reversed = Object.fromEntries(Object.entries(CENTRE).reverse())
    expect((await errs([file(withMeta(0, { ...m, genome: reversed }))])).errors).toEqual([])
  })

  it("meta.method changing between a learner's files is rejected; a file without meta after one with it is accepted", async () => {
    const a = await metaOf(), b = { ...a, method: "uniform" as const, drawSeed: 7, posteriorSha: "c".repeat(64) }
    expect(await msg([file(withMeta(0, a)), file(withMeta(1, b))])).toContain("meta.method differs from the learner's earlier files")
    expect(await msg([file(withMeta(1, b))], [withMeta(0, a)])).toContain("meta.method differs")
    expect(await msg([file(withMeta(1, { ...a, evalPanel: 4 }))], [withMeta(0, a)])).toContain("meta.evalPanel differs")
    expect(await msg([file(withMeta(2, { ...a, evalPanel: 4 }))], [withMeta(0, a), withMeta(1, undefined)])).toContain("meta.evalPanel differs")
    // Absence is allowed in any file (old code): meta, none, meta again; none, then meta.
    expect((await errs([file(withMeta(1, undefined)), file(withMeta(2, a))], [withMeta(0, a)])).errors).toEqual([])
    expect((await errs([file(withMeta(1, a))], [withMeta(0, undefined)])).errors).toEqual([])
    // filters, heldOut and end may differ: filters at 512 only, end only in the final file.
    const filters = Buffer.from(new Float32Array(104 * 16).buffer).toString("base64")
    const f9 = withMeta(9, { ...a, filters }), f10 = withMeta(10, { ...a, end: "budget" }, { final: true })
    expect((await errs([file(f9), file(f10)], [withMeta(8, a)])).errors).toEqual([])
  })

  it("two files carrying filters for level 512 are rejected", async () => {
    const a = await metaOf()
    const filters = Buffer.from(new Float32Array(104 * 16).buffer).toString("base64")
    const at = (seq: number, cu: number) => withMeta(seq, { ...a, filters }, { cu, stepsTotal: 400 * cu, curve: [] })
    expect(await msg([file(at(9, 512)), file(at(10, 700))])).toContain("a second file carries meta.filters for level 512")
    expect(await msg([file(at(10, 700))], [at(9, 512)])).toContain("a second file carries meta.filters for level 512")
    // One per level is fine: 512, 1024, 2048.
    expect((await errs([file(at(10, 1024)), file(at(11, 2048))], [at(9, 512)])).errors).toEqual([])
  })
})
