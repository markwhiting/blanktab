import { describe, it, expect } from "vitest"
import { validateSessionPr, type PrFile } from "../validate.ts"
import type { LogFile } from "@blanktab/harness/log"

const ID = "lrn_0123456789abcdef"
const log = (seq: number, over: Partial<LogFile> = {}): LogFile => ({
  schemaVersion: 1, learnerId: ID, seq, codeSha: "abc123", knowledgeSha: "0".repeat(64),
  writtenAt: "2026-10-19T12:00:00.000Z", stepsTotal: 100 * (seq + 1), settings: { lr: 0.1, mode: "a" },
  hostSig: { cores: 8, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  sessionsSeenAtBirth: 0, final: false, ...over,
})
const file = (l: LogFile, id = ID): PrFile => ({ path: `data/runs/${id}/${l.seq}.json`, content: JSON.stringify(l) })
const errs = (files: PrFile[], prev: LogFile[] = []) => validateSessionPr(files, prev)

describe("validateSessionPr", () => {
  it("accepts seq 0-2 (with heartbeat)", () => {
    const hb: PrFile = { path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: "2026-10-19T12:00:00.000Z" }) }
    const r = errs([file(log(0)), file(log(1)), file(log(2, { final: true })), hb])
    expect(r).toEqual({ ok: true, errors: [], kind: "log" })
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
  it("rejects files over 1 MB (bytes)", () => {
    const big = log(0, { settings: { ["é".repeat(600_000)]: 1 } }) // 1.2M bytes, 600k chars
    const r = errs([file(big)])
    expect(r.errors.join("\n")).toContain("exceeds 1 MB")
  })
  it("accepts a file of exactly 1 MB", () => {
    const base = JSON.stringify(log(0, { settings: { "": 1 } }))
    const pad = 1_048_576 - new TextEncoder().encode(base).length
    const l = log(0, { settings: { ["a".repeat(pad)]: 1 } })
    const f = file(l)
    expect(new TextEncoder().encode(f.content).length).toBe(1_048_576)
    expect(errs([f]).ok).toBe(true)
  })
  it("rejects duplicate seq", () => {
    const r = errs([file(log(0)), { path: `data/runs/${ID}/0.json`, content: JSON.stringify(log(0)) }])
    expect(r.errors.join("\n")).toContain("duplicate seq 0")
  })
  it("rejects seq not continuing from previous", () => {
    expect(errs([file(log(3))], [log(0), log(1)]).errors.join("\n")).toContain("expected seq 2")
    expect(errs([file(log(1))], [log(0), log(1)]).errors.join("\n")).toContain("expected seq 2")
  })
  it("rejects first seq not 0 with no previous", () => {
    expect(errs([file(log(1))]).errors.join("\n")).toContain("expected seq 0")
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
    expect(errs([file(log(0, { schemaVersion: 2 as 1 }))]).errors.join("\n")).toContain("schemaVersion")
  })
  it("rejects a bad heartbeat", () => {
    const r = errs([{ path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID }) }])
    expect(r.errors.join("\n")).toContain("writtenAt")
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
  it("rejects unknown heartbeat keys", () => {
    const hb = { path: `data/runs/${ID}/heartbeat.json`, content: JSON.stringify({ learnerId: ID, writtenAt: "2026-10-19T12:00:00.000Z", x: 1 }) }
    expect(errs([hb]).errors.join("\n")).toContain("unknown key")
  })
  it("rejects overlong settings strings", () => {
    expect(errs([file(log(0, { settings: { s: "a".repeat(201) } }))]).errors.join("\n")).toContain("200")
    expect(errs([file(log(0, { settings: { s: "a".repeat(200) } }))]).ok).toBe(true)
  })
  it("rejects an empty PR", () => {
    expect(errs([]).ok).toBe(false)
  })
})
