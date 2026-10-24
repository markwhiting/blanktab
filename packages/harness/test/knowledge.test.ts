import { describe, it, expect } from "vitest"
import committedPrior from "../../../knowledge/caregiver-prior.json"
import committedGenomeSpace from "../../../knowledge/genome-space.json"
import committedGenome from "../../../knowledge/genome.json"
import { CAREGIVER_DEFAULTS, CAREGIVER_RANGES } from "../src/caregiver"
import { loadKnowledge, loadManifest, OFFLINE_KNOWLEDGE_SHA, type Knowledge } from "../src/knowledge"

const row = (id: string) => ({
  learnerId: id, settings: { moveProbability: 0.5 }, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "dev", knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0,
})
const FILES: Record<string, unknown> = {
  "summary.json": { schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa"), row("lrn_bbbbbbbbbbbbbbbb")] },
  "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev" },
  "genome-space.json": committedGenomeSpace,
  "genome.json": committedGenome,
}
/** A file served as these exact bytes (not as JSON). */
const raw = (bytes: string | Uint8Array) => ({ [RAW]: bytes })
const RAW = Symbol("raw")

function fakeFetch(files: Record<string, unknown>, opts: { status?: number; throws?: boolean } = {}) {
  const urls: string[] = []
  const fn = (async (input: RequestInfo | URL) => {
    const url = String(input)
    urls.push(url)
    if (opts.throws) throw new TypeError("Failed to fetch")
    const name = url.split("/").pop()!
    if (!(name in files)) return new Response("not found", { status: 404 })
    const f = files[name] as { [RAW]?: string | Uint8Array }
    const body = typeof f === "object" && f !== null && RAW in f ? f[RAW]! : JSON.stringify(files[name])
    return new Response(body as BodyInit, { status: opts.status ?? 200 })
  }) as typeof fetch
  return { fn, urls }
}

describe("loadKnowledge", () => {
  it("fetches its six files under <base>knowledge/ and returns them; the meta-learner's raw (the page parses them)", async () => {
    const f = fakeFetch({ ...FILES, "posterior.json": raw('{"x": 1}\n') })
    const k = await loadKnowledge("/learner/", f.fn)
    expect([...f.urls].sort()).toEqual(["caregiver-prior", "genome-space", "genome", "manifest", "posterior", "summary"].map(n => `/learner/knowledge/${n}.json`))
    expect(k.offline).toBe(false)
    expect(k.summary.sessions).toHaveLength(2)
    expect(k.manifest.knowledgeSha).toBe("c".repeat(64))
    expect(k.meta).toEqual({ genomeSpace: committedGenomeSpace, genome: committedGenome, posteriorText: '{"x": 1}\n' })
    expect(k).not.toHaveProperty("settingsSpace")
  })
  it("posterior.json is its exact text: a BOM is kept, invalid UTF-8 is no posterior, a 404 is no posterior", async () => {
    const bom = new Uint8Array([0xef, 0xbb, 0xbf, 0x7b, 0x7d])
    expect((await loadKnowledge("/", fakeFetch({ ...FILES, "posterior.json": raw(bom) }).fn)).meta!.posteriorText).toBe("\ufeff{}")
    const bad = new Uint8Array([0x7b, 0xff, 0x7d])
    expect((await loadKnowledge("/", fakeFetch({ ...FILES, "posterior.json": raw(bad) }).fn)).meta!.posteriorText).toBeNull()
    const k = await loadKnowledge("/", fakeFetch(FILES).fn)
    expect(k.offline).toBe(false)
    expect(k.meta).toEqual({ genomeSpace: committedGenomeSpace, genome: committedGenome, posteriorText: null })
  })
  it("without genome-space.json or genome.json (absent or not JSON) meta is null, and the knowledge is not offline", async () => {
    for (const name of ["genome-space.json", "genome.json"]) {
      const gone = { ...FILES }
      delete gone[name]
      for (const files of [gone, { ...FILES, [name]: raw("<html>") }]) {
        const k = await loadKnowledge("/", fakeFetch({ ...files, "posterior.json": raw("{}") }).fn)
        expect(k.offline, name).toBe(false)
        expect(k.meta, name).toBeNull()
        expect(k.manifest.knowledgeSha).toBe("c".repeat(64))
      }
    }
  })

  const offline = (k: Knowledge) => {
    expect(k.offline).toBe(true)
    expect(k.summary).toEqual({ schemaVersion: 1, sessions: [] })
    expect(k.manifest.knowledgeSha).toBe(OFFLINE_KNOWLEDGE_SHA)
    expect(OFFLINE_KNOWLEDGE_SHA).toBe("0".repeat(64))
    expect(k.meta).toBeNull()
  }
  it("falls back to an empty, offline knowledge when the network fails", async () => {
    offline(await loadKnowledge("/learner/", fakeFetch(FILES, { throws: true }).fn))
  })
  it("falls back when a file is missing (404)", async () => {
    const rest = { ...FILES }
    delete rest["manifest.json"]
    offline(await loadKnowledge("/learner/", fakeFetch(rest).fn))
  })
  it("drops malformed summary rows instead of failing; keeps a valid lastWrittenAt, strips an invalid one", async () => {
    const good = { ...row("lrn_aaaaaaaaaaaaaaaa"), lastWrittenAt: "2026-10-20T01:02:03.000Z" }
    const oddTime = { ...row("lrn_bbbbbbbbbbbbbbbb"), lastWrittenAt: "yesterday" }
    const bad = [
      { ...row("lrn_cccccccccccccccc"), learnerId: "nope" }, { ...row("lrn_cccccccccccccccc"), hostSig: null },
      { ...row("lrn_cccccccccccccccc"), hostSig: { cores: "4", webgpu: false } }, { ...row("lrn_cccccccccccccccc"), settings: [1] },
      { ...row("lrn_cccccccccccccccc"), stepsTotal: "3000" }, { ...row("lrn_cccccccccccccccc"), sessionsSeenAtBirth: null },
      { ...row("lrn_cccccccccccccccc"), final: "yes" }, null, "row", 7,
    ]
    const k = await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": { schemaVersion: 1, sessions: [bad[0], good, ...bad.slice(1), oddTime] } }).fn)
    expect(k.offline).toBe(false)
    expect(k.summary.sessions).toEqual([{ ...good, cu: null, lossRatio: null, heldOutRatio: null, checkupScore: null }, { ...row("lrn_bbbbbbbbbbbbbbbb"), cu: null, lossRatio: null, heldOutRatio: null, checkupScore: null }])
  })
  it("reads a row's cu and lossRatio as numbers >= 0 or null (never drops the row for them)", async () => {
    const rows = [
      { ...row("lrn_aaaaaaaaaaaaaaaa"), cu: 12.5, lossRatio: 0.9, heldOutRatio: 1.1 },
      { ...row("lrn_bbbbbbbbbbbbbbbb"), cu: "12", lossRatio: -1, heldOutRatio: -1 },
      { ...row("lrn_cccccccccccccccc"), cu: null, lossRatio: { evil: 1 }, heldOutRatio: "0.9" },
    ]
    const k = await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": { schemaVersion: 1, sessions: rows } }).fn)
    expect(k.summary.sessions.map(r => [r.cu, r.lossRatio, r.heldOutRatio])).toEqual([[12.5, 0.9, 1.1], [null, null, null], [null, null, null]])
  })
  it("falls back when a file is malformed", async () => {
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": { schemaVersion: 2, sessions: [] } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "manifest.json": { knowledgeSha: "nope", codeSha: "dev" } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": "<html>" }).fn))
  })
})

describe("caregiver prior", () => {
  const prior = { caregiverPeriod: 300, caregiverTau: 0.1, caregiverEpsilon: 0.2, caregiverThreshold: 0.02, caregiverStuckSteps: 1500 }
  it("loads caregiver-prior.json; missing or malformed gives CAREGIVER_DEFAULTS and offline stays false", async () => {
    const f = fakeFetch({ ...FILES, "caregiver-prior.json": { schemaVersion: 1, settings: prior } })
    const k = await loadKnowledge("/", f.fn)
    expect(k.offline).toBe(false)
    expect(k.caregiverPrior).toEqual(prior)
    expect(f.urls).toContain("/knowledge/caregiver-prior.json")
    const bads: unknown[] = [
      undefined, "<html>", { schemaVersion: 2, settings: prior }, { schemaVersion: 1 }, { schemaVersion: 1, settings: [] },
      { schemaVersion: 1, settings: { ...prior, caregiverTau: "0.1" } }, { schemaVersion: 1, settings: { ...prior, caregiverPeriod: 5 } },
      { schemaVersion: 1, settings: { ...prior, caregiverEpsilon: 2 } }, { schemaVersion: 1, settings: { ...prior, caregiverPeriod: 300.5 } },
      { schemaVersion: 1, settings: { ...prior, caregiverStuckSteps: 1500.5 } }, { schemaVersion: 1, settings: { caregiverPeriod: 300 } },
    ]
    for (const bad of bads) {
      const files = { ...FILES, ...(bad === undefined ? {} : { "caregiver-prior.json": bad }) }
      const kk = await loadKnowledge("/", fakeFetch(files).fn)
      expect(kk.offline).toBe(false)
      expect(kk.caregiverPrior).toEqual(CAREGIVER_DEFAULTS)
      expect(kk.manifest.knowledgeSha).toBe("c".repeat(64))
    }
  })
  it("the offline fallback carries the defaults too", async () => {
    const k = await loadKnowledge("/", fakeFetch(FILES, { throws: true }).fn)
    expect(k.caregiverPrior).toEqual(CAREGIVER_DEFAULTS)
  })
  it("the committed prior is the caregiver's defaults, inside its ranges (the genome's ranges: tools/genome-centre.test.ts)", () => {
    const file = committedPrior
    expect(file).toEqual({ schemaVersion: 1, settings: CAREGIVER_DEFAULTS })
    for (const [name, value] of Object.entries(file.settings as Record<string, number>)) {
      const [lo, hi] = CAREGIVER_RANGES[name as keyof typeof CAREGIVER_RANGES]
      expect(value, name).toBeGreaterThanOrEqual(lo)
      expect(value, name).toBeLessThanOrEqual(hi)
    }
  })
})

describe("the manifest's learnerStateVersion", () => {
  it("is read when present (a positive integer) and left out when absent (a manifest from before it existed)", async () => {
    const withV = await loadKnowledge("/", fakeFetch({ ...FILES, "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev", learnerStateVersion: 3 } }).fn)
    expect(withV.manifest).toEqual({ knowledgeSha: "c".repeat(64), codeSha: "dev", learnerStateVersion: 3 })
    const without = await loadKnowledge("/", fakeFetch(FILES).fn)
    expect(without.manifest).toEqual({ knowledgeSha: "c".repeat(64), codeSha: "dev" })
  })
  it("a malformed one makes the manifest malformed", async () => {
    for (const v of [0, -1, 1.5, "1", null]) {
      expect((await loadKnowledge("/", fakeFetch({ ...FILES, "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev", learnerStateVersion: v } }).fn)).offline).toBe(true)
    }
  })
})

describe("loadManifest", () => {
  it("fetches only <base>knowledge/manifest.json and parses it", async () => {
    const f = fakeFetch({ "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "a".repeat(40), learnerStateVersion: 1 } })
    expect(await loadManifest("/learner/", f.fn)).toEqual({ knowledgeSha: "c".repeat(64), codeSha: "a".repeat(40), learnerStateVersion: 1 })
    expect(f.urls).toEqual(["/learner/knowledge/manifest.json"])
  })
  it("is null when unreachable, missing or malformed (never throws)", async () => {
    expect(await loadManifest("/", fakeFetch(FILES, { throws: true }).fn)).toBeNull()
    expect(await loadManifest("/", fakeFetch({}).fn)).toBeNull()
    expect(await loadManifest("/", fakeFetch({ "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "nope" } }).fn)).toBeNull()
  })
})
