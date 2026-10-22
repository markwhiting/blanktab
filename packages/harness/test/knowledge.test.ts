import { describe, it, expect } from "vitest"
import { loadKnowledge, loadManifest, chooseSettings, OFFLINE_KNOWLEDGE_SHA, type Knowledge } from "../src/knowledge"

const row = (id: string) => ({
  learnerId: id, settings: { moveProbability: 0.5 }, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "dev", knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0,
})
const FILES: Record<string, unknown> = {
  "summary.json": { schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa"), row("lrn_bbbbbbbbbbbbbbbb")] },
  "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev" },
  "settings-space.json": { schemaVersion: 1, settings: [{ name: "moveProbability", min: 0.1, max: 1 }, { name: "other", min: -2, max: -1 }] },
}

function fakeFetch(files: Record<string, unknown>, opts: { status?: number; throws?: boolean } = {}) {
  const urls: string[] = []
  const fn = (async (input: RequestInfo | URL) => {
    const url = String(input)
    urls.push(url)
    if (opts.throws) throw new TypeError("Failed to fetch")
    const name = url.split("/").pop()!
    if (!(name in files)) return new Response("not found", { status: 404 })
    return new Response(JSON.stringify(files[name]), { status: opts.status ?? 200 })
  }) as typeof fetch
  return { fn, urls }
}

describe("loadKnowledge", () => {
  it("fetches the three files under <base>knowledge/ and returns them", async () => {
    const f = fakeFetch(FILES)
    const k = await loadKnowledge("/learner/", f.fn)
    expect([...f.urls].sort()).toEqual(["/learner/knowledge/manifest.json", "/learner/knowledge/settings-space.json", "/learner/knowledge/summary.json"])
    expect(k.offline).toBe(false)
    expect(k.summary.sessions).toHaveLength(2)
    expect(k.manifest.knowledgeSha).toBe("c".repeat(64))
    expect(k.settingsSpace.settings.map(s => s.name)).toEqual(["moveProbability", "other"])
  })

  const offline = (k: Knowledge) => {
    expect(k.offline).toBe(true)
    expect(k.summary).toEqual({ schemaVersion: 1, sessions: [] })
    expect(k.manifest.knowledgeSha).toBe(OFFLINE_KNOWLEDGE_SHA)
    expect(OFFLINE_KNOWLEDGE_SHA).toBe("0".repeat(64))
    expect(k.settingsSpace.settings).toEqual([])
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
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "settings-space.json": { settings: [{ name: "x", min: 2, max: 1 }] } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "settings-space.json": { settings: [{ name: "bad key", min: 0, max: 1 }] } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": "<html>" }).fn))
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

describe("chooseSettings", () => {
  const k = (settings: { name: string; min: number; max: number }[]): Knowledge => ({
    offline: false, summary: { schemaVersion: 1, sessions: [] }, manifest: { knowledgeSha: "c".repeat(64), codeSha: "dev" }, settingsSpace: { settings },
  })
  it("draws each setting uniformly within its range", () => {
    const space = k([{ name: "moveProbability", min: 0.1, max: 1 }, { name: "other", min: -2, max: -1 }])
    expect(chooseSettings(space, () => 0)).toEqual({ moveProbability: 0.1, other: -2 })
    expect(chooseSettings(space, () => 0.5)).toEqual({ moveProbability: 0.55, other: -1.5 })
    let seed = 1
    const rng = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    for (let i = 0; i < 1000; i++) {
      const s = chooseSettings(space, rng)
      expect(s.moveProbability).toBeGreaterThanOrEqual(0.1)
      expect(s.moveProbability).toBeLessThanOrEqual(1)
      expect(s.other).toBeGreaterThanOrEqual(-2)
      expect(s.other).toBeLessThanOrEqual(-1)
    }
  })
  it("a degenerate range gives its single value; an empty space gives no settings", () => {
    expect(chooseSettings(k([{ name: "a", min: 3, max: 3 }]), () => 0.7)).toEqual({ a: 3 })
    expect(chooseSettings(k([]), () => 0.7)).toEqual({})
  })
})
