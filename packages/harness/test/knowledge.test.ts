import { describe, it, expect } from "vitest"
import { loadKnowledge, chooseSettings, OFFLINE_KNOWLEDGE_SHA, type Knowledge } from "../src/knowledge"

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
    const k = await loadKnowledge("/blanktab/", f.fn)
    expect([...f.urls].sort()).toEqual(["/blanktab/knowledge/manifest.json", "/blanktab/knowledge/settings-space.json", "/blanktab/knowledge/summary.json"])
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
    offline(await loadKnowledge("/blanktab/", fakeFetch(FILES, { throws: true }).fn))
  })
  it("falls back when a file is missing (404)", async () => {
    const rest = { ...FILES }
    delete rest["manifest.json"]
    offline(await loadKnowledge("/blanktab/", fakeFetch(rest).fn))
  })
  it("falls back when a file is malformed", async () => {
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": { schemaVersion: 2, sessions: [] } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "manifest.json": { knowledgeSha: "nope", codeSha: "dev" } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "settings-space.json": { settings: [{ name: "x", min: 2, max: 1 }] } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "settings-space.json": { settings: [{ name: "bad key", min: 0, max: 1 }] } }).fn))
    offline(await loadKnowledge("/", fakeFetch({ ...FILES, "summary.json": "<html>" }).fn))
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
