// What a newborn tab reads before it starts: the published summary, the manifest, and the settings space.
// Browser-safe. Unreachable or malformed knowledge never stops the learner: it falls back to an empty, offline view.
import { CODE_SHA_RE, KNOWLEDGE_SHA_RE, SETTING_KEY_RE, parseSessionRow, type SessionRow, type Summary } from "./log"

export interface SettingRange { name: string; min: number; max: number }

export interface Knowledge {
  /** True when the knowledge files could not be loaded; the fields below are then the empty fallback. */
  offline: boolean
  summary: Summary
  manifest: { knowledgeSha: string; codeSha: string }
  settingsSpace: { settings: SettingRange[] }
}

/** knowledgeSha recorded by a learner born without knowledge. */
export const OFFLINE_KNOWLEDGE_SHA = "0".repeat(64)

export const offlineKnowledge = (): Knowledge => ({
  offline: true,
  summary: { schemaVersion: 1, sessions: [] },
  manifest: { knowledgeSha: OFFLINE_KNOWLEDGE_SHA, codeSha: "dev" },
  settingsSpace: { settings: [] },
})

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)

function asSummary(v: unknown): Summary {
  if (!isObj(v) || v.schemaVersion !== 1 || !Array.isArray(v.sessions)) throw new Error("bad summary.json")
  // One bad row must not take the rest (or the page) down: drop it.
  return { schemaVersion: 1, sessions: v.sessions.map(parseSessionRow).filter((r): r is SessionRow => r !== null) }
}
function asManifest(v: unknown): Knowledge["manifest"] {
  if (!isObj(v) || typeof v.knowledgeSha !== "string" || !KNOWLEDGE_SHA_RE.test(v.knowledgeSha)) throw new Error("bad manifest.json knowledgeSha")
  if (typeof v.codeSha !== "string" || !CODE_SHA_RE.test(v.codeSha)) throw new Error("bad manifest.json codeSha")
  return { knowledgeSha: v.knowledgeSha, codeSha: v.codeSha }
}
function asSettingsSpace(v: unknown): Knowledge["settingsSpace"] {
  if (!isObj(v) || !Array.isArray(v.settings)) throw new Error("bad settings-space.json")
  const settings = v.settings.map((s: unknown): SettingRange => {
    if (!isObj(s) || typeof s.name !== "string" || !SETTING_KEY_RE.test(s.name) || !isNum(s.min) || !isNum(s.max) || s.min > s.max)
      throw new Error("bad settings-space.json entry")
    return { name: s.name, min: s.min, max: s.max }
  })
  return { settings }
}

export async function loadKnowledge(baseUrl: string, fetchFn: typeof fetch): Promise<Knowledge> {
  const get = async (name: string): Promise<unknown> => {
    const res = await fetchFn(`${baseUrl}knowledge/${name}`, { cache: "no-cache" })
    if (!res.ok) throw new Error(`${name}: ${res.status}`)
    return res.json()
  }
  try {
    const [summary, manifest, settingsSpace] = await Promise.all([get("summary.json"), get("manifest.json"), get("settings-space.json")])
    return { offline: false, summary: asSummary(summary), manifest: asManifest(manifest), settingsSpace: asSettingsSpace(settingsSpace) }
  } catch {
    return offlineKnowledge()
  }
}

/** Placeholder policy: each setting uniform within its range. */
export function chooseSettings(k: Knowledge, rng: () => number): Record<string, number> {
  const out: Record<string, number> = {}
  for (const { name, min, max } of k.settingsSpace.settings) out[name] = Math.min(max, min + rng() * (max - min))
  return out
}
