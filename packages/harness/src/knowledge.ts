// What a tab reads before its learner starts, and again while it lives: the published summary, the manifest, and the settings space.
// Browser-safe. Unreachable or malformed knowledge never stops the learner: it falls back to an empty, offline view.
import { CAREGIVER_DEFAULTS, CAREGIVER_RANGES, type CaregiverSettings } from "./caregiver"
import { CODE_SHA_RE, KNOWLEDGE_SHA_RE, SETTING_KEY_RE, parseSessionRow, type SessionRow, type Summary } from "./log"

export interface SettingRange { name: string; min: number; max: number }

/** knowledge/manifest.json, as the deploy writes it (tools/gate/manifest.ts). */
export interface Manifest {
  knowledgeSha: string
  codeSha: string
  /**
   * The checkpoint format (LearnerState.version) the deployed code reads and writes: a tab reloads onto new code only
   * if it matches its own (decision 0014). Absent in manifests written before it was added.
   */
  learnerStateVersion?: number
}

export interface Knowledge {
  /** True when the knowledge files could not be loaded; the fields below are then the empty fallback. */
  offline: boolean
  summary: Summary
  manifest: Manifest
  settingsSpace: { settings: SettingRange[] }
  /** knowledge/caregiver-prior.json; CAREGIVER_DEFAULTS when absent or malformed (that alone does not make the knowledge offline). */
  caregiverPrior: CaregiverSettings
}

/** knowledgeSha recorded by a learner born without knowledge. */
export const OFFLINE_KNOWLEDGE_SHA = "0".repeat(64)

export const offlineKnowledge = (): Knowledge => ({
  offline: true,
  summary: { schemaVersion: 1, sessions: [] },
  manifest: { knowledgeSha: OFFLINE_KNOWLEDGE_SHA, codeSha: "dev" },
  settingsSpace: { settings: [] },
  caregiverPrior: { ...CAREGIVER_DEFAULTS },
})

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)

function asSummary(v: unknown): Summary {
  if (!isObj(v) || v.schemaVersion !== 1 || !Array.isArray(v.sessions)) throw new Error("bad summary.json")
  // One bad row must not take the rest (or the page) down: drop it.
  return { schemaVersion: 1, sessions: v.sessions.map(parseSessionRow).filter((r): r is SessionRow => r !== null) }
}
function asManifest(v: unknown): Manifest {
  if (!isObj(v) || typeof v.knowledgeSha !== "string" || !KNOWLEDGE_SHA_RE.test(v.knowledgeSha)) throw new Error("bad manifest.json knowledgeSha")
  if (typeof v.codeSha !== "string" || !CODE_SHA_RE.test(v.codeSha)) throw new Error("bad manifest.json codeSha")
  const version = v.learnerStateVersion
  if (version !== undefined && !(Number.isInteger(version) && (version as number) >= 1)) throw new Error("bad manifest.json learnerStateVersion")
  return { knowledgeSha: v.knowledgeSha, codeSha: v.codeSha, ...(version !== undefined ? { learnerStateVersion: version as number } : {}) }
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

const INTEGER_KEYS: ReadonlySet<string> = new Set(["caregiverPeriod", "caregiverStuckSteps"])

/** Strict: every caregiver setting present, finite and inside its range (counts integral), else the whole prior is malformed. */
function asCaregiverPrior(v: unknown): CaregiverSettings {
  if (!isObj(v) || v.schemaVersion !== 1 || !isObj(v.settings)) throw new Error("bad caregiver-prior.json")
  const out = { ...CAREGIVER_DEFAULTS }
  for (const key of Object.keys(CAREGIVER_RANGES) as (keyof CaregiverSettings)[]) {
    const x = v.settings[key]
    const [lo, hi] = CAREGIVER_RANGES[key]
    if (!isNum(x) || x < lo || x > hi || (INTEGER_KEYS.has(key) && !Number.isInteger(x))) throw new Error(`bad caregiver-prior.json ${key}`)
    out[key] = x
  }
  return out
}

/** One knowledge file, revalidated with the server (a polling tab must see a new deploy). Throws on any failure. */
async function getJson(baseUrl: string, fetchFn: typeof fetch, name: string): Promise<unknown> {
  const res = await fetchFn(`${baseUrl}knowledge/${name}`, { cache: "no-cache" })
  if (!res.ok) throw new Error(`${name}: ${res.status}`)
  return res.json()
}

export async function loadKnowledge(baseUrl: string, fetchFn: typeof fetch): Promise<Knowledge> {
  const get = (name: string) => getJson(baseUrl, fetchFn, name)
  try {
    // The caregiver prior is optional: a tab born before it was published still runs, on the defaults.
    const prior = get("caregiver-prior.json").then(asCaregiverPrior).catch(() => ({ ...CAREGIVER_DEFAULTS }))
    const [summary, manifest, settingsSpace, caregiverPrior] = await Promise.all([get("summary.json"), get("manifest.json"), get("settings-space.json"), prior])
    return { offline: false, summary: asSummary(summary), manifest: asManifest(manifest), settingsSpace: asSettingsSpace(settingsSpace), caregiverPrior }
  } catch {
    return offlineKnowledge()
  }
}

/** Just the deployed manifest (what a running tab polls for new code), or null if it is unreachable or malformed. */
export async function loadManifest(baseUrl: string, fetchFn: typeof fetch): Promise<Manifest | null> {
  try {
    return asManifest(await getJson(baseUrl, fetchFn, "manifest.json"))
  } catch {
    return null
  }
}

/** Placeholder policy: each setting uniform within its range. */
export function chooseSettings(k: Knowledge, rng: () => number): Record<string, number> {
  const out: Record<string, number> = {}
  for (const { name, min, max } of k.settingsSpace.settings) out[name] = Math.min(max, min + rng() * (max - min))
  return out
}
