// What a tab reads before its learner starts, and again while it lives: the published summary, the manifest, the
// caregiver prior and the meta-learner's files (the genome space, the promoted genome and the gate's posterior).
// Browser-safe. Unreachable or malformed knowledge never stops the learner: it falls back to an empty, offline view.
import { CAREGIVER_DEFAULTS, CAREGIVER_RANGES, type CaregiverSettings } from "./caregiver"
import { CODE_SHA_RE, KNOWLEDGE_SHA_RE, parseSessionRow, type SessionRow, type Summary } from "./log"

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
  /** knowledge/caregiver-prior.json; CAREGIVER_DEFAULTS when absent or malformed (that alone does not make the knowledge offline). */
  caregiverPrior: CaregiverSettings
  /**
   * The meta-learner's files (Plan 05a), raw: this package does not read genomes, the page parses them (@blanktab/meta).
   * `genomeSpace` and `genome` are knowledge/genome-space.json and knowledge/genome.json as JSON; null when either
   * cannot be fetched or is not JSON. `posteriorText` is knowledge/posterior.json (the deploy copies the gate's
   * data/meta/posterior.json there) as its exact UTF-8 text, so its sha is of the bytes the gate wrote; null when
   * it cannot be fetched or is not UTF-8. None of this makes the knowledge offline. Offline knowledge has null.
   */
  meta: { genomeSpace: unknown; genome: unknown; posteriorText: string | null } | null
}

/** knowledgeSha recorded by a learner born without knowledge. */
export const OFFLINE_KNOWLEDGE_SHA = "0".repeat(64)

export const offlineKnowledge = (): Knowledge => ({
  offline: true,
  summary: { schemaVersion: 1, sessions: [] },
  manifest: { knowledgeSha: OFFLINE_KNOWLEDGE_SHA, codeSha: "dev" },
  caregiverPrior: { ...CAREGIVER_DEFAULTS },
  meta: null,
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

/** One knowledge file's response, revalidated with the server (a polling tab must see a new deploy). Throws unless 2xx. */
async function getFile(baseUrl: string, fetchFn: typeof fetch, name: string): Promise<Response> {
  const res = await fetchFn(`${baseUrl}knowledge/${name}`, { cache: "no-cache" })
  if (!res.ok) throw new Error(`${name}: ${res.status}`)
  return res
}
/**
 * One knowledge file's exact text: strict UTF-8 (invalid bytes throw) with a byte-order mark kept, so encoding the
 * text again gives back the bytes served (Response.text() would drop a BOM and replace invalid bytes).
 */
async function getText(baseUrl: string, fetchFn: typeof fetch, name: string): Promise<string> {
  const bytes = await (await getFile(baseUrl, fetchFn, name)).arrayBuffer()
  return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes)
}

/** The meta-learner's files (Knowledge.meta): null without the space or the genome; posteriorText null without it. */
async function loadMeta(get: (name: string) => Promise<unknown>, text: (name: string) => Promise<string>): Promise<Knowledge["meta"]> {
  const posterior = text("posterior.json").catch(() => null)
  try {
    const [genomeSpace, genome] = await Promise.all([get("genome-space.json"), get("genome.json")])
    return { genomeSpace, genome, posteriorText: await posterior }
  } catch {
    return null
  }
}

export async function loadKnowledge(baseUrl: string, fetchFn: typeof fetch): Promise<Knowledge> {
  const get = async (name: string): Promise<unknown> => (await getFile(baseUrl, fetchFn, name)).json()
  const text = (name: string) => getText(baseUrl, fetchFn, name)
  try {
    // The caregiver prior is optional: a tab born before it was published still runs, on the defaults. So are the
    // meta-learner's files: without them a learner runs DEFAULT_SETTINGS (the page says so).
    const prior = get("caregiver-prior.json").then(asCaregiverPrior).catch(() => ({ ...CAREGIVER_DEFAULTS }))
    const meta = loadMeta(get, text)
    const [summary, manifest, caregiverPrior] = await Promise.all([get("summary.json"), get("manifest.json"), prior])
    return { offline: false, summary: asSummary(summary), manifest: asManifest(manifest), caregiverPrior, meta: await meta }
  } catch {
    return offlineKnowledge()
  }
}

/** Just the deployed manifest (what a running tab polls for new code), or null if it is unreachable or malformed. */
export async function loadManifest(baseUrl: string, fetchFn: typeof fetch): Promise<Manifest | null> {
  try {
    return asManifest(await (await getFile(baseUrl, fetchFn, "manifest.json")).json())
  } catch {
    return null
  }
}
