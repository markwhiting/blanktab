// The posterior file, data/meta/posterior.json, schemaVersion 1 (spec §7.2): what the gate tells every tab about the
// search. 05a writes `models: {}`, `incumbent: null` and `confirmation: null`; 05b fills them. A tab that cannot read
// the file draws the centre (spec §4.1 step 1), so parsing never throws: any shape error is null.
import { MAX_GENES, SETTING_KEY_RE, type Genome } from "./genome"

export type Mode = "calibrate" | "search" | "paused" | "flat"
export interface Posterior {
  schemaVersion: 1; genomeSchema: number; evaluator: string; sourceCommit: string; mode: Mode
  centre: { genomeSha: string; values: Genome }
  calibration: { genomes: { genomeSha: string; values: Genome }[]; need: number[] }
  models: Record<string, unknown>; incumbent: unknown | null; confirmation: unknown | null
  diagnostics: Record<string, unknown>
  page: { eligibleLives: number; lastPromotion: string | null; calibrationLives: number; calibrationTarget: number; looks: number }
}

const MODES: readonly Mode[] = ["calibrate", "search", "paused", "flat"]
const TOP_KEYS = ["schemaVersion", "genomeSchema", "evaluator", "sourceCommit", "mode", "centre", "calibration", "models",
  "incumbent", "confirmation", "diagnostics", "page"]
const PAGE_KEYS = ["eligibleLives", "lastPromotion", "calibrationLives", "calibrationTarget", "looks"]
const HEX64 = /^[0-9a-f]{64}$/
const COMMIT = /^([0-9a-f]{40}|[0-9a-f]{64})$/ // a sha-1 or sha-256 git object id

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isCount = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0
const exactKeys = (v: Record<string, unknown>, keys: readonly string[]) => {
  const ks = Object.keys(v)
  return ks.length === keys.length && keys.every(k => Object.hasOwn(v, k))
}

/** Genome-shaped: at most MAX_GENES SETTING_KEY_RE names, each a finite number or a string (bounds are the space's). */
function isGenome(v: unknown): v is Genome {
  if (!isObj(v)) return false
  const ks = Object.keys(v)
  return ks.length <= MAX_GENES && ks.every(k => SETTING_KEY_RE.test(k) && (typeof v[k] === "string" || Number.isFinite(v[k])))
}
const isShaGenome = (v: unknown): v is { genomeSha: string; values: Genome } =>
  isObj(v) && exactKeys(v, ["genomeSha", "values"]) && typeof v.genomeSha === "string" && HEX64.test(v.genomeSha) && isGenome(v.values)

/** posterior.json's text → a Posterior, or null on any shape error (the tab then draws centre). */
export function parsePosterior(text: string): Posterior | null {
  let v: unknown
  try { v = JSON.parse(text) } catch { return null }
  if (!isObj(v) || !exactKeys(v, TOP_KEYS)) return null
  if (v.schemaVersion !== 1) return null
  if (!Number.isSafeInteger(v.genomeSchema) || (v.genomeSchema as number) < 1) return null
  if (typeof v.evaluator !== "string" || v.evaluator.length === 0) return null
  if (typeof v.sourceCommit !== "string" || !COMMIT.test(v.sourceCommit)) return null
  if (!MODES.includes(v.mode as Mode)) return null
  if (!isShaGenome(v.centre)) return null

  const cal = v.calibration
  if (!isObj(cal) || !exactKeys(cal, ["genomes", "need"])) return null
  if (!Array.isArray(cal.genomes) || !cal.genomes.every(isShaGenome)) return null
  if (!Array.isArray(cal.need) || !cal.need.every(isCount) || cal.need.length !== cal.genomes.length) return null
  if (v.mode === "calibrate" && cal.genomes.length === 0) return null // nothing to calibrate on

  if (!isObj(v.models) || !isObj(v.diagnostics)) return null
  if (v.incumbent !== null && !isObj(v.incumbent)) return null
  if (v.confirmation !== null && !isObj(v.confirmation)) return null

  const page = v.page
  if (!isObj(page) || !exactKeys(page, PAGE_KEYS)) return null
  if (!isCount(page.eligibleLives) || !isCount(page.calibrationLives) || !isCount(page.calibrationTarget) || !isCount(page.looks)) return null
  if (page.lastPromotion !== null && typeof page.lastPromotion !== "string") return null
  return v as unknown as Posterior
}
