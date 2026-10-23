// The newborn reference file (spec §2.3), strictly parsed: a person pins it, and the search score reads it.
import { LEARNER_ID_RE } from "@blanktab/harness/log"
import type { MilestoneKey, NewbornProvenance, NewbornReference, ReferenceParser } from "./types"

export const REFERENCE_MILESTONES: readonly MilestoneKey[] = ["M1", "M2", "M3", "M4", "M5", "M6"]
const KEYS = ["evaluator", "genomeSchema", "heldOutMedian", "learnerIds", "milestones", "suiteVersion", "widthsClass"]
const STAT_KEYS = ["median", "passRate", "sustainedPassRate"]
const PROVENANCE_KEYS = ["browser", "codeSha", "date", "machine"]
/** A provenance text (browser, machine): 1 to 120 printable ASCII characters. */
export const PROVENANCE_TEXT_RE = /^[\x20-\x7E]{1,120}$/
const CODE_SHA_RE = /^([0-9a-f]{40}|dev)$/
const DATE_RE = /^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$/

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const sameKeys = (o: Record<string, unknown>, keys: readonly string[]) => {
  const k = Object.keys(o).sort()
  return k.length === keys.length && k.every((x, i) => x === [...keys].sort()[i])
}
const isRate = (v: unknown): v is number => isNum(v) && v >= 0 && v <= 1

/** A NewbornProvenance, or null: exactly its keys, a codeSha (40 hex or "dev"), short printable texts, a YYYY-MM-DD date. */
function parseProvenance(v: unknown): NewbornProvenance | null {
  if (!isObj(v) || !sameKeys(v, PROVENANCE_KEYS)) return null
  const { codeSha, browser, machine, date } = v
  if (typeof codeSha !== "string" || !CODE_SHA_RE.test(codeSha)) return null
  if (typeof browser !== "string" || !PROVENANCE_TEXT_RE.test(browser) || typeof machine !== "string" || !PROVENANCE_TEXT_RE.test(machine)) return null
  if (typeof date !== "string" || !DATE_RE.test(date)) return null
  return { codeSha, browser, machine, date }
}

/**
 * A NewbornReference, or null if `v` is not exactly one: the fixed header, sorted unique learnerIds, each milestone's
 * median (finite, >= 0: measures are) and rates in [0, 1], a held-out median that is finite and > 0, and optionally
 * a strict `provenance` (no evaluator reads it).
 */
export function parseNewbornReference(v: unknown): NewbornReference | null {
  if (!isObj(v)) return null
  let provenance: NewbornProvenance | undefined
  if ("provenance" in v) {
    const p = parseProvenance(v.provenance)
    if (p === null) return null
    provenance = p
  }
  if (!sameKeys(Object.fromEntries(Object.entries(v).filter(([k]) => k !== "provenance")), KEYS)) return null
  if (v.evaluator !== "newborn-baseline@1" || v.suiteVersion !== 1 || v.genomeSchema !== 1 || v.widthsClass !== "gpu") return null
  const ids = v.learnerIds
  if (!Array.isArray(ids) || ids.length === 0) return null
  for (const [i, id] of (ids as unknown[]).entries()) {
    if (typeof id !== "string" || !LEARNER_ID_RE.test(id)) return null
    if (i > 0 && !(id > (ids[i - 1] as string))) return null
  }
  const ms = v.milestones
  if (!isObj(ms) || !sameKeys(ms, REFERENCE_MILESTONES)) return null
  const milestones = {} as NewbornReference["milestones"]
  for (const m of REFERENCE_MILESTONES) {
    const s = ms[m]
    if (!isObj(s) || !sameKeys(s, STAT_KEYS)) return null
    if (!(isNum(s.median) && s.median >= 0 && isRate(s.passRate) && isRate(s.sustainedPassRate))) return null
    milestones[m] = { median: s.median, passRate: s.passRate, sustainedPassRate: s.sustainedPassRate }
  }
  if (!(isNum(v.heldOutMedian) && v.heldOutMedian > 0)) return null
  return {
    evaluator: "newborn-baseline@1", suiteVersion: 1, genomeSchema: 1, widthsClass: "gpu",
    learnerIds: [...(ids as string[])], milestones, heldOutMedian: v.heldOutMedian, ...(provenance ? { provenance } : {}),
  }
}

/** Every known reference name → its parser (registered alongside the evaluators; an unknown name fails config load). */
export const REFERENCE_PARSERS: Readonly<Record<string, ReferenceParser>> = Object.freeze({ newborn: parseNewbornReference })
