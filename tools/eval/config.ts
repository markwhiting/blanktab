// The pinned evaluation config (spec §2.2; decision 0020). Read-only: the gate and tools read config/, never write it.
// A change to a pin is a person's or coding agent's edit under a recorded decision.
import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { parseNewbornReference, type References } from "@blanktab/evaluators"

export interface EvalConfig {
  schemaVersion: 1; search: string; confirm: string; signal: string; baseline: string; budgetCU: number; levels: number[]
  references: { newborn: string }; decision: string
}

export const EVAL_CONFIG_PATH = "config/evaluation.json"
const ID_AT_VERSION_RE = /^[a-z][a-z0-9-]*@[1-9][0-9]*$/
/** A reference path: a JSON file directly under config/references/. */
const REFERENCE_PATH_RE = /^config\/references\/[A-Za-z0-9@._-]+\.json$/
const DECISION_PATH_RE = /^docs\/decisions\/\d{4}-[a-z0-9-]+\.md$/
const KEYS = ["schemaVersion", "search", "confirm", "signal", "baseline", "budgetCU", "levels", "references", "decision"]

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)

/** config/evaluation.json under `root`, strictly checked; throws on anything malformed. */
export function loadEvalConfig(root: string): EvalConfig {
  const v: unknown = JSON.parse(readFileSync(join(root, EVAL_CONFIG_PATH), "utf8"))
  const fail = (why: string): never => { throw new Error(`${EVAL_CONFIG_PATH}: ${why}`) }
  if (!isObj(v)) return fail("must be a JSON object")
  const extra = Object.keys(v).find(k => !KEYS.includes(k))
  if (extra !== undefined) fail(`unknown key ${JSON.stringify(extra)}`)
  if (v.schemaVersion !== 1) fail("schemaVersion must be 1")
  for (const k of ["search", "confirm", "signal", "baseline"] as const) {
    if (typeof v[k] !== "string" || !ID_AT_VERSION_RE.test(v[k] as string)) fail(`${k} must be an id@version`)
  }
  if (!(Number.isInteger(v.budgetCU) && (v.budgetCU as number) > 0)) fail("budgetCU must be a positive integer")
  const lv = v.levels
  if (!Array.isArray(lv) || lv.length === 0 || !lv.every((x, i) => Number.isInteger(x) && x > 0 && (i === 0 || x > lv[i - 1]))) {
    fail("levels must be a non-empty increasing list of positive integers")
  }
  const refs = v.references
  if (!isObj(refs) || Object.keys(refs).length !== 1 || typeof refs.newborn !== "string" || !REFERENCE_PATH_RE.test(refs.newborn)) {
    fail("references must be { newborn: a .json file directly under config/references/ }")
  }
  if (typeof v.decision !== "string" || !DECISION_PATH_RE.test(v.decision)) fail("decision must be a docs/decisions/NNNN-title.md path")
  const r = refs as { newborn: string }
  return {
    schemaVersion: 1, search: v.search as string, confirm: v.confirm as string, signal: v.signal as string, baseline: v.baseline as string,
    budgetCU: v.budgetCU as number, levels: [...(lv as number[])], references: { newborn: r.newborn }, decision: v.decision as string,
  }
}

/** The pinned references. An absent reference file gives {} (scores are then ineligible); a malformed one throws. */
export function loadReferences(root: string, c: EvalConfig): References {
  const path = join(root, c.references.newborn)
  if (!existsSync(path)) return {}
  const newborn = parseNewbornReference(JSON.parse(readFileSync(path, "utf8")))
  if (!newborn) throw new Error(`${c.references.newborn}: not a valid newborn reference`)
  return { newborn }
}
