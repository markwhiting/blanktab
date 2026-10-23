// The pinned evaluation config (spec §2.2; decision 0020). Read-only: the gate and tools read config/, never write it.
// A change to a pin is a person's or coding agent's edit under a recorded decision. Swapping an evaluator is a new
// module, one REGISTRY line and a config edit: what references load follows the pinned evaluators' `needs`.
import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { REFERENCE_PARSERS, REGISTRY, type Evaluator, type ReferenceParser, type References } from "@blanktab/evaluators"

export interface EvalConfig {
  schemaVersion: 1; search: string; confirm: string; signal: string; baseline: string; budgetCU: number; levels: number[]
  /** Reference name → its file, directly under config/references/. Exactly the names the pinned evaluators need. */
  references: Record<string, string>; decision: string
}

/** What config loading resolves against (tests pass their own). */
export interface EvalRegistry { evaluators: Readonly<Record<string, Evaluator>>; parsers: Readonly<Record<string, ReferenceParser>> }
const DEFAULT_REGISTRY: EvalRegistry = { evaluators: REGISTRY, parsers: REFERENCE_PARSERS }


export const EVAL_CONFIG_PATH = "config/evaluation.json"
const ID_AT_VERSION_RE = /^[a-z][a-z0-9-]*@[1-9][0-9]*$/
const REFERENCE_PATH_RE = /^config\/references\/[A-Za-z0-9@._-]+\.json$/
const DECISION_PATH_RE = /^docs\/decisions\/\d{4}-[a-z0-9-]+\.md$/
const PIN_KEYS = ["search", "confirm", "signal", "baseline"] as const
const KEYS = ["schemaVersion", ...PIN_KEYS, "budgetCU", "levels", "references", "decision"]

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)

/** The evaluators the config pins. */
function pinnedEvaluators(c: Pick<EvalConfig, typeof PIN_KEYS[number]>, reg: EvalRegistry): Evaluator[] {
  return PIN_KEYS.flatMap(k => (Object.hasOwn(reg.evaluators, c[k]) ? [reg.evaluators[c[k]]!] : []))
}

/** The reference names the pinned evaluators need, sorted and unique. */
export function neededReferences(c: EvalConfig, reg: EvalRegistry = DEFAULT_REGISTRY): string[] {
  return [...new Set(pinnedEvaluators(c, reg).flatMap(e => e.needs))].sort()
}

/** config/evaluation.json under `root`, strictly checked against the registry; throws on anything malformed. */
export function loadEvalConfig(root: string, reg: EvalRegistry = DEFAULT_REGISTRY): EvalConfig {
  const v: unknown = JSON.parse(readFileSync(join(root, EVAL_CONFIG_PATH), "utf8"))
  const fail = (why: string): never => { throw new Error(`${EVAL_CONFIG_PATH}: ${why}`) }
  if (!isObj(v)) return fail("must be a JSON object")
  const extra = Object.keys(v).find(k => !KEYS.includes(k))
  if (extra !== undefined) fail(`unknown key ${JSON.stringify(extra)}`)
  if (v.schemaVersion !== 1) fail("schemaVersion must be 1")
  for (const k of PIN_KEYS) {
    const pin = v[k]
    if (typeof pin !== "string" || !ID_AT_VERSION_RE.test(pin)) fail(`${k} must be an id@version`)
    if (!Object.hasOwn(reg.evaluators, pin as string)) fail(`${k}: unknown evaluator ${JSON.stringify(pin)}`)
    // Each pin's evaluator fits its role: the search one scores and ranks lives; the others are cohort tests.
    const e = reg.evaluators[pin as string]!
    const fits = k === "search" ? e.kind === "life" && !!e.evaluateLife && !!e.rankAverage && !!e.targets : e.kind === "cohort" && !!e.evaluateCohort
    if (!fits) fail(`${k}: ${pin as string} is not a ${k === "search" ? "life evaluator with rankAverage and targets" : "cohort evaluator"}`)
  }
  if (!(Number.isInteger(v.budgetCU) && (v.budgetCU as number) > 0)) fail("budgetCU must be a positive integer")
  const lv = v.levels
  if (!Array.isArray(lv) || lv.length === 0 || !lv.every((x, i) => Number.isInteger(x) && x > 0 && (i === 0 || x > lv[i - 1]))) {
    fail("levels must be a non-empty increasing list of positive integers")
  }
  const refs = v.references
  if (!isObj(refs)) return fail("references must be an object of name → path")
  for (const [name, path] of Object.entries(refs)) {
    if (typeof path !== "string" || !REFERENCE_PATH_RE.test(path)) fail(`references.${name} must be a .json file directly under config/references/`)
  }
  if (typeof v.decision !== "string" || !DECISION_PATH_RE.test(v.decision)) fail("decision must be a docs/decisions/NNNN-title.md path")
  const c: EvalConfig = {
    schemaVersion: 1, search: v.search as string, confirm: v.confirm as string, signal: v.signal as string, baseline: v.baseline as string,
    budgetCU: v.budgetCU as number, levels: [...(lv as number[])], references: { ...(refs as Record<string, string>) }, decision: v.decision as string,
  }
  // The pinned evaluators' needs are the source of truth: each needed reference has a parser and a pinned path, and
  // the config pins nothing else.
  for (const e of pinnedEvaluators(c, reg)) {
    for (const name of e.needs) {
      if (!Object.hasOwn(reg.parsers, name)) fail(`${e.id}@${e.version} needs reference ${JSON.stringify(name)}, which no parser is registered for`)
      if (!Object.hasOwn(c.references, name)) fail(`${e.id}@${e.version} needs reference ${JSON.stringify(name)}, which references does not pin`)
    }
  }
  // A pinned evaluator that declares a budget reads exactly the config's (the harness runs lives on the config's).
  for (const e of pinnedEvaluators(c, reg)) {
    if (e.budget && !(e.budget.budgetCU === c.budgetCU && e.budget.levels.length === c.levels.length && e.budget.levels.every((x, i) => x === c.levels[i]))) {
      fail(`${e.id}@${e.version} reads a budget of ${e.budget.budgetCU} CU at ${e.budget.levels.join(", ")}, not the config's ${c.budgetCU} CU at ${c.levels.join(", ")}`)
    }
  }
  const needed = neededReferences(c, reg)
  const unused = Object.keys(c.references).find(name => !needed.includes(name))
  if (unused !== undefined) fail(`references.${unused} is needed by no pinned evaluator`)
  return c
}

/**
 * The references the pinned evaluators need, each parsed by its registered parser. An absent file is left out (scores
 * needing it are then ineligible); a malformed one throws.
 */
export function loadReferences(root: string, c: EvalConfig, reg: EvalRegistry = DEFAULT_REGISTRY): References {
  const out: References = {}
  for (const name of neededReferences(c, reg)) {
    const rel = c.references[name]
    const parse = reg.parsers[name]
    if (rel === undefined || !parse) throw new Error(`reference ${JSON.stringify(name)} is not pinned or has no parser (load the config with loadEvalConfig)`)
    const path = join(root, rel)
    if (!existsSync(path)) continue
    const parsed = parse(JSON.parse(readFileSync(path, "utf8")))
    if (parsed === null) throw new Error(`${rel}: not a valid ${name} reference`)
    out[name] = parsed
  }
  return out
}
