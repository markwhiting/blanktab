// The checkpoint validator as it was before Plan 04c (commit 5658165: checkpoint.ts checkCheckpoint and checkCheckups,
// caregiver.ts checkCaregiverState, log.ts parseCheckups and parseCaregiver), frozen here so a test can check that what
// 04c code writes still loads on the code before it (decision 0017, rollback). Copied logic, kept as it was: do not
// update it with the live validators. Only isLogLevel (unchanged by 04c) is imported.
import { isLogLevel } from "../src/log"

const SUITE_1 = ["M1", "M2", "M3", "M4", "M5", "M6"]
const CAREGIVER_SETTING_KEYS = ["caregiverPeriod", "caregiverTau", "caregiverEpsilon", "caregiverThreshold", "caregiverStuckSteps"]
const CODE_SHA_RE = /^([0-9a-f]{40}|dev)$/
const KNOWLEDGE_SHA_RE = /^[0-9a-f]{64}$/
const MAX_CHECKUPS_PER_FILE = 8
const CHECKUP_KEYS = ["level", "steps", "passed", "measures", "evalCU", "error"]
const CAREGIVER_KEYS = ["carries", "to", "safetyNet"]
const CARE_DEST_RE = /^[a-z]{1,16}$/
const MAX_CARE_DESTS = 16
const STATE_KEYS = ["version", "settings", "rng", "rooms", "sinceDecision", "stuckFor", "m6Streak", "safetyNet", "carries", "to"]

const isRec = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const isNonNegInt = (v: unknown): v is number => isNum(v) && Number.isInteger(v) && v >= 0
const isNonNeg = (v: unknown): v is number => isNum(v) && v >= 0
const onlyKeys = (o: Record<string, unknown>, keys: readonly string[]) => Object.keys(o).every(k => keys.includes(k))
const unknownKey = (o: Record<string, unknown>, allowed: readonly string[], where: string): string | null => {
  const k = Object.keys(o).find(key => !allowed.includes(key))
  return k === undefined ? null : `unknown key ${JSON.stringify(k.slice(0, 80))} in ${where}`
}

function parseCheckups(v: unknown, cu: number, stepsTotal: number): string | { level: number }[] {
  if (!Array.isArray(v)) return "checkups must be an array"
  if (v.length > MAX_CHECKUPS_PER_FILE) return "too many checkups"
  const ids = SUITE_1
  const out: { level: number }[] = []
  for (const [i, c] of (v as unknown[]).entries()) {
    const at = `checkups[${i}]`
    if (!isRec(c)) return `${at} must be an object`
    const uk = unknownKey(c, CHECKUP_KEYS, at)
    if (uk) return uk
    if (!isNonNegInt(c.level) || !isLogLevel(c.level)) return `${at}.level must be a log-schedule level`
    if (c.level > cu) return `${at}.level is ahead of the file's cu`
    const prev = out[i - 1]
    if (prev && c.level <= prev.level) return `${at}.level must be above the previous checkup's`
    if (!isNonNegInt(c.steps) || c.steps > stepsTotal) return `${at}.steps must be a non-negative integer no more than stepsTotal`
    if (!Array.isArray(c.passed)) return `${at}.passed must be an array`
    let last = -1
    for (const id of c.passed as unknown[]) {
      const idx = typeof id === "string" ? ids.indexOf(id) : -1
      if (idx < 0) return `${at}.passed has an id outside the suite`
      if (idx <= last) return `${at}.passed must be in suite order without duplicates`
      last = idx
    }
    const meas = c.measures
    if (!isRec(meas)) return `${at}.measures must be an object`
    const mk = unknownKey(meas, ids, `${at}.measures`)
    if (mk) return mk
    for (const id of ids) if (!(meas[id] === null || isNum(meas[id]))) return `${at}.measures.${id} must be a finite number or null`
    if (!isNonNeg(c.evalCU)) return `${at}.evalCU must be a finite number >= 0`
    if ("error" in c && c.error !== true) return `${at}.error must be true when present`
    out.push({ level: c.level })
  }
  return out
}

function parseCaregiver(v: unknown): string | null {
  if (!isRec(v)) return "caregiver must be an object"
  const uk = unknownKey(v, CAREGIVER_KEYS, "caregiver")
  if (uk) return uk
  if (!isNonNegInt(v.carries)) return "caregiver.carries must be a non-negative integer"
  if (typeof v.safetyNet !== "boolean") return "caregiver.safetyNet must be a boolean"
  if (!isRec(v.to)) return "caregiver.to must be an object"
  const entries = Object.entries(v.to)
  if (entries.length > MAX_CARE_DESTS) return "caregiver.to has too many keys"
  let sum = 0
  for (const [k, n] of entries) {
    if (!CARE_DEST_RE.test(k)) return `caregiver.to key ${JSON.stringify(k.slice(0, 80))} is not allowed`
    if (!isNonNegInt(n)) return `caregiver.to.${k} must be a non-negative integer`
    sum += n
  }
  return sum === v.carries ? null : "caregiver.to values must sum to carries"
}

function checkCaregiverState(v: unknown, fail: (what: string) => never): void {
  if (!isRec(v) || !onlyKeys(v, STATE_KEYS)) fail("not a caregiver state")
  const r = v as Record<string, unknown>
  if (r.version !== 1) fail("version must be 1")
  const st = r.settings
  if (!isRec(st) || !onlyKeys(st, CAREGIVER_SETTING_KEYS) || !CAREGIVER_SETTING_KEYS.every(k => isNum(st[k]))) fail("settings")
  if (!(Number.isInteger(r.rng) && (r.rng as number) >= 0 && (r.rng as number) < 2 ** 32)) fail("rng must be a uint32")
  if (!isRec(r.rooms)) fail("rooms must be an object")
  for (const [room, e] of Object.entries(r.rooms as Record<string, unknown>)) {
    if (!isRec(e) || !onlyKeys(e, ["fast", "slow"]) || !isNum(e.fast) || !isNum(e.slow)) fail(`rooms.${room.slice(0, 80)}`)
  }
  for (const k of ["sinceDecision", "stuckFor", "m6Streak"]) if (!isNonNegInt(r[k])) fail(`${k} must be a non-negative integer`)
  const summary = parseCaregiver({ carries: r.carries, to: r.to, safetyNet: r.safetyNet })
  if (summary !== null) fail(summary)
}

/** Null if the pre-04c code accepts `c` as `learnerId`'s checkpoint record, else what it rejects. */
export function pre04cCheckpointError(c: unknown, learnerId: string): string | null {
  const fail = (what: string): never => { throw new RangeError(what) }
  try {
    if (!isRec(c)) fail("not an object")
    const r = c as Record<string, unknown>
    if (r.learnerId !== learnerId) fail("learnerId")
    if (typeof r.savedAt !== "string") fail("savedAt must be a string")
    if (!isNonNegInt(r.steps) || !isNonNegInt(r.seq)) fail("steps and seq must be non-negative integers")
    if (!isNonNeg(r.cu) || !isNonNeg(r.lostCU)) fail("cu and lostCU must be finite numbers >= 0")
    if (!isNonNegInt(r.restores) || !isNonNegInt(r.lostSkips)) fail("restores and lostSkips must be non-negative integers")
    if (!Array.isArray(r.curveTail)) fail("curveTail must be an array")
    if (r.knowledgeSha !== undefined && !(typeof r.knowledgeSha === "string" && KNOWLEDGE_SHA_RE.test(r.knowledgeSha))) fail("knowledgeSha")
    if (r.sessionsSeenAtBirth !== undefined && !isNonNegInt(r.sessionsSeenAtBirth)) fail("sessionsSeenAtBirth")
    if (r.codeSha !== undefined && !(typeof r.codeSha === "string" && CODE_SHA_RE.test(r.codeSha))) fail("codeSha")
    if (r.checkups !== undefined) {
      const v = r.checkups
      if (!isRec(v) || !onlyKeys(v, ["lastLevel", "tail"])) fail("checkups must be { lastLevel, tail }")
      const { lastLevel, tail } = v as Record<string, unknown>
      if (!(lastLevel === 0 || (typeof lastLevel === "number" && isLogLevel(lastLevel) && lastLevel <= (r.cu as number)))) fail("checkups.lastLevel")
      const parsed = parseCheckups(tail, r.cu as number, r.steps as number)
      if (typeof parsed === "string") fail(`checkups.tail: ${parsed}`)
      if ((parsed as { level: number }[]).some(x => x.level > (lastLevel as number))) fail("checkups.tail has a level above lastLevel")
    }
    if (r.caregiver !== undefined) checkCaregiverState(r.caregiver, what => fail(`caregiver state: ${what}`))
    if (!("learnerState" in r)) fail("no learnerState")
    return null
  } catch (e) {
    return e instanceof Error ? e.message : String(e)
  }
}
