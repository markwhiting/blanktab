// The caregiver: a harness component (never part of the learner) that watches the learner's prediction loss per room and
// periodically carries it to the room where it is learning fastest, via the world's bringToFront. Pure and deterministic
// given its seed; browser-safe. The carry is the harness's choice, the learner never sees it (AGENTS.md invariants 2, 4).
import { createRng } from "@blanktab/tensor"
import { parseCaregiver, type CaregiverSummary } from "./log"

export interface CaregiverSettings {
  /** Steps between decisions (decision 0016: not also after each sleep, which would make any period ≥ the sleep interval inert). */
  caregiverPeriod: number
  /** Softmax temperature over learning progress. */
  caregiverTau: number
  /** Probability mass spread uniformly over all rooms. */
  caregiverEpsilon: number
  /** Safety net: a room with learning progress below this is "stuck". */
  caregiverThreshold: number
  /** Safety net: steps in a row with every visited room stuck before a carry. */
  caregiverStuckSteps: number
}

export const CAREGIVER_DEFAULTS: CaregiverSettings = {
  caregiverPeriod: 500, caregiverTau: 0.05, caregiverEpsilon: 0.1, caregiverThreshold: 0.01, caregiverStuckSteps: 2000,
}

export const CAREGIVER_RANGES = {
  caregiverPeriod: [50, 10_000], caregiverTau: [0.001, 10], caregiverEpsilon: [0, 1], caregiverThreshold: [0, 0.5], caregiverStuckSteps: [100, 100_000],
} as const

const INTEGER_SETTINGS: ReadonlySet<keyof CaregiverSettings> = new Set(["caregiverPeriod", "caregiverStuckSteps"])

/** Settings from untrusted input: non-finite (or non-number) values take the default, the rest are clamped into range; counts are rounded. */
export function clampCaregiverSettings(s: Partial<Record<keyof CaregiverSettings, unknown>>): CaregiverSettings {
  const out = { ...CAREGIVER_DEFAULTS }
  for (const key of Object.keys(CAREGIVER_RANGES) as (keyof CaregiverSettings)[]) {
    const v = s[key]
    if (typeof v !== "number" || !Number.isFinite(v)) continue
    const [lo, hi] = CAREGIVER_RANGES[key]
    out[key] = Math.min(hi, Math.max(lo, INTEGER_SETTINGS.has(key) ? Math.round(v) : v))
  }
  return out
}

export const LP_FAST_ALPHA = 0.1
export const LP_SLOW_ALPHA = 0.01

export interface RoomLP { fast: number; slow: number }

/** Relative learning progress (decision 0016): (slow − fast) / max(slow, 1e-6). */
export function learningProgress(r: RoomLP): number {
  return (r.slow - r.fast) / Math.max(r.slow, 1e-6)
}

export interface CaregiverState {
  version: 1
  settings: CaregiverSettings
  /** The Rng's state (a uint32). */
  rng: number
  rooms: Record<string, RoomLP>
  sinceDecision: number
  stuckFor: number
  m6Streak: number
  safetyNet: boolean
  /** Carries (and where to) since the last takeSummary. */
  carries: number
  to: Record<string, number>
  /**
   * Carries since the last takeSummary that restores threw away (decision 0017): they went with the learner's state
   * the restore discarded, so they are no longer in `carries`. Absent (as in states from before it) when there are none.
   */
  lost?: number
}

export interface Caregiver {
  /** The loss of the transition just learned from, seen in `room`. NaN and ±Infinity are ignored. */
  observe(room: string, loss: number): void
  /** Once per learner step, after observe: a room to carry the learner to, or null. Decides every caregiverPeriod steps (safety net: after caregiverStuckSteps stuck steps). */
  decide(current: string, rooms: readonly string[]): string | null
  /** After each completed checkup. */
  onCheckup(passed: readonly string[]): void
  /** Carries since the last take, with `lostCarries` when some were lost (then both reset). */
  takeSummary(): CaregiverSummary
  /** Adds `n` (an integer >= 0; RangeError otherwise) carries lost to a restore to `lost`. */
  noteLost(n: number): void
  state(): CaregiverState
}

const STATE_KEYS = ["version", "settings", "rng", "rooms", "sinceDecision", "stuckFor", "m6Streak", "safetyNet", "carries", "to", "lost"]
const isRec = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const onlyKeys = (o: Record<string, unknown>, keys: readonly string[]) => Object.keys(o).every(k => keys.includes(k))
const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const isNonNegInt = (v: unknown) => Number.isInteger(v) && (v as number) >= 0

/**
 * Throws RangeError unless `v` is a caregiver state (from a checkpoint, so untrusted): a restored state must not smuggle
 * a non-finite learning progress or a bad RNG state into pick(), nor carries a log file would refuse (log.ts parseCaregiver).
 */
export function checkCaregiverState(v: unknown): asserts v is CaregiverState {
  const fail = (what: string): never => { throw new RangeError(`caregiver state: ${what}`) }
  if (!isRec(v) || !onlyKeys(v, STATE_KEYS)) fail("not a caregiver state")
  const r = v as Record<string, unknown>
  if (r.version !== 1) fail("version must be 1")
  const st = r.settings
  const settingKeys = Object.keys(CAREGIVER_DEFAULTS)
  if (!isRec(st) || !onlyKeys(st, settingKeys) || !settingKeys.every(k => isFiniteNum(st[k]))) fail("settings must hold the five caregiver settings as finite numbers")
  if (!(Number.isInteger(r.rng) && (r.rng as number) >= 0 && (r.rng as number) < 2 ** 32)) fail(`rng must be a uint32, got ${String(r.rng)}`)
  if (!isRec(r.rooms)) fail("rooms must be an object")
  for (const [room, e] of Object.entries(r.rooms as Record<string, unknown>)) {
    if (!isRec(e) || !onlyKeys(e, ["fast", "slow"]) || !isFiniteNum(e.fast) || !isFiniteNum(e.slow)) fail(`rooms.${room.slice(0, 80)} must be { fast, slow }, finite numbers`)
  }
  for (const k of ["sinceDecision", "stuckFor", "m6Streak"]) if (!isNonNegInt(r[k])) fail(`${k} must be a non-negative integer`)
  if ("lost" in r && !isNonNegInt(r.lost)) fail("lost must be a non-negative integer when present")
  const summary = parseCaregiver({ carries: r.carries, to: r.to, safetyNet: r.safetyNet })
  if (typeof summary === "string") fail(summary)
}

export function createCaregiver(init: { settings: CaregiverSettings; seed: number } | { state: CaregiverState }): Caregiver {
  if ("state" in init) checkCaregiverState(init.state)
  const rng = createRng("state" in init ? init.state.rng : init.seed)
  const st: CaregiverState = "state" in init
    ? structuredClone(init.state)
    : { version: 1, settings: init.settings, rng: rng.getState(), rooms: {}, sinceDecision: 0, stuckFor: 0, m6Streak: 0, safetyNet: false, carries: 0, to: {} }
  st.settings = clampCaregiverSettings(st.settings) // tau 0 etc. must not reach the softmax
  
  /** Learning progress per room; an unvisited room is as promising as the best visited one (0 if none is visited). */
  function progress(rooms: readonly string[]): number[] {
    const visited = Object.values(st.rooms).map(learningProgress)
    const fill = visited.length ? Math.max(...visited) : 0
    return rooms.map(r => (st.rooms[r] ? learningProgress(st.rooms[r]!) : fill))
  }

  function pick(rooms: readonly string[]): string {
    const lp = progress(rooms)
    const top = Math.max(...lp)
    const w = lp.map(x => Math.exp((x - top) / st.settings.caregiverTau))
    const z = w.reduce((a, b) => a + b, 0)
    const u = rng()
    let acc = 0
    for (let i = 0; i < rooms.length; i++) {
      acc += (1 - st.settings.caregiverEpsilon) * (w[i]! / z) + st.settings.caregiverEpsilon / rooms.length
      if (u < acc) return rooms[i]!
    }
    return rooms[rooms.length - 1]!
  }

  return {
    observe(room, loss) {
      if (!Number.isFinite(loss)) return
      const r = st.rooms[room]
      if (!r) st.rooms[room] = { fast: loss, slow: loss }
      else {
        r.fast += LP_FAST_ALPHA * (loss - r.fast)
        r.slow += LP_SLOW_ALPHA * (loss - r.slow)
      }
    },
    decide(current, rooms) {
      st.sinceDecision++
      let go: boolean
      if (st.safetyNet) {
        const visited = Object.values(st.rooms)
        const stuck = visited.length > 0 && visited.every(r => learningProgress(r) < st.settings.caregiverThreshold)
        st.stuckFor = stuck ? st.stuckFor + 1 : 0
        go = st.stuckFor >= st.settings.caregiverStuckSteps
        if (go) st.stuckFor = 0
      } else {
        go = st.sinceDecision >= st.settings.caregiverPeriod
      }
      if (!go || rooms.length === 0) return null
      st.sinceDecision = 0
      const to = pick(rooms)
      st.rng = rng.getState()
      if (to === current) return null
      st.carries++
      st.to[to] = (st.to[to] ?? 0) + 1
      return to
    },
    onCheckup(passed) {
      st.m6Streak = passed.includes("M6") ? st.m6Streak + 1 : 0
      if (st.m6Streak >= 2) st.safetyNet = true
    },
    takeSummary() {
      const lost = st.lost ?? 0
      const out: CaregiverSummary = { carries: st.carries, to: st.to, safetyNet: st.safetyNet, ...(lost > 0 ? { lostCarries: lost } : {}) }
      st.carries = 0
      st.to = {}
      delete st.lost
      return out
    },
    noteLost(n) {
      if (!(Number.isInteger(n) && n >= 0)) throw new RangeError(`caregiver.noteLost: ${n} is not an integer >= 0`)
      const lost = (st.lost ?? 0) + n
      if (lost > 0) st.lost = lost
    },
    state() {
      st.rng = rng.getState()
      return structuredClone(st)
    },
  }
}
