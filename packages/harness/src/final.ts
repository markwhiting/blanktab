// The final log file of a life that ended without writing its own: a life that failed (session.ts) or an orphan the
// startup sweep deletes (orphans.ts). One builder, so both end on the record by the same rules.
import { clampSettings } from "@blanktab/learner"
import type { LearnerSettings, LearnerStats, Widths } from "@blanktab/learner"
import type { CaregiverState } from "./caregiver"
import type { Checkpoint } from "./checkpoint"
import { MAX_CURVE_POINTS, MILESTONE_SUITE_VERSION, type CurvePoint, type LogFile, type LogFileV1, type LogFileV2, type MetaBlock } from "./log"

export const DEFAULT_MAX_STEPS_PER_SEC = 30

/** Keeps every other point, always the newest, until at most MAX_CURVE_POINTS remain. */
export function thin(curve: CurvePoint[]): CurvePoint[] {
  let c = curve
  while (c.length > MAX_CURVE_POINTS) c = c.filter((_, i) => (c.length - 1 - i) % 2 === 0)
  return c
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const WIDTH_KEYS = ["conv1", "conv2", "enc", "hidden"] as const

/** A learner's birth facts. */
export type Birth = { seed: number; widths: Widths; settings: LearnerSettings }

/** A checkpointed learner's birth facts, from its state. RangeError if they are missing or malformed. */
export function birthOf(state: unknown, id: string): Birth {
  const fail = (what: string): never => { throw new RangeError(`learner ${id}: the checkpoint's learnerState ${what}`) }
  if (!isRecord(state)) return fail("is not an object")
  const { seed, widths, settings } = state
  if (!(Number.isInteger(seed) && (seed as number) >= 0 && (seed as number) < 2 ** 32)) fail("has no uint32 seed")
  if (!isRecord(widths) || !WIDTH_KEYS.every(k => Number.isInteger(widths[k]) && (widths[k] as number) >= 1)) fail("has malformed widths")
  if (!isRecord(settings)) fail("has no settings")
  let clamped: LearnerSettings
  try {
    clamped = clampSettings(settings as unknown as LearnerSettings)
  } catch (e) {
    return fail(`has bad settings (${e instanceof Error ? e.message : String(e)})`)
  }
  const w = widths as Record<string, number>
  return { seed: seed as number, widths: { conv1: w.conv1!, conv2: w.conv2!, enc: w.enc!, hidden: w.hidden! }, settings: clamped }
}

/** What a final file is labelled with that its checkpoint and latest file may not say. */
export interface FinalContext {
  learnerId: string
  now: () => Date
  /** The host writing the file (this tab's). */
  hostSig: LogFileV1["hostSig"]
  /** Used only when neither the checkpoint nor the latest file names the code the life was lived on. */
  codeSha: string
  maxStepsPerSec: number
  /** The knowledge in force when the life ended, and the birth fact. */
  knowledgeSha: string
  sessionsSeenAtBirth: number
}

/** The life's own counters, from a running session (an orphan has none). */
export interface Lived {
  steps: number
  curve: CurvePoint[]
  restores: number
  lostCU: number
  lostSkips: number
  stats: Pick<LearnerStats, "cu" | "skippedUpdates" | "evalCU">
}

/**
 * The final log of a life that ended without its own: from its checkpoint (if usable), else its latest v2 file, else
 * what it lived; null if nothing to record (a life that took no step and wrote no log, or whose latest file is final).
 *
 * - From the checkpoint, when its seq follows the latest file and its birth facts are known (`birth`, else from its
 *   learnerState: a state of another version may not give them): its steps, CU, curve tail and restores. lostCU adds
 *   the CU lived since it (`lived`), skippedUpdates those since it. Its checkup tail goes in under the suite it was
 *   recorded in (a tail without evalCU throws: it is never dropped). Its caregiver's carries and `to`, with
 *   `lostCarries` = `carriesSinceSave` (the carries made since it, thrown away with the life) + the checkpoint's own
 *   `lost`. An orphan's carries since its last save are unknown (its tab is gone): only the checkpoint's `lost` counts.
 * - Else from the latest v2 file: its counters, not its checkups or carries (those are on the record already), an
 *   empty curve.
 * - Else, with no log file yet, from what it lived (with `birth`).
 * Counters never go below the latest file's. Labelled with the code the life was lived on: the checkpoint's, else the
 * latest file's, else `ctx.codeSha`.
 *
 * Its meta block (Plan 05a) is the checkpoint's, else the latest file's (without that file's `filters`, `heldOut` or
 * `end`), with `end` set to `o.end` when given. A life without a block, or `o.noMeta`, gets none (and so no `end`).
 */
export function finalLogOf(o: {
  ctx: FinalContext
  cp?: Checkpoint<unknown> | undefined
  latest?: LogFile | undefined
  lived?: Lived | undefined
  /** The life's birth facts when the caller knows them (a newborn's are its config's); else taken from cp.learnerState. */
  birth?: Birth | undefined
  /** The caregiver's carries since `cp` was saved (a live session counts them; an orphan's are unknown: 0). */
  carriesSinceSave?: number
  /** Why the life ended (meta.end): "failed" from the session, "orphan" from the startup sweep. */
  end?: "failed" | "orphan"
  /** No meta block (a run whose block was not what it ran writes none, its final file included). */
  noMeta?: boolean
}): LogFileV2 | null {
  const { ctx, cp, latest: last, lived } = o
  if (last?.final) return null
  const id = ctx.learnerId
  const nextSeq = (last?.seq ?? -1) + 1
  const prior = last?.schemaVersion === 2 ? last : undefined
  // A life that took no step and wrote no log file has nothing to put on the record.
  if (!last && !(Math.max(cp?.steps ?? 0, lived?.steps ?? 0) > 0)) return null
  let birth = o.birth
  if (!birth && cp) {
    try {
      birth = birthOf(cp.learnerState, id)
    } catch {
      birth = undefined // a state of another version, or a stub: its birth facts are not known
    }
  }
  const metaFrom = o.noMeta ? undefined : cp?.meta ?? prior?.meta
  let meta: MetaBlock | undefined
  if (metaFrom) {
    const { filters: _f, heldOut: _h, end: _e, ...constant } = structuredClone(metaFrom)
    void _f
    void _h
    void _e
    meta = { ...constant, ...(o.end ? { end: o.end } : {}) }
  }
  const metaField = meta ? { meta } : {}
  const base = {
    schemaVersion: 2 as const, learnerId: id, seq: nextSeq, codeSha: cp?.codeSha ?? prior?.codeSha ?? ctx.codeSha, writtenAt: ctx.now().toISOString(), hostSig: { ...ctx.hostSig },
    final: true, condition: "A" as const, ...metaField,
  }
  const birthFields = (b: Birth, care?: CaregiverState) => ({
    settings: { ...b.settings, maxStepsPerSec: ctx.maxStepsPerSec, seed: b.seed, ...(care ? care.settings : {}) }, widths: { ...b.widths },
    knowledgeSha: ctx.knowledgeSha, sessionsSeenAtBirth: ctx.sessionsSeenAtBirth,
  })
  // Never below the latest file's counters (they never decrease across seq).
  const atLeast = (f: LogFileV2): LogFileV2 => !prior ? f : {
    ...f, stepsTotal: Math.max(f.stepsTotal, prior.stepsTotal), cu: Math.max(f.cu, prior.cu), restores: Math.max(f.restores, prior.restores),
    lostCU: Math.max(f.lostCU, prior.lostCU), skippedUpdates: Math.max(f.skippedUpdates, prior.skippedUpdates),
    // Never from a number back to absent: a checkpoint without evalCU (an older or stub state) keeps the last file's.
    ...(f.evalCU !== undefined || prior.evalCU !== undefined ? { evalCU: Math.max(f.evalCU ?? 0, prior.evalCU ?? 0) } : {}),
  }
  /** evalCU, when the stats have it (a learner state from before the held-out check, or a stub, may not). */
  const evalOf = (v: unknown) => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? { evalCU: v } : {})
  if (cp && birth && cp.seq === nextSeq) {
    const st = (cp.learnerState as { stats?: Partial<LearnerStats> } | null)?.stats
    const savedSkips = typeof st?.skippedUpdates === "number" ? st.skippedUpdates : 0
    // The scoring done since the checkpoint is thrown away with the life (it is not training: not in lostCU).
    const f = atLeast({
      ...evalOf(st?.evalCU),
      ...base, ...birthFields(birth, cp.caregiver), stepsTotal: cp.steps, cu: cp.cu, curve: thin([...cp.curveTail]), restores: cp.restores,
      // The CU lived since the checkpoint is thrown away with the life: on the record, as a restore's is.
      lostCU: cp.lostCU + Math.max(0, (lived?.stats.cu ?? cp.cu) - cp.cu),
      skippedUpdates: cp.lostSkips + Math.max(savedSkips, lived?.stats.skippedUpdates ?? 0),
    })
    // The checkpoint's checkups not yet logged, under the suite they were recorded in, billed inside the file's evalCU.
    // A checkpoint with a tail always has the learner's evalCU (checkCheckpoint), so a tail is never dropped here.
    const ck = cp.checkups
    if (ck && ck.tail.length > 0) {
      if (f.evalCU === undefined) throw new Error("final log: a checkup tail without evalCU")
      f.checkups = ck.tail.map(c => structuredClone(c))
      f.milestoneSuiteVersion = ck.suiteVersion ?? 1
    } else if (ck && f.evalCU !== undefined) {
      f.checkups = []
      f.milestoneSuiteVersion = MILESTONE_SUITE_VERSION
    }
    if (cp.caregiver) {
      // The carries made since the checkpoint are thrown away with the life, as are those its restores threw away.
      const lost = (o.carriesSinceSave ?? 0) + (cp.caregiver.lost ?? 0)
      f.caregiver = { carries: cp.caregiver.carries, to: { ...cp.caregiver.to }, safetyNet: cp.caregiver.safetyNet, ...(lost > 0 ? { lostCarries: lost } : {}) }
    }
    return f
  }
  if (prior) {
    // Its counters, not its checkups or carries: those are on the record already, once.
    const { checkups: _c, caregiver: _g, meta: _m, ...counters } = prior
    void _c
    void _g
    void _m
    return { ...counters, ...base, knowledgeSha: prior.knowledgeSha, sessionsSeenAtBirth: prior.sessionsSeenAtBirth, curve: [] }
  }
  if (!last && birth && lived && lived.steps > 0) {
    return {
      ...evalOf(lived.stats.evalCU),
      ...base, ...birthFields(birth), stepsTotal: lived.steps, cu: lived.stats.cu, curve: thin([...lived.curve]), restores: lived.restores,
      lostCU: lived.lostCU, skippedUpdates: lived.stats.skippedUpdates + lived.lostSkips,
    }
  }
  return null
}
