// A learner's life: steps in the world, measured in compute units (CU), with sleeps, checkpoints and log files (v2).
import type { Observation, WorldView } from "@blanktab/interface"
import { clampSettings } from "@blanktab/learner"
import type { LearnerSettings, LearnerState, TrainableLearner, Widths } from "@blanktab/learner"
import { deleteCheckpoint, loadCheckpoint, saveCheckpoint, type Checkpoint } from "./checkpoint"
import { MAX_CURVE_POINTS, type CurvePoint, type LogFileV2 } from "./log"
import { openQueue } from "./queue"

/** 2^10 CU. */
export const DEFAULT_LIFETIME_CU = 1024
export const DEFAULT_SLEEP_INTERVAL = 250
export const DEFAULT_MAX_STEPS_PER_SEC = 30
/** This many skipped updates in a row (a learner gone non-finite) restore the last checkpoint. */
export const RESTORE_AFTER_SKIPS = 20
/** Restores from one checkpoint before the life is given up as failing (no new checkpoint in between). */
export const MAX_RESTORES_PER_CHECKPOINT = 3

export interface SessionConfig {
  learnerId: string
  codeSha: string
  knowledgeSha: string
  sessionsSeenAtBirth: number
  hostSig: LogFileV2["hostSig"]
  /** Birth facts: a resumed learner's checkpoint must match them. */
  seed: number
  widths: Widths
  /** Clamped (clampSettings) before use; the log records the clamped values. */
  learnerSettings: LearnerSettings
  /** Default DEFAULT_LIFETIME_CU. */
  lifetimeCU?: number
  /** Steps between sleeps. Default DEFAULT_SLEEP_INTERVAL. */
  sleepInterval?: number
  /** Default DEFAULT_MAX_STEPS_PER_SEC. */
  maxStepsPerSec?: number
}

/** What the session needs of a learner (a TrainableLearner has it). */
export type SessionLearner = Pick<TrainableLearner, "step" | "stats" | "sleep" | "save" | "dispose">
/** How the session asks for a learner: newborn, or from a checkpoint's state (createLearner's options minus the backend). */
export type LearnerInit = { seed: number; widths: Widths; settings: LearnerSettings } | { state: LearnerState }

/** Milliseconds. */
export interface Clock { now(): number; sleep(ms: number): Promise<void> }
const realClock: Clock = { now: () => performance.now(), sleep: ms => new Promise(r => setTimeout(r, ms)) }

export type SessionEvent =
  | { type: "resumed"; steps: number }
  /** After RESTORE_AFTER_SKIPS skipped updates in a row at `fromSteps`, the learner went back to its checkpoint at `toSteps`. */
  | { type: "restored"; fromSteps: number; toSteps: number }

export interface SessionDeps {
  /** drainTrace, if the world has one, is called after every action (nothing reads the trace yet). */
  world: WorldView & { drainTrace?(): unknown }
  createLearner(init: LearnerInit): Promise<SessionLearner>
  /** Holds the outbox and the checkpoints (one database, so a log and its checkpoint commit together). */
  idb: IDBFactory
  now: () => Date
  /** For the speed cap. Default: performance.now and setTimeout. */
  clock?: Clock
  /** Awaited after every step. */
  onTick?: (s: { steps: number; cu: number; obs: Observation }) => void | Promise<void>
  onEvent?: (e: SessionEvent) => void
}

export interface SessionResult {
  steps: number
  cu: number
  /** The next seq (= the number of log files this learner has written). */
  seq: number
  /** The checkpoint's step count when the life was resumed, else null. */
  resumedFrom: number | null
  /** The learner's last log file was already final: nothing ran (and any leftover checkpoint was removed). */
  alreadyFinal: boolean
}

/**
 * The speed cap: awaited before every step, it holds consecutive steps at least 1000 / maxStepsPerSec ms apart
 * (never more than maxStepsPerSec a second, even after a stall). It always yields to the event loop.
 */
export function createPacer(maxStepsPerSec: number, clock: Clock = realClock): () => Promise<void> {
  const interval = 1000 / maxStepsPerSec
  let last = Number.NEGATIVE_INFINITY
  return async () => {
    await clock.sleep(Math.max(0, last + interval - clock.now()))
    last = clock.now()
  }
}

/** The smallest power of two above `cu` (the next log crossing), at least 1 = 2^0. */
export function nextCrossing(cu: number): number {
  let t = 1
  while (t <= cu) t *= 2
  return t
}

/** Keeps every other point, always the newest, until at most MAX_CURVE_POINTS remain. */
function thin(curve: CurvePoint[]): CurvePoint[] {
  let c = curve
  while (c.length > MAX_CURVE_POINTS) c = c.filter((_, i) => (c.length - 1 - i) % 2 === 0)
  return c
}

const sameSettings = (a: LearnerSettings, b: LearnerSettings) =>
  (Object.keys(a) as (keyof LearnerSettings)[]).every(k => a[k] === b[k]) && Object.keys(a).length === Object.keys(b).length

/**
 * Lives one learner's life (or the rest of it) until it has used `lifetimeCU` compute units.
 *
 * - Every `sleepInterval` steps the learner sleeps; a curve point (predLoss vs the copy-last baseline) is added if
 *   both are finite; then a checkpoint is saved.
 * - A log file (v2) is queued each time CU crosses the next 2^k (k = 0, 1, ...) and at the end of life, carrying the
 *   curve points since the previous file. Each one is queued in one transaction with the checkpoint of that moment,
 *   so the checkpoint is never behind the latest log file: resuming never reuses a seq or goes back in steps or CU.
 * - No checkpoint is saved (and so no non-final log is written) while the learner has skipped updates outstanding;
 *   after RESTORE_AFTER_SKIPS in a row it is restored from the last checkpoint.
 * - The final log file removes the checkpoint in the same transaction.
 * - With a checkpoint for this learnerId the life resumes from it (its steps, CU, seq and curve); a learner with log
 *   files but no checkpoint cannot be resumed and throws.
 */
export async function runSession(cfg: SessionConfig, deps: SessionDeps): Promise<SessionResult> {
  const lifetimeCU = cfg.lifetimeCU ?? DEFAULT_LIFETIME_CU
  const sleepInterval = cfg.sleepInterval ?? DEFAULT_SLEEP_INTERVAL
  const maxStepsPerSec = cfg.maxStepsPerSec ?? DEFAULT_MAX_STEPS_PER_SEC
  if (!(Number.isFinite(lifetimeCU) && lifetimeCU > 0)) throw new RangeError(`lifetimeCU must be finite and > 0, got ${lifetimeCU}`)
  if (!(Number.isInteger(sleepInterval) && sleepInterval >= 1)) throw new RangeError(`sleepInterval must be an integer >= 1, got ${sleepInterval}`)
  if (!(Number.isFinite(maxStepsPerSec) && maxStepsPerSec > 0)) throw new RangeError(`maxStepsPerSec must be finite and > 0, got ${maxStepsPerSec}`)
  const settings = clampSettings(cfg.learnerSettings)
  const { learnerId: id } = cfg
  const { world, idb, now, onTick, onEvent } = deps
  const queue = await openQueue(idb)

  const latest = await queue.latest(id)
  if (latest?.final) {
    await deleteCheckpoint(idb, id)
    return { steps: latest.stepsTotal, cu: latest.schemaVersion === 2 ? latest.cu : 0, seq: latest.seq + 1, resumedFrom: null, alreadyFinal: true }
  }
  let cp = await loadCheckpoint(idb, id)
  if (!cp && latest) throw new Error(`learner ${id} has log files up to seq ${latest.seq} but no checkpoint: its life cannot be resumed`)
  if (cp) {
    if (latest && latest.seq >= cp.seq) throw new Error(`learner ${id}: the checkpoint (seq ${cp.seq}) is behind log file ${latest.seq}`)
    const s = cp.learnerState
    if (s.seed !== cfg.seed || (Object.keys(cfg.widths) as (keyof Widths)[]).some(k => s.widths?.[k] !== cfg.widths[k]) || !s.settings || !sameSettings(s.settings, settings)) {
      throw new Error(`learner ${id}: the checkpoint's seed, widths or settings differ from this session's`)
    }
  }

  let learner = await deps.createLearner(cp ? { state: cp.learnerState } : { seed: cfg.seed, widths: cfg.widths, settings })
  try {
    let steps = cp?.steps ?? 0
    let seq = cp?.seq ?? 0
    let curve: CurvePoint[] = cp ? [...cp.curveTail] : []
    let crossing = nextCrossing(learner.stats().cu)
    let restores = 0
    const resumedFrom = cp ? cp.steps : null
    if (resumedFrom !== null) onEvent?.({ type: "resumed", steps: resumedFrom })

    const checkpointNow = async (): Promise<Checkpoint> => ({
      learnerId: id, savedAt: now().toISOString(), steps, cu: learner.stats().cu, seq, curveTail: [...curve], learnerState: await learner.save(),
    })
    const logFile = (final: boolean): LogFileV2 => {
      const st = learner.stats()
      return {
        schemaVersion: 2, learnerId: id, seq, codeSha: cfg.codeSha, knowledgeSha: cfg.knowledgeSha, writtenAt: now().toISOString(),
        stepsTotal: steps, settings: { ...settings, maxStepsPerSec, seed: cfg.seed }, hostSig: { ...cfg.hostSig },
        sessionsSeenAtBirth: cfg.sessionsSeenAtBirth, final, cu: st.cu, condition: "A", widths: { ...cfg.widths }, curve: [...curve],
        skippedUpdates: st.skippedUpdates,
      }
    }
    const addPoint = () => {
      const { cu, predLoss, baselineLoss } = learner.stats()
      if (!Number.isFinite(predLoss) || !Number.isFinite(baselineLoss)) return // never NaN in a log
      if (curve.at(-1)?.steps === steps) return
      curve = thin([...curve, { cu, steps, predLoss, baselineLoss }])
    }
    const saved = () => { restores = 0 }

    // A newborn's first checkpoint, so there is always one to restore.
    if (!cp) {
      cp = await checkpointNow()
      await saveCheckpoint(idb, cp)
    }
    const pace = createPacer(maxStepsPerSec, deps.clock ?? realClock)

    while (learner.stats().cu < lifetimeCU) {
      await pace()
      const obs = world.observe()
      world.act(await learner.step(obs))
      world.drainTrace?.()
      steps++
      await onTick?.({ steps, cu: learner.stats().cu, obs })
      let checkpointDue = false
      if (steps % sleepInterval === 0) {
        await learner.sleep()
        addPoint()
        checkpointDue = true
      }
      const st = learner.stats()
      if (st.consecutiveSkips >= RESTORE_AFTER_SKIPS) {
        if (++restores > MAX_RESTORES_PER_CHECKPOINT) {
          throw new Error(`learner ${id}: still skipping updates after ${MAX_RESTORES_PER_CHECKPOINT} restores from the checkpoint at step ${cp.steps}`)
        }
        const back = await loadCheckpoint(idb, id)
        if (!back) throw new Error(`learner ${id}: no checkpoint to restore`)
        // Every log file carries its checkpoint, so nothing was logged after this one: seq must not move.
        if (back.seq !== seq) throw new Error(`learner ${id}: the checkpoint's seq ${back.seq} is not the next seq ${seq}`)
        learner.dispose()
        learner = await deps.createLearner({ state: back.learnerState })
        onEvent?.({ type: "restored", fromSteps: steps, toSteps: back.steps })
        steps = back.steps
        curve = [...back.curveTail]
        crossing = nextCrossing(learner.stats().cu)
        cp = back
        continue
      }
      if (st.cu >= lifetimeCU) break
      if (st.consecutiveSkips > 0) continue // no checkpoint now; a crossing waits until updates apply again
      if (st.cu >= crossing) {
        const f = logFile(false)
        seq++
        curve = []
        crossing = nextCrossing(st.cu)
        cp = await checkpointNow()
        await queue.put(f, { save: cp })
        saved()
      } else if (checkpointDue) {
        cp = await checkpointNow()
        await saveCheckpoint(idb, cp)
        saved()
      }
    }

    addPoint()
    await queue.put(logFile(true), { remove: id })
    seq++
    return { steps, cu: learner.stats().cu, seq, resumedFrom, alreadyFinal: false }
  } finally {
    learner.dispose()
  }
}
