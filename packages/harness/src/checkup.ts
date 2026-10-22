// Checkups (spec §8.3, decision 0016): the milestone suite, each milestone in its own fresh variant world, driven by
// the learner's frozen run. A frozen run changes nothing in the learner but stats.evalCU, so there is nothing to
// restore afterwards. Everything a milestone reads (state, trace) is privileged and harness only.
import type { TrainableLearner } from "@blanktab/learner"
import { hashSeed, type CanvasLike } from "@blanktab/world"
import type { CheckupResult } from "./log"
import { MILESTONES_V1, recordRun, type Milestone } from "./milestones"

/** hashSeed(0, `${learnerId}/${level}/${milestoneId}`): a variant this learner has never seen. */
export function checkupSeed(learnerId: string, level: number, milestoneId: string): number {
  return hashSeed(0, `${learnerId}/${level}/${milestoneId}`)
}

export interface CheckupDeps {
  learnerId: string
  /** The log-schedule CU threshold this checkup is for. */
  level: number
  /** The learner's steps when it runs. */
  steps: number
  learner: Pick<TrainableLearner, "frozenRun" | "stats">
  /** A fresh canvas for each variant world (and for M5's reference). */
  canvas(): CanvasLike
  suite?: readonly Milestone[] // default MILESTONES_V1
  /** Errors the session handles itself (e.g. a lost device): rethrown after cleanup, not recorded. */
  recoverable?: (e: unknown) => boolean
  /** Checked before each milestone and after the last: true abandons the checkup. */
  shouldStop?: () => boolean
  /** Awaited after every frozen step (the session's pacer); without it, recordRun yields (setTimeout 0) every YIELD_EVERY steps. */
  yieldNow?: () => Promise<void>
}

export type CheckupOutcome = { status: "done"; result: CheckupResult } | { status: "abandoned" }

/**
 * Runs every milestone of the suite in order. Milestone m acts in m.makeVariant(checkupSeed(learnerId, level, m.id))
 * through a frozen run seeded hashSeed(that seed, "policy"), for m.budgetSteps steps. A milestone that throws (not
 * recoverable) is not passed, its measure is null and the result has `error: true`; the rest still run. A recoverable
 * error is rethrown once the open run is disposed. A halt (shouldStop) abandons the checkup: no result. `evalCU` is the
 * learner's stats().evalCU delta over the whole checkup.
 */
export async function runCheckup(d: CheckupDeps): Promise<CheckupOutcome> {
  const suite = d.suite ?? MILESTONES_V1
  const evalCU0 = d.learner.stats().evalCU
  const passed: string[] = []
  const measures: Record<string, number | null> = {}
  let error = false
  for (const m of suite) {
    if (d.shouldStop?.()) return { status: "abandoned" }
    const seed = checkupSeed(d.learnerId, d.level, m.id)
    let run: ReturnType<typeof d.learner.frozenRun> | null = null
    try {
      const world = m.makeVariant(seed, d.canvas())
      const reference = m.reference?.(seed, d.canvas())
      const r = (run = d.learner.frozenRun({ seed: hashSeed(seed, "policy"), measureLoss: m.measuresLoss }))
      const trace = await recordRun(world, o => r.step(o), m.budgetSteps, d.yieldNow)
      const stats = r.result()
      const value = m.measure(trace, stats, reference)
      const pass = m.passes(trace, stats, reference)
      measures[m.id] = Number.isFinite(value) ? value : null
      if (pass) passed.push(m.id)
    } catch (e) {
      if (d.recoverable?.(e)) throw e
      measures[m.id] = null
      error = true
    } finally {
      run?.dispose()
    }
  }
  if (d.shouldStop?.()) return { status: "abandoned" }
  const result: CheckupResult = { level: d.level, steps: d.steps, passed, measures, evalCU: d.learner.stats().evalCU - evalCU0 }
  if (error) result.error = true
  return { status: "done", result }
}
